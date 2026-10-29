import sys
import unittest
import concurrent.futures
import threading
import types
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks/pronunciation-assessment/src")]
from mdd.canonical_fix import CanonicalBuilder
from mdd.word_judge import KoCanonical
from mdd.observed_reference_adapter import ObservedReferenceAdapter


class AdapterTest(unittest.TestCase):
    def app(self, barrier=None):
        builder = CanonicalBuilder(KoCanonical())
        def forward(decoder, wavs):
            if barrier:
                barrier.wait(timeout=5)
            return torch.tensor([[[float(wavs[0][0])]]]), [1]
        app = types.SimpleNamespace(S={"cb": builder, "fwd": forward, "sz": None})
        def analyze(wav, text, country):
            app.S["fwd"](None, [wav])
            canonical = builder.fixed(text)
            sites = [dict(i=i, canon=p, pred="ㅆ" if p == "ㅅ" else p) for i, p in enumerate(canonical)]
            return ({"pred_jamo": [s["pred"] for s in sites]}, {}, {"canon": canonical, "sites": sites})
        app.analyze = analyze
        return app

    def test_concurrent_requests_use_their_own_captured_posterior(self):
        app = self.app(threading.Barrier(2))
        original = app.S["fwd"]
        def rescore(app, before, wav, lp, canonical, country):
            self.assertEqual(float(wav[0]), float(lp[0, 0]))
            sites = [dict(i=i, canon=p, pred=p) for i, p in enumerate(canonical)]
            return ({"pred_jamo": [s["pred"] for s in sites]}, {}, {"canon": canonical, "sites": sites, "value": float(lp[0, 0])})
        adapter = ObservedReferenceAdapter(app, rescore)
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(adapter.analyze_with_trace, np.array([v]), "여덟 시", "") for v in (1., 2.)]
                result = [f.result() for f in futures]
            self.assertEqual([x[0][2]["value"] for x in result], [1., 2.])
            self.assertTrue(all(x[1]["forward_calls"] == 1 for x in result))
            self.assertIsNone(adapter.context.get())
        finally:
            adapter.close()
        self.assertIs(app.S["fwd"], original)

    def test_context_is_reset_if_original_analysis_fails(self):
        app = self.app()
        def fail(*args):
            raise RuntimeError("failed request")
        app.analyze = fail
        adapter = ObservedReferenceAdapter(app)
        try:
            with self.assertRaises(RuntimeError):
                adapter.analyze(np.array([1.]), "여덟 시", "")
            self.assertIsNone(adapter.context.get())
        finally:
            adapter.close()

    def test_optional_detector_is_refused_before_changing_callbacks(self):
        app = self.app()
        original = app.S["fwd"]
        app.S["sz"] = object()
        with self.assertRaises(ValueError):
            ObservedReferenceAdapter(app)
        self.assertIs(app.S["fwd"], original)


if __name__ == "__main__":
    unittest.main()
