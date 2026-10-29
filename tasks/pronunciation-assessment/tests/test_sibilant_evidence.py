"""Signal-grounded checks and isolated HTTP tests for the listening workflow."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    if ROOT.parent == ROOT:
        raise RuntimeError("repository root not found")
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tasks/pronunciation-assessment/src")]

import importlib.util
import hashlib
import json
import os
import socket
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
import wave

import numpy as np
from scipy.signal import lfilter
from mdd.sibilant_acoustics import frame_tracks, harmonic_amplitudes, validate_landmarks, partition_balance_tracks


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class NativeReferenceLabels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.indexer = module("seoul_sibilants_test", ROOT / "tasks/pronunciation-assessment/scripts/data/index_seoul_sibilants.py")

    def test_duplicate_tiers_quotes_and_truncated_input(self):
        text = ('File type = "ooTextFile"\nObject class = "TextGrid"\nsize = 2\nitem []:\n'
                '    item [1]:\n        class = "IntervalTier"\n        name = "same"\n'
                '        intervals: size = 1\n        intervals [1]:\n'
                '            xmin = 0\n            xmax = 1\n            text = "a""b"\n'
                '    item [2]:\n        class = "IntervalTier"\n        name = "same"\n'
                '        intervals: size = 1\n        intervals [1]:\n'
                '            xmin = 0\n            xmax = 1\n            text = "다른 층"\n')
        parsed = self.indexer.parse_textgrid(text.encode("utf-16"))
        self.assertEqual([t["name"] for t in parsed], ["same", "same"])
        self.assertEqual(parsed[0]["intervals"][0][2], 'a"b')
        self.assertEqual(parsed[1]["intervals"][0][2], "다른 층")
        with self.assertRaises(ValueError):
            self.indexer.parse_textgrid(text.replace('text = "다른 층"', '').encode())

    def test_context_trimming_preserves_onsets_and_excludes_private_region(self):
        tiers = [{"name": n, "intervals": [(0., 1., "safe")]} for n in self.indexer.EXPECTED_TIERS]
        tiers[0]["intervals"] = [(0., .1, "<PRIVATE.INFO>"), (.1, .2, "s0"),
                                  (.2, .3, "ii"), (.3, .4, "<SIL>"), (.4, 1., "aa")]
        for t in tiers[1:]:
            t["intervals"] = [(0., .1, "<PRIVATE.INFO>"), (.1, .3, "시"), (.3, 1., "다음")]
        rows, rejected = self.indexer.collect_cv(tiers, "s01m16f1")
        self.assertEqual(len(rows), 1)
        self.assertFalse(rejected)
        self.assertEqual(rows[0]["context_interval"], [.1, .3])
        self.assertTrue(rows[0]["at_labeled_utterance_start"])
        self.assertTrue(rows[0]["context_trimmed_at_marked_interval"])
        self.assertEqual(rows[0]["previous_phone"], "<PRIVATE.INFO>")
        self.assertEqual(rows[0]["word_pronounced_hangul"], "시")


class AcousticMeasurements(unittest.TestCase):
    def test_silence_does_not_become_periodic_or_harmonic(self):
        x = np.zeros(16000)
        self.assertTrue(np.isnan(frame_tracks(x)["periodicity"]).all())
        self.assertIsNone(harmonic_amplitudes(x, 16000, .5, 200))
        self.assertIsNone(harmonic_amplitudes(np.ones(16000), 16000, .5, float("nan")))

    def test_known_harmonic_amplitude_ratio(self):
        t = np.arange(16000) / 16000
        x = .1 * (np.sin(2 * np.pi * 200 * t) + .5 * np.sin(2 * np.pi * 400 * t))
        r = harmonic_amplitudes(x, 16000, .5, 200)
        self.assertAlmostEqual(r["h1h2_raw"], 20 * np.log10(2), delta=.15)
        self.assertIsNone(r["h1h2_corrected"])

    def test_overlapping_harmonic_windows_do_not_mutate_pcm(self):
        t = np.arange(16000) / 16000
        x = .2 + .1 * np.sin(2 * np.pi * 173 * t)
        before = x.copy()
        for center in (.500, .505, .510):
            harmonic_amplitudes(x, 16000, center, 173)
        np.testing.assert_array_equal(x, before)

    def test_complete_spectral_partition_includes_middle_band(self):
        t = np.arange(16000) / 16000
        # 2500Hz was absent from the old low/high ratio. It must affect a
        # 3kHz partition, with the same value after an overall gain change.
        x = .1 * (np.sin(2 * np.pi * 2500 * t) + .5 * np.sin(2 * np.pi * 5000 * t))
        a = partition_balance_tracks(x)["upper_lower_db_3000"]
        b = partition_balance_tracks(x * .1)["upper_lower_db_3000"]
        self.assertAlmostEqual(float(np.median(a)), -20 * np.log10(2), delta=.02)
        np.testing.assert_allclose(a, b, atol=1e-8)
        self.assertTrue(np.isnan(partition_balance_tracks(np.zeros(16000))["upper_lower_db_3000"]).all())
        self.assertTrue(np.isnan(frame_tracks(np.zeros(16000))["centroid_hz"]).all())

    def test_known_energy_transition_preserves_time_axis(self):
        sr = 16000
        t = np.arange(sr) / sr
        x = np.where(t < .5, .1 * np.sin(2 * np.pi * 5000 * t), .1 * np.sin(2 * np.pi * 300 * t))
        r = frame_tracks(x)
        before = (r["times"] > .1) & (r["times"] < .45)
        after = (r["times"] > .55) & (r["times"] < .9)
        self.assertGreater(np.median(r["high_low_db"][before]), 30)
        self.assertLess(np.median(r["high_low_db"][after]), -30)
        self.assertAlmostEqual(r["times"][0], 199.5 / sr)

    def test_formant_correction_recovers_known_source_ratio(self):
        root = os.environ.get("SIBILANT_OPENSAUCE_ROOT")
        if not root:
            self.skipTest("set SIBILANT_OPENSAUCE_ROOT for the externally versioned correction check")
        correction = module("opensauce_correction_test", Path(root) / "opensauce/harmonics.py").correction_iseli_i
        sr, f0 = 16000, 200.
        t = np.arange(sr) / sr
        source = .03 * sum(np.sin(2 * np.pi * k * f0 * t) / k for k in range(1, 21))
        formants, bandwidths = [700., 1700.], [80., 120.]
        filtered = source.copy()
        for f, b in zip(formants, bandwidths):
            radius = np.exp(-np.pi * b / sr)
            denominator = [1., -2 * radius * np.cos(2 * np.pi * f / sr), radius ** 2]
            filtered = lfilter([sum(denominator)], denominator, filtered)
        actual = harmonic_amplitudes(filtered, sr, .5, f0, formants, bandwidths, correction)
        self.assertAlmostEqual(actual["h1h2_corrected"], 20 * np.log10(2), delta=.25)
        self.assertAlmostEqual(actual["h1a1_corrected"], 20 * np.log10(actual["a1_harmonic"]), delta=.4)
        self.assertGreater(abs(actual["h1a1_raw"] - actual["h1a1_corrected"]), 5)
        # High vowels can place F1 near H1: H1-A1 then contains no independent contrast.
        high = harmonic_amplitudes(source, sr, .5, f0, [240., 1800.], [70., 100.], correction)
        self.assertEqual(high["formant_quality"], "h1_equals_a1")
        self.assertIsNone(high["h1a1_corrected"])

    def test_voiced_frication_is_allowed_and_provenance_required(self):
        r = {"frication_start": .2, "frication_end": .4, "voicing_first": .35,
             "voicing_stable": .43, "actor_type": "synthetic_test", "transition_kind": "overlap"}
        self.assertEqual(validate_landmarks(r, 1.)["voicing_first"], .35)
        with self.assertRaises(ValueError):
            validate_landmarks({**r, "actor_type": "human"}, 1.)
        with self.assertRaises(ValueError):
            validate_landmarks({**r, "voicing_stable": .1}, 1.)


class BlindListeningHTTP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import uvicorn
        cls.tmp = tempfile.TemporaryDirectory(prefix="sibilant-review-synthetic-test-")
        cls.root = Path(cls.tmp.name)
        packet = cls.root / "packet"
        packet.mkdir()
        (packet / "audio").mkdir()
        (packet / "items.private.jsonl").write_text(json.dumps({"id": "hidden-item", "prompt": "PRIVATE_PROMPT", "actual": "PRIVATE_LABEL"}) + "\n")
        trials = {w: [{"trial": w + "-trial", "item": "hidden-item"}] for w in "ABC"}
        (packet / "trials.private.json").write_text(json.dumps(trials))
        for mode in ("local", "context"):
            with wave.open(str(packet / "audio" / ("hidden-item_" + mode + ".wav")), "wb") as f:
                f.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                f.writeframes(np.zeros(1600, np.int16).tobytes())
        cls.api = module("sibilant_review_test_app", ROOT / "deploy/sibilant-review/app.py")
        cls.keys = {w: w + "-synthetic-test-key" for w in "ABC"}
        app = cls.api.make_app(packet, cls.keys, cls.root / "synthetic_responses")
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        cls.url = "http://127.0.0.1:" + str(sock.getsockname()[1])
        cls.server = uvicorn.Server(uvicorn.Config(app, log_level="error", access_log=False))
        cls.thread = threading.Thread(target=cls.server.run, kwargs={"sockets": [sock]}, daemon=True)
        cls.thread.start()
        deadline = time.monotonic() + 5
        while not cls.server.started and time.monotonic() < deadline:
            time.sleep(.02)
        if not cls.server.started:
            raise RuntimeError("isolated test server did not start")

    @classmethod
    def tearDownClass(cls):
        cls.server.should_exit = True
        cls.thread.join(timeout=5)
        cls.tmp.cleanup()

    def request(self, path, who=None, data=None):
        headers = {"x-review-key": self.keys[who]} if who else {}
        if data is not None:
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(self.url + path, headers=headers,
                                     data=json.dumps(data).encode() if data is not None else None)
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, r.read(), dict(r.headers)

    def test_private_labels_paths_and_other_trials_are_hidden(self):
        status, raw, headers = self.request("/trials", "B")
        self.assertEqual(status, 200)
        for secret in (b"PRIVATE_PROMPT", b"PRIVATE_LABEL", b"hidden-item", b"A-trial", b"source_wav"):
            self.assertNotIn(secret, raw)
        self.assertEqual(headers["cache-control"], "no-store")
        with self.assertRaises(urllib.error.HTTPError) as e:
            self.request("/trials")
        self.assertEqual(e.exception.code, 401)
        with self.assertRaises(urllib.error.HTTPError) as e:
            self.request("/audio/A-trial/local", "B")
        self.assertEqual(e.exception.code, 404)
        self.assertTrue(self.request("/audio/B-trial/local", "B")[1].startswith(b"RIFF"))

    def test_submit_resume_and_no_cross_reviewer_answer(self):
        data = {"trial": "A-trial", "choice": "uncertain", "confidence": 1,
                "native_korean": True, "plays": {"local": 1, "context": 0}, "elapsed_seconds": 4}
        self.assertEqual(self.request("/label", "A", data)[0], 200)
        restored = json.loads(self.request("/trials", "A")[1])["trials"][0]
        self.assertEqual(restored["answer"]["choice"], "uncertain")
        self.assertIsNone(json.loads(self.request("/trials", "B")[1])["trials"][0]["answer"])
        self.request("/label", "A", {**data, "choice": "ss", "confidence": 3})
        saved = self.api.read_jsonl(self.root / "synthetic_responses/labels_A.jsonl")
        self.assertEqual(len(saved), 2)
        self.assertEqual(saved[-1]["choice"], "ss")
        for invalid in ({**data, "plays": {}}, {**data, "confidence": True}, {**data, "native_korean": False}, {**data, "trial": "B-trial"}):
            with self.assertRaises(urllib.error.HTTPError) as e:
                self.request("/label", "A", invalid)
            self.assertEqual(e.exception.code, 400)
        self.assertEqual(len(self.api.read_jsonl(self.root / "synthetic_responses/labels_A.jsonl")), 2)


class ListeningConsensus(unittest.TestCase):
    def test_repeats_are_not_independent_listeners_and_missing_is_not_agreement(self):
        summary = module("listening_consensus_test", ROOT / "tasks/pronunciation-assessment/scripts/eval/summarize_sibilant_listening.py")
        with tempfile.TemporaryDirectory(prefix="sibilant-consensus-synthetic-test-") as tmp:
            p = Path(tmp)
            private = p / "items.private.jsonl"
            private.write_text(json.dumps({"id": "one", "actual": "ㅆ"}) + "\n")
            trials = {"A": [{"trial": "a1", "item": "one"}, {"trial": "a2", "item": "one"}],
                      "B": [{"trial": "b1", "item": "one"}]}
            (p / "trials.private.json").write_text(json.dumps(trials))
            sha = hashlib.sha256(private.read_bytes()).hexdigest()
            responses = p / "responses"
            responses.mkdir()
            empty = summary.summarize(p, responses)
            self.assertFalse(empty["all_items_have_two_ratings"])
            self.assertIsNone(empty["between_listener_agreement"]["A_B"]["agreement"])

            def write(who, rows):
                with (responses / ("labels_" + who + ".jsonl")).open("w") as f:
                    for trial, choice in rows:
                        rec = {"reviewer": who, "packet_sha256": sha, "trial": trial, "item": "one",
                               "actor_type": "human_self_declared", "choice": choice, "confidence": 3,
                               "native_korean": True, "plays": {"local": 1, "context": 0}}
                        f.write(json.dumps(rec) + "\n")
            write("A", [("a1", "ss"), ("a2", "ss")])
            single = summary.summarize(p, responses)
            self.assertEqual(single["within_listener_repeat_agreement"]["A"]["agreement"], 1.)
            self.assertEqual(single["items_with_at_least_two_first_pass_ratings"], 0)
            self.assertIsNone(single["records"][0]["consensus"])
            write("B", [("b1", "s")])
            disagreement = summary.summarize(p, responses)
            self.assertTrue(disagreement["all_items_have_two_ratings"])
            self.assertEqual(disagreement["records"][0]["status"], "uncertain_or_disagreement")
            write("B", [("b1", "ss")])
            agreed = summary.summarize(p, responses)
            self.assertEqual(agreed["TAG_comparable_consensus_items"], 1)
            self.assertTrue(agreed["records"][0]["TAG_agrees"])


class EvidenceAudits(unittest.TestCase):
    def test_bandwidth_change_moves_spectral_cues_without_changing_input(self):
        audit = module("bandwidth_audit_test", ROOT / "tasks/pronunciation-assessment/scripts/eval/audit_sibilant_bandwidth.py")
        t = np.arange(16000) / 16000
        x = .1 * (np.sin(2 * np.pi * 2000 * t) + np.sin(2 * np.pi * 7000 * t))
        before = x.copy()
        y = audit.roundtrip_8k(x)
        times, a = audit.tracks(x)
        other_times, b = audit.tracks(y)
        np.testing.assert_array_equal(x, before)
        np.testing.assert_array_equal(times, other_times)
        inside = (times > .1) & (times < .9)
        self.assertAlmostEqual(np.median(a["centroid_1_to_7p9khz_hz"][inside]), 4500, delta=1)
        self.assertAlmostEqual(np.median(b["centroid_1_to_7p9khz_hz"][inside]), 2000, delta=1)
        self.assertLess(np.median(b["balance_4000hz_db"][inside]), -45)
        low = .1 * np.sin(2 * np.pi * 2000 * t)
        _, default_filter = audit.tracks(audit.roundtrip_8k(low))
        _, strict_filter = audit.tracks(audit.roundtrip_8k(low, "strict"))
        # More stop-band rejection suppresses interpolation images; upsampling
        # cannot turn those images into newly observed speech information.
        self.assertLess(np.median(strict_filter["balance_4000hz_db"][inside]),
                        np.median(default_filter["balance_4000hz_db"][inside]) - 15)
        _, zero = audit.tracks(np.zeros(16000))
        self.assertTrue(all(np.isnan(v).all() for v in zero.values()))

    def test_pcm_constant_gap_preserves_original_time_and_does_not_label_error(self):
        audit = module("pcm_audit_test", ROOT / "tasks/pronunciation-assessment/scripts/eval/audit_sibilant_pcm.py")
        x = np.tile(np.array([100, -100], dtype=np.int16), 800)
        x[320:448] = 0
        runs = audit.constant_runs(x, 16000, offset=4.)
        self.assertEqual(len(runs), 1)
        self.assertAlmostEqual(runs[0]["start"], 4.020)
        self.assertAlmostEqual(runs[0]["end"], 4.028)
        self.assertEqual(runs[0]["samples"], 128)
        observation = audit.inspect_interval(x, 16000, 0., .1)
        self.assertEqual(observation["max_constant_run_ms"], 8.)
        self.assertNotIn("pronunciation_error", observation)

    def test_missing_landmark_is_not_replaced_by_search_range(self):
        comparison = module("landmark_comparison_test", ROOT / "tasks/pronunciation-assessment/scripts/eval/compare_sibilant_landmarks.py")
        row = {"frication_start": None, "frication_start_range": [.1, .2], "voicing_first": .3, "voicing_first_range": [.29, .31]}
        self.assertEqual(comparison.difference_ms(row, "frication_start", "voicing_first"), (None, None))
        row["frication_start"] = .15
        value, interval = comparison.difference_ms(row, "frication_start", "voicing_first")
        self.assertAlmostEqual(value, 150.)
        np.testing.assert_allclose(interval, [90., 210.])
        self.assertEqual(comparison.describe([None, 150.])["total"], 2)
        self.assertEqual(comparison.describe([None, 150.])["missing"], 1)


class ExpertBoundaryInputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reader = module("expert_boundary_summary_test", ROOT / "tasks/pronunciation-assessment/scripts/eval/summarize_sibilant_boundaries.py")
        cls.exporter = module("expert_boundary_export_test", ROOT / "tasks/pronunciation-assessment/scripts/data/export_sibilant_boundary_packet.py")

    def test_empty_and_uncertain_annotations_never_become_observed_points(self):
        r = self.reader
        empty = {k: None for k in r.LANDMARKS}
        result = r.observation(empty, None, "expert_A", 1., 4.)
        self.assertEqual(result["status"], "empty_template")
        self.assertEqual(result["actor_type"], "unattested")
        # Fictional notes, only in an isolated unit test.
        note = {"actor_type": "human_expert", "annotator": "expert_A",
                "target_association": "uncertain", "transition_kind": "uncertain",
                "landmark_status": {k: "uncertain" for k in r.LANDMARKS},
                "uncertain_ranges_relative_seconds": {"first_vowel_voicing": [.2, .4]}}
        result = r.observation(empty, note, "expert_A", 1., 4.)
        self.assertEqual(result["status"], "complete_review")
        self.assertIsNone(result["points_original_seconds"]["first_vowel_voicing"])
        self.assertEqual(result["uncertain_ranges_original_seconds"]["first_vowel_voicing"], [4.2, 4.4])
        with self.assertRaises(ValueError):
            r.observation({**empty, "first_vowel_voicing": .3}, note, "expert_A", 1., 4.)

    def test_praat_points_use_clip_time_and_allow_voiced_frication(self):
        try:
            import parselmouth
        except ImportError:
            self.skipTest("Parselmouth required for native TextGrid validation")
        r = self.reader
        values = dict(zip(r.LANDMARKS, [.1, .3, .25, .27, .30]))
        lines, name = [], None
        for line in self.exporter.template(1.).splitlines():
            if "name =" in line:
                name = line.split('"')[1]
            if "points: size = 0" in line:
                lines.extend(['        points: size = 1', '        points [1]:',
                              '            number = %.9f' % values[name], '            mark = ""'])
            else:
                lines.append(line)
        with tempfile.TemporaryDirectory(prefix="sibilant-expert-synthetic-test-") as tmp:
            p = Path(tmp) / "one.TextGrid"
            p.write_text("\n".join(lines) + "\n")
            points = r.read_points(p, 1.)
            self.assertEqual(points, values)
            result = r.observation(points, None, "expert_A", 1., 4.)
            self.assertEqual(result["status"], "unattested_points")
            self.assertEqual(result["points_original_seconds"]["first_vowel_voicing"], 4.25)
            with self.assertRaises(ValueError):
                r.read_points(p, .5)

    def test_annotation_audio_fingerprint_is_checked_before_reading_points(self):
        with tempfile.TemporaryDirectory(prefix="sibilant-expert-synthetic-test-") as tmp:
            p = Path(tmp)
            (p / "audio").mkdir()
            (p / "source_offsets.private.json").write_text(json.dumps([{"id": "one", "audio_sha256": "expected"}]))
            (p / "annotation_protocol.json").write_text(json.dumps({"items": 1}))
            (p / "audio/one.wav").write_bytes(b"changed input")
            with self.assertRaisesRegex(ValueError, "exported audio changed"):
                self.reader.summarize(p, ["expert_A"])


class CanonicalConsistency(unittest.TestCase):
    def test_ambiguous_and_deleted_targets_are_never_filled(self):
        root = ROOT / "tasks/pronunciation-assessment/scripts/eval"
        audit = module("canonical_consistency_test", root / "audit_sibilant_canonical.py")
        mapper = module("canonical_mapper_test", root / "eval_sibilant_directional.py").all_optimal_mapping
        # Equally good deletions of repeated s tokens must not become a label.
        ambiguous = audit.site_status(["ㅅ", "ㅅ", "ㅏ"], ["ㅅ", "ㅏ"], 0, mapper)
        self.assertEqual(ambiguous["status"], "ambiguous")
        self.assertTrue(ambiguous["deletion_possible"])
        self.assertIsNone(ambiguous["unique_target_phone"])
        deleted = audit.site_status(["ㅅ", "ㅏ"], ["ㅏ"], 0, mapper)
        self.assertEqual(deleted["status"], "deleted")
        self.assertIsNone(deleted["target_next_phone"])
        mismatch = audit.site_status(["ㅆ", "ㅗ"], ["ㅅ", "ㅗ"], 0, mapper)
        self.assertEqual(mismatch["status"], "disagrees")
        self.assertEqual(mismatch["unique_target_phone"], "ㅅ")


class RegionDependence(unittest.TestCase):
    def test_intervention_preserves_outside_pcm_and_tapers_edges(self):
        m = module("region_dependence_test", ROOT / "tasks/pronunciation-assessment/scripts/eval/probe_sibilant_region_dependence.py")
        x = np.linspace(-.8, .8, 16000, dtype=np.float32)
        before = x.copy()
        lo, hi = 4800, 5440
        y = m.intervene(x, .3, .34, "attenuate_20db")
        np.testing.assert_array_equal(x, before)
        np.testing.assert_array_equal(y[:lo + 1], x[:lo + 1])
        np.testing.assert_array_equal(y[hi - 1:], x[hi - 1:])
        np.testing.assert_allclose(y[lo + 80:hi - 80], x[lo + 80:hi - 80] * .1, atol=1e-8)
        half = m.intervene(x, .3, .34, "attenuate_20db", strength=.5)
        ratio = np.sum((half.astype(np.float64) - x) ** 2) / np.sum((y.astype(np.float64) - x) ** 2)
        self.assertAlmostEqual(ratio, .25, places=6)
        z = m.intervene(x, .3, .34, "reverse")
        np.testing.assert_array_equal(z[lo + 80:hi - 80], x[lo:hi][::-1][80:-80])
        with self.assertRaises(ValueError):
            m.intervene(x, -.01, .03, "reverse")

    def test_ctc_pair_difference_matches_enumerated_paths(self):
        import itertools
        m = module("region_dependence_ctc_test", ROOT / "tasks/pronunciation-assessment/scripts/eval/probe_sibilant_region_dependence.py")
        # Alphabet blank, s, ss, a. Enumerate all 4**5 CTC paths independently.
        p = np.array([[.1, .1, .1, .7], [.7, .1, .1, .1], [.1, .2, .6, .1],
                      [.7, .1, .1, .1], [.1, .1, .1, .7]])
        expected = {1: 0., 2: 0.}
        for path in itertools.product(range(4), repeat=5):
            decoded = [k for k, _ in itertools.groupby(path) if k != 0]
            for middle in expected:
                if decoded == [3, middle, 3]:
                    expected[middle] += float(np.prod([p[t, k] for t, k in enumerate(path)]))
        score = m.pair_scores(np.log(p), [3, 1, 3], 1, 1, 2, 0)
        self.assertAlmostEqual(score["logP_s"], np.log(expected[1]), places=5)
        self.assertAlmostEqual(score["logP_ss"], np.log(expected[2]), places=5)
        self.assertGreater(score["ss_minus_s"], 0)


if __name__ == "__main__":
    unittest.main()
