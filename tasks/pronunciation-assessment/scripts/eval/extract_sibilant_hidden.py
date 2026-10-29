"""Replay frozen service inputs; retain selected layer trajectories and verify parity."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    if ROOT.parent == ROOT:
        raise RuntimeError("repository root not found")
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tasks/pronunciation-assessment/src")]

import argparse
import hashlib
import json
import shutil
import time

import numpy as np
import torch

import eval_sibilant_directional as service
from mdd.sibilant_hidden import hidden_window


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0, help="smoke only; never train on limited extraction")
    args = parser.parse_args()
    protocol = json.loads(Path(args.protocol).read_text())
    for definition in protocol["cache_sources"].values():
        for path, digest in definition["sha256"].items():
            if service.sha(path) != digest:
                raise ValueError("cached baseline input changed: " + path)
    previous = json.loads(Path(protocol["service_metadata"]).read_text())["service_metadata"]
    for path, digest in dict(previous["source_sha256"], **previous["scoring_sha256"]).items():
        if service.sha(path) != digest:
            raise ValueError("baseline source/scorer changed: " + path)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    paths = {Path(p).name: p for p in previous["scoring_sha256"]}
    config = argparse.Namespace(threads=2, scorer=paths["score_models_v2_w2vbert.joblib"], norms=paths["norms.json"],
                               calib=paths["calibration.json"], flu_rule=paths["fluency_rule.json"],
                               utt_pct=paths["utt_percentile.json"])
    app = service.app_module(config, previous["base_ckpt"])
    dec = app.S["dec"]
    if dec.model.config.add_adapter or dec.model.config.use_intermediate_ffn_before_adapter:
        raise ValueError("pilot requires the frozen non-adapted model")
    layers = protocol["layers"]
    captures, observed, handles = {}, {}, []

    def make_hook(layer):
        def hook(module, inputs, output):
            if layer in captures:
                raise AssertionError("unexpected repeated layer forward")
            captures[layer] = output[0].detach().clone()
        return hook

    for layer in layers:
        handles.append(dec.model.wav2vec2_bert.encoder.layers[layer - 1].register_forward_hook(make_hook(layer)))
    original = app.S["fwd"]

    def forward(decoder, wavs):
        if len(wavs) != 1:
            raise ValueError("singleton production inference required")
        lp, lengths = original(decoder, wavs)
        observed.update(wav=wavs[0].copy(), lp=lp[0, :lengths[0]].numpy().copy())
        return lp, lengths

    app.S["fwd"] = forward
    metadata = {"protocol": protocol, "protocol_sha256": service.sha(args.protocol), "limit": args.limit,
                "runtime_inputs": "cached symmetric runtime anchors, same VAD PCM and model pass; no TAG boundary",
                "hook_reference_audit": None, "splits": {}}
    for path in [Path(__file__), ROOT / "tasks/pronunciation-assessment/src/mdd/sibilant_hidden.py"]:
        shutil.copy2(path, out / path.name)
    try:
        for split, definition in protocol["cache_sources"].items():
            source = Path(definition["path"])
            responses = service.read_jsonl(source / (split + ".responses.jsonl"))
            features = service.read_jsonl(source / (split + ".features.jsonl"))
            arrays = dict(np.load(source / (split + ".npz")))
            hidden = {layer: np.lib.format.open_memmap(str(out / (split + ".h%d.npy" % layer)), mode="w+",
                      dtype=np.float16, shape=(len(features), 31, dec.model.config.hidden_size)) for layer in layers}
            filled = np.zeros(len(features), bool)
            elapsed, n_accepted = [], 0
            started = time.monotonic()
            for number, record in enumerate(responses[:args.limit or None], 1):
                if record["raw"] is not None:
                    captures.clear()
                    observed.clear()
                    x = service.load_audio(record["row"])
                    code, _, _ = app.check_audio(x[:320000], record["row"]["prompt"])
                    if code is not None:
                        raise AssertionError("accepted input changed status")
                    tick = time.monotonic()
                    _, _, raw = app.analyze(x[:320000], record["row"]["prompt"], "")
                    body = service.response(app, record["row"], raw)
                    elapsed.append(time.monotonic() - tick)
                    if body != record["body"] or raw["canon"] != record["raw"]["canon"]:
                        raise AssertionError("frozen baseline response changed: " + record["uid"])
                    if set(captures) != set(layers):
                        raise AssertionError("layer capture incomplete")
                    n_accepted += 1
                    wav, lp = observed["wav"], observed["lp"]
                    current = {k: v[0, :len(lp)].cpu().numpy().copy() for k, v in captures.items()}
                    if metadata["hook_reference_audit"] is None:
                        # One explicit full-hidden-state pass validates layer indexing.
                        captures.clear()
                        p = next(dec.model.parameters())
                        enc = dec.fe([wav], sampling_rate=16000, return_tensors="pt", padding=True)
                        with torch.inference_mode():
                            ref = dec.model(input_features=enc["input_features"].to(p.device, p.dtype),
                                            attention_mask=enc["attention_mask"].to(p.device), output_hidden_states=True)
                        for layer in layers:
                            np.testing.assert_array_equal(current[layer], ref.hidden_states[layer][0, :len(lp)].cpu().numpy())
                        np.testing.assert_array_equal(lp, ref.logits[0, :len(lp)].float().log_softmax(-1).cpu().numpy())
                        metadata["hook_reference_audit"] = {"uid": record["uid"], "exact_layers": layers, "exact_logp": True}
                        del ref
                    digest = hashlib.sha256(wav.tobytes()).hexdigest()
                    for q, ix in record["feature_indices"].items():
                        feature = features[ix]
                        if feature["uid"] != record["uid"] or feature["q_runtime"] != int(q):
                            raise AssertionError("feature identity mismatch")
                        if digest != feature["vad_pcm_sha256"]:
                            raise AssertionError("VAD input changed")
                        if not np.array_equal(arrays["posterior"][ix][arrays["mask"][ix]],
                                              np.clip(lp[max(0, feature["anchor_frame"] - 15):
                                                         min(len(lp), feature["anchor_frame"] + 16)], -20, 0) / 20):
                            raise AssertionError("baseline local posteriors changed")
                        for layer in layers:
                            window, mask = hidden_window(current[layer], feature["anchor_frame"])
                            np.testing.assert_array_equal(mask, arrays["mask"][ix])
                            hidden[layer][ix] = window
                        filled[ix] = True
                if number % 100 == 0 or number == len(responses[:args.limit or None]):
                    print(split, number, "accepted", n_accepted, "features", int(filled.sum()),
                          "seconds", round(time.monotonic() - started, 1), flush=True)
            if not args.limit and not filled.all():
                raise AssertionError("missing features in full extraction")
            for layer in layers:
                hidden[layer].flush()
            np.save(out / (split + ".filled.npy"), filled)
            metadata["splits"][split] = {"responses": len(responses), "accepted_reproduced": n_accepted,
                                       "filled": int(filled.sum()), "seconds": time.monotonic() - started,
                                       "instrumented_service_seconds_quantiles": np.quantile(elapsed, [.5, .95]).tolist() if elapsed else None}
            (out / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    finally:
        app.S["fwd"] = original
        for handle in handles:
            handle.remove()
    (out / "finished.json").write_text(json.dumps(metadata["splits"], indent=2) + "\n")


if __name__ == "__main__":
    main()
