"""Observe pretrained encoder on frozen service PCM/anchors, without TAG timing."""
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
from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertModel

import eval_sibilant_directional as service
from mdd.sibilant_hidden import hidden_window
from mdd.vad import trim


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0, help="smoke only, cannot train on limited output")
    parser.add_argument("--splits", nargs="+", choices=["train", "cal", "test"], default=["train", "cal", "test"])
    args = parser.parse_args()
    protocol = json.loads(Path(args.protocol).read_text())
    for path, digest in protocol["source_sha256"].items():
        if service.sha(path) != digest:
            raise ValueError("frozen extraction source changed")
    download = json.loads(Path(protocol["model_record"]).read_text())
    for path, digest in download["sha256"].items():
        if service.sha(path) != digest:
            raise ValueError("pretrained model changed")
    previous = json.loads(Path(protocol["previous_linear_protocol"]).read_text())
    old_hidden = Path(previous["hidden_features"])
    cache_protocol = json.loads((old_hidden / "metadata.json").read_text())["protocol"]
    baseline = json.loads(Path(cache_protocol["service_metadata"]).read_text())["service_metadata"]
    for path, digest in baseline["source_sha256"].items():
        if service.sha(path) != digest:
            raise ValueError("baseline source changed")
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("this frozen comparison requires the baseline GPU FP16 precision")
    model = Wav2Vec2BertModel.from_pretrained(download["local_path"], local_files_only=True).eval().to(device, torch.float16)
    if model.config.add_adapter or model.config.use_intermediate_ffn_before_adapter or model.config.num_hidden_layers != 24:
        raise ValueError("matched 24-layer encoder without adapter required")
    extractor = SeamlessM4TFeatureExtractor.from_pretrained(download["local_path"], local_files_only=True)
    baseline_extractor = SeamlessM4TFeatureExtractor.from_pretrained(baseline["base_ckpt"], local_files_only=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, out / Path(__file__).name)
    metadata = {"protocol": protocol, "limit": args.limit, "model_record": download,
                "input_contract": "original VAD PCM and symmetric runtime anchors; no TAG intervals or actual labels",
                "precision": "GPU FP16, as cached operating-family encoder", "first_input_audit": None, "splits": {}}
    for split in args.splits:
        definition = cache_protocol["cache_sources"][split]
        for path, digest in definition["sha256"].items():
            if service.sha(path) != digest:
                raise ValueError("baseline cache changed")
        cache = Path(definition["path"])
        responses = service.read_jsonl(cache / (split + ".responses.jsonl"))
        features = service.read_jsonl(cache / (split + ".features.jsonl"))
        arrays = dict(np.load(cache / (split + ".npz")))
        hidden = np.lib.format.open_memmap(str(out / (split + ".h24.npy")), mode="w+", dtype=np.float16,
                                           shape=(len(features), 31, model.config.hidden_size))
        filled = np.zeros(len(features), bool)
        accepted, started, elapsed = 0, time.monotonic(), []
        for number, record in enumerate(responses[:args.limit or None], 1):
            if record["raw"] is not None:
                wav, _ = trim(service.load_audio(record["row"])[:320000], 16000)
                digest = hashlib.sha256(wav.tobytes()).hexdigest()
                if any(features[i]["vad_pcm_sha256"] != digest for i in record["feature_indices"].values()):
                    raise ValueError("cached VAD PCM not reproduced")
                enc = extractor([wav], sampling_rate=16000, return_tensors="pt", padding=True)
                length = int(enc["attention_mask"].sum())
                first = metadata["first_input_audit"] is None
                if first:
                    reference = baseline_extractor([wav], sampling_rate=16000, return_tensors="pt", padding=True)
                    for key in ("input_features", "attention_mask"):
                        torch.testing.assert_close(enc[key], reference[key], rtol=0, atol=0)
                tick = time.monotonic()
                with torch.inference_mode():
                    result = model(input_features=enc["input_features"].to(device, torch.float16),
                                   attention_mask=enc["attention_mask"].to(device), output_hidden_states=first)
                if first:
                    torch.testing.assert_close(result.last_hidden_state, result.hidden_states[24], rtol=0, atol=0)
                    metadata["first_input_audit"] = {"uid": record["uid"], "feature_extractor_inputs_exact": True,
                                                       "last_state_equals_layer24": True}
                values = result.last_hidden_state[0, :length].cpu().numpy()
                elapsed.append(time.monotonic() - tick)
                if not np.isfinite(values).all():
                    raise ValueError("nonfinite pretrained hidden states")
                for q, ix in record["feature_indices"].items():
                    feature = features[ix]
                    if feature["uid"] != record["uid"] or feature["q_runtime"] != int(q):
                        raise ValueError("feature identity mismatch")
                    if abs(len(wav) / (16000. * length) - feature["frame_seconds"]) > 1e-12:
                        raise ValueError("pretrained and baseline frame grids differ")
                    window, mask = hidden_window(values, feature["anchor_frame"])
                    np.testing.assert_array_equal(mask, arrays["mask"][ix])
                    hidden[ix] = window
                    filled[ix] = True
                accepted += 1
                del result
            if number % 100 == 0 or number == len(responses[:args.limit or None]):
                print(split, number, "accepted", accepted, "features", int(filled.sum()), round(time.monotonic() - started, 1), flush=True)
        if not args.limit and not filled.all():
            raise ValueError("full extraction has missing observations")
        hidden.flush()
        np.save(out / (split + ".filled.npy"), filled)
        metadata["splits"][split] = {"responses": len(responses), "accepted": accepted, "filled": int(filled.sum()),
                                     "seconds": time.monotonic() - started,
                                     "instrumented_model_seconds_p50_p95": np.quantile(elapsed, [.5, .95]).tolist() if elapsed else None}
        (out / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    (out / "finished.json").write_text(json.dumps(metadata["splits"], indent=2) + "\n")


if __name__ == "__main__":
    main()
