"""Cache pre-CMVN fbanks on the exact existing automatic local clips.

No new admission, location or TAG selection. Reproduce VAD/clip hashes and every
observed normalized fbank before accepting its unnormalized counterpart. This
prepares an acoustic-information ablation, not new pronunciation labels.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    if ROOT.parent == ROOT:
        raise RuntimeError("repository root not found")
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tasks/pronunciation-assessment/src")]
import argparse
import collections
import hashlib
import json
import shutil
import time
import numpy as np
from transformers import SeamlessM4TFeatureExtractor
import eval_sibilant_directional as service
from mdd.vad import trim


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    p, out = json.loads(Path(a.protocol).read_text()), Path(a.out)
    for path, digest in p["sha256"].items():
        if service.sha(path) != digest:
            raise ValueError("frozen source changed: " + path)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(a.protocol, out / "protocol.json")
    shutil.copy2(__file__, out / Path(__file__).name)
    fe = SeamlessM4TFeatureExtractor.from_pretrained(p["checkpoint"])
    report = {"scope": p["scope"], "deployment": False, "cohorts": {}}
    for name, prefix in p["prefixes"].items():
        started = time.monotonic()
        data = dict(np.load(prefix + ".npz"))
        features = service.read_jsonl(prefix + ".features.jsonl")
        by_uid = collections.defaultdict(list)
        for i, feature in enumerate(features):
            by_uid[feature["uid"]].append(i)
        records = service.read_jsonl(prefix + ".responses.jsonl")
        acoustic = np.zeros_like(data["x"], dtype=np.float32)
        seen, counts = set(), collections.Counter()
        for record in records:
            row = record["row"]
            indices = by_uid.get(row["uid"], [])
            if not indices:
                continue
            wav, _ = trim(service.load_audio(row)[:320000], 16000)
            vad_digest = hashlib.sha256(wav.tobytes()).hexdigest()
            for i in indices:
                f = features[i]
                if i in seen or vad_digest != f["vad_pcm_sha256"]:
                    raise AssertionError("duplicate input or VAD waveform differs")
                clip = wav[f["left"]:f["right"]]
                if hashlib.sha256(clip.tobytes()).hexdigest() != f["clip_sha256"]:
                    raise AssertionError("local waveform differs")
                normalized = fe([clip], sampling_rate=16000, return_tensors="np", padding=True)
                raw = fe([clip], sampling_rate=16000, return_tensors="np", padding=True,
                         do_normalize_per_mel_bins=False)
                length = int(normalized["attention_mask"][0].sum())
                if length != int(data["lengths"][i]) or length != int(raw["attention_mask"][0].sum()):
                    raise AssertionError("observed acoustic grid differs")
                np.testing.assert_array_equal(normalized["input_features"][0, :length], data["x"][i, :length])
                acoustic[i, :length] = raw["input_features"][0, :length]
                if not np.isfinite(acoustic[i]).all():
                    raise ValueError("nonfinite pre-CMVN fbank")
                seen.add(i)
                counts["normalized_inputs_exact"] += 1
            counts["vad_waveforms_exact"] += 1
            if counts["vad_waveforms_exact"] % 500 == 0:
                print(name, dict(counts), "seconds", round(time.monotonic() - started, 1), flush=True)
        if seen != set(range(len(features))):
            raise AssertionError("incomplete pre-CMVN feature coverage")
        np.savez(out / (name + ".npz"), acoustic=acoustic, lengths=data["lengths"])
        report["cohorts"][name] = {"all_input_features": len(features), "counts": dict(counts),
                                   "seconds": time.monotonic() - started,
                                   "source_npz_sha256": service.sha(prefix + ".npz"),
                                   "acoustic_sha256": service.sha(out / (name + ".npz")),
                                   "no_TAG_model_inputs": True}
        (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(name, json.dumps(report["cohorts"][name], ensure_ascii=False), flush=True)
    (out / "finished.json").write_text(json.dumps({"sha256": service.sha(out / "report.json")}) + "\n")


if __name__ == "__main__":
    main()
