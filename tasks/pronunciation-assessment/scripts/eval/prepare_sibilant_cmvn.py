"""Change only local mel normalization scope, preserving cached runtime support."""
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
import numpy as np
from transformers import SeamlessM4TFeatureExtractor
import eval_sibilant_directional as service
from mdd.vad import trim
from mdd.sibilant_cmvn import statistics, stack_normalized


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    p = json.loads(Path(a.protocol).read_text())
    for path, digest in p["sha256"].items():
        if service.sha(path) != digest:
            raise ValueError("frozen source changed: " + path)
    out, source = Path(a.out), Path(p["feature_source"])
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(a.protocol, out / "protocol.json")
    shutil.copy2(__file__, out / Path(__file__).name)
    fe = SeamlessM4TFeatureExtractor.from_pretrained(p["checkpoint"])
    report = {"scope": p["scope"], "splits": {}, "deployment": False}
    for split in ("train", "cal", "test"):
        data = dict(np.load(source / (split + ".npz")))
        features = service.read_jsonl(source / (split + ".features.jsonl"))
        records = service.read_jsonl(source / (split + ".responses.jsonl"))
        by_uid = collections.defaultdict(list)
        for i, feature in enumerate(features):
            by_uid[feature["uid"]].append(i)
        changed, differences, counts = np.zeros_like(data["x"]), [], collections.Counter()
        for record in records:
            row = record["row"]
            if not by_uid[row["uid"]]:
                continue
            wav = trim(service.load_audio(row)[:320000], 16000)[0]
            digest = hashlib.sha256(wav.tobytes()).hexdigest()
            full_raw = fe._extract_fbank_features(wav)
            full_mean, full_var = statistics(full_raw)
            for i in by_uid[row["uid"]]:
                f, length = features[i], int(data["lengths"][i])
                clip = wav[f["left"]:f["right"]]
                if f["vad_pcm_sha256"] != digest or hashlib.sha256(clip.tobytes()).hexdigest() != f["clip_sha256"]:
                    raise AssertionError("cached runtime PCM or support changed")
                raw = fe._extract_fbank_features(clip)
                control = stack_normalized(raw, *statistics(raw), length=length, stride=fe.stride)
                np.testing.assert_array_equal(control, data["x"][i, :length])
                new = stack_normalized(raw, full_mean, full_var, length, fe.stride)
                changed[i, :length] = new
                differences.append(float(np.sqrt(np.mean((new - control) ** 2))))
                counts["control_exact_features"] += 1
            counts["runtime_utterances"] += 1
            if counts["runtime_utterances"] % 250 == 0:
                print(split, dict(counts), flush=True)
        if counts["control_exact_features"] != len(features):
            raise AssertionError("missing runtime feature")
        data["x"] = changed
        np.savez(out / (split + ".npz"), **data)
        for suffix in ("features", "labels", "responses"):
            (out / (split + "." + suffix + ".jsonl")).symlink_to((source / (split + "." + suffix + ".jsonl")).resolve())
        original = np.load(source / (split + ".npz"))
        for k in data:
            if k != "x":
                np.testing.assert_array_equal(data[k], original[k])
        report["splits"][split] = {"counts": dict(counts), "all_non_x_arrays_exact": True,
                                   "feature_rms_change_p10_p50_p90": np.quantile(differences, [.1, .5, .9]).tolist(),
                                   "input_shape": list(data["x"].shape), "output_sha256": service.sha(out / (split + ".npz"))}
        (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(split, json.dumps(report["splits"][split], ensure_ascii=False), flush=True)
    (out / "finished.json").write_text(json.dumps({"sha256": service.sha(out / "report.json")}) + "\n")


if __name__ == "__main__":
    main()
