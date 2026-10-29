"""Score a matched frozen-encoder head from NPU pooled parts, auditing payload parity."""
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r:
        raise RuntimeError("Repository root not found")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import argparse
import hashlib
import json
import numpy as np


def mean_log(parts):
    lp = np.log(np.clip(np.stack(parts), 1e-12, 1)).mean(0)
    p = np.exp(lp - lp.max())
    return p / p.sum()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npu-dir", type=_pl.Path, required=True)
    ap.add_argument("--base-head", type=_pl.Path, required=True)
    ap.add_argument("--candidate", type=_pl.Path, required=True)
    ap.add_argument("--candidate-tag", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--out", type=_pl.Path, required=True)
    args = ap.parse_args()
    if not 0 <= args.alpha <= 1:
        ap.error("Alpha must be 0..1")
    with np.load(args.base_head, allow_pickle=True) as f:
        base = {k: f[k].copy() for k in f.files}
    with np.load(args.candidate, allow_pickle=True) as f:
        candidate = {k: f[k].copy() for k in f.files}
    for key in ("mu", "sd", "labels"):
        if not np.array_equal(base[key], candidate[key]):
            raise ValueError("Head/encoder conditions differ: " + key)
    labels = list(map(str, base["labels"]))
    ki = labels.index("Korea")
    head = dict(base)
    for key in ("W", "b"):
        head[key] = ((1 - args.alpha) * base[key] + args.alpha * candidate[key]).astype(np.float32)

    def payload(pooled, pg, values):
        parts = []
        for x in pooled:
            logits = ((x[None, :] - values["mu"]) / values["sd"]) @ values["W"].T + values["b"]
            p = np.exp(logits - logits.max())
            p /= p.sum()
            parts.append(p[0])
        p = mean_log(parts)
        if pg[ki] > .9 or p.argmax() == ki:
            if pg[ki] > p[ki]:
                p = pg.copy()
            if p.argmax() != ki:
                p[ki] = p.max() + 1e-6
                p /= p.sum()
        return {"top5": [{"label": labels[i], "prob": round(float(p[i]), 4)} for i in np.argsort(-p)[:5]],
                "gate_korea": float(pg[ki])}

    args.out.mkdir(parents=True, exist_ok=True)
    count = 0
    for path in sorted(args.npu_dir.glob("shard_*.jsonl")):
        output = []
        for line in path.read_text().splitlines():
            row = json.loads(line)
            with np.load(args.npu_dir / "pooled" / (row["capture_rid"] + ".npz")) as f:
                pooled, gate = f["pooled"], f["gate"]
                if list(map(str, f["labels"])) != labels or len(pooled) != row["n"]:
                    raise ValueError("Cache labels/parts differ")
            pg = mean_log(gate)
            for original_head, tag in ((base, "baseline"), (candidate, args.candidate_tag)):
                recomputed = payload(pooled, pg, original_head)
                if recomputed["top5"] != row["variants"][tag]["top5"]:
                    raise ValueError("Cached head payload differs: {} {}".format(row["job_id"], tag))
            row["variants"][args.tag] = payload(pooled, pg, head)
            output.append(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
        (args.out / path.name).write_text("".join(output))
    np.savez(args.out / "head_np.npz", **head)
    info = {"alpha": args.alpha, "tag": args.tag, "candidate_tag": args.candidate_tag,
            "base_head": str(args.base_head), "base_sha256": hashlib.sha256(args.base_head.read_bytes()).hexdigest(),
            "candidate_head": str(args.candidate), "candidate_sha256": hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
            "npu_cached_groups": count, "payload_parity": "All baseline and candidate top5 payloads match exactly",
            "conditions": "Frozen deployed run_zeroth encoder and gate; original NPU pooled parts",
            "selection": "Interpolation coefficient chosen on repeated original validation; exploratory evidence"}
    (args.out / "cache_scoring_manifest.json").write_text(json.dumps(info, indent=2))
    print("[done] {} NPU cached groups; complete baseline/candidate payload parity".format(count), flush=True)


if __name__ == "__main__":
    main()
