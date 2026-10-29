"""Select frozen-encoder head refinements on original validation only.

Fresh evaluation records are hashed, never scored during selection. Critical
country recalls must meet both production and anchor validation recalls.
"""
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r:
        raise RuntimeError("Repository root not found")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent / "train"))

import argparse
import json
import numpy as np
from refine_nationality_head import load_head, validation_parts
from pilot_youtube_head import sha, dump_json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-head", type=_pl.Path, required=True)
    ap.add_argument("--anchor-head", type=_pl.Path, required=True)
    ap.add_argument("--runs-dir", type=_pl.Path, required=True)
    ap.add_argument("--validation-responses", type=_pl.Path, required=True)
    ap.add_argument("--npu-validation", type=_pl.Path, required=True)
    ap.add_argument("--fresh-records", type=_pl.Path, required=True)
    ap.add_argument("--critical-countries", default="Uzbekistan,Philippines")
    ap.add_argument("--mixes", default="0.1,0.25,0.5")
    ap.add_argument("--out", type=_pl.Path, required=True)
    args = ap.parse_args()
    mixes = [float(v) for v in args.mixes.split(",")]
    if any(not 0 < v < 1 for v in mixes):
        ap.error("Interpolation weights must be between zero and one")
    args.out.mkdir(parents=True, exist_ok=True)
    base, anchor = load_head(args.base_head), load_head(args.anchor_head)
    labels = list(map(str, base["labels"]))
    critical = [s for s in args.critical_countries.split(",") if s]
    if set(critical) - set(labels):
        ap.error("Unknown critical country")
    records, x, starts, counts, gate = validation_parts(args, base)
    if set(critical) - {r["country"] for r in records}:
        ap.error("Critical countries must have validation examples")
    y = np.array([labels.index(r["country"]) for r in records])
    korea = labels.index("Korea")
    foreign = y != korea

    def evaluate(head):
        for key in ("mu", "sd", "labels"):
            if not np.array_equal(head[key], base[key]):
                raise ValueError("Encoder/head normalization differs: " + key)
        z = x @ head["W"].T + head["b"]
        p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        logp = np.log(np.clip(p, 1e-12, 1))
        aggregate = np.add.reduceat(logp, starts, axis=0) / counts[:, None]
        pred = aggregate.argmax(1); pred[gate] = korea
        hits = pred == y
        per = {c: float(hits[y == labels.index(c)].mean())
               for c in sorted({r["country"] for r in records})}
        return {"micro": float(hits[foreign].mean()),
                "macro": float(np.mean([v for c, v in per.items() if c != "Korea"])),
                "korean": per.get("Korea"), "per_country": per}, pred

    bm, bp = evaluate(base)
    expected = np.array([labels.index(r["variants"]["baseline"]["top5"][0]["label"])
                         for r in records])
    if not np.array_equal(bp, expected):
        raise ValueError("Validation does not match recorded NPU baseline")
    am, _ = evaluate(anchor)
    paths = {"baseline": args.base_head, "previous": args.anchor_head}
    trials = []
    for path in sorted(args.runs_dir.glob("*/diagnostic_head.npz")):
        tag = path.parent.name
        head = load_head(path)
        paths[tag] = path
        for weight in [1.] + mixes:
            if weight == 1.:
                values, name, target = head, tag, path
            else:
                values = dict(anchor)
                values["W"] = (1 - weight) * anchor["W"] + weight * head["W"]
                values["b"] = (1 - weight) * anchor["b"] + weight * head["b"]
                name = tag + "_blend{:.2f}".format(weight)
                target = args.out / (name + ".npz")
                np.savez(target, **values)
            metrics, _ = evaluate(values)
            failed = []
            if metrics["micro"] < am["micro"]:
                failed.append("foreign_micro_below_anchor")
            if metrics["korean"] < am["korean"]:
                failed.append("korean_recall_below_anchor")
            for c in critical:
                if metrics["per_country"].get(c, 0) < max(bm["per_country"].get(c, 0), am["per_country"].get(c, 0)):
                    failed.append(c + "_recall_below_reference")
            trials.append(dict(tag=name, path=str(target), fraction=weight,
                               eligible=not failed, failed_constraints=failed, **metrics))
    eligible = [t for t in trials if t["eligible"] and t["micro"] + t["macro"] > am["micro"] + am["macro"]]
    best = max(eligible, key=lambda t: t["micro"] + t["macro"]) if eligible else None
    if best:
        paths["selected"] = _pl.Path(best["path"])
    dump_json(args.out / "validation_selection.json", {
        "production": bm, "anchor": am, "trials": trials, "selected": best,
        "scope": "Original validation only; repeated exploration may overestimate gains",
        "fresh_evaluation_not_scored": True})
    dump_json(args.out / "candidate_freeze.json", {
        "heads": {tag: {"path": str(p), "sha256": sha(p)} for tag, p in paths.items()},
        "head_sha256": {tag: sha(p) for tag, p in paths.items()}, "profiles": {},
        "test_records_sha256": sha(args.fresh_records),
        "validation_responses_sha256": sha(args.validation_responses),
        "selection": "Validation foreign micro + country mean, preserving critical recalls",
        "selected": best, "production_replacement": False})
    print(json.dumps({"trials": len(trials), "selected": best}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
