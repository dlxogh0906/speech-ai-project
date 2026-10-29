"""Score frozen-encoder heads against complete NPU pooled caches, including rejections."""
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r:
        raise RuntimeError("Repository root not found")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import argparse
import collections
import hashlib
import json
import numpy as np
from score_npu_cached_head import mean_log


def load_head(path):
    with np.load(path, allow_pickle=True) as values:
        return {k: values[k].copy() for k in values.files}


def payload(pooled, pg, values, labels):
    parts = []
    for x in pooled:
        logits = ((x[None, :] - values["mu"]) / values["sd"]) @ values["W"].T + values["b"]
        p = np.exp(logits - logits.max()); p /= p.sum()
        parts.append(p[0])
    p = mean_log(parts)
    ki = labels.index("Korea")
    if pg[ki] > .9 or p.argmax() == ki:
        if pg[ki] > p[ki]:
            p = pg.copy()
        if p.argmax() != ki:
            p[ki] = p.max() + 1e-6; p /= p.sum()
    return {"top5": [{"label": labels[i], "prob": round(float(p[i]), 4)} for i in np.argsort(-p)[:5]],
            "gate_korea": float(pg[ki])}


def metrics(rows, tag):
    countries = np.array([r["country"] for r in rows])
    predicted = np.array([r["variants"][tag]["top5"][0]["label"] if r["variants"][tag]["top5"] else "REJECTED" for r in rows])
    hit = predicted == countries
    hit5 = np.array([r["country"] in [p["label"] for p in r["variants"][tag]["top5"]] for r in rows])
    foreign = countries != "Korea"
    accepted = predicted != "REJECTED"
    per = {}
    for country in sorted(set(countries)):
        mask = countries == country
        per[country] = {"n": int(mask.sum()), "speakers": len({r["speaker"] for r in rows if r["country"] == country}),
                        "correct_top1": int(hit[mask].sum()), "top1": float(hit[mask].mean()), "top5": float(hit5[mask].mean()),
                        "errors": dict(collections.Counter(predicted[mask & ~hit].tolist()))}
    speaker = collections.defaultdict(list)
    for r, correct in zip(rows, hit):
        if r["country"] != "Korea":
            speaker[r["speaker"]].append(correct)
    return {"foreign_n": int(foreign.sum()), "foreign_top1": float(hit[foreign].mean()),
            "foreign_top5": float(hit5[foreign].mean()),
            "foreign_macro_top1": float(np.mean([v["top1"] for c, v in per.items() if c != "Korea"])),
            "foreign_speaker_mean_top1": float(np.mean([np.mean(v) for v in speaker.values()])),
            "foreign_correct_top1": int(hit[foreign].sum()),
            "foreign_accepted": int((foreign & accepted).sum()),
            "foreign_top1_given_prediction": float(hit[foreign & accepted].mean()) if (foreign & accepted).any() else None,
            "korean_top1": float(hit[~foreign].mean()) if (~foreign).any() else None,
            "rejections": int((predicted == "REJECTED").sum()), "per_country": per}, hit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npu-dir", type=_pl.Path, required=True)
    parser.add_argument("--base-head", type=_pl.Path, required=True)
    parser.add_argument("--head", action="append", default=[], help="TAG=HEAD_NP.NPZ")
    parser.add_argument("--expected-records", type=_pl.Path, required=True)
    parser.add_argument("--out", type=_pl.Path, required=True)
    parser.add_argument("--selection-freeze", type=_pl.Path, help="Required immutable candidate hashes for independent test")
    parser.add_argument("--profile", action="append", default=[],
                        help="TAG=LONG_HEAD_TAG,SHORT_HEAD_TAG; use short head only when actual n=1")
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    args = parser.parse_args()
    paths = {"baseline": args.base_head}
    for item in args.head:
        tag, path = item.split("=", 1)
        if tag in paths:
            raise ValueError("Duplicate head tag")
        paths[tag] = _pl.Path(path)
    heads = {tag: load_head(p) for tag, p in paths.items()}
    profiles = {}
    for item in args.profile:
        tag, pair = item.split("=", 1)
        long_tag, short_tag = pair.split(",", 1)
        if tag in heads or tag in profiles or long_tag not in heads or short_tag not in heads:
            raise ValueError("Invalid routed head profile")
        profiles[tag] = {"long": long_tag, "short": short_tag, "selector": "actual_n_equals_1"}
    base = heads["baseline"]
    for head in heads.values():
        for key in ("mu", "sd", "labels"):
            if not np.array_equal(base[key], head[key]):
                raise ValueError("Encoder/head conditions differ: " + key)
    hashes = {tag: hashlib.sha256(p.read_bytes()).hexdigest() for tag, p in paths.items()}
    if args.selection_freeze:
        freeze = json.loads(args.selection_freeze.read_text())
        for tag, digest in hashes.items():
            if freeze["head_sha256"].get(tag) != digest:
                raise ValueError("Candidate changed after freeze: " + tag)
        if profiles != freeze.get("profiles", {}):
            raise ValueError("Routed inference profile changed after freeze")
        record_hash = hashlib.sha256(args.expected_records.read_bytes()).hexdigest()
        if record_hash not in {freeze.get("test_records_sha256"), freeze.get("word_test_records_sha256")}:
            raise ValueError("Independent evaluation request definitions changed after freeze")
    expected = {json.loads(s)["job_id"]: json.loads(s) for s in args.expected_records.read_text().splitlines()}
    labels = list(map(str, base["labels"]))
    args.out.mkdir(parents=True, exist_ok=True)
    found, scored = set(), []
    for path in sorted(args.npu_dir.glob("shard_*.jsonl")):
        output = []
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["job_id"] not in expected:
                continue
            if row["job_id"] in found:
                raise ValueError("Duplicate cached request")
            found.add(row["job_id"])
            if any(row[key] != expected[row["job_id"]][key] for key in ("country", "speaker")):
                raise ValueError("Cached evaluation request metadata differs")
            if row["n"]:
                with np.load(args.npu_dir / "pooled" / (row["capture_rid"] + ".npz")) as f:
                    if list(map(str, f["labels"])) != labels or len(f["pooled"]) != row["n"]:
                        raise ValueError("Cached part metadata differs")
                    pooled, pg = f["pooled"], mean_log(f["gate"])
                variants = {tag: payload(pooled, pg, head, labels) for tag, head in heads.items()}
                if variants["baseline"]["top5"] != row["variants"]["baseline"]["top5"]:
                    raise ValueError("Complete cached baseline payload parity failed: " + row["job_id"])
            else:
                variants = {tag: {"top5": [], "gate_korea": None} for tag in heads}
            for tag, profile in profiles.items():
                variants[tag] = variants[profile["short"] if row["n"] == 1 else profile["long"]]
            row["variants"] = variants
            output.append(json.dumps(row, ensure_ascii=False) + "\n")
            scored.append(row)
        (args.out / path.name).write_text("".join(output))
    if found != set(expected):
        raise ValueError("Incomplete NPU evaluation: {} of {}".format(len(found), len(expected)))
    scored.sort(key=lambda row: row["job_id"])
    result, hits = {}, {}
    for tag in list(heads) + list(profiles):
        result[tag], hits[tag] = metrics(scored, tag)
    foreign = np.array([r["country"] != "Korea" for r in scored])
    speakers = sorted({r["speaker"] for r in scored if r["country"] != "Korea"})
    sizes = np.array([sum(r["speaker"] == s and r["country"] != "Korea" for r in scored) for s in speakers])
    draws = np.random.default_rng(42).integers(0, len(speakers), (args.bootstrap_reps, len(speakers)))
    comparisons = {}
    for tag in list(heads) + list(profiles):
        if tag == "baseline":
            continue
        for reference in ("baseline", "previous", "routed_control"):
            if reference not in result or reference == tag:
                continue
            difference = hits[tag].astype(int) - hits[reference].astype(int)
            speaker_delta = np.array([sum(difference[i] for i, r in enumerate(scored) if r["speaker"] == s and foreign[i]) for s in speakers])
            changes = speaker_delta[draws].sum(1) / sizes[draws].sum(1) * 100
            comparisons[tag + "_vs_" + reference] = {
                "foreign_top1_delta_pp": float(difference[foreign].mean() * 100),
                "country_mean_delta_pp": (result[tag]["foreign_macro_top1"] - result[reference]["foreign_macro_top1"]) * 100,
                "speaker_bootstrap_95ci_pp": np.percentile(changes, [2.5, 97.5]).tolist(),
                "improved_groups": [r["job_id"] for i, r in enumerate(scored) if foreign[i] and difference[i] == 1],
                "regressed_groups": [r["job_id"] for i, r in enumerate(scored) if foreign[i] and difference[i] == -1]}
    summary = {"groups": len(scored), "speakers": len({r["speaker"] for r in scored}),
               "actual_parts": dict(collections.Counter(str(r["n"]) for r in scored)),
               "metrics": result, "comparisons": comparisons, "head_sha256": hashes,
               "baseline_payload_parity": "all complete rounded top5 payloads match actual NPU",
               "scope": "independent_test" if args.selection_freeze else "validation or auxiliary evaluation",
               "selection_freeze": str(args.selection_freeze) if args.selection_freeze else None}
    summary["profiles"] = profiles
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    for tag, m in result.items():
        print(tag, {k: v for k, v in m.items() if k != "per_country"}, flush=True)


if __name__ == "__main__":
    main()
