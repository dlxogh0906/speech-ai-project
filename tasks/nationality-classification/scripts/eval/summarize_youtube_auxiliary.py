"""Report unseen YouTube speakers separately from a known validation word error."""
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r:
        raise RuntimeError("Repository root not found")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import argparse
import json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npu-dir", type=_pl.Path, required=True)
    ap.add_argument("--expected", type=_pl.Path, required=True)
    ap.add_argument("--out", type=_pl.Path, required=True)
    args = ap.parse_args()
    expected = {r["job_id"]: r for r in map(json.loads, args.expected.read_text().splitlines())}
    found = {}
    for path in args.npu_dir.glob("shard_*.jsonl"):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["job_id"] in expected:
                if row["job_id"] in found:
                    raise ValueError("Duplicate auxiliary job")
                found[row["job_id"]] = row
    if set(found) != set(expected):
        raise ValueError("Incomplete auxiliary evaluation")
    variants = sorted(next(iter(found.values()))["variants"])
    rows = [found[k] for k in sorted(found) if k.startswith("youtube_holdout:")]
    known = [found[k] for k in sorted(found) if not k.startswith("youtube_holdout:")]
    if any(r["variants"]["baseline"]["top5"][0]["label"] !=
           expected[r["job_id"]]["payload"]["top5"][0]["label"] for r in found.values()):
        raise ValueError("Baseline auxiliary top-1 differs from recorded service")
    metrics = {}
    for variant in variants:
        speakers = {}
        for speaker in sorted({r["speaker"] for r in rows}):
            clips = [r for r in rows if r["speaker"] == speaker]
            predictions = [r["variants"][variant]["top5"][0] for r in clips]
            speakers[speaker] = {"country": clips[0]["country"], "groups": len(clips),
                                 "correct": sum(p["label"] == r["country"] for p, r in zip(predictions, clips)),
                                 "predictions": predictions}
        metrics[variant] = {"groups": len(rows), "correct": sum(v["correct"] for v in speakers.values()),
                            "speakers": speakers}
    summary = {"holdout": metrics, "known_word_errors": known,
               "limits": ["Only two held-out YouTube speakers; seven nonoverlapping three-clip groups",
                          "Not a replacement for an independent corpus test or actual app recordings",
                          "The word error is not evidence of the cause of the unrecorded historic user incident"],
               "source_records": str(args.expected), "source_predictions": str(args.npu_dir)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    for name, score in metrics.items():
        print(name, "holdout", score["correct"], "/", score["groups"], flush=True)
    for row in known:
        print("known error", row["job_id"],
              {name: row["variants"][name]["top5"][0] for name in variants}, flush=True)


if __name__ == "__main__":
    main()
