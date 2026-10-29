"""Compare fixed normal-preservation arms using frozen production responses.

Only reads completed directional evaluations. Does not select thresholds,
checkpoints, or change any inference output.
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
import json

from eval_sibilant_directional import cluster_delta, read_jsonl, sha


ARMS = ("old_rank", "normal1", "normal4")
SCOPES = ("mapped_expert_ss", "mapped_conservative_context")


def compare(root):
    root = Path(root)
    runs, reports, metadata = {}, {}, {}
    for arm in ARMS:
        path = root / arm
        if not (path / "candidate.finished").exists():
            raise RuntimeError("Incomplete evaluation: " + arm)
        runs[arm] = read_jsonl(path / "predictions.jsonl")
        reports[arm] = json.loads((path / "comparison.json").read_text())
        metadata[arm] = json.loads((path / "metadata.json").read_text())
        if reports[arm]["metadata"] != metadata[arm]:
            raise AssertionError("Report metadata is stale: " + arm)
    first = runs[ARMS[0]]
    identity_fields = ("uid", "speaker", "prompt", "cohort", "item", "rejected", "targets")
    for arm in ARMS[1:]:
        if len(runs[arm]) != len(first):
            raise AssertionError("Different row counts")
        for left, right in zip(first, runs[arm]):
            for field in identity_fields[:-1]:
                if left.get(field) != right.get(field):
                    raise AssertionError("Data/acceptance changed: " + field)
            if len(left["targets"]) != len(right["targets"]):
                raise AssertionError("Target count changed")
            for a, b in zip(left["targets"], right["targets"]):
                if {k: v for k, v in a.items() if k not in ("conservative", "broad")} != {
                        k: v for k, v in b.items() if k not in ("conservative", "broad")}:
                    raise AssertionError("Target mapping/truth/baseline changed")
            if left["variants"].get("baseline") != right["variants"].get("baseline"):
                raise AssertionError("Full baseline body differs")
        for key in ("base_ckpt", "data_sha256", "source_sha256", "scoring_sha256", "configuration"):
            if metadata[arm][key] != metadata[ARMS[0]][key]:
                raise AssertionError("Frozen metadata differs: " + key)

    result = {
        "sources": {arm: {"path": str(root / arm / "predictions.jsonl"),
                          "sha256": sha(root / arm / "predictions.jsonl")} for arm in ARMS},
        "primary_candidate": "normal4; predetermined before training/evaluation",
        "note": "Conservative response only; normal1 is a matched new-data/new-direction control. Old rank uses earlier data and loss, not a matched ablation. No threshold tuning.",
        "cohorts": {},
    }
    for cohort in sorted({r["cohort"] for r in first}):
        positions = [i for i, r in enumerate(first) if r["cohort"] == cohort]
        c = {k: reports["normal4"]["cohorts"][cohort][k] for k in (
            "utterances", "speakers", "prompts", "rejections", "mapping", "unknown_speaker_utterances")}
        c["scopes"] = {}
        for scope in SCOPES:
            matched = []
            for i in positions:
                for j, t in enumerate(first[i]["targets"]):
                    if t["mapping_status"] != "mapped":
                        continue
                    if scope == "mapped_conservative_context" and not (
                            t["lab_conservative_context"] and t["runtime_conservative_context"]):
                        continue
                    v = {**t, **{arm: runs[arm][i]["targets"][j]["conservative"] for arm in ARMS}}
                    matched.append((first[i], v))
            rates = {}
            for arm in ("baseline",) + ARMS:
                errors = [t for _, t in matched if t["actual"] == "ㅅ"]
                normals = [t for _, t in matched if t["actual"] == "ㅆ"]
                tp = sum(t[arm]["ss_to_s"] for t in errors)
                fp = sum(t[arm]["ss_to_s"] for t in normals)
                rates[arm] = {"tp": tp, "errors": len(errors), "fp": fp, "normals": len(normals),
                              "recall": tp / len(errors) if errors else None,
                              "fpr": fp / len(normals) if normals else None}
            paired = {}
            for reference in ("baseline", "old_rank", "normal1"):
                relabeled = [(r, dict(t, baseline=t[reference])) for r, t in matched]
                paired["normal4_minus_" + reference] = {
                    unit: {"recall": cluster_delta(relabeled, "normal4", "ss_to_s", unit, "ㅅ"),
                           "fpr": cluster_delta(relabeled, "normal4", "ss_to_s", unit, "ㅆ")}
                    for unit in ("speaker", "prompt")}
            c["scopes"][scope] = {"rates": rates, "paired_differences": paired}
        c["burden"] = {"baseline": reports["normal4"]["cohorts"][cohort]["burden"]["baseline"]}
        for arm in ARMS:
            c["burden"][arm] = reports[arm]["cohorts"][cohort]["burden"]["conservative"]
        primary = c["scopes"]["mapped_expert_ss"]["rates"]
        b, one, four = (primary[k] for k in ("baseline", "normal1", "normal4"))
        c["checks"] = {
            "errors_present": bool(four["errors"]), "normals_present": bool(four["normals"]),
            "normal4_more_tp_than_baseline": four["tp"] > b["tp"],
            "normal4_no_more_fp_than_baseline": four["fp"] <= b["fp"],
            "normal4_fewer_fp_than_normal1": four["fp"] < one["fp"],
            "normal4_no_fewer_tp_than_normal1": four["tp"] >= one["tp"],
            "all_non_target_changes_zero": all(
                reports[arm]["cohorts"][cohort]["other_phoneme_changes"] == 0 for arm in ARMS),
        }
        result["cohorts"][cohort] = c
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evaluations", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    output = Path(a.out)
    if output.exists():
        raise FileExistsError(output)
    result = compare(a.evaluations)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    for cohort, c in result["cohorts"].items():
        print(cohort, json.dumps(c["scopes"]["mapped_expert_ss"]["rates"], ensure_ascii=False))


if __name__ == "__main__":
    main()
