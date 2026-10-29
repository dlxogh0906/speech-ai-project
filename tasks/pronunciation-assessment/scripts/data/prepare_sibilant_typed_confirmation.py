"""Normalize prior cached cohorts for frozen typed-confidence comparisons."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    if ROOT.parent == ROOT:
        raise RuntimeError("repository root not found")
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tasks/pronunciation-assessment/src"),
               str(ROOT / "tasks/pronunciation-assessment/scripts/eval")]
import argparse
import json
import shutil
from eval_sibilant_directional import read_jsonl, all_optimal_mapping, sha, write_line
from mdd.sibilant_labels import label_sites


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="existing transition experiment directory")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, out / Path(__file__).name)
    calibration = {r["uid"] for r in read_jsonl(root / "complex_data_v1/data/cal.jsonl")}
    plan = [("expanded_cal", "cal", "expanded_margin_v1/features", "time_margin_v1", "time_reference_v1"),
            ("expanded_test", "test", "expanded_margin_v1/features", "time_margin_v1", "time_reference_v1"),
            ("confirmation", "confirmation", "margin_confirmation_v1/features", "time_margin_confirmation_v1", "time_reference_confirmation_v1"),
            ("preserve_confirmation", "preserve_confirmation", "margin_confirmation_v1/features", "time_margin_confirmation_v1", "time_reference_confirmation_v1")]
    report = {"scope": "Same old recordings and cached responses, all sibilant outcomes rederived from full human TAG/LAB sequences; repeated exploratory regression, no independent data claim", "cohorts": {}}
    for name, split, folder, integrated, reference in plan:
        paths = [root / folder / (split + ".responses.jsonl"),
                 root / integrated / "evaluation" / (split + ".responses.jsonl"),
                 root / reference / "evaluation" / (split + ".reference_changes.jsonl")]
        bases, finals, changes = [read_jsonl(p) for p in paths]
        if len(bases) != len(finals):
            raise AssertionError("cohort sizes differ")
        changed_raw = {r["uid"]: r["candidate_raw"] for r in changes}
        count, overlap = 0, 0
        with (out / (name + ".jsonl")).open("w") as stream:
            for base, final in zip(bases, finals):
                if base["uid"] != final["uid"]:
                    raise AssertionError("cohort identity differs")
                row = dict(base["row"])
                row["sites"] = label_sites(row["canonical"], row["actual"], all_optimal_mapping(row["canonical"], row["actual"]))
                raw = changed_raw.get(base["uid"], base["raw"])
                original = {k: base[k] for k in ("raw", "body", "rejected")}
                existing = {"raw": raw, "body": final["body"], "rejected": final["rejected"]}
                if existing["rejected"] != original["rejected"]:
                    raise AssertionError("admission changed")
                write_line(stream, {"row": row, "original": original, "existing": existing})
                count += len(row["sites"])
                overlap += row["uid"] in calibration
        report["cohorts"][name] = {"utterances": len(bases), "all_sibilant_sites": count,
                                   "overlap_with_typed_policy_calibration_uids": overlap,
                                   "input_sha256": {str(p): sha(p) for p in paths},
                                   "output_sha256": sha(out / (name + ".jsonl"))}
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["cohorts"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
