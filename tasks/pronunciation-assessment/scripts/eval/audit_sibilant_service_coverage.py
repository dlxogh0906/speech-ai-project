"""Expose visible alerts on TAG-normal sites even when runtime canonical differs.

This supplements, rather than changes, frozen contrast experiments. It does
not resolve whether a disputed reference pronunciation is linguistically
mandatory. Input rejection and ambiguous positions remain explicit.
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

PAIR = {"ㅅ": "ㅆ", "ㅆ": "ㅅ"}


def audit(rows):
    output = {}
    for expected, alternate in PAIR.items():
        group = [r for r in rows if r["expected"] == expected]
        normals = [r for r in group if r["actual"] == expected]
        errors = [r for r in group if r["actual"] == alternate]

        def position_known(r):
            return (r.get("q_runtime") is not None and r.get("baseline_phone") is not None
                    and r["state"] not in ("input_rejected", "runtime_mapping_ambiguous"))

        known = [r for r in normals if position_known(r)]
        result = {"normal_all": len(normals), "normal_position_known_accepted": len(known),
                  "normal_excluded_states": dict(collections.Counter(r["state"] for r in normals if not position_known(r))),
                  "error_all": len(errors), "error_states": dict(collections.Counter(r["state"] for r in errors)),
                  "normal_canonical_disagreement_examples": [r for r in known if r["state"] == "runtime_canonical_disagrees"]}
        for label, key in (("baseline", "baseline_phone"), ("candidate", "candidate_phone")):
            alerts = [r for r in known if (r.get(key) or {}).get("errorType")]
            mismatches = [r for r in alerts if r["state"] == "runtime_canonical_disagrees"]
            result[label] = {"tag_normal_visible_alerts": len(alerts),
                             "tag_normal_visible_alert_rate": len(alerts) / len(known) if known else None,
                             "alerts_on_canonical_disagreements": len(mismatches),
                             "correct_direction_error_detections": sum(
                                 position_known(r) and bool((r.get(key) or {}).get("errorType"))
                                 and r[key].get("alpha_jamo") == expected
                                 and r[key].get("actual_jamo") == alternate for r in errors)}
        output[expected + "→" + alternate] = result
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    report = {"scope": "Existing TAG-normal labels, known runtime position, actual outward alerts; includes canonical disagreements. Not an independent human correctness judgment.",
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "files": {}}
    for name in args.sites:
        path = Path(name)
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        report["files"][str(path)] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "audit": audit(rows)}
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    for path, result in report["files"].items():
        print(Path(path).name, {k: {"known_normals": v["normal_position_known_accepted"],
                                  "baseline": v["baseline"], "candidate": v["candidate"]}
                               for k, v in result["audit"].items()}, flush=True)


if __name__ == "__main__":
    main()
