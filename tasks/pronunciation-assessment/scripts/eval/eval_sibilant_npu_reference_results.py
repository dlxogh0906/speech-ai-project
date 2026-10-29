"""Score isolated NPU reference responses using the existing full TAG metrics.

The selected time-phrase/guard subset does not estimate overall service FPR or
new actual-phone recognition. Input rejects remain in error denominators.
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
import json
import shutil
import eval_sibilant_directional as service
from eval_speech_rescue import observe
from eval_sibilant_typed_margin import comparison


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--result", required=True)
    ap.add_argument("--source-rows", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    source, out = Path(a.result), Path(a.out)
    if json.loads((source / "finished.json").read_text())["sha256"] != service.sha(source / "report.json"):
        raise ValueError("NPU validation incomplete or changed")
    runtime = json.loads((source / "report.json").read_text())
    if runtime["output_sha256"]["responses.jsonl"] != service.sha(source / "responses.jsonl"):
        raise ValueError("NPU responses changed")
    rows = service.read_jsonl(a.source_rows)
    inputs = {r["uid"]: r for r in rows}
    records = service.read_jsonl(source / "responses.jsonl")
    if len(inputs) != len(rows) or {r["uid"] for r in records} != set(inputs) or len(records) != len(inputs):
        raise ValueError("NPU input coverage differs")
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, out / Path(__file__).name)
    labels, counts = [], collections.Counter()
    with (out / "sites.jsonl").open("x") as stream:
        for record in records:
            row = inputs[record["uid"]]
            base, candidate = record["baseline"], record["candidate"]
            if base["rejected"] != candidate["rejected"]:
                raise AssertionError("input admission changed")
            if not base["rejected"]:
                before, after = service.phonemes(base["body"]), service.phonemes(candidate["body"])
                accepted = record["trace"]["accepted_positions"]
                if len(before) != len(after) or any(a != b for q, (a, b) in enumerate(zip(before, after)) if q not in accepted):
                    raise AssertionError("non-target API phoneme changed")
                counts["other_API_phoneme_changes"] += 0
                counts["changed_reference_utterances"] += record["trace"]["reference_changed"]
            views = {key: observe(record[key], row) for key in ("baseline", "candidate")}
            for site in row["sites"]:
                label = dict(site, expected=site["canon"], uid=row["uid"], item=row["item"],
                             speaker=row["speaker"], language=row.get("language"))
                label.update({key: view[site["q"]] for key, view in views.items()})
                labels.append(label)
                service.write_line(stream, label)
    result = comparison(labels, "baseline", "candidate")
    report = {"scope": __doc__, "device": runtime["device"], "inputs": len(inputs),
              "metrics": result, "counts": dict(counts), "subgroups": {},
              "source_sha256": {a.source_rows: service.sha(a.source_rows),
                                str(source / "report.json"): service.sha(source / "report.json")},
              "label_source": "existing TAG actual/LAB expected; no new listeners, reused selected cases",
              "deployment": False}
    for key in ("expected", "label_kind", "following", "position", "item", "language"):
        report["subgroups"][key] = {str(v): comparison([r for r in labels if r[key] == v], "baseline", "candidate")
                                   for v in sorted({r[key] for r in labels}, key=str)}
    (out / "report.json").write_text(json.dumps(service.jsonable(report), ensure_ascii=False, indent=2) + "\n")
    (out / "finished.json").write_text(json.dumps({"sha256": service.sha(out / "report.json")}) + "\n")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
