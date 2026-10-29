"""Rerun original scoring under a source-attested observed time reference.

Only runtime text, allowed variants and decoded ㅆ trigger the adapter. Retain
TAG denominators and recognized phones; report reference/score changes apart
from acoustic recognition. Third phones and deletions do not trigger it.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    if ROOT.parent == ROOT:
        raise RuntimeError("repository root not found")
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tasks/pronunciation-assessment/src"),
               str(ROOT / "tasks/pronunciation-assessment/scripts/train")]
import argparse
import collections
import copy
import json
import shutil
import numpy as np
import eval_sibilant_directional as service
from eval_speech_rescue import observe
from eval_sibilant_typed_margin import app_from, render
from eval_sibilant_cv_agreement import detail
from train_sibilant_local_encoder import write
from mdd.canonical_time import TimePhraseBuilder
from mdd.canonical_observed_time import observed_time_reference, FixedReference
from mdd.sibilant_margin import annotate_margin


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    p = json.loads(Path(a.protocol).read_text())
    for path, digest in p["sha256"].items():
        if service.sha(path) != digest:
            raise ValueError("frozen source changed: " + path)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(a.protocol, out / "protocol.json")
    shutil.copy2(__file__, out / Path(__file__).name)
    app, captured = app_from(p), {}
    app.S["cb"] = TimePhraseBuilder(app.S["cb"])
    builder, forward = app.S["cb"], app.S["fwd"]

    def capture(decoder, wavs):
        if captured.get("reuse"):
            np.testing.assert_array_equal(wavs[0], captured["wav"])
            return captured["lp"].clone(), copy.deepcopy(captured["lengths"])
        lp, lengths = forward(decoder, wavs)
        captured.update(wav=wavs[0].copy(), lp=lp.clone(), lengths=copy.deepcopy(lengths))
        return lp, lengths

    app.S["fwd"] = capture
    report = {"scope": p["scope"], "deployment": False, "cohorts": {}}
    for name, config in p["cohorts"].items():
        rows, counts, audits = [], collections.Counter(), []
        with (out / (name + ".responses.jsonl")).open("x") as stream:
            for record in service.read_jsonl(config["source"]):
                row, base = record["row"], record[config["base_key"]]
                candidate = dict(base)
                if base["raw"] is not None:
                    raw = base["raw"]
                    proposed, accepted, reason = observed_time_reference(builder, row["prompt"], raw["canon"], raw["sites"])
                    counts[reason] += 1
                    phones = service.phonemes(base["body"])
                    needed = proposed != raw["canon"] or any(phones[q].get("actual_jamo") != "ㅆ" or phones[q].get("errorType") for q in accepted)
                    if accepted and needed:
                        x = service.load_audio(row)[:320000]
                        captured["reuse"] = False
                        first, _, old = app.analyze(x, row["prompt"], "")
                        lp = captured["lp"][0, :captured["lengths"][0]].numpy()
                        dec = app.S["dec"]
                        sites, _ = annotate_margin(lp, old["canon"], old["sites"], dec.d.index, dec.blank, p["pair_thresholds"])
                        if render(app, row, old, sites) != base["body"]:
                            raise AssertionError("prior API output did not reproduce")
                        try:
                            app.S["cb"] = FixedReference(row["prompt"], proposed)
                            captured["reuse"] = True
                            second, _, current = app.analyze(x, row["prompt"], "")
                        finally:
                            app.S["cb"] = builder
                            captured["reuse"] = False
                        if first["pred_jamo"] != second["pred_jamo"] or current["canon"] != proposed:
                            raise AssertionError("recognition or forced reference changed unexpectedly")
                        annotated, _ = annotate_margin(lp, current["canon"], current["sites"], dec.d.index, dec.blank, p["pair_thresholds"])
                        for q in accepted:
                            annotated[q].pop("sz", None)
                        body = render(app, row, current, annotated)
                        new = service.phonemes(body)
                        outside = [i for i, (old_p, new_p) in enumerate(zip(phones, new)) if i not in accepted
                                   and (old_p.get("actual_jamo") != new_p.get("actual_jamo") or old_p.get("errorType") != new_p.get("errorType"))]
                        if len(new) != len(phones) or outside or any(new[q].get("actual_jamo") != "ㅆ" or new[q].get("errorType") for q in accepted):
                            counts["deferred_alignment_or_other_phone_change"] += 1
                        else:
                            candidate.update(raw=current, body=body)
                            counts["changed_utterances"] += 1
                            counts["non_target_score_changes"] += sum(a.get("pronunciation") != b.get("pronunciation") for i, (a, b) in enumerate(zip(phones, new)) if i not in accepted)
                            audits.append({"uid": row["uid"], "accepted_positions": accepted, "old_canonical": raw["canon"],
                                           "new_canonical": proposed, "prediction_unchanged": True,
                                           "target_before": [phones[q] for q in accepted], "target_after": [new[q] for q in accepted],
                                           "utterance_scores_before": raw["scores"], "utterance_scores_after": current["scores"]})
                else:
                    counts["input_rejected"] += 1
                viewed = {"original": observe(record["original"], row), "reference": observe(base, row),
                          "prior": observe(base, row), "candidate": observe(candidate, row)}
                for site in row["sites"]:
                    value = dict(site, expected=site["canon"], uid=row["uid"], speaker=row["speaker"], item=row["item"], language=row.get("language"))
                    value.update({k: v[site["q"]] for k, v in viewed.items()})
                    rows.append(value)
                service.write_line(stream, {"uid": row["uid"], "body": candidate["body"], "rejected": candidate["rejected"]})
        for suffix, values in (("sites", rows), ("reference_changes", audits)):
            with (out / (name + "." + suffix + ".jsonl")).open("x") as stream:
                for value in values:
                    service.write_line(stream, value)
        result = detail(rows, counts)
        report["cohorts"][name] = result
        write(out / "report.json", report)
        print(name, json.dumps(result["versus_prior"], ensure_ascii=False), flush=True)
    write(out / "finished.json", {"sha256": service.sha(out / "report.json")})


if __name__ == "__main__":
    main()
