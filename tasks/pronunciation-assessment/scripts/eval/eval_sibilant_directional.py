# -*- coding: utf-8 -*-
"""Offline screen-response evaluation of a narrowly scoped ㅆ→ㅅ add-on.

Run baseline and candidate in separate processes (one GPU model at a time).
The documented production source/configuration is reused; this does not attest
that a remote deployment currently runs that exact source or binary.
No threshold, checkpoint, canonical pronunciation, or score is fitted here.
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
import importlib.util
import itertools
import json
import math
import time
import wave

import numpy as np

VOWELS = set("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
VARIANTS = ("baseline", "conservative", "broad")
COHORTS = ("test_rp", "test_rs029", "native_burden", "diagnostic_other", "reproduction")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("nonfinite value in captured production response")
    return value


def write_line(f, row):
    f.write(json.dumps(jsonable(row), ensure_ascii=False, allow_nan=False) + "\n")
    f.flush()


def load_rows(path):
    p = Path(path)
    files = ([p] if p.is_file() else ([p / "eval.jsonl"] if (p / "eval.jsonl").exists()
             else [p / (c + ".jsonl") for c in COHORTS if (p / (c + ".jsonl")).exists()]))
    if not files:
        raise ValueError("no supported evaluation cohorts found")
    rows, hashes = [], {}
    for f in files:
        hashes[f.name] = sha(f)
        seen = set()
        for r in read_jsonl(f):
            if r["uid"] in seen:
                raise ValueError("duplicate uid within cohort: " + r["uid"])
            seen.add(r["uid"])
            r = dict(r)
            r["cohort"] = r.get("cohort", f.stem)
            if not r.get("prompt"):
                raise ValueError("API refText is missing: " + r["uid"])
            for s in r.get("sites", []):
                q, a = s["q"], s.get("a")
                if r["canonical"][q] != s["canon"]:
                    raise ValueError("LAB index mismatch: " + r["uid"])
                if a is not None and r.get("actual") is not None and r["actual"][a] != s.get("actual"):
                    raise ValueError("TAG index mismatch: " + r["uid"])
            rows.append(r)
    return rows, hashes


def load_audio(row):
    with wave.open(row["wav"], "rb") as f:
        if (f.getframerate(), f.getnchannels(), f.getsampwidth()) != (16000, 1, 2):
            raise ValueError("expected mono PCM16 16k WAV: " + row["uid"])
        if row.get("frames") is not None and f.getnframes() != row["frames"]:
            raise ValueError("WAV frame mismatch: " + row["uid"])
        return np.frombuffer(f.readframes(f.getnframes()), np.int16).astype(np.float32) / 32768.0


def app_module(a, checkpoint):
    import torch
    torch.set_num_threads(a.threads)
    spec = importlib.util.spec_from_file_location("directional_production_source", ROOT / "deploy/pronunciation-demo/app.py")
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    args = argparse.Namespace(hf_ckpt=checkpoint, rbln=None, scorer=a.scorer, norms=a.norms,
                              calib=a.calib, flu_rule=a.flu_rule, utt_pct=a.utt_pct, utt_comp=False,
                              canon_fix=True, canon_fix_margin=1.0, del_run_min=1, del_score_max=50,
                              liquid_rule=True, ss_rule=False, sz_detector=None,
                              nomatch_reject=0.0, too_long_reject=False, no_korean_reject=False)
    app.load_all(args)
    # This annotation is not read by the speech formatter and has no API effect.
    # Avoid loading country-specific feedback for an offline evaluation with country="".
    app.S["l1fb"] = None
    return app


def phonemes(body):
    return [p for w in body["result"]["words"] for p in w["phonemes"]]


def response(app, row, raw, sites=None):
    return app.S["ss"].build(row["prompt"], raw["canon"], sites if sites is not None else raw["sites"],
                             raw["margins"], utt_scores=raw["scores"], record_id=None,
                             vowels=app.S["vowels"], feats=raw.get("feats"))


def metadata(a, hashes):
    source_files = [ROOT / "deploy/pronunciation-demo/app.py", ROOT / "tasks/pronunciation-assessment/src/mdd/speechsuper.py",
                    ROOT / "tasks/pronunciation-assessment/src/mdd/gop_feats_v2.py",
                    ROOT / "tasks/pronunciation-assessment/src/mdd/canonical_fix.py", Path(__file__)]
    helper = ROOT / "tasks/pronunciation-assessment/src/mdd/sibilant_directional.py"
    if helper.exists():
        source_files.append(helper)
    return {"data_sha256": hashes, "base_ckpt": a.base_ckpt, "candidate_ckpt": a.candidate_ckpt,
            "source_sha256": {str(p): sha(p) for p in source_files},
            "scoring_sha256": {str(p): sha(p) for p in (a.scorer, a.norms, a.calib, a.flu_rule, a.utt_pct)},
            "configuration": {"canon_fix": True, "canon_margin": 1.0, "del_run_min": 1,
                              "del_score_max": 50, "liquid_rule": True, "ss_rule": False,
                              "baseline_sz_rule": False, "min_candidate_probability": 0.6,
                              "no_match_reject": False, "too_long_reject": False, "no_korean_reject": False,
                              "common_input_gates": "0.3s minimum; production low_speech; first20s",
                              "primary": "conservative: onset ㅆ; utterance first or previous canonical vowel",
                              "diagnostic": "broad: all onset ㅆ including post-coda"},
            "note": "Offline current source with documented operating flags; no remote deployment identity attested."}


def baseline(a):
    rows, hashes = load_rows(a.data)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "baseline.jsonl"
    if path.exists():
        raise FileExistsError(path)
    app = app_module(a, a.base_ckpt)
    meta = metadata(a, hashes)
    (out / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    started = time.time()
    with path.open("w", encoding="utf-8") as f:
        for n, row in enumerate(rows, 1):
            x = load_audio(row)
            rec = {"row": row, "input_frames": len(x), "truncated_frames": max(0, len(x) - 320000)}
            code = "too_short" if len(x) < 4800 else ("no_reftext" if not row["prompt"].strip() else None)
            info = None
            if code is None:
                code, _msg, info = app.check_audio(x[:320000], row["prompt"])
            if code:
                rec.update(rejected=code, audio_gate=info, body=None, raw=None)
            else:
                _, _, raw = app.analyze(x[:320000], row["prompt"], "")
                body = response(app, row, raw)
                ph = phonemes(body)
                if [p["alpha_jamo"] for p in ph] != raw["canon"]:
                    raise AssertionError("formatter canonical position mismatch: " + row["uid"])
                rec.update(rejected=None, audio_gate=info, body=body, raw=raw)
            write_line(f, rec)
            if n % 25 == 0 or n == len(rows):
                print("[baseline] %d/%d %.1fs" % (n, len(rows), time.time() - started), flush=True)
    (out / "baseline.finished").write_text(str(len(rows)) + "\n")


def all_optimal_mapping(reference, target):
    """Each reference position's possible target indices over ALL optimal edits.

    A unique edlib path can hide repeated-token ambiguity. Here an ambiguous
    location (including an equally good deletion) is explicitly unscorable.
    """
    if reference == target:
        return [{i} for i in range(len(reference))]
    n, m = len(reference), len(target)
    f = np.empty((n + 1, m + 1), np.int32)
    b = np.empty_like(f)
    f[:, 0], f[0, :] = np.arange(n + 1), np.arange(m + 1)
    for i in range(n):
        for j in range(m):
            f[i + 1, j + 1] = min(f[i, j + 1] + 1, f[i + 1, j] + 1,
                                  f[i, j] + (reference[i] != target[j]))
    b[:, m], b[n, :] = np.arange(n, -1, -1), np.arange(m, -1, -1)
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            b[i, j] = min(b[i + 1, j] + 1, b[i, j + 1] + 1,
                          b[i + 1, j + 1] + (reference[i] != target[j]))
    d, opts = int(f[n, m]), [set() for _ in reference]
    for i in range(n):
        for j in range(m + 1):
            if f[i, j] + 1 + b[i + 1, j] == d:
                opts[i].add(None)
            if j < m and f[i, j] + (reference[i] != target[j]) + b[i + 1, j + 1] == d:
                opts[i].add(j)
    return opts


def onset(tokens, q):
    return q + 1 < len(tokens) and tokens[q + 1] in VOWELS


def decisions(p):
    if p is None:
        return {"any_error": False, "ss_to_s": False, "heard_s": False, "actual": None,
                "errorType": None, "uncertain": None, "pronunciation": None}
    shown = bool(p.get("errorType"))
    return {"any_error": shown, "ss_to_s": shown and p.get("actual_jamo") == "ㅅ",
            "heard_s": p.get("actual_jamo") == "ㅅ", "actual": p.get("actual_jamo"),
            "errorType": p.get("errorType"), "uncertain": bool(p.get("uncertain")),
            "pronunciation": p.get("pronunciation")}


def label_records(row, canon, bodies, rejected):
    lab = row["canonical"]
    opts = all_optimal_mapping(lab, canon) if canon is not None else [set() for _ in lab]
    phs = {k: phonemes(v) if v else [] for k, v in bodies.items()}
    targets = []
    for site in row.get("sites", []):
        if site["canon"] != "ㅆ":
            continue
        q = site["q"]
        possible = opts[q]
        r = next(iter(possible)) if len(possible) == 1 else None
        if rejected:
            status = "rejected_" + rejected
        elif site.get("alignment_ambiguous"):
            status = "expert_TAG_alignment_ambiguous"
        elif not possible or possible == {None}:
            status = "runtime_unmapped"
        elif len(possible) != 1:
            status = "runtime_alignment_ambiguous"
        elif r is None or canon[r] != lab[q]:
            status = "runtime_canonical_disagrees"
        elif not onset(lab, q) or not onset(canon, r):
            status = "not_onset_both_axes"
        else:
            status = "mapped"
        target = {"q": q, "a": site.get("a"), "canon": "ㅆ", "actual": site.get("actual"),
                  "label_kind": site.get("label_kind"), "alignment_ambiguous": bool(site.get("alignment_ambiguous")),
                  "runtime_q": r, "mapping_status": status,
                  "runtime_possible_indices": sorted(possible, key=lambda x: -1 if x is None else x),
                  "lab_conservative_context": onset(lab, q) and (q == 0 or lab[q - 1] in VOWELS),
                  "runtime_conservative_context": r is not None and canon is not None and onset(canon, r)
                                                    and (r == 0 or canon[r - 1] in VOWELS)}
        for k in VARIANTS:
            target[k] = decisions(phs[k][r] if status == "mapped" else None)
        targets.append(target)
    return targets


def candidate(a):
    from mdd.sibilant_directional import annotate_directional
    from mdd.gop_feats_v2 import attach_pred_probs
    from mdd import speechsuper
    out = Path(a.out)
    if not (out / "baseline.finished").exists():
        raise RuntimeError("baseline is incomplete")
    path = out / "predictions.jsonl"
    if path.exists():
        raise FileExistsError(path)
    rows, hashes = load_rows(a.data)
    saved = read_jsonl(out / "baseline.jsonl")
    meta = json.loads((out / "metadata.json").read_text())
    if hashes != meta["data_sha256"] or len(rows) != len(saved):
        raise ValueError("data changed after baseline")
    current = metadata(a, hashes)
    for key in ("base_ckpt", "candidate_ckpt", "scoring_sha256"):
        if current[key] != meta[key]:
            raise ValueError("configuration changed after baseline: " + key)
    for p, h in meta["source_sha256"].items():
        if sha(p) != h:
            raise ValueError("production/evaluation source changed after baseline: " + p)
    app = app_module(a, a.candidate_ckpt)
    dec = app.S["dec"]
    started = time.time()
    with path.open("w", encoding="utf-8") as f:
        for n, (row, base) in enumerate(zip(rows, saved), 1):
            if row != base["row"]:
                raise AssertionError("row changed after baseline")
            rec = {"uid": row["uid"], "speaker": row["speaker"], "prompt": row["prompt"],
                   "cohort": row["cohort"], "item": row.get("item"), "role": row.get("role"),
                   "has_expert_actual": row.get("actual") is not None, "rejected": base["rejected"],
                   "truncated_frames": base["truncated_frames"], "candidate": None,
                   "variants": {}, "other_phoneme_changes": 0}
            if base["rejected"]:
                rec["targets"] = label_records(row, None, {k: None for k in VARIANTS}, base["rejected"])
            else:
                raw, canon = base["raw"], base["raw"]["canon"]
                # Verify JSON round-trip leaves the complete baseline response unchanged.
                speechsuper.SZ_RULE = False
                rebuilt = jsonable(response(app, row, raw))
                if rebuilt != base["body"]:
                    raise AssertionError("baseline formatter reconstruction changed: " + row["uid"])
                x = load_audio(row)
                w, vad = app.S["trim"](x[:320000], 16000)
                if jsonable(vad) != raw["vad"]:
                    raise AssertionError("different VAD input: " + row["uid"])
                lp, lens = app.S["fwd"](dec, [w])
                l = lp[0, :lens[0]].numpy()
                hyp = [dec.d[int(v)] for v, _ in itertools.groupby(l.argmax(-1)) if int(v) != dec.blank]
                hyp = [t for t in hyp if len(t) == 1 and 0x3131 <= ord(t) <= 0x3163]
                sites = app.align(canon, hyp)
                attach_pred_probs(dec, l, sites, hyp)
                rec["candidate"] = {"hypothesis": hyp, "sites": sites}
                bodies = {"baseline": base["body"]}
                audits = {}
                for name, broad in (("conservative", False), ("broad", True)):
                    merged, audit = annotate_directional(canon, raw["sites"], sites, app.S["vowels"],
                                                         min_prob=0.6, allow_post_coda=broad)
                    allowed = {i for i, s in enumerate(merged) if s.get("sz")}
                    speechsuper.SZ_RULE = True
                    body = jsonable(response(app, row, raw, merged))
                    speechsuper.SZ_RULE = False
                    old, new = phonemes(base["body"]), phonemes(body)
                    if len(old) != len(new):
                        raise AssertionError("phoneme count changed")
                    changed = [i for i, (p, q) in enumerate(zip(old, new)) if p != q]
                    other = [i for i in changed if i not in allowed]
                    if other:
                        raise AssertionError("non-target phonemes changed: %r" % other)
                    if any(p.get("pronunciation") != q.get("pronunciation") for p, q in zip(old, new)):
                        raise AssertionError("baseline phoneme confidence changed")
                    if any(canon[i] != "ㅆ" or new[i].get("actual_jamo") != "ㅅ" for i in changed):
                        raise AssertionError("change escaped ㅆ→ㅅ scope")
                    at, owners, word_changes, word_score_changes = 0, set(), [], []
                    for wi, word in enumerate(base["body"]["result"]["words"]):
                        if any(q in allowed for q in range(at, at + len(word["phonemes"]))):
                            owners.add(wi)
                        at += len(word["phonemes"])
                    for wi, (p, q) in enumerate(zip(base["body"]["result"]["words"], body["result"]["words"])):
                        if p != q:
                            word_changes.append(wi)
                            if wi not in owners:
                                raise AssertionError("non-target word changed")
                            if {k: v for k, v in p.items() if k not in ("phonemes", "score", "readType")} != {
                                    k: v for k, v in q.items() if k not in ("phonemes", "score", "readType")}:
                                raise AssertionError("target owner non-aggregate fields changed")
                        if p["score"] != q["score"]:
                            word_score_changes.append({"word_index": wi, "before": p["score"], "after": q["score"],
                                                       "absolute_delta": abs((q["score"] or 0) - (p["score"] or 0))})
                    score_changes = [k for k, value in base["body"]["result"].items()
                                     if k != "words" and body["result"][k] != value]
                    score_deltas = {k: {"before": base["body"]["result"][k], "after": body["result"][k],
                                       "absolute_delta": abs((body["result"][k] or 0) - (base["body"]["result"][k] or 0))}
                                    for k in score_changes}
                    audit = dict(audit)
                    audit.update(annotated_indices=sorted(allowed), changed_indices=changed, other_phoneme_changes=0,
                                 changed_owner_words=word_changes, owner_score_changes=word_score_changes,
                                 aggregate_score_changes=score_changes, aggregate_score_deltas=score_deltas)
                    audits[name] = audit
                    bodies[name] = body
                rec["runtime_canonical"] = canon
                rec["targets"] = label_records(row, canon, bodies, None)
                mapped = {s["runtime_q"] for s in rec["targets"] if s["mapping_status"] == "mapped"}
                for name, body in bodies.items():
                    ph = phonemes(body)
                    ssq = [i for i, t in enumerate(canon) if t == "ㅆ" and onset(canon, i)]
                    rec["variants"][name] = {"body": body, "audit": audits.get(name),
                        "phone_n": len(ph), "visible_error_n": sum(bool(p.get("errorType")) for p in ph),
                        "ss_onset_n": len(ssq),
                        "ss_visible_error_n": sum(bool(ph[i].get("errorType")) for i in ssq),
                        "ss_to_s_visible_n": sum(bool(ph[i].get("errorType")) and ph[i].get("actual_jamo") == "ㅅ" for i in ssq),
                        "unmapped_ss_to_s_indices": [i for i in ssq if i not in mapped and bool(ph[i].get("errorType")) and ph[i].get("actual_jamo") == "ㅅ"]}
            write_line(f, rec)
            if n % 25 == 0 or n == len(rows):
                print("[candidate] %d/%d %.1fs" % (n, len(rows), time.time() - started), flush=True)
    (out / "candidate.finished").write_text(str(len(rows)) + "\n")


def cluster_delta(records, variant, measure, unit, truth, iterations=1000):
    buckets = collections.defaultdict(lambda: [0, 0, 0])
    for row, target in records:
        if target["actual"] != truth:
            continue
        b = buckets[row[unit]]
        b[0] += 1
        b[1] += target["baseline"][measure]
        b[2] += target[variant][measure]
    if not buckets:
        return {"units": 0, "n": 0, "difference": None, "ci95": None}
    x = np.asarray(list(buckets.values()), dtype=np.float64)
    rng = np.random.RandomState(231023)
    draws = []
    for _ in range(iterations):
        s = x[rng.randint(len(x), size=len(x))].sum(0)
        draws.append((s[2] - s[1]) / s[0])
    s = x.sum(0)
    return {"units": len(x), "n": int(s[0]), "difference": float((s[2] - s[1]) / s[0]),
            "ci95": np.percentile(draws, [2.5, 97.5]).tolist(), "resamples": iterations}


def summarize(a):
    out = Path(a.out)
    if not (out / "candidate.finished").exists():
        raise RuntimeError("candidate is incomplete")
    rows = read_jsonl(out / "predictions.jsonl")
    result = {"metadata": json.loads((out / "metadata.json").read_text()), "cohorts": {},
              "denominator_policy": "All listed expert ㅆ sites retained. Rejection, mapping ambiguity/disagreement, or missing output counts as not detected; separate counts below. Native labels absent: burden only, never false-positive rate.",
              "ci_note": "Paired cluster bootstrap with fixed preregistered rule, no test tuning. Speaker and prompt cluster intervals are separate, not joint or an unseen-prompt guarantee.",
              "endpoint_note": "Response phoneme/errorType and word readType from production formatter. Dynamic HTTP metadata such as timing/token/userId omitted."}
    for cohort in sorted({r["cohort"] for r in rows}):
        subset = [r for r in rows if r["cohort"] == cohort]
        recs = [(r, t) for r in subset for t in r["targets"]]
        report = {"utterances": len(subset), "speakers": len({r["speaker"] for r in subset if r["speaker"] is not None}),
                  "prompts": len({r["prompt"] for r in subset}),
                  "unknown_speaker_utterances": sum(r["speaker"] is None for r in subset),
                  "rejections": dict(collections.Counter(r["rejected"] for r in subset if r["rejected"])),
                  "truncated_utterances": sum(r["truncated_frames"] > 0 for r in subset),
                  "mapping": dict(collections.Counter(t["mapping_status"] for _, t in recs)),
                  "expert_actual": dict(collections.Counter(str(t["actual"]) for _, t in recs)),
                  "other_phoneme_changes": sum(r["other_phoneme_changes"] for r in subset), "scopes": {}, "burden": {}}
        for scope in ("all_expert_ss", "lab_conservative_context", "mapped_expert_ss", "mapped_conservative_context"):
            selected = recs
            if "conservative_context" in scope:
                selected = [(r, t) for r, t in selected if t["lab_conservative_context"]]
            if scope.startswith("mapped_"):
                selected = [(r, t) for r, t in selected if t["mapping_status"] == "mapped"]
            if scope == "mapped_conservative_context":
                selected = [(r, t) for r, t in selected if t["runtime_conservative_context"]]
            summary = {}
            for variant in VARIANTS:
                vc = {}
                for measure in ("any_error", "ss_to_s", "heard_s"):
                    errors = [t for _, t in selected if t["actual"] == "ㅅ"]
                    normals = [t for _, t in selected if t["actual"] == "ㅆ"]
                    tp = sum(t[variant][measure] for t in errors)
                    fp = sum(t[variant][measure] for t in normals)
                    vc[measure] = {"tp": tp, "errors": len(errors), "recall": tp / len(errors) if errors else None,
                                   "fp": fp, "normals": len(normals), "fpr": fp / len(normals) if normals else None}
                    if variant != "baseline":
                        vc[measure]["paired_differences"] = {unit: {
                            "recall": cluster_delta(selected, variant, measure, unit, "ㅅ"),
                            "fpr": cluster_delta(selected, variant, measure, unit, "ㅆ")}
                            for unit in ("speaker", "prompt")}
                summary[variant] = vc
            report["scopes"][scope] = summary
        for variant in VARIANTS:
            counted = [r["variants"][variant] for r in subset if variant in r["variants"]]
            keys = ("phone_n", "visible_error_n", "ss_onset_n", "ss_visible_error_n", "ss_to_s_visible_n")
            b = {k: sum(r[k] for r in counted) for k in keys}
            b["accepted_utterances"] = len(counted)
            b["unmapped_ss_to_s_n"] = sum(len(r["unmapped_ss_to_s_indices"]) for r in counted)
            b["error_flags_per100phones"] = 100 * b["visible_error_n"] / b["phone_n"] if b["phone_n"] else None
            if variant != "baseline":
                b["annotated_sites"] = sum(len(r["audit"]["annotated_indices"]) for r in counted)
                b["changed_sites"] = sum(len(r["audit"]["changed_indices"]) for r in counted)
                b["aggregate_score_changed_utterances"] = sum(bool(r["audit"]["aggregate_score_changes"]) for r in counted)
                b["aggregate_score_field_changes"] = dict(collections.Counter(
                    key for r in counted for key in r["audit"]["aggregate_score_changes"]))
                b["aggregate_score_max_absolute_delta"] = max(
                    [v["absolute_delta"] for r in counted for v in r["audit"]["aggregate_score_deltas"].values()] or [0])
                b["owner_score_changed_words"] = sum(len(r["audit"]["owner_score_changes"]) for r in counted)
                b["owner_score_max_absolute_delta"] = max(
                    [v["absolute_delta"] for r in counted for v in r["audit"]["owner_score_changes"]] or [0])
            report["burden"][variant] = b
        result["cohorts"][cohort] = report
    (out / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for cohort, report in result["cohorts"].items():
        print(cohort, "utterances", report["utterances"], "mapping", report["mapping"])
        for variant, scores in report["scopes"]["all_expert_ss"].items():
            print(" ", variant, json.dumps(scores["ss_to_s"], ensure_ascii=False))


def main():
    from shared.paths import MDD_OUTPUTS
    sp = Path(MDD_OUTPUTS) / "score_phoneme"
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=("baseline", "candidate", "report"), required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--base-ckpt", required=True)
    ap.add_argument("--candidate-ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--scorer", default=str(Path(MDD_OUTPUTS) / "score_v2_w2vbert/score_models_v2_w2vbert.joblib"))
    ap.add_argument("--norms", default=str(sp / "norms.json"))
    ap.add_argument("--calib", default=str(sp / "calibration.json"))
    ap.add_argument("--flu-rule", default=str(sp / "fluency_rule.json"))
    ap.add_argument("--utt-pct", default=str(sp / "utt_percentile.json"))
    a = ap.parse_args()
    {"baseline": baseline, "candidate": candidate, "report": summarize}[a.stage](a)


if __name__ == "__main__":
    main()
