"""Freeze a blinded listening and acoustic-inspection set from existing TAG data.

This is an exploratory measurement set, not a new held-out performance test.
TAG phone intervals are candidate locations, NOT frication/aspiration boundaries.
No new human judgements or acoustic boundary labels are fabricated here.
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
import json
import random
import re
import wave

from shared.paths import DATA_ROOT, MDD_OUTPUTS

VOWELS = set("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
FOCUS = {"RP003": "쓰기", "RW137": "있어요", "RS058": "늦었어",
         "RS015": "보셨어요", "RS020": "닮았어요"}


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(s) for s in f if s.strip()]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def stable(text):
    return hashlib.sha256(("sibilant-evidence-v1:" + text).encode()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tiers(path):
    b = Path(path).read_bytes()
    text = b.decode("utf-16") if b[:2] in (b"\xff\xfe", b"\xfe\xff") else b.decode("utf-8-sig")
    out = {}
    for m in re.finditer(r'name = "([^"]+)"(.*?)(?=name = "|\Z)', text, re.S):
        ivs = re.findall(r'xmin = ([-\d.eE]+)\s*xmax = ([-\d.eE]+)\s*text = "([^"]*)"', m.group(2))
        out[m.group(1)] = [(float(a), float(b), t.strip()) for a, b, t in ivs]
    return out


def jamo(c):
    return "\u3131" <= c <= "\u3163"


def locate(row, site, tag_dirs):
    stem = row.get("parent_uid", row["uid"])
    matches = [d / (stem + ".TextGrid") for d in tag_dirs if (d / (stem + ".TextGrid")).is_file()]
    if len(matches) != 1:
        raise ValueError("expected one original TAG file: " + stem)
    path = matches[0]
    tt = tiers(path)
    segment = row.get("source_segment", [0., float("inf")])
    tokens = []
    for start, end, label in tt["phones"]:
        if not segment[0] <= (start + end) / 2 < segment[1]:
            continue
        for c in label:
            if jamo(c):
                tokens.append((start, end, c, label))
    if [p[2] for p in tokens] != row["actual"]:
        raise ValueError("original TAG/frozen actual mismatch: " + row["uid"])
    a = site["a"]
    c, v = tokens[a], tokens[a + 1]
    if c[2] != site["actual"] or v[2] not in VOWELS:
        raise ValueError("ambiguous phone interval: " + row["uid"])
    if c[1] > v[0] + .001 or c[1] <= c[0] or v[1] <= v[0]:
        raise ValueError("invalid candidate intervals")
    words = [(s, e, w) for s, e, w in tt.get("words", []) if w and s <= (c[0] + c[1]) / 2 < e]
    source_wav = Path(row.get("source_wav", row["wav"]))
    with wave.open(str(source_wav), "rb") as f:
        if (f.getframerate(), f.getnchannels(), f.getsampwidth()) != (16000, 1, 2):
            raise ValueError("expected original mono PCM16/16k")
        duration = f.getnframes() / 16000
    word = words[0] if len(words) == 1 else (c[0], v[1], "")
    return {"tag_file": str(path), "tag_sha256": digest(path), "source_wav": str(source_wav),
            "source_sha256": digest(source_wav), "source_duration": duration,
            "tag_consonant": list(c[:2]), "tag_vowel": list(v[:2]),
            "tag_consonant_label": c[3], "tag_vowel_label": v[3],
            "tag_intervals_isolated": len(c[3]) == 1 and len(v[3]) == 1,
            "tag_word": list(word), "actual_vowel": v[2],
            "previous_tag_phone": list(tokens[a - 1][:3]) if a else None,
            "local_interval": [max(0., c[0] - .12), min(duration, v[1] + .04)],
            "context_interval": [max(0., word[0] - .12), min(duration, word[1] + .12)],
            "analysis_interval": [max(0., c[0] - .20), min(duration, v[1] + .15)]}


def crop(source, interval, path):
    with wave.open(str(source), "rb") as f:
        sr = f.getframerate()
        start = max(0, int(round(interval[0] * sr)))
        end = min(f.getnframes(), int(round(interval[1] * sr)))
        f.setpos(start)
        data = f.readframes(end - start)
    with wave.open(str(path), "wb") as f:
        f.setparams((1, 2, sr, 0, "NONE", "not compressed"))
        f.writeframes(data)
    return {"start_sample": start, "end_sample": end, "sample_rate": sr,
            "seconds": (end - start) / sr, "sha256": digest(path)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=MDD_OUTPUTS / "analysis/ss_preserve_2026-10-23/data/eval.jsonl")
    ap.add_argument("--predictions", type=Path, default=MDD_OUTPUTS / "analysis/ss_preserve_2026-10-23/evaluations/normal4/predictions.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--normal-per-item", type=int, default=8)
    ap.add_argument("--controls", type=int, default=50)
    a = ap.parse_args()
    if a.out.exists():
        ap.error("refusing to overwrite an existing evidence packet")
    ev = load_module("directional_evidence_mapping", ROOT / "tasks/pronunciation-assessment/scripts/eval/eval_sibilant_directional.py")
    rows = read_jsonl(a.data)
    preds = {r["uid"]: r for r in read_jsonl(a.predictions)}
    focus_pool = collections.defaultdict(list)
    selected = []
    for row in rows:
        if row["item"] not in FOCUS:
            continue
        cohort = "reused_fresh_rp" if row["item"] == "RP003" else "fresh_speaker_confirmation"
        if row["cohort"] != cohort:
            continue
        pred = preds[row["uid"]]
        for target in pred["targets"]:
            if target["mapping_status"] != "mapped" or not target["runtime_conservative_context"]:
                continue
            if target["actual"] not in ("ㅅ", "ㅆ"):
                continue
            # RP003 target is the canonical /s*/ of 쓰기; other focus items have one eligible target.
            q = target["q"]
            if row["item"] == "RP003" and row["canonical"][q + 1] != "ㅡ":
                continue
            rec = {"row": row, "site": target, "selection_role": "focus_error" if target["actual"] == "ㅅ" else "focus_normal"}
            focus_pool[(row["item"], target["actual"])].append(rec)
    used = set()
    for item in FOCUS:
        errors = sorted(focus_pool[(item, "ㅅ")], key=lambda x: stable(x["row"]["uid"]))
        selected.extend(errors)
        prompts = {r["row"]["norm_prompt"] for r in errors}
        normal = [r for r in focus_pool[(item, "ㅆ")] if r["row"]["norm_prompt"] in prompts]
        normal.sort(key=lambda r: (not r["site"]["conservative"]["ss_to_s"], stable(r["row"]["uid"])))
        selected.extend(normal[:a.normal_per_item])
    used.update((r["row"]["uid"], r["site"]["q"]) for r in selected)
    used_uids = {u for u, q in used}
    # Context-varied controls prevent a listening set consisting only of expected /s*/.
    pools = collections.defaultdict(list)
    for row in rows:
        actual, canonical = row.get("actual"), row.get("canonical")
        if not actual or not canonical or row["uid"] in used_uids:
            continue
        opts = ev.all_optimal_mapping(canonical, actual)
        for q, c in enumerate(canonical[:-1]):
            if c not in ("ㅅ", "ㅆ") or canonical[q + 1] not in VOWELS or len(opts[q]) != 1:
                continue
            j = next(iter(opts[q]))
            if j is None or j + 1 >= len(actual) or actual[j] not in ("ㅅ", "ㅆ") or actual[j + 1] != canonical[q + 1]:
                continue
            site = {"q": q, "a": j, "canon": c, "actual": actual[j], "alignment_ambiguous": False}
            context = "initial" if q == 0 else "vowel" if canonical[q - 1] in VOWELS else "coda"
            pools[(c, actual[j], canonical[q + 1], context)].append({"row": row, "site": site, "selection_role": "context_control"})
    for bucket in pools.values():
        bucket.sort(key=lambda r: stable(r["row"]["uid"] + str(r["site"]["q"])))
    keys = sorted(pools, key=lambda k: stable(str(k)))
    counts = collections.Counter()
    for _ in range(a.controls):
        choices = []
        for key in keys:
            while pools[key] and (pools[key][0]["row"]["uid"] in used_uids or counts[pools[key][0]["row"]["speaker"]] >= 2):
                pools[key].pop(0)
            if pools[key]:
                choices.append(key)
        if not choices:
            break
        key = min(choices, key=lambda k: (counts[("cell", k)], stable(str(k))))
        rec = pools[key].pop(0)
        selected.append(rec)
        used_uids.add(rec["row"]["uid"])
        counts[rec["row"]["speaker"]] += 1
        counts[("cell", key)] += 1
    tag_dirs = list((DATA_ROOT / "aihub").glob("0*/01-1.정식개방데이터/*/02.라벨링데이터/*_pronunciation_tag"))
    if not tag_dirs:
        ap.error("original TAG directories are missing")
    a.out.mkdir(parents=True)
    (a.out / "audio").mkdir()
    records, exclusions = [], []
    for rec in selected:
        row, site = rec["row"], rec["site"]
        try:
            info = locate(row, site, tag_dirs)
        except (ValueError, KeyError) as e:
            exclusions.append({"uid": row["uid"], "q": site["q"], "reason": str(e)})
            continue
        item_id = stable(row["uid"] + ":" + str(site["q"]))[:16]
        record = {"id": item_id, "uid": row["uid"], "speaker": row["speaker"], "prompt": row["prompt"],
                  "norm_prompt": row["norm_prompt"], "item": row["item"], "cohort": row["cohort"],
                  "source_split": row.get("source_split"), "selection_role": rec["selection_role"],
                  "canon": site["canon"], "actual": site["actual"], "q": site["q"], "a": site["a"],
                  "following_vowel": row["canonical"][site["q"] + 1],
                  "preceding_canonical": row["canonical"][site["q"] - 1] if site["q"] else None,
                  "language": row.get("language"), "sex_filename": row["uid"].split("-")[1],
                  "frozen_decisions": {k: site.get(k) for k in ("baseline", "conservative")},
                  "label_source": "existing_human_TAG_not_new_blind_consensus", **info}
        for mode in ("local", "context"):
            path = a.out / "audio" / (item_id + "_" + mode + ".wav")
            record[mode + "_audio"] = crop(info["source_wav"], info[mode + "_interval"], path)
        records.append(record)
    records.sort(key=lambda r: r["id"])
    with (a.out / "items.private.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n")
    trials = {}
    for reviewer in ("A", "B", "C"):
        rng = random.Random("sibilant-v1-" + reviewer)
        order = [r["id"] for r in records]
        rng.shuffle(order)
        repeat_pool = order[:max(1, len(order) // 4)]
        repeats = rng.sample(repeat_pool, min(10, len(repeat_pool)))
        # Repeats appear only after a long first-pass block; each is an opaque trial.
        tail = order[len(order) // 2:] + repeats
        rng.shuffle(tail)
        order = order[:len(order) // 2] + tail
        trials[reviewer] = [{"trial": stable(reviewer + ":" + str(i) + ":" + item)[:20], "item": item}
                            for i, item in enumerate(order)]
    (a.out / "trials.private.json").write_text(json.dumps(trials, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {"purpose": "exploratory acoustic measurement and fresh blinded listening; not deployment validation",
               "items": len(records), "speakers": len({r["speaker"] for r in records}),
               "roles": dict(collections.Counter(r["selection_role"] for r in records)),
               "label_cells": dict(collections.Counter(r["canon"] + "->" + r["actual"] for r in records)),
               "vowels": dict(collections.Counter(r["following_vowel"] for r in records)),
               "coupled_tag_intervals_requiring_review": [r["id"] for r in records if not r["tag_intervals_isolated"]],
               "trials_per_reviewer": {k: len(v) for k, v in trials.items()},
               "exclusions": exclusions,
               "source_hashes": {str(p): digest(p) for p in (a.data, a.predictions, Path(__file__))},
               "human_blind_responses": 0, "human_acoustic_boundary_annotations": 0,
               "limitations": ["Existing TAG is one transcription layer; independent new ratings are pending.",
                               "TAG phone intervals are candidate locations, not validated acoustic landmarks.",
                               "Focus pairs share prompts but differ in speakers; not controlled minimal pairs.",
                               "Repeatedly inspected data cannot establish untouched-test generalization.",
                               "Raw PCM crops preserve amplitude; no noise suppression, normalization or synthetic audio."]}
    (a.out / "preparation.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
