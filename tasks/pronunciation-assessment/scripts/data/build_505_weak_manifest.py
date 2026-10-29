# -*- coding: utf-8 -*-
"""AIHub 505(외국인 한국어 발화) → 약지도 학습용 매니페스트 — 문장 8,988종 · 화자 1,910명 · 오류 라벨 없음.

TAG(040~043)는 오류 라벨이 있지만 문항이 287개뿐이고, 505 는 라벨은 없지만 **문장이 30배 다양하고 모바일 녹음**(앱과 같은 환경)이다.
여기서는 읽기 발화(`ReadingLabelText`)만 골라 표준발음(KoG2P → 46자모)을 붙인다. 어느 자리를 믿을지는 다음 단계
(`gate_505_wildcard.py`)가 정한다 — 이 스크립트는 **정답을 주장하지 않는다**(`pool.ltr` 은 표준발음 복사본, 자리표시자).

제외: 한글 외 문자 · 길이 밖 · 사전 밖 자모 · 음성 없음 · **홀드아웃 문항과 같은 문장**(test_unseen 의 "안 본 문장" 조건 유지).
      홀드아웃 문항이 부분 문자열로 들어 있는 문장 수는 stats 에 적는다(단어 문항 '입'·'김치' 등은 어느 코퍼스에나 있다).
상한: 문장당 --per_sentence · 화자당 --per_speaker (seed 고정) — 첫 라운드는 수백 시간이면 충분하다.

출력 <out>/: pool.tsv · pool.canon · pool.ltr · pool.meta.jsonl · dict.ltr.txt · stats.json

    $PY tasks/pronunciation-assessment/scripts/data/build_505_weak_manifest.py --out $MDD_OUTPUTS/manifest_505_weak_v1
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.fairseq_guard import ensure_fairseq
from shared.paths import DATA_ROOT, MDD_OUTPUTS

import argparse, collections, glob, json, os, random, shutil, time, wave
from multiprocessing import Pool

H = lambda c: 0xAC00 <= ord(c) <= 0xD7A3
L1_MAP = {"베트남어": "Vietnamese", "영어": "English", "일본어": "Japanese", "중국어": "Chinese", "태국어": "Thai",
          "러시아어": "Russian", "인도네시아어": "Indonesian", "몽골어": "Mongolian", "스페인어": "Spanish; Castilian",
          "프랑스어": "French", "독일어": "German", "우즈베크어": "Uzbek", "터키어": "Turkish", "아랍어": "Arabic",
          "힌디어": "Hindi", "포르투갈어": "Portuguese", "이탈리아어": "Italian", "미얀마어": "Burmese", "크메르어": "Central Khmer",
          "네팔어": "Nepali", "카자흐어": "Kazakh", "타갈로그어": "Tagalog", "필리핀어": "Tagalog", "말레이어": "Malay"}


def read_label(p):
    try:
        j = json.load(open(p, encoding="utf-8"))
        t, s, f = j.get("transcription", {}), j.get("skill_info", {}), j.get("file_info", {})
        return {"stem": os.path.basename(p)[:-5], "text": (t.get("ReadingLabelText") or "").strip(), "answer": bool(t.get("AnswerLabelText")),
                "spk": f.get("speakerID") or j.get("SpeakerID"), "sent": f.get("sentenceID"), "unit": f.get("recordUnit"),
                "sec": float(f.get("recordTime") or 0), "mother": s.get("motherTongue"), "lang_class": s.get("languageClass"),
                "topik": s.get("topikGrade"), "self": s.get("selfAssessment"), "learn_months": s.get("LearningPeriod"),
                "sent_lv": t.get("SentenceSpeechLV")}
    except Exception:
        return None


def wav_frames(p):
    try:
        with wave.open(p, "rb") as w:
            if w.getframerate() != 16000 or w.getnchannels() != 1 or w.getsampwidth() != 2:
                return -1
            return w.getnframes()
    except Exception:
        return -1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aihub_root", default=str(DATA_ROOT / "aihub"))
    ap.add_argument("--out", default=str(MDD_OUTPUTS / "manifest_505_weak_v1"))
    ap.add_argument("--holdout_manifest", default=str(MDD_OUTPUTS / "manifest_foreign_clean_v2"))
    ap.add_argument("--dict", default=None, help="기본: holdout_manifest 의 dict.ltr.txt")
    ap.add_argument("--part", default="1.Training")
    ap.add_argument("--per_sentence", type=int, default=15); ap.add_argument("--per_speaker", type=int, default=250)
    ap.add_argument("--min_sec", type=float, default=1.5); ap.add_argument("--max_sec", type=float, default=15.0)
    ap.add_argument("--workers", type=int, default=12); ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    ensure_fairseq()
    from mdd import pipeline as MD
    from g2p import readRules, ver_info
    MD.S["ri"], MD.S["ro"] = readRules(ver_info[0], MD.RULEBOOK)
    t0 = time.time()
    roots = glob.glob(os.path.join(a.aihub_root, "131.*", "*", a.part))
    assert len(roots) == 1, roots
    root = roots[0]
    wav_of = {}
    for d in glob.glob(os.path.join(root, "원천데이터*")):
        for r, _, fs in os.walk(d):
            for f in fs:
                if f.endswith(".wav"):
                    wav_of[f[:-4]] = os.path.join(r, f)
    labels = [os.path.join(r, f) for r, _, fs in os.walk(os.path.join(root, "라벨링데이터")) for f in fs if f.endswith(".json")]
    print(f"[scan] wav {len(wav_of):,} · 라벨 {len(labels):,} ({time.time()-t0:.0f}s)", flush=True)
    with Pool(a.workers) as pool:
        recs = [r for r in pool.imap_unordered(read_label, labels, chunksize=2000) if r]
    print(f"[label] 읽음 {len(recs):,} ({time.time()-t0:.0f}s)", flush=True)

    hold = set()
    for sp in ("test_unseen", "valid_unseen"):
        p = os.path.join(a.holdout_manifest, f"{sp}.meta.jsonl")
        if os.path.exists(p):
            hold |= {json.loads(l).get("norm_prompt", "") for l in open(p, encoding="utf-8")}
    hold.discard("")
    dict_path = a.dict or os.path.join(a.holdout_manifest, "dict.ltr.txt")
    dict_jamo = set(l.split()[0] for l in open(dict_path, encoding="utf-8") if l.strip())

    why = collections.Counter(); canon_of, sub_hit = {}, 0
    keep = []
    for r in recs:
        tx = r["text"]
        if not tx:
            why["answer_or_empty"] += 1; continue
        if r["stem"] not in wav_of:
            why["no_audio"] += 1; continue
        if not all(H(c) or c == " " for c in tx):
            why["non_hangul"] += 1; continue
        if not a.min_sec <= r["sec"] <= a.max_sec:
            why["duration"] += 1; continue
        norm = tx.replace(" ", "")
        if norm in hold:
            why["holdout_sentence"] += 1; continue
        if tx not in canon_of:
            try:
                c = list(MD.canonical(tx)[0])
            except Exception:
                c = []
            canon_of[tx] = c if c and set(c) <= dict_jamo else None
            if canon_of[tx] and any(h in norm for h in hold):
                sub_hit += 1
        if not canon_of[tx]:
            why["g2p_or_oov"] += 1; continue
        keep.append(r)
    print(f"[filter] 후보 {len(keep):,} · 제외 {dict(why)} · 고유 문장 {sum(1 for v in canon_of.values() if v):,}", flush=True)

    rng = random.Random(a.seed); rng.shuffle(keep)
    ns, nk, sel = collections.Counter(), collections.Counter(), []
    for r in keep:
        if ns[r["text"]] >= a.per_sentence or nk[r["spk"]] >= a.per_speaker:
            continue
        ns[r["text"]] += 1; nk[r["spk"]] += 1; sel.append(r)
    with Pool(a.workers) as pool:
        frames = pool.map(wav_frames, [wav_of[r["stem"]] for r in sel], chunksize=500)
    sel = [(r, n) for r, n in zip(sel, frames) if n > 0 and a.min_sec * 16000 <= n <= a.max_sec * 16000]
    sel.sort(key=lambda x: x[0]["stem"])

    os.makedirs(a.out, exist_ok=True)
    with open(f"{a.out}/pool.tsv", "w", encoding="utf-8") as ft, open(f"{a.out}/pool.canon", "w", encoding="utf-8") as fc, \
         open(f"{a.out}/pool.ltr", "w", encoding="utf-8") as fl, open(f"{a.out}/pool.meta.jsonl", "w", encoding="utf-8") as fm:
        ft.write("/\n")
        for r, n in sel:
            c = " ".join(canon_of[r["text"]])
            ft.write(f"{wav_of[r['stem']].lstrip('/')}\t{n}\n"); fc.write(c + "\n"); fl.write(c + "\n")
            fm.write(json.dumps({"stem": r["stem"], "spk": r["spk"], "ds": "505", "task": "READ", "item": r["sent"], "prompt": r["text"],
                                 "language": L1_MAP.get(r["mother"] or "", "?"), "mother_tongue": r["mother"], "nationality": None,
                                 "proficiency": None, "topik": r["topik"], "self_assessment": r["self"], "learn_months": r["learn_months"],
                                 "sentence_level": r["sent_lv"], "record_unit": r["unit"], "sec": round(n / 16000, 3)}, ensure_ascii=False) + "\n")
    shutil.copy(dict_path, f"{a.out}/dict.ltr.txt")
    hours = sum(n for _, n in sel) / 16000 / 3600
    stats = {"selected": len(sel), "hours": round(hours, 1), "speakers": len({r["spk"] for r, _ in sel}),
             "unique_sentences": len({r["text"] for r, _ in sel}), "candidates": len(keep), "excluded": dict(why),
             "holdout_prompts": len(hold), "sentences_containing_a_holdout_prompt_as_substring": sub_hit,
             "language": dict(collections.Counter(L1_MAP.get(r["mother"] or "", "?") for r, _ in sel).most_common()),
             "topik": dict(collections.Counter(str(r["topik"]) for r, _ in sel).most_common()),
             "record_unit": dict(collections.Counter(str(r["unit"]) for r, _ in sel).most_common()),
             "caps": {"per_sentence": a.per_sentence, "per_speaker": a.per_speaker}, "seed": a.seed, "sec": round(time.time() - t0, 1)}
    json.dump(stats, open(f"{a.out}/stats.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in stats.items() if k not in ("language",)}, ensure_ascii=False), flush=True)
    print("언어:", dict(list(stats["language"].items())[:10]), flush=True)


if __name__ == "__main__":
    main()
