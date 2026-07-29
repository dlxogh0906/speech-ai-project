# -*- coding: utf-8 -*-
"""단어 누락 검출 정량평가 — record(LAR)를 정답지로 대량 채점.

GT: record의 prompt vs orthographic_clean(전사)을 음절 단위 NW 정렬 →
    프롬프트 단어의 음절 70%+ 삭제 = 그 단어를 실제로 빼먹음 (정답 누락).
예측: 데모와 동일 파이프라인 — KoG2P canonical + tag_all CTC 디코드 + 자모 NW 정렬 →
    단어 span의 음소 삭제율 임계값(0.5~0.9 스윕) → 누락 판정.
지표: 누락 클래스 precision / recall / F1 (단어 단위).

출력: eval_word_skip.json + 예측 캐시 word_skip_preds.jsonl (재채점 CPU-only)
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.fairseq_guard import ensure_fairseq
ensure_fairseq()
from shared.paths import DATA_ROOT, MDD_ANALYSIS, MDD_OUTPUTS, ensure_kog2p

import sys, os, json, glob, random, argparse, time, math

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=20000)
ap.add_argument("--out", default=f"{MDD_OUTPUTS}/eval_word_skip.json")
ap.add_argument("--preds", default=f"{MDD_OUTPUTS}/word_skip_preds.jsonl")
ap.add_argument("--ckpt", default=f"{DATA_ROOT}/mdd_ckpt_tag_all/checkpoint_best.pt")
# quality_flag != ok (collapsed / restart_suspect) 발화는 기본적으로 제외되는데,
# 실측 결과 그 집단이 실제 단어누락의 69.6%를 품고 있다(누락률 19.76% vs 0.96%).
# 즉 기본 설정은 "잡으려는 현상의 대부분을 빼고" 재는 것이므로, 전수 채점 옵션을 둔다.
ap.add_argument("--include_lowq", action="store_true",
                help="quality_flag != ok 발화도 채점에 포함 (전수 268,533 기준)")
A = ap.parse_args()

RECORDS = f"{MDD_ANALYSIS}/040_041_042_043_both/records.jsonl"
AUDIO_DIRS = [f"{DATA_ROOT}/aihub/european_korean_speech/selected_speech_fix",
              f"{DATA_ROOT}/aihub/asian_korean_speech/selected_speech",
              f"{DATA_ROOT}/aihub/asian_korean_speech/selected_speech_manifest",
              # 041 원천 Speech_sound (LAR 74k — 2026-07-20 인덱스 확장)
              f"{DATA_ROOT}/aihub/european_korean_speech/041.교육용_유럽어_모국어_사용자의_한국어_음성_데이터/01-1.정식개방데이터/Training/01.원천데이터/extracted/TS_Speech_sound_1",
              f"{DATA_ROOT}/aihub/european_korean_speech/041.교육용_유럽어_모국어_사용자의_한국어_음성_데이터/01-1.정식개방데이터/Training/01.원천데이터/extracted/TS_Speech_sound_2",
              f"{DATA_ROOT}/aihub/european_korean_speech/041.교육용_유럽어_모국어_사용자의_한국어_음성_데이터/01-1.정식개방데이터/Training/01.원천데이터/extracted/TS_Speech_sound_3"]
CKPT = A.ckpt
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]

ensure_kog2p()
from mdd import pipeline as MD                     # canonical/word_spans/_nw 재사용 (모델은 안 뜸)
from g2p import readRules, ver_info
MD.S["ri"], MD.S["ro"] = readRules(ver_info[0], MD.RULEBOOK)

H = lambda c: 0xAC00 <= ord(c) <= 0xD7A3
def clean(s):
    return "".join(c for c in s if H(c) or c == " ").strip()
def hangul_only(s):
    return all(H(c) or c in " .?!,~" for c in s.strip())

# ── 1) 오디오 인덱스 + record 선별 ──
aidx = {}
# record 전량 16k 뱅크 (구 트랙 산출물, 040~043 LAR 268,533개 — 최우선 소스)
BANK = f"{DATA_ROOT}/mdd_wav16k"
for sh in sorted(os.listdir(BANK)):
    p = os.path.join(BANK, sh)
    if os.path.isdir(p):
        for f in os.listdir(p):
            if "LAR" in f and f.endswith(".wav"):
                aidx.setdefault(f[:-4], os.path.join(p, f))
for d in AUDIO_DIRS:
    for f in os.listdir(d):
        if "LAR" in f and f.endswith(".wav"):
            aidx.setdefault(f[:-4], os.path.join(d, f))
print(f"[audio] LAR wav {len(aidx):,}", flush=True)

cands = []
for line in open(RECORDS, encoding="utf-8"):
    r = json.loads(line)
    if r["task"] != "LAR":
        continue
    if not A.include_lowq and r.get("quality_flag") != "ok":
        continue
    stem = r["file"][:-5]
    if stem not in aidx:
        continue
    pr, tr = r.get("prompt", ""), r.get("orthographic_clean", "")
    if not (pr and hangul_only(pr) and hangul_only(tr)):
        continue
    pw = clean(pr).split()
    if len(pw) < 2:
        continue
    cands.append({"stem": stem, "wav": aidx[stem], "prompt": clean(pr), "transcript": clean(tr)})
random.seed(0); random.shuffle(cands)
cands = cands[: A.n]
print(f"[record] 사용 발화 {len(cands):,}", flush=True)

# ── 2) GT: 음절 NW 로 프롬프트 단어별 실제 누락 ──
for c in cands:
    pw = c["prompt"].split()
    psyl = [ch for w in pw for ch in w]
    spans = []; i = 0
    for w in pw:
        spans.append((i, i + len(w))); i += len(w)
    ops = MD._nw(psyl, [ch for ch in c["transcript"].replace(" ", "")])
    ndel = [0] * len(pw)
    for op in ops:
        if op["kind"] == "del":
            for wi, (lo, hi) in enumerate(spans):
                if lo <= op["tpos"] < hi:
                    ndel[wi] += 1; break
    c["gt_missing"] = [ndel[wi] >= max(1, math.ceil(0.7 * len(pw[wi]))) for wi in range(len(pw))]

gt_pos = sum(sum(c["gt_missing"]) for c in cands)
gt_tot = sum(len(c["gt_missing"]) for c in cands)
print(f"[GT] 단어 {gt_tot:,}개 중 실제 누락 {gt_pos:,} ({gt_pos/gt_tot:.1%})", flush=True)

# ── 3) 디코드 (캐시 있으면 스킵) ──
preds = {}
if os.path.exists(A.preds):
    for line in open(A.preds, encoding="utf-8"):
        d = json.loads(line); preds[d["stem"]] = d["pred"]
todo = [c for c in cands if c["stem"] not in preds]
print(f"[decode] cached {len(preds):,} · to-decode {len(todo):,}", flush=True)
if todo:
    import mdd_eval; mdd_eval.CKPT = CKPT
    from mdd_eval import Decoder
    from country_classifier.audio import load_wav_mono
    dec = Decoder()
    t0 = time.time(); done = 0
    CH = 256
    with open(A.preds, "a", encoding="utf-8") as f:
        for k in range(0, len(todo), CH):
            chunk = todo[k:k + CH]
            waves = [load_wav_mono(c["wav"], 16000)[0].numpy()[: 20 * 16000] for c in chunk]
            order = sorted(range(len(waves)), key=lambda i: len(waves[i]))
            for b0 in range(0, len(order), 24):
                idxs = order[b0:b0 + 24]
                outs = dec.decode([waves[i] for i in idxs])
                for i, p in zip(idxs, outs):
                    p = [t for t in p if t in MD.JAMO]
                    preds[chunk[i]["stem"]] = p
                    f.write(json.dumps({"stem": chunk[i]["stem"], "pred": p}, ensure_ascii=False) + "\n")
            done += len(chunk)
            if (k // CH) % 8 == 0:
                print(f"  decode {done:,}/{len(todo):,} ({done/max(1e-9,time.time()-t0):.0f}/s)", flush=True)

# ── 4) 예측 누락 판정 (임계값 스윕) + 채점 ──
res = {t: {"tp": 0, "fp": 0, "fn": 0} for t in THRESHOLDS}
used = skipped = 0
examples = {"fp": [], "fn": []}
for c in cands:
    pred = preds.get(c["stem"])
    if pred is None:
        skipped += 1; continue
    canon, _ = MD.canonical(c["prompt"])
    spans = MD.word_spans(c["prompt"], canon)
    if not spans or len(spans) != len(c["gt_missing"]):
        skipped += 1; continue
    ops = MD._nw(canon, pred)
    ndel = [0] * len(spans)
    for op in ops:
        if op["kind"] == "del":
            for wi, w in enumerate(spans):
                if w["start"] <= op["tpos"] < w["end"]:
                    ndel[wi] += 1; break
    used += 1
    for t in THRESHOLDS:
        for wi, w in enumerate(spans):
            n_ph = w["end"] - w["start"]
            pm = ndel[wi] >= max(2, int(t * n_ph))
            gm = c["gt_missing"][wi]
            if pm and gm: res[t]["tp"] += 1
            elif pm and not gm: res[t]["fp"] += 1
            elif gm and not pm: res[t]["fn"] += 1
    # 대표 임계 0.7의 오답 예시 수집
    for wi, w in enumerate(spans):
        n_ph = w["end"] - w["start"]
        pm = ndel[wi] >= max(2, int(0.7 * n_ph))
        gm = c["gt_missing"][wi]
        key = "fp" if (pm and not gm) else ("fn" if (gm and not pm) else None)
        if key and len(examples[key]) < 8:
            examples[key].append({"stem": c["stem"], "word": w["word"], "prompt": c["prompt"],
                                  "transcript": c["transcript"]})

table = {}
for t in THRESHOLDS:
    tp, fp, fn = res[t]["tp"], res[t]["fp"], res[t]["fn"]
    pr = tp / max(1, tp + fp); rc = tp / max(1, tp + fn)
    table[t] = {"tp": tp, "fp": fp, "fn": fn, "precision": round(pr, 4), "recall": round(rc, 4),
                "f1": round(2 * pr * rc / max(1e-9, pr + rc), 4)}

out = {"n_utts_used": used, "n_utts_skipped": skipped, "n_words": gt_tot,
       "gt_missing_words": gt_pos, "gt_missing_rate": round(gt_pos / max(1, gt_tot), 4),
       "ckpt": CKPT, "thresholds": table, "examples_at_0.7": examples}
json.dump(out, open(A.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

print(f"\n========== 단어 누락 검출 정량평가 (발화 {used:,} · 단어 {gt_tot:,} · 실제누락률 {gt_pos/gt_tot:.1%}) ==========")
print(f"{'임계값':>6} {'precision':>10} {'recall':>8} {'F1':>8}   (tp/fp/fn)")
for t in THRESHOLDS:
    v = table[t]
    print(f"{t:>6} {v['precision']:>10} {v['recall']:>8} {v['f1']:>8}   ({v['tp']}/{v['fp']}/{v['fn']})")
print(f"[saved] {A.out}")
