# -*- coding: utf-8 -*-
"""발음 진단(MDD) 웹 데모 — 문장 제시 → 사용자 녹음 → 오류 위치·유형 진단.

파이프라인: 문장 --KoG2P--> 표준 음소열(canonical)
           오디오 --XLS-R 300m+CTC(TAG 학습, F1 0.956)--> 실제 발음 자모열
           → NW 정렬 → sub/del/ins → 유형 규칙(경음화·격음화·비음화·유음화·구개음화 등)
           → L1-prior(81개국) 가중 → 진단 JSON.  자모/KoG2P 71음소 병기(후처리 변환).

Run:  python mdd_demo.py --port 8901   (fairseq_xlsr env, GPU)
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists(): _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.fairseq_guard import ensure_fairseq
ensure_fairseq()
from shared.paths import DATA_ROOT, G2P_TOOLS, MDD_ANALYSIS, REPO_ROOT, ensure_kog2p

import sys, os, json, time, subprocess, argparse
import numpy as np

from mdd import mdd_eval
mdd_eval.CKPT = f"{DATA_ROOT}/mdd_ckpt_tag_all_lar/checkpoint_best.pt"   # 전 언어권 + LAR 도메인 적응 (F1 0.9547 유지, 누락정밀도 0.07→0.68)
from mdd.mdd_eval import Decoder
from mdd.jamo_to_kog2p import jamo_to_phones, phones_to_jamo, INV, VOWEL, ONSET, CODA
JAMO = set(VOWEL) | set(ONSET) | set(CODA)
ensure_kog2p()
_argv, sys.argv = sys.argv, sys.argv[:1]   # g2p.py가 import 시 optparse로 argv를 파싱함
from g2p import readRules, graph2prono, ver_info
sys.argv = _argv

# 데모 HTML 위치. deploy/mdd-demo/app.py 가 자기 디렉터리로 덮어쓴다.
HERE = str(REPO_ROOT / "deploy" / "mdd-demo")
RULEBOOK = f"{G2P_TOOLS}/KoG2P/rulebook.txt"
L1_JSON = f"{MDD_ANALYSIS}/l1_prior_product/l1_prior_weights.json"
SR = 16000
S = {}

# ── s1_speechlab_mdd.py _nw() 사본 (mdd_eval_pos 와 동일; 서버 독립성 위해 복사) ──
def _nw(target, actual, match=0, mismatch=1, gap=1):
    n, m = len(target), len(actual)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1): D[i][0] = i * gap
    for j in range(1, m + 1): D[0][j] = j * gap
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = match if target[i - 1] == actual[j - 1] else mismatch
            D[i][j] = min(D[i - 1][j - 1] + c, D[i - 1][j] + gap, D[i][j - 1] + gap)
    ops, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i][j] == D[i - 1][j - 1] + (match if target[i - 1] == actual[j - 1] else mismatch):
            ops.append({"kind": "match" if target[i - 1] == actual[j - 1] else "sub",
                        "target": target[i - 1], "actual": actual[j - 1], "tpos": i - 1, "apos": j - 1})
            i, j = i - 1, j - 1
        elif i > 0 and D[i][j] == D[i - 1][j] + gap:
            ops.append({"kind": "del", "target": target[i - 1], "actual": None, "tpos": i - 1, "apos": None})
            i -= 1
        else:
            ops.append({"kind": "ins", "target": None, "actual": actual[j - 1], "tpos": i, "apos": j - 1})
            j -= 1
    return ops[::-1]

# ── 유형 규칙 (mdd_demo_pipeline.py 와 동일) + 초/종성 세분 ──
VOWELS = set(VOWEL)
TENSE = {'ㄱ': 'ㄲ', 'ㄷ': 'ㄸ', 'ㅂ': 'ㅃ', 'ㅅ': 'ㅆ', 'ㅈ': 'ㅉ'}
ASP = {'ㄱ': 'ㅋ', 'ㄷ': 'ㅌ', 'ㅂ': 'ㅍ', 'ㅈ': 'ㅊ'}
NASAL = {'ㄱ': 'ㅇ', 'ㄷ': 'ㄴ', 'ㅂ': 'ㅁ', 'ㅋ': 'ㅇ', 'ㅌ': 'ㄴ', 'ㅍ': 'ㅁ'}
DESC = {"경음화": "된소리(경음)를 평음으로, 또는 그 반대로 발음",
        "격음화": "거센소리(격음) 관련 오류 — ㅋ·ㅌ·ㅍ·ㅊ와 평음 혼동",
        "비음화": "비음화 규칙(예: 국민→[궁민]) 미적용 또는 과적용",
        "유음화": "ㄴ↔ㄹ 유음화(예: 신라→[실라]) 오류",
        "모음오류": "모음의 음가가 표준과 다름",
        "음운탈락": "발음해야 할 소리를 빠뜨림",
        "음운첨가": "표준 발음에 없는 소리를 추가함",
        "초성": "음절 첫 자음 발음 오류", "종성": "받침 발음 오류"}

def ctype(t, a, kind, is_coda):
    if kind == "del": return "음운탈락"
    if kind == "ins": return "음운첨가"
    if (t in VOWELS) or (a in VOWELS): return "모음오류"
    if TENSE.get(t) == a or TENSE.get(a) == t: return "경음화"
    if ASP.get(t) == a or ASP.get(a) == t: return "격음화"
    if NASAL.get(t) == a: return "비음화"
    if {t, a} == {'ㄴ', 'ㄹ'}: return "유음화"
    return "종성" if is_coda else "초성"

# ctype 결과 → L1-prior 유형 키
PRIOR_KEY = {"경음화": "경음화", "격음화": "격음화", "비음화": "비음화", "유음화": "유음화",
             "모음오류": "모음오류", "음운탈락": "음운탈락", "초성": "초성", "종성": "종성",
             "음운첨가": "ㄴ-첨가"}

PRESETS = [  # KoG2P canonical 검증 완료 문장만
    {"text": "괜찮다", "focus": "격음화"}, {"text": "줄넘기", "focus": "유음화·경음화"},
    {"text": "국민", "focus": "비음화"}, {"text": "같이", "focus": "구개음화"},
    {"text": "신라", "focus": "유음화"}, {"text": "독립", "focus": "비음화"},
    {"text": "학교", "focus": "경음화"}, {"text": "꽃잎", "focus": "ㄴ첨가"},
    {"text": "나뭇잎", "focus": "ㄴ첨가·비음화"}, {"text": "빛을", "focus": "연음"},
    {"text": "밥 먹었어요", "focus": "문장·비음화"}, {"text": "머리를 잘라야 한다", "focus": "문장"},
    {"text": "책상 위에 있어요", "focus": "문장·경음화"},
]

HANGUL = lambda c: 0xAC00 <= ord(c) <= 0xD7A3


def syllable_ids(canon):
    """canonical 자모열의 각 위치 → 음절 번호. 음운규칙은 음절 수를 바꾸지 않으므로
    모음 1개 = 음절 1개. 초성(모음 앞 자음, ㅇ 제외)은 다음 음절, 종성은 앞 음절 소속."""
    n = len(canon)
    is_v = [t in VOWELS for t in canon]
    is_on = [(not is_v[i]) and canon[i] != "ㅇ" and i + 1 < n and is_v[i + 1] for i in range(n)]
    ids, cur = [], -1
    for i in range(n):
        if is_on[i] or (is_v[i] and not (i > 0 and is_on[i - 1])):
            cur += 1
        ids.append(max(cur, 0))
    return ids


def word_spans(sentence, canon):
    """단어별 음절 수로 canonical 자모열을 단어 span으로 분할.
    반환: [{word, start, end(excl)}]. 음절 수 불일치 시(숫자 등) None."""
    words = [w for w in sentence.split() if any(HANGUL(c) for c in w)]
    if not words:
        return None
    syls = [sum(1 for c in w if HANGUL(c)) for w in words]
    sid = syllable_ids(canon)
    total = (sid[-1] + 1) if sid else 0
    if total != sum(syls):
        return None                      # G2P가 음절 수를 바꿨거나 비한글 포함 → 단어 진단 생략
    spans, s0 = [], 0
    for w, k in zip(words, syls):
        lo = next(i for i, s in enumerate(sid) if s == s0)
        hi = max(i for i, s in enumerate(sid) if s == s0 + k - 1) + 1
        spans.append({"word": w, "start": lo, "end": hi})
        s0 += k
    return spans


def canonical(sentence):
    """문장 → (표준발음 자모열, 71음소열). KoG2P 규칙 적용(경음화·비음화·ㄴ첨가 포함)."""
    phs = [p for p in graph2prono(sentence, S["ri"], S["ro"]).split() if p in INV]
    return phones_to_jamo(phs), phs


def decode_audio(raw: bytes) -> np.ndarray:
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    p = subprocess.run([ff, "-nostdin", "-i", "pipe:0", "-f", "f32le", "-ac", "1",
                        "-ar", str(SR), "pipe:1"], input=raw, capture_output=True)
    if p.returncode != 0 or not p.stdout:
        raise ValueError("could not decode audio")
    return np.frombuffer(p.stdout, dtype=np.float32).copy()


def load():
    t0 = time.time()
    S["ri"], S["ro"] = readRules(ver_info[0], RULEBOOK)
    S["l1"] = json.load(open(L1_JSON, encoding="utf-8"))["priors"]
    S["dec"] = Decoder()
    print(f"[load] ready in {time.time()-t0:.0f}s | L1 countries={len(S['l1'])}", flush=True)


def diagnose(wave, sentence, country=None):
    t0 = time.time()
    dur = len(wave) / SR
    if dur < 0.3 or float(np.sqrt((wave ** 2).mean())) < 1.5e-3:
        return {"no_speech": True, "reason": "too_short_or_silent", "duration_sec": round(dur, 2)}
    pred = S["dec"].decode([wave[: 20 * SR]])[0]
    pred = [t for t in pred if t in JAMO]        # special 토큰 제거, 자모만
    if len(pred) < 1:
        return {"no_speech": True, "reason": "no_phones_decoded", "duration_sec": round(dur, 2)}

    canon, canon_ph = canonical(sentence)
    pred_ph = jamo_to_phones(pred)
    ops = _nw(canon, pred)

    priors = S["l1"].get(country or "", {}).get("weights", {})
    errors = []
    for op in ops:
        if op["kind"] == "match":
            continue
        tpos = op["tpos"]
        # 초/종성 판정: canonical 위치의 다음 토큰이 모음이면 초성 (ㅇ은 항상 종성 ng)
        is_coda = False
        if op["kind"] != "ins" and canon[tpos] not in VOWELS:
            is_coda = (canon[tpos] == "ㅇ") or not (tpos + 1 < len(canon) and canon[tpos + 1] in VOWELS)
        et = ctype(op.get("target") or "", op.get("actual") or "", op["kind"], is_coda)
        e = {"pos": tpos, "kind": op["kind"], "type": et, "desc": DESC.get(et, ""),
             "target": op.get("target") or "∅", "actual": op.get("actual") or "∅",
             "target_ph": canon_ph[tpos] if op["kind"] != "ins" else "∅",
             "actual_ph": (pred_ph[op["apos"]] if op.get("apos") is not None else "∅")}
        pk = PRIOR_KEY.get(et)
        if priors and pk in priors:
            w = priors[pk].get("prior_weight", 1.0)
            e["l1_weight"] = round(w, 2)
            e["l1_typical"] = bool(w >= 1.3)
        errors.append(e)

    # ── 단어 수준 집계: 음절 span별 sub/del/ins → OK / 오류 / 누락 ──
    spans = word_spans(sentence, canon)
    words = None
    if spans:
        for e in errors:                          # 오류 카드에 단어 표시
            for w in spans:
                if w["start"] <= min(e["pos"], len(canon) - 1) < w["end"]:
                    e["word"] = w["word"]; break
        words = []
        for w in spans:
            n_ph = w["end"] - w["start"]
            n_del = sum(1 for op in ops if op["kind"] == "del" and w["start"] <= op["tpos"] < w["end"])
            n_err = sum(1 for op in ops if op["kind"] in ("sub", "del") and w["start"] <= op["tpos"] < w["end"]) \
                  + sum(1 for op in ops if op["kind"] == "ins" and w["start"] <= min(op["tpos"], len(canon) - 1) < w["end"])
            status = "missing" if n_del >= max(2, int(0.8 * n_ph)) else ("error" if n_err else "ok")
            # 0.8 = record LAR 2만 발화 정량평가의 F1 최적점 (precision 0.683/recall 0.682)
            words.append({"word": w["word"], "start": w["start"], "end": w["end"],
                          "n_phones": n_ph, "n_err": n_err, "status": status})

    # ── 신뢰도 가드레일 (LAR 정량평가 실측 기반) ──
    # 모델 붕괴 = 삭제가 문장 전체에 퍼짐 / 진짜 누락 = 한 단어에 집중.
    # → '누락 후보 밖(배경)' 삭제율로 판별해야 참누락을 보류로 강등하지 않음.
    n_syl = (syllable_ids(canon)[-1] + 1) if canon else 0
    rate = n_syl / max(0.1, dur)
    miss_rng = [(w["start"], w["end"]) for w in (words or []) if w["status"] == "missing"]
    in_miss = lambda p: any(lo <= p < hi for lo, hi in miss_rng)
    bg_n = sum(1 for p in range(len(canon)) if not in_miss(p))
    bg_del = sum(1 for op in ops if op["kind"] == "del" and not in_miss(op["tpos"])) / max(1, bg_n)
    low_conf = rate > 4.0 or bg_del > 0.20
    if low_conf and words:
        for w in words:
            if w["status"] == "missing":
                w["status"] = "uncertain"          # 오탐 가능성 높음 → 누락 단정 금지

    n_err_pos = len({e["pos"] for e in errors if e["kind"] != "ins"})
    return {"no_speech": False, "sentence": sentence, "words": words,
            "reliability": {"speaking_rate": round(rate, 2), "bg_del_frac": round(bg_del, 3),
                            "low_confidence": low_conf},
            "canon": canon, "canon_ph": canon_ph, "pred": pred, "pred_ph": pred_ph,
            "ops": [{k: op[k] for k in ("kind", "target", "actual", "tpos", "apos")} for op in ops],
            "errors": errors,
            "match_rate": round(1.0 - n_err_pos / max(1, len(canon)), 3),
            "duration_sec": round(dur, 2), "took_sec": round(time.time() - t0, 3)}


def make_app():
    from fastapi import FastAPI, File, UploadFile, Form, HTTPException, APIRouter
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse
    from typing import Optional

    app = FastAPI(title="Korean MDD demo")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    r = APIRouter()

    @r.get("/", response_class=HTMLResponse)
    def index():
        return open(os.path.join(HERE, "mdd_demo.html"), encoding="utf-8").read()

    @r.get("/health")
    def health():
        return {"status": "ok" if "dec" in S else "loading", "model": "xlsr300m-ctc-tag-all-lar",
                "f1_weighted": 0.9547, "coverage": "EN/ZH/JA/유럽/아시아 전 언어권 + 문장낭독(LAR) 적응"}

    @r.get("/meta")
    def meta():
        return {"presets": PRESETS, "countries": sorted(S.get("l1", {}).keys())}

    @r.post("/predict")
    async def predict(audio: UploadFile = File(...), sentence: str = Form(...),
                      country: Optional[str] = Form(None)):
        if "dec" not in S:
            raise HTTPException(503, "loading")
        sentence = (sentence or "").strip()
        if not sentence or len(sentence) > 60:
            raise HTTPException(400, "sentence required (<=60 chars)")
        try:
            wave = decode_audio(await audio.read())
        except ValueError:
            raise HTTPException(400, "could not decode audio")
        return diagnose(wave, sentence, country)

    app.include_router(r)
    app.include_router(r, prefix="/mdd")   # tailscale serve path-mount 대응(프리픽스 보존형)
    return app


if __name__ == "__main__":
    raise SystemExit(
        "이 파일은 라이브러리다. 데모 서버는 다음으로 띄운다:\n"
        "    $PY deploy/mdd-demo/app.py --port 8901"
    )
