# -*- coding: utf-8 -*-
"""한국어 발음 평가 — 지금 운영 vs 넣을 버전 나란히 비교 데모 (내부 확인용).

같은 녹음 하나를 두 경로로 채점해 화면 판정(errorType)을 나란히 보여 준다. 채점 경로는 발음 API 를 그대로 옮긴
`mdd.api_offline.ApiEmulator`(30/30 응답 일치 확인) — FLAG_MAX 화면 문턱·말소리 검사까지 같다.
  ※ 2026-10-21 01:58 에 B(입력 거르기 셋만 빼고)가 NPU 운영이 됐다 — 페이지는 A = 이전 운영 · B = 지금 운영으로 표시한다.
  A 이전 운영(NPU pron-api 8932, 2026-10-18 20:16 ~ 10-21 01:58):
       w2vbert_synth_tense_v2/interp_a0.5_step3000(기준·2차 가중치 반반) + 표준발음 보정(`mdd.canonical_fix`, margin 1.0)
       + 말소리 기준(기준 문장 음절 수에 맞춘 최소 말한 시간) + 센소리 예외(`speechsuper.STRONG_OF`)
  B 넣을 버전(제안, 2026-10-19): w2vbert_synth_tense_v2/step_3000(2차 학습 100%) + A 와 같은 보정·기준·예외
       + 화면 규칙 2개(빠진 소리는 강제정렬 점수 50 미만일 때만 지적 `DEL_SCORE_MAX` · ㄴ↔ㄹ 예외 `LIQUID_NASAL`)
       + 입력 거르기 3개(no_korean · too_long 말소리 20초 초과 · no_match 기준 소리 중 다르게 들린 비율 0.9 초과)
       — 운영 app.py 로는 `--del-score-max 50 --liquid-rule --nomatch-reject 0.9 --too-long-reject` 와 같다.
       (10-19 오후 안은 `--del-run-min 3` 이었다 — 학습자의 진짜 빠뜨림을 거의 다 숨겨서 바꿨다. speechsuper.DEL_SCORE_MAX 주석)
  문장 점수(발음·유창·이해)는 두 쪽 다 운영과 같은 점수기(step_38000 짝)를 쓴다.
  GPU 에서 HF 모델로 돌리므로 NPU(.rbln) 응답과 숫자가 아주 조금 다를 수 있다.
화면 규칙은 `mdd.speechsuper` 의 모듈 전역값이라 채점 잠금 안에서 쪽마다 바꿔 끼운다(`rules()`).
녹음은 저장하지 않는다. 127.0.0.1 에만 연다(외부 공개는 지시가 있을 때만).

    PYTHONPATH=/data/speech-ai/tools/hf_py38 $PY deploy/pronunciation-ab-demo/app.py --port 8936
    → VS Code 포트 전달로 http://localhost:8936/ (마이크는 localhost 나 HTTPS 에서만 열린다)
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import DATA_ROOT, FFMPEG, MDD_OUTPUTS

import argparse, contextlib, os, re, threading, time

HERE = _pl.Path(__file__).resolve().parent
SR = 16000
MAX_SPEECH_SEC = 20.0          # 운영 app.py 와 같다(too_long)
NOMATCH_B = 0.9                # 운영 app.py --nomatch-reject 권장값(no_match)
CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JONG = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ",
        "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]
S = {}


@contextlib.contextmanager
def rules(**kw):
    """`mdd.speechsuper` 화면 규칙(모듈 전역값)을 잠깐 바꿔 끼운다 — 두 쪽이 한 프로세스를 같이 쓰므로 채점 잠금 안에서만."""
    from mdd import speechsuper as ss
    old = {k: getattr(ss, k) for k in kw}
    for k, v in kw.items():
        setattr(ss, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(ss, k, v)


def side_rules():
    """→ {"A": 지금 운영, "B": 넣을 버전}. A 는 라이브러리 기본값이지만 기본값이 바뀌어도 운영과 같게 값을 박아 둔다."""
    from mdd import speechsuper as ss
    return {"A": {"DEL_RUN_MIN": 1, "DEL_SCORE_MAX": None, "LIQUID_NASAL": {}, "SS_RULE": False},
            "B": {"DEL_RUN_MIN": 1, "DEL_SCORE_MAX": 50, "LIQUID_NASAL": dict(ss.LIQUID_NASAL_PAIRS), "SS_RULE": False}}


def mismatch_ratio(sites):
    """운영 app.py mismatch_ratio 와 같다 — 기준 소리 중 다르게 들린(다름·삭제) 비율. 소리가 4개 미만이면 None."""
    if len(sites) < 4:
        return None
    return sum(1 for x in sites if x.get("verdict") != "일치") / len(sites)


def to_hangul(jamo):
    """발음 자모열 → 읽기용 한글(받침 ㅇ 과 첫소리 ㅇ 이 겹치는 곳은 첫소리로 읽는다 — 화면 표시용)."""
    j, out, i = [x for x in jamo if x], [], 0
    V = set(JUNG)
    while i < len(j):
        on = "ㅇ"
        if j[i] not in V:
            if i + 1 < len(j) and j[i + 1] in V:
                on = j[i]; i += 1
            else:
                out.append(j[i]); i += 1; continue
        v = j[i]; i += 1
        if v == "ㅗ" and i < len(j) and j[i] == "ㅐ":
            v = "ㅙ"; i += 1
        elif v == "ㅜ" and i < len(j) and j[i] == "ㅔ":
            v = "ㅞ"; i += 1
        co = ""
        if i < len(j) and j[i] not in V and not (i + 1 < len(j) and j[i + 1] in V) and j[i] in JONG:
            co = j[i]; i += 1
        out.append(chr(0xAC00 + CHO.index(on) * 588 + JUNG.index(v) * 28 + JONG.index(co)))
    return "".join(out)


def site_probs(lp, canon, dec, k=3):
    """표준발음 자리마다 모델이 준 소리 확률 — 녹음을 표준발음에 강제 정렬(CTC)해 그 자리에 붙은 구간의 확률을 평균 내고,
    blank 를 뺀 자모끼리 다시 나눠 상위 k 개. → [[(자모, 확률), …], …] (정렬이 안 되면 None)."""
    import numpy as np
    import torch
    import torchaudio.functional as AF
    try:
        y = [dec.d.index(t) for t in canon]
        lpt = torch.as_tensor(lp).float().unsqueeze(0)
        ali, sc = AF.forced_align(lpt, torch.tensor([y], dtype=torch.int32), blank=dec.blank)
        spans = AF.merge_tokens(ali[0], sc[0].exp())
    except Exception:
        return None
    if len(spans) != len(canon):
        return None
    P = np.exp(np.asarray(lp, dtype=np.float32))
    jamo_ids = [i for i in range(P.shape[1]) if i != dec.blank and len(dec.d[i]) == 1 and 0x3131 <= ord(dec.d[i]) <= 0x3163]
    out = []
    for sp in spans:
        m = P[sp.start:sp.end, jamo_ids].mean(0)
        m = m / max(1e-9, m.sum())
        top = np.argsort(-m)[:k]
        out.append([(dec.d[jamo_ids[t]], round(float(m[t]) * 100)) for t in top])
    return out


def summarize(body, raw, emu, lp=None):
    words = []
    n_err = n_unc = 0
    probs = site_probs(lp, raw["canon"], emu.dec) if lp is not None and raw.get("canon") else None
    idx = 0
    for w in body["result"]["words"]:
        ph = []
        for p in w["phonemes"]:
            e = p.get("errorType")
            n_err += bool(e); n_unc += bool(p.get("uncertain"))
            ph.append({"c": p.get("alpha_jamo"), "h": p.get("actual_jamo"), "s": p.get("pronunciation"), "e": e,
                       "u": bool(p.get("uncertain")), "top": (probs[idx] if probs and idx < len(probs) else None),
                       "ca": p.get("alpha"), "ha": p.get("actual")})
            idx += 1
        words.append({"word": w["word"], "score": w.get("score"), "ph": ph,
                      "canon": to_hangul([q["c"] for q in ph]), "heard": to_hangul([q["h"] for q in ph if q["h"]])})
    r = body["result"]
    return {"scores": {"overall": r.get("overall"), "pronunciation": r.get("pronunciation"), "fluency": r.get("fluency"),
                       "comprehensibility": r.get("comprehensibility")},
            "canon": to_hangul(raw["canon"]), "words": words, "n_err": n_err, "n_unc": n_unc,
            "eojeol": eojeols(body.get("refText") or "", words)}


def eojeols(text, words):
    """어절 단위로 묶은 표준 발음 / 들린 그대로(연음이 보이게 어절 안 자모를 이어서 한글로) + KoG2P 기호.
    응답의 words 는 원문 한글 글자마다 하나라서 원문 띄어쓰기로 다시 묶는다."""
    import re
    groups, k = [], 0
    for tok in re.sub(r"\s+", " ", text).strip().split(" "):
        n = len(re.findall(r"[가-힣]", tok))
        if not n:
            continue
        groups.append([tok, [q for w in words[k:k + n] for q in w["ph"]]]); k += n
    # 경계 ㄴ 첨가처럼 다음 어절 첫소리인데 앞 어절에 붙은 자음(받침 뒤 자음 하나 더)은 다음 어절 앞으로 옮긴다
    V = set(JUNG)
    for i in range(len(groups) - 1):
        ph = groups[i][1]
        if len(ph) >= 3 and ph[-1]["c"] and ph[-1]["c"] not in V and ph[-2]["c"] and ph[-2]["c"] not in V:
            groups[i + 1][1].insert(0, ph.pop())
    out = []
    for tok, ph in groups:
        out.append({"text": tok, "canon": to_hangul([q["c"] for q in ph]), "heard": to_hangul([q["h"] for q in ph if q["h"]]),
                    "e": any(q["e"] for q in ph), "u": any(q["u"] for q in ph),
                    "sym": [[q["ca"], q["ha"], bool(q["e"]), bool(q["u"])] for q in ph]})
    return out


def make_app():
    from fastapi import FastAPI, File, Form, UploadFile
    from fastapi.responses import FileResponse, JSONResponse
    from starlette.concurrency import run_in_threadpool
    from mdd_en.audio import decode_audio
    from mdd import canonical_fix as cf
    from mdd.api_offline import min_voiced_sec
    app = FastAPI(title="한국어 발음 평가 — 지금 운영 vs 넣을 버전")
    lock = threading.Lock()

    @app.get("/")
    def index():
        return FileResponse(str(HERE / "static" / "index.html"), headers={"Cache-Control": "no-store"})

    @app.get("/health")
    def health():
        return {"ok": True, "a": os.path.basename(S["a_ckpt"].rstrip("/")), "b": os.path.basename(S["b_ckpt"].rstrip("/")),
                "rules_b": {"del_run_min": S["rules"]["B"]["DEL_RUN_MIN"], "del_score_max": S["rules"]["B"]["DEL_SCORE_MAX"],
                            "liquid_nasal": bool(S["rules"]["B"]["LIQUID_NASAL"]),
                            "nomatch_reject": NOMATCH_B, "too_long_sec": MAX_SPEECH_SEC, "no_korean": True}}

    def canon_str(f, text):
        try:
            return to_hangul(f(text))
        except Exception:                               # 한글 없는 문장 등 — 화면 표시용이라 비워 둔다
            return ""

    def pick_canon(emu, l, text):
        """운영 analyze() 와 같은 순서 — KoG2P 표준발음 → 보정 + 허용 변이를 이 쪽 모델의 녹음 CTC 우도로 고름."""
        canon, used, base = emu.canonical(text), [], []
        if canon:
            base, alts = S["builder"].variants(text)
            if base:
                canon, used = cf.choose(l, base, alts, emu.dec.d.index, blank=emu.dec.blank, margin=S["margin"])
        return canon, [{"kind": k, "from": to_hangul(base[s:e]) or "∅", "to": to_hangul(r) or "∅", "gain": round(g, 2)}
                       for s, e, r, k, g in used]

    def run(raw, text):
        t0 = time.perf_counter()
        x = decode_audio(raw, FFMPEG)
        if len(x) < SR * 0.3:
            return 400, {"code": "too_short"}
        A, Bm = S["A"], S["B"]
        rej = {"A": None, "B": None}
        # 넣을 버전의 입력 거르기 — 운영 app.py sent_eval_kr 와 같은 순서(no_korean → too_long → 말소리 → no_match)
        if not re.search(r"[가-힣]", text):
            rej["B"] = {"code": "no_korean", "msg": "기준 문장에 한글이 없어요. (앱에는 한국어 문장을 보내라는 오류가 갑니다)"}
        elif len(x) > SR * MAX_SPEECH_SEC:
            w2, _ = Bm.trim(x, SR)
            if len(w2) > SR * MAX_SPEECH_SEC:
                rej["B"] = {"code": "too_long", "msg": "녹음이 너무 길어요. 말한 부분 %.0f초 · %.0f초까지 받습니다. (지금 운영은 앞 %.0f초만 채점)"
                                                      % (len(w2) / SR, MAX_SPEECH_SEC, MAX_SPEECH_SEC)}
        bad, info = A.check_audio(x[: SR * 20], text)            # 짧은 문장 말소리 기준 — 두 쪽 같다(10-18 운영 반영)
        if bad:
            low = {"code": "low_speech", "msg": "말소리가 거의 안 들려요. 말한 시간 %.1f초 · 이 문장은 %.1f초 이상 필요 · 소리 비율 %d%%."
                                                % (info["voiced_sec"], min_voiced_sec(text), round(100 * info["voiced_ratio"]))}
            rej["A"] = low
            rej["B"] = rej["B"] or low
        out = {"text": text, "sec": round(len(x) / SR, 2), "api_canon": canon_str(A.canonical, text),
               "fixed_canon": canon_str(S["builder"].fixed, text), "ms": {}}
        with lock:
            for side, emu in (("A", A), ("B", Bm)):
                if rej[side]:
                    out[side] = {"rejected": rej[side]}
                    continue
                t1 = time.perf_counter()
                lp = emu.lprobs(x)
                canon, used = pick_canon(emu, lp[0], text)
                with rules(**S["rules"][side]):
                    body, r = emu.assess(x, text, canon=canon, lp=lp, gate=False)
                out["ms"][side] = round(1000 * (time.perf_counter() - t1))
                if body is None:
                    out[side] = {"rejected": {"code": "assess_failed", "msg": "채점하지 못했어요."}}
                    continue
                mm = mismatch_ratio(r["sites"])
                if side == "B" and mm is not None and mm > NOMATCH_B:
                    out[side] = {"rejected": {"code": "no_match", "msg": "기준 문장 소리 중 %d%%가 다르게 들려 말소리를 못 알아들었다고 거절합니다. (%d%% 넘으면 거절)"
                                                                       % (round(100 * mm), round(100 * NOMATCH_B))}}
                    continue
                out[side] = summarize(body, r, emu, lp[0])
                out[side].update(variants=used, mismatch=None if mm is None else round(mm, 2))
        out["ms"]["total"] = round(1000 * (time.perf_counter() - t0))
        return 200, out

    @app.post("/compare")
    async def compare(audio: UploadFile = File(...), refText: str = Form(...)):
        text = (refText or "").strip()
        if not text:
            return JSONResponse({"code": "no_reftext"}, status_code=400)
        raw = await audio.read()
        try:
            code, out = await run_in_threadpool(run, raw, text)
        except ValueError as e:
            return JSONResponse({"code": "decode_failed", "error": str(e)[:200]}, status_code=400)
        return JSONResponse(out, status_code=code)

    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8936)
    ap.add_argument("--a-ckpt", dest="a_ckpt", help="지금 운영 모델",
                    default=os.path.join(str(DATA_ROOT), "checkpoints", "w2vbert_synth_tense_v2", "interp_a0.5_step3000"))
    ap.add_argument("--b-ckpt", dest="b_ckpt", help="넣을 버전 모델",
                    default=os.path.join(str(DATA_ROOT), "checkpoints", "w2vbert_synth_tense_v2", "step_3000"))
    ap.add_argument("--scorer", default=os.path.join(str(MDD_OUTPUTS), "score_v2_w2vbert", "score_models_v2_w2vbert.joblib"))
    ap.add_argument("--margin", type=float, default=1.0, help="표준발음 보정 변이 margin(운영 --canon-fix-margin)")
    a = ap.parse_args()
    from mdd.api_offline import ApiEmulator
    from mdd.canonical_fix import CanonicalBuilder
    A = ApiEmulator(a.a_ckpt, scorer=a.scorer)
    Bm = ApiEmulator(a.b_ckpt, scorer=a.scorer)
    S.update(A=A, B=Bm, builder=CanonicalBuilder(A.canonical), a_ckpt=a.a_ckpt, b_ckpt=a.b_ckpt, margin=a.margin,
             rules=side_rules())
    import uvicorn
    print("[serve] http://127.0.0.1:%d/ · A 지금 운영 %s · B 넣을 버전 %s · B 규칙 %s · no_match %.1f · too_long %.0f초"
          % (a.port, a.a_ckpt, a.b_ckpt, {k: (v if k != "LIQUID_NASAL" else bool(v)) for k, v in S["rules"]["B"].items()},
             NOMATCH_B, MAX_SPEECH_SEC), flush=True)
    uvicorn.run(make_app(), host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
