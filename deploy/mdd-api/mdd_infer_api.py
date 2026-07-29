# -*- coding: utf-8 -*-
"""발음진단 통합 추론 API — MDD(어디를 틀렸나) + 점수 3종(얼마나 잘하나)을 한 번에.

핵심: 점수용 GOP 특징은 MDD 인코더 forward 의 **부산물**이다. 따라서 모델을 한 번만
돌리고 두 결과를 모두 낸다(공식 팩 방식이라면 모델 4번 + 모국어 판별이 필요하다).

  POST /diagnose        오디오 + 문장  → 음소별 진단 (+ 세션 누적 시 점수)
  GET  /session/{id}    세션 점수 조회
  DELETE /session/{id}  세션 초기화
  GET  /health

⚠️ 점수는 **세션 단위로만 유효**하다. 발화 1개 PCC는 0.43, 20~30발화면 0.66~0.76으로 포화한다.
   그래서 발화 수가 MIN_UTTS 미만이면 점수를 내지 않고 `pending` 으로 돌려준다.
   (사람 채점자도 발화가 아니라 학습자를 보고 점수를 매겼다 — 라벨 분산의 74~94%가 화자로 설명됨)

Run:  python mdd_infer_api.py --port 8903
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
from shared.paths import MDD_OUTPUTS

import os, sys, json, time, argparse, threading, collections
import numpy as np
import torch
import torch.nn.functional as F
import editdistance
import joblib

from mdd import gop_feats_v2 as GF                     # forward_lprobs / viterbi_align / align_feats
from mdd import pipeline as MD                         # canonical / _nw / ctype / syllable_ids / decode_audio

from mdd.mdd_eval import Decoder
from mdd.jamo_to_kog2p import jamo_to_phones

SR = 16000
MIN_UTTS = 5                                  # 이 밑에서는 점수를 내지 않는다
GOOD_UTTS = 20                                # 여기부터 신뢰 구간 (30에서 포화)
MODELS_PATH = f"{MDD_OUTPUTS}/score_models_v2.joblib"
S = {"dec": None, "bundle": None, "lock": threading.Lock(),
     "sessions": collections.defaultdict(list)}


def load():
    t0 = time.time()
    import g2p
    MD.S["ri"], MD.S["ro"] = g2p.readRules(g2p.ver_info[0], MD.RULEBOOK)
    S["dec"] = Decoder()
    S["bundle"] = joblib.load(MODELS_PATH)
    print(f"[load] 모델 1개 + 회귀기 3개 · 특징 {len(S['bundle']['feature_names'])}개 "
          f"({time.time()-t0:.0f}s)", flush=True)


@torch.inference_mode()
def analyze(wave, canon_toks):
    """MDD 진단 + 점수 특징을 **같은 forward 한 번**으로 뽑는다."""
    dec = S["dec"]
    lp, Ls = GF.forward_lprobs(dec, [wave])
    l = lp[0, :Ls[0]].numpy()
    dur = len(wave) / SR
    frame_sec = dur / max(1, l.shape[0])

    y = [dec.d.index(t) for t in canon_toks]
    if dec.d.unk() in y:
        raise ValueError("표준발음에 사전 밖 기호가 있습니다")

    g = l.argmax(-1)
    import itertools
    pred_idx = [int(v) for v, _ in itertools.groupby(g) if v != dec.blank]
    pred = [dec.d[v] for v in pred_idx]
    pred = [p for p in pred if p in MD.JAMO]
    per = editdistance.eval(pred_idx, y) / max(1, len(y))

    segs = GF.viterbi_align(l, y, blank=dec.blank)
    if segs is None:
        gops = [-10.0] * len(y)
    else:
        gops = []
        for si, fs in enumerate(segs):
            if not fs:
                gops.append(-10.0); continue
            fa = np.asarray(fs)
            gops.append(float((l[fa, y[si]] - l[fa].max(-1)).mean()))
    ga = np.asarray(gops)

    blanks = (g == dec.blank)
    spans = [(fs[0] * frame_sec, (fs[-1] + 1) * frame_sec) for fs in (segs or []) if fs]
    if len(spans) >= 2:
        lead_sil = spans[0][0]; trail_sil = max(0.0, dur - spans[-1][1])
        speech_span = max(0.1, spans[-1][1] - spans[0][0])
        gaps = [b0 - e0 for (_, e0), (b0, _) in zip(spans, spans[1:])]
        pauses = [gp for gp in gaps if gp >= GF.PAUSE_SEC]
    else:
        lead_sil = trail_sil = 0.0; speech_span = max(0.1, dur); pauses = []
    pause_tot = float(sum(pauses)); speak = max(0.1, speech_span - pause_tot)

    tl = torch.from_numpy(l).unsqueeze(1)
    nll = float(F.ctc_loss(tl, torch.tensor([y]), torch.tensor([l.shape[0]]),
                           torch.tensor([len(y)]), blank=dec.blank,
                           reduction="sum", zero_infinity=True)) / max(1, len(y))

    feat = {
        "gop": float(ga.mean()), "gop_p10": float(np.percentile(ga, 10)),
        "gop_p25": float(np.percentile(ga, 25)), "gop_med": float(np.median(ga)),
        "gop_min": float(ga.min()), "gop_std": float(ga.std()),
        "gop_bad": float((ga < -2.0).mean()),
        "neg_nll": -nll, "neg_per": -per, "dur": dur, "n_canon": len(y),
        "rate": len(y) / max(0.1, dur), "speech_rate": len(y) / speech_span,
        "artic_rate": len(y) / speak, "blank_frac": float(blanks.mean()),
        "lead_sil": lead_sil, "trail_sil": trail_sil, "speech_span": speech_span,
        "n_pause": len(pauses), "pause_tot": pause_tot,
        "pause_mean": float(np.mean(pauses)) if pauses else 0.0,
        "max_pause": float(max(pauses)) if pauses else 0.0,
        "pause_rate": len(pauses) / speech_span, "pause_frac": pause_tot / speech_span,
    }
    feat.update(GF.align_feats(canon_toks, pred))
    spans_out = [{"start": round(s0, 3), "end": round(e0, 3)} for s0, e0 in spans]
    return pred, feat, gops, spans_out


import re
_PAREN = re.compile(r"[（(][^）)]*[）)]")


def read_text(sentence):
    """낭독 대상만 남긴다. AIHub 프롬프트는 괄호 안에 '지시문'을 넣는데
    (예: "창문이 열렸네요. (닫아 주세요.)") 화자는 그 부분을 읽지 않는다.
    그대로 표준발음에 넣으면 멀쩡한 발화가 탈락 9개로 잡힌다 — 실측 확인."""
    t = _PAREN.sub(" ", sentence)
    return " ".join(t.split())


def diagnose_ops(canon, pred):
    """정렬 결과를 UI 가 그대로 그릴 수 있는 형태로."""
    ops = MD._nw(canon, pred)
    errors = []
    n = {"sub": 0, "del": 0, "ins": 0}
    for op in ops:
        if op["kind"] == "match":
            continue
        n[op["kind"]] += 1
        tpos = op["tpos"]
        is_coda = False
        if op["kind"] != "ins" and tpos < len(canon) and canon[tpos] not in MD.VOWELS:
            is_coda = (canon[tpos] == "ㅇ") or not (tpos + 1 < len(canon)
                                                    and canon[tpos + 1] in MD.VOWELS)
        errors.append({"pos": tpos, "kind": op["kind"],
                       "type": MD.ctype(op.get("target") or "", op.get("actual") or "",
                                        op["kind"], is_coda),
                       "target": op.get("target") or "∅", "actual": op.get("actual") or "∅"})
    hit = len({e["pos"] for e in errors if e["kind"] != "ins"})
    return errors, n, round(1.0 - hit / max(1, len(canon)), 3)


def session_scores(sid):
    """세션에 쌓인 발화들로 점수 3종. 발화가 적으면 점수를 내지 않는다."""
    rows = S["sessions"].get(sid) or []
    b = S["bundle"]; names = b["feature_names"]
    out = {"n_utts": len(rows), "min_utts": MIN_UTTS, "reliable_at": GOOD_UTTS}
    if len(rows) < MIN_UTTS:
        out["status"] = "pending"
        out["message"] = f"{MIN_UTTS - len(rows)}문장 더 필요합니다 (권장 {GOOD_UTTS}문장)"
        out["scores"] = None
        return out
    X = np.array([[r["feat"][k] for k in names] for r in rows])
    scores = {}
    for dim, meta in b["metrics"].items():
        v = float(np.mean(b["models"][dim].predict(X)))
        scores[meta["name"]] = {"score": round(min(5.0, max(1.0, v)), 2),
                                "key": dim, "pcc": meta["pcc_spk"]}
    out["status"] = "ok" if len(rows) >= GOOD_UTTS else "provisional"
    out["confidence"] = "높음" if len(rows) >= GOOD_UTTS else "낮음 (문장 수 부족)"
    out["scores"] = scores
    # 점수의 근거 — 팩과 달리 왜 그 점수인지 말할 수 있다
    agg = lambda k: float(np.mean([r["feat"][k] for r in rows]))
    out["evidence"] = {
        "평균_쉼_횟수": round(agg("n_pause"), 2), "평균_최장쉼_초": round(agg("max_pause"), 2),
        "조음속도_음소per초": round(agg("artic_rate"), 2),
        "음소오류율": round(-agg("neg_per"), 3),
        "초성오류율": round(agg("r_onset"), 3), "종성오류율": round(agg("r_coda"), 3),
        "모음오류율": round(agg("r_vowel"), 3), "오류_집중도": round(agg("err_clump"), 3),
    }
    return out


def make_app():
    from fastapi import FastAPI, File, UploadFile, Form, HTTPException, APIRouter
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(title="발음진단 통합 API (MDD + 점수 3종)")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                       allow_headers=["*"])
    r = APIRouter()

    @r.get("/health")
    def health():
        b = S["bundle"]
        return {"status": "ok" if S["dec"] else "loading",
                "ckpt": "mdd_ckpt_tag_all_lar/checkpoint_best.pt",
                "mdd_f1_weighted": 0.9547,
                "score_pcc": {m["name"]: m["pcc_spk"] for m in (b["metrics"].values() if b else [])},
                "n_features": len(b["feature_names"]) if b else 0,
                "note": "점수는 세션 단위. 발화 1개로는 신뢰 불가(PCC 0.43)"}

    @r.post("/diagnose")
    async def diagnose(audio: UploadFile = File(...), sentence: str = Form(...),
                       session_id: str = Form(None)):
        if not S["dec"]:
            raise HTTPException(503, "loading")
        sentence = (sentence or "").strip()
        if not sentence or len(sentence) > 200:
            raise HTTPException(400, "sentence required (<=200 chars)")
        try:
            w = MD.decode_audio(await audio.read())
        except ValueError:
            raise HTTPException(400, "could not decode audio")

        dur = len(w) / SR
        if dur < 0.3 or float(np.sqrt((w ** 2).mean())) < 1.5e-3:
            return {"no_speech": True, "duration_sec": round(dur, 2)}

        read = read_text(sentence)
        if not read:
            raise HTTPException(400, "낭독할 내용이 없습니다 (괄호 지시문만 있음)")
        canon, _ = MD.canonical(read)
        if not canon:
            raise HTTPException(400, "표준발음 생성 실패 (숫자·영문 확인)")

        t0 = time.time()
        with S["lock"]:                       # 단일 GPU — 직렬화
            pred, feat, gops, spans = analyze(w[: 30 * SR], canon)
        errors, n, match_rate = diagnose_ops(canon, pred)

        res = {
            "no_speech": False, "sentence": sentence, "read_text": read,
            "duration_sec": round(dur, 2),
            "took_sec": round(time.time() - t0, 3),
            "canonical": {"jamo": canon, "phones": jamo_to_phones(canon)},
            "recognized": {"jamo": pred, "phones": jamo_to_phones(pred) if pred else []},
            "mdd": {"errors": errors, "n_sub": n["sub"], "n_del": n["del"],
                    "n_ins": n["ins"], "match_rate": match_rate},
            "per_phone": [{"i": i, "phone": canon[i], "gop": round(gops[i], 2),
                           **({"span": spans[i]} if i < len(spans) else {})}
                          for i in range(len(canon))],
        }
        if session_id:
            S["sessions"][session_id].append({"feat": feat, "sentence": sentence})
            res["session"] = session_scores(session_id)
        else:
            res["session"] = {"status": "no_session",
                              "message": "session_id 를 보내면 문장이 누적되어 점수가 산출됩니다"}
        return res

    @r.get("/session/{sid}")
    def get_session(sid: str):
        if sid not in S["sessions"]:
            raise HTTPException(404, "no such session")
        out = session_scores(sid)
        out["sentences"] = [r["sentence"] for r in S["sessions"][sid]]
        return out

    @r.delete("/session/{sid}")
    def del_session(sid: str):
        S["sessions"].pop(sid, None)
        return {"deleted": sid}

    app.include_router(r)
    app.include_router(r, prefix="/api")      # funnel --set-path 는 프리픽스를 벗기지 않는다
    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8903)
    a = ap.parse_args()
    load()
    import uvicorn
    uvicorn.run(make_app(), host="0.0.0.0", port=a.port)
