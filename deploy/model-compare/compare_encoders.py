# -*- coding: utf-8 -*-
"""두 모델 나란히 비교 데모 — 인코더 세대 비교 — XLS-R(2021) vs omniASR(2025).

  왼쪽  tag_all      : RW/RS(발음문항)만 학습. LAR 문장낭독 미적응.
  오른쪽 tag_all_lar : 위 체크포인트에 무오류 LAR 3.9만을 섞어 이어학습.

같은 오디오를 두 모델에 동시에 넣고 음소열·정렬·오류수를 좌우로 보여준다.
입력 3가지: ① 실제 AIHub 샘플(내장) ② 마이크 녹음 ③ 파일 업로드.

표준발음(canonical) 출처:
  · AIHub 발음문항 샘플 → manifest의 .canon (LAB JSON 원본, 가장 정확)
  · 그 외(LAR·녹음·업로드) → KoG2P(문장)

Run:  python compare_models.py --port 8902
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
from shared.paths import DATA_ROOT

import os, sys, json, time, argparse
import numpy as np

from mdd import mdd_eval                                  # Decoder는 모듈 전역 CKPT를 읽는다
from mdd import pipeline as MD                            # canonical/_nw/ctype/decode_audio 재사용
from mdd.mdd_eval import Decoder
from mdd.jamo_to_kog2p import jamo_to_phones

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLES_JSON = os.path.join(HERE, "compare_samples.json")
SR = 16000

MODELS = [
    {"key": "xlsr_native", "ckpt": f"{DATA_ROOT}/mdd_ckpt_tag_all_native/checkpoint_best.pt",
     "title": "XLS-R (2021)", "sub": "xlsr_native — 원어민 포함 최종 모델 · TAG F1 0.9558",
     "record_f1": 0.9558, "record_per": 0.0858},
    {"key": "omniasr", "ckpt": f"{DATA_ROOT}/mdd_ckpt_omniw2v/checkpoint_best.pt",
     "title": "omniASR (2025)", "sub": "omniASR-W2V-300M — 동일 데이터 학습 · TAG F1 0.9551",
     "record_f1": 0.9551, "record_per": 0.0866},
]

S = {"dec": {}, "samples": []}


def load():
    t0 = time.time()
    MD.S["ri"], MD.S["ro"] = __import__("g2p").readRules(__import__("g2p").ver_info[0], MD.RULEBOOK)
    for m in MODELS:
        mdd_eval.CKPT = m["ckpt"]                # Decoder() 가 이 전역을 읽는다
        S["dec"][m["key"]] = Decoder()
        print(f"[load] {m['key']} ready ({time.time()-t0:.0f}s)", flush=True)
    if os.path.exists(SAMPLES_JSON):
        S["samples"] = json.load(open(SAMPLES_JSON, encoding="utf-8"))
    print(f"[load] 총 {time.time()-t0:.0f}s · 샘플 {len(S['samples'])}개", flush=True)


def load_wav(path):
    import wave
    with wave.open(path, "rb") as w:
        n = w.getnframes()
        return np.frombuffer(w.readframes(n), np.int16).astype(np.float32) / 32768.0


def run_one(key, wave_arr, canon):
    """모델 1개 디코드 + 표준발음 대비 정렬."""
    t0 = time.time()
    pred = [t for t in S["dec"][key].decode([wave_arr[: 20 * SR]])[0] if t in MD.JAMO]
    took = time.time() - t0

    ops = MD._nw(canon, pred)
    errors, n = [], {"sub": 0, "del": 0, "ins": 0}
    for op in ops:
        if op["kind"] == "match":
            continue
        n[op["kind"]] += 1
        tpos = op["tpos"]
        is_coda = False
        if op["kind"] != "ins" and canon[tpos] not in MD.VOWELS:
            is_coda = (canon[tpos] == "ㅇ") or not (tpos + 1 < len(canon) and canon[tpos + 1] in MD.VOWELS)
        errors.append({
            "pos": tpos, "kind": op["kind"],
            "type": MD.ctype(op.get("target") or "", op.get("actual") or "", op["kind"], is_coda),
            "target": op.get("target") or "∅", "actual": op.get("actual") or "∅",
        })
    hit = len({e["pos"] for e in errors if e["kind"] != "ins"})
    return {
        "pred": pred, "pred_ph": jamo_to_phones(pred) if pred else [],
        "ops": [{k: op[k] for k in ("kind", "target", "actual", "tpos", "apos")} for op in ops],
        "errors": errors, "n_err": len(errors), "n_sub": n["sub"], "n_del": n["del"], "n_ins": n["ins"],
        "match_rate": round(1.0 - hit / max(1, len(canon)), 3),
        "n_pred": len(pred), "took_sec": round(took, 3),
    }


def compare(wave_arr, sentence=None, canon_override=None, ref=None):
    dur = len(wave_arr) / SR
    if dur < 0.3 or float(np.sqrt((wave_arr ** 2).mean())) < 1.5e-3:
        return {"no_speech": True, "duration_sec": round(dur, 2)}

    if canon_override:
        canon = canon_override.split() if isinstance(canon_override, str) else list(canon_override)
        src = "LAB(AIHub 원본)"
    else:
        canon, _ = MD.canonical(sentence)
        src = "KoG2P(규칙 생성)"
    if not canon:
        return {"error": "표준발음을 만들 수 없습니다 (숫자·영문이 섞였는지 확인)"}

    out = {m["key"]: run_one(m["key"], wave_arr, canon) for m in MODELS}
    a, b = out[MODELS[0]["key"]], out[MODELS[1]["key"]]
    n_syl = (MD.syllable_ids(canon)[-1] + 1) if canon else 0
    return {
        "no_speech": False, "sentence": sentence, "canon": canon,
        "canon_ph": jamo_to_phones(canon), "canon_src": src,
        "duration_sec": round(dur, 2), "speaking_rate": round(n_syl / max(0.1, dur), 2),
        "models": out, "ref": ref,
        "delta": {"n_err": a["n_err"] - b["n_err"], "n_del": a["n_del"] - b["n_del"],
                  "match_rate": round(b["match_rate"] - a["match_rate"], 3),
                  "n_pred": a["n_pred"] - b["n_pred"]},
    }


def make_app():
    from fastapi import FastAPI, File, UploadFile, Form, HTTPException, APIRouter
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse, FileResponse
    from typing import Optional

    app = FastAPI(title="MDD 모델 비교 (LAR 적응 전/후)")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    r = APIRouter()

    @r.get("/", response_class=HTMLResponse)
    def index():
        return open(os.path.join(HERE, "compare_encoders.html"), encoding="utf-8").read()

    @r.get("/health")
    def health():
        return {"status": "ok" if S["dec"] else "loading",
                "models": [{k: m[k] for k in ("key", "title", "sub", "record_f1", "record_per")} for m in MODELS],
                "n_samples": len(S["samples"])}

    @r.get("/samples")
    def samples():
        return [{"i": i, "task": s["task"], "prompt": s["prompt"], "note": s["note"],
                 "dur": s["dur"], "stem": s["stem"],
                 "ref_kind": "TAG 전문가 음소전사" if s.get("ltr") else ("전사(맞춤법)" if s.get("tr") else None)}
                for i, s in enumerate(S["samples"])]

    @r.get("/sample_audio/{i}")
    def sample_audio(i: int):
        if not 0 <= i < len(S["samples"]):
            raise HTTPException(404, "no such sample")
        return FileResponse(S["samples"][i]["wav"], media_type="audio/wav")

    @r.post("/compare_sample")
    def compare_sample(i: int = Form(...)):
        if not S["dec"]:
            raise HTTPException(503, "loading")
        if not 0 <= i < len(S["samples"]):
            raise HTTPException(404, "no such sample")
        s = S["samples"][i]
        ref = None
        if s.get("ltr"):
            ref = {"kind": "TAG 전문가 음소전사 (사람이 들은 대로)", "jamo": s["ltr"].split()}
        elif s.get("tr"):
            ref = {"kind": "전사 (사람이 맞춤법으로 받아적음)", "text": s["tr"]}
        res = compare(load_wav(s["wav"]), sentence=s["prompt"],
                      canon_override=s.get("canon"), ref=ref)
        res["sample"] = {"stem": s["stem"], "task": s["task"], "note": s["note"]}
        return res

    @r.post("/compare")
    async def compare_upload(audio: UploadFile = File(...), sentence: str = Form(...)):
        if not S["dec"]:
            raise HTTPException(503, "loading")
        sentence = (sentence or "").strip()
        if not sentence or len(sentence) > 80:
            raise HTTPException(400, "sentence required (<=80 chars)")
        try:
            w = MD.decode_audio(await audio.read())
        except ValueError:
            raise HTTPException(400, "could not decode audio")
        return compare(w, sentence=sentence)

    app.include_router(r)
    # tailscale funnel --set-path 는 프리픽스를 벗기지 않고 그대로 전달하므로 같은 라우터를 한 번 더 건다
    app.include_router(r, prefix="/compare")
    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8902)
    a = ap.parse_args()
    load()
    import uvicorn
    uvicorn.run(make_app(), host="0.0.0.0", port=a.port)
