# -*- coding: utf-8 -*-
"""문장 독립 발음 진단 데모 — **아무 문장이나** 입력하고 읽으면 자모 단위로 진단한다.

서버에는 로직이 없다. 진단은 `mdd.phone_detector.PhoneDetector`, 설명은 `mdd.phone_explain`, 오디오 디코딩은 `mdd.pipeline.decode_audio`.

  GET  /            데모 화면
  GET  /health      올라간 모델·임계값
  GET  /samples     내장 샘플 목록 (PD_SAMPLES_JSON 이 있을 때)      GET /sample_audio/{i}
  POST /diagnose    audio + sentence (+ l1, mode) → 음절·자모별 판정 · 들린 소리 · 조음 설명 · 시간 구간
                    mode: strict(기본, 정상 자리 오지적 1% 동작점) | balanced(F1 최대)

설정(환경변수): PD_ENSEMBLE_JSON — anchor · dict_dir · detectors · threshold_f1 · operating_points 를 담은 JSON
               (기본 $MDD_OUTPUTS/../novelty/detector/ensemble_v3.json 을 찾지 않는다 — 명시할 것)

Run:  PD_ENSEMBLE_JSON=<json> $PY deploy/phone-detector-demo/app.py --port 8921      (tmux 안에서)
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import argparse, json, os, threading, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
S = {"det": None, "cfg": None, "lock": threading.Lock(), "samples": []}
L1_CHOICES = ["?", "Vietnamese", "Chinese", "Japanese", "English", "Indonesian", "Thai", "Russian", "Spanish; Castilian", "Mongolian",
              "Uzbek", "French", "German", "Turkish"]


def load():
    from mdd.phone_detector import PhoneDetector
    p = os.environ.get("PD_ENSEMBLE_JSON")
    if not p or not os.path.exists(p):
        raise SystemExit("PD_ENSEMBLE_JSON 을 지정할 것 (anchor · dict_dir · detectors · threshold_f1 · operating_points)")
    cfg = json.load(open(p, encoding="utf-8"))
    t0 = time.time()
    S["det"] = PhoneDetector(cfg["anchor"], cfg["detectors"], cfg["dict_dir"], threshold=cfg.get("threshold_f1"))
    S["cfg"] = cfg
    sj = os.environ.get("PD_SAMPLES_JSON")
    if sj and os.path.exists(sj):
        S["samples"] = json.load(open(sj, encoding="utf-8"))
    print(f"[load] anchor {os.path.basename(os.path.dirname(cfg['anchor']))} · 판별기 {len(cfg['detectors'])}개 · 샘플 {len(S['samples'])} ({time.time()-t0:.0f}s)", flush=True)


def run(wave, sentence, l1, mode):
    from mdd import phone_explain as PE
    det, cfg = S["det"], S["cfg"]
    th = cfg["operating_points"]["frr0.01"]["threshold"] if mode == "strict" else cfg["threshold_f1"]
    dur = len(wave) / 16000.0
    if dur < 0.3 or float(np.sqrt((wave ** 2).mean())) < 1.5e-3:
        return {"no_speech": True, "duration_sec": round(dur, 2)}
    t0 = time.time()
    with S["lock"]:                                     # 단일 GPU 모델 — 직렬화
        out = det.diagnose(wave[: 30 * 16000], sentence=sentence, l1=None if l1 in (None, "", "?") else l1)
    if not out["align_ok"]:
        return {"no_speech": False, "align_ok": False, "message": "문장과 음성을 맞추지 못했어요. 문장 전체를 읽었는지 확인해 주세요."}
    canon = out["canonical"]; pos = PE.positions(canon)
    phones = []
    for ph, ps in zip(out["phones"], pos):
        heard = ph["heard_top3"][0][0]
        flag = ph["p_err"] >= th
        ex = PE.explain(ph["jamo"], heard, ps) if flag else None
        if ex and ex["native_merger"]:
            flag = False                                # 원어민도 구별하지 않는 쌍(ㅐ/ㅔ 등)은 지적하지 않는다
        phones.append({**ph, "pos": ps, "flag": bool(flag), "explain": ex if flag else None})
    syl = [{"text": s["text"], "idx": s["idx"], "p_max": max(phones[j]["p_err"] for j in s["idx"]),
            "flag": any(phones[j]["flag"] for j in s["idx"])} for s in PE.syllabify(canon)]
    return {"no_speech": False, "align_ok": True, "sentence": sentence, "pron": "".join(s["text"] for s in syl), "mode": mode,
            "threshold": round(th, 4), "duration_sec": round(dur, 2), "took_sec": round(time.time() - t0, 3),
            "n_flag": sum(p["flag"] for p in phones), "syllables": syl, "phones": phones}


def make_app():
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import FileResponse, HTMLResponse
    from mdd import pipeline as MD
    app = FastAPI(title="문장 독립 발음 진단 데모")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return open(os.path.join(HERE, "index.html"), encoding="utf-8").read()

    @app.get("/health")
    def health():
        c = S["cfg"] or {}
        return {"status": "ok" if S["det"] else "loading", "anchor": c.get("anchor"), "n_detectors": len(c.get("detectors", [])),
                "threshold_f1": c.get("threshold_f1"), "operating_points": c.get("operating_points"), "l1_choices": L1_CHOICES,
                "note": "안 본 문장 기준 F1 0.50 · 오지적 2.4% (results/15). strict 모드는 오지적 1% 동작점."}

    @app.get("/samples")
    def samples():
        return [{"i": i, "sentence": s["sentence"], "l1": s.get("l1"), "note": s.get("note", "")} for i, s in enumerate(S["samples"])]

    @app.get("/sample_audio/{i}")
    def sample_audio(i: int):
        if not 0 <= i < len(S["samples"]):
            raise HTTPException(404, "no such sample")
        return FileResponse(S["samples"][i]["wav"], media_type="audio/wav")

    @app.post("/diagnose")
    def diagnose(audio: UploadFile = File(...), sentence: str = Form(...), l1: str = Form("?"), mode: str = Form("strict")):
        if not S["det"]:
            raise HTTPException(503, "loading")
        sentence = (sentence or "").strip()
        if not sentence or len(sentence) > 120:
            raise HTTPException(400, "문장을 입력해 주세요 (120자 이하)")
        try:
            w = MD.decode_audio(audio.file.read())
        except ValueError:
            raise HTTPException(400, "오디오를 읽지 못했어요")
        try:
            return run(w, sentence, l1, mode if mode in ("strict", "balanced") else "strict")
        except ValueError as e:
            raise HTTPException(400, f"표준발음을 만들지 못했어요 — 한글 문장인지 확인해 주세요 ({e})")

    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8921)
    a = ap.parse_args()
    load()
    import uvicorn
    uvicorn.run(make_app(), host="0.0.0.0", port=a.port)
