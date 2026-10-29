# -*- coding: utf-8 -*-
"""성대모사 교정 Phase 0 데모 — 대사를 고르고(또는 원본을 올리고) 브라우저에서 녹음하면 어긋난 곳을 짚는다.

계획서 Phase 0 완료 기준("짚은 글자가 팀원 귀로 듣기에도 어긋난 곳이 맞음")을 팀원이 직접 확인하는 용도다.
분석은 전부 `tasks/mimicry-coach/src/mimicry` 가 한다 — 이 파일엔 라이브러리 로직을 넣지 않는다.

    GET  /api/clips                 견본 대사 목록(make_sample_clips.py 산출물의 clips.json)
    GET  /api/clips/{id}/ref.wav    원본 음성(원작이면 목소리만) · /mix.wav 배경음 포함(원작 발췌만)
    POST /api/compare               take(녹음 파일) + clip(견본 id) 또는 ref(원본 파일)+text(대사) → 보고서 HTML·지적·수치
    POST /api/label                 지적 하나에 대한 사람 판단(맞아요·아니에요·모르겠어요) — `--labels` 를 줬을 때만.
                                    허용치 보정(계획서 8장)·Phase 0 귀 확인의 재료. 녹음·접속 정보는 안 남기고 대사·지적·잰 값·판단만.

    PYTHONPATH=/data/speech-ai/tools/hf_py38:/data/speech-ai/tools/mimic_py38 HF_HUB_OFFLINE=1 \\
        $PY deploy/mimicry-demo/app.py --clips /data/speech-ai/out/mimic/original_clips_v1 --port 8962
    (TTS 견본도 같이: --clips .../original_clips_v1 --clips .../sample_clips_v1)

⚠️ 원작 발췌(original_clips_v1)는 권리 확인 전 내부 시험용 — funnel 공개 주소로 내보내지 말 것(계획서 7.3).
⚠️ 로컬 확인용이다. 인증이 없다. 녹음은 메모리에서만 처리하고 디스크에 남기지 않는다(디코드용 임시 파일은 즉시 삭제).
   마이크는 보안 컨텍스트에서만 열린다 — http://127.0.0.1:<port> 로(SSH 포트 포워딩) 접속할 것.
   팀원에게 열 땐 `tailscale funnel --bg --set-path` (serve 는 funnel 을 끈다 — CLAUDE.md). 이미 쓰는 포트는 피할 것.
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import argparse, datetime, json, os, threading, time, uuid

import numpy as np

SR = 16000
MAX_UPLOAD = 20 * 1024 * 1024    # 녹음 파일 한 개 최대 바이트(15초 녹음은 opus 로 수백 KB)
S = {}


def load_clips(dirs):
    """clips.json 폴더들 → {id: clip}. 원작 발췌(prepare_clips.py)·TTS 견본(make_sample_clips.py) 둘 다 받는다.
    두 번째 폴더부터는 id 가 겹치면 앞에 종류를 붙인다. 품질 검사 탈락 클립은 뺀다."""
    out = {}
    for d in dirs or []:
        meta = json.load(open(os.path.join(d, "clips.json"), encoding="utf-8"))
        kind = "original" if meta.get("kind") == "original-excerpt" else "tts"
        for c in meta["clips"]:
            qc = c.get("qc") or {}
            ok = qc.get("ok", (qc.get("ref") or {}).get("ok", True))  # 원작 발췌: qc.ok · TTS 견본: qc.ref.ok
            if not ok:
                print("[mimicry-demo] 품질 검사 탈락이라 뺌: %s %s" % (c["id"], c.get("line")), flush=True)
                continue
            cid = c["id"] if c["id"] not in out else "%s-%s" % (kind, c["id"])
            src = c.get("source") or {}
            credit = ("출처: YouTube · %s" % src["channel"]) if src.get("channel") else ("TTS 견본(원작 음성 아님)" if kind == "tts" else "")
            out[cid] = dict(c, id=cid, dir=d, kind=kind, credit=credit)
    return out


def plain(o):
    """numpy 수(쉼이 있으면 경계 값이 numpy float 로 남는다)를 JSON 이 받는 기본형으로."""
    if isinstance(o, dict):
        return {k: plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [plain(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    return o


def checks_brief(C):
    """전/후 비교용 — 음절별 항목 차이만 추린다(페이지가 지난 시도와 맞춰 본다)."""
    return [dict(i=row["i"], uncertain=row["uncertain"],
                 checks={k: dict(diff=v["diff"], ratio=round(v["ratio"], 3), flag=v["flag"]) for k, v in row["checks"].items()})
            for row in C["rows"]]


def make_app(static_dir):
    from fastapi import FastAPI, File, Form, UploadFile
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from mimicry.audio import decode_bytes, load_audio
    from mimicry.report import html, terminal

    app = FastAPI(title="mimicry-demo")
    lock = threading.Lock()                         # 모델 하나를 같이 쓴다
    label_lock = threading.Lock()
    cache = {}                                      # 견본 원본 분석(원본은 한 번만 분석)

    @app.get("/health")
    def health():
        return {"ok": True, "device": S["coach"].al.device, "clips": len(S["clips"])}

    @app.get("/api/clips")
    def clips():
        return {"labels": bool(S.get("labels")),
                "clips": [dict(id=c["id"], line=c["line"], title=c["title"], kind=c["kind"], credit=c["credit"],
                               has_mix="mix" in c["files"] and c["kind"] == "original") for c in S["clips"].values()]}

    VERDICTS = ("yes", "no", "unsure")
    ITEM_KEYS = ("type", "i", "i_end", "dir", "diff", "ratio", "what", "tier")

    @app.post("/api/label")
    async def label(body: dict):
        """{result, clip, line, item: {type, i, i_end, dir, diff, ratio, what, tier}, verdict} → 한 줄 JSONL."""
        if not S.get("labels"):
            return JSONResponse({"error": "판단 모으기가 꺼져 있어요(--labels)"}, status_code=404)
        item = body.get("item") or {}
        if body.get("verdict") not in VERDICTS or not isinstance(item, dict) or not item.get("type"):
            return JSONResponse({"error": "verdict(yes·no·unsure)와 item.type 이 필요해요"}, status_code=400)
        rec = dict(ts=datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z", result=str(body.get("result", ""))[:40],
                   clip=str(body.get("clip", ""))[:40], line=str(body.get("line", ""))[:80], verdict=body["verdict"],
                   item={k: item.get(k) for k in ITEM_KEYS})
        with label_lock, open(S["labels"], "a", encoding="utf-8") as f:
            f.write(json.dumps(plain(rec), ensure_ascii=False) + "\n")
        return {"ok": True}

    @app.get("/api/clips/{cid}/{name}.wav")
    def clip_audio(cid: str, name: str):
        """ref = 분석에 쓰는 원본(원작이면 목소리만 분리한 것) · mix = 배경음 포함(원작 발췌만)."""
        c = S["clips"].get(cid)
        if not c or name not in ("ref", "mix") or name not in c["files"]:
            return JSONResponse({"error": "없는 대사"}, status_code=404)
        return FileResponse(os.path.join(c["dir"], c["files"][name]), media_type="audio/wav")

    def check_audio(w, what):
        if len(w) < SR * 0.4:
            raise ValueError("%s이 너무 짧아요 (0.4초 미만)" % what)
        if float(np.sqrt(np.mean(w[: SR * 20] ** 2))) < 1.5e-3:
            raise ValueError("%s에 소리가 거의 없어요 — 마이크를 확인해 주세요" % what)
        return w[: SR * 15]

    @app.post("/api/compare")
    async def compare_api(take: UploadFile = File(...), clip: str = Form(""), text: str = Form(""),
                          ref: UploadFile = File(None)):
        t0 = time.time()
        try:
            raw = await take.read()
            if len(raw) > MAX_UPLOAD:
                return JSONResponse({"error": "녹음 파일이 너무 커요(20 MB 넘음)"}, status_code=413)
            xt = check_audio(decode_bytes(raw), "녹음")
            if clip:
                c = S["clips"].get(clip)
                if not c:
                    return JSONResponse({"error": "없는 대사"}, status_code=404)
                line, title = c["line"], c["title"]
                xr = load_audio(os.path.join(c["dir"], c["files"]["ref"]))
            else:
                if ref is None or not text.strip():
                    return JSONResponse({"error": "견본을 고르거나, 원본 파일과 대사를 함께 주세요"}, status_code=400)
                line, title = text.strip(), "내가 올린 원본"
                xr = check_audio(decode_bytes(await ref.read()), "원본")
            with lock:
                pre = cache.get(clip) if clip else None
                if clip and pre is None:
                    pre = cache[clip] = S["coach"].analyze(xr, line)
                A, B, C = S["coach"].compare(xr, xt, line, ref=pre)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        except RuntimeError as e:
            return JSONResponse({"error": "분석 실패: %s" % e}, status_code=500)
        return plain({"ms": int((time.time() - t0) * 1000), "result": uuid.uuid4().hex[:12], "line": line,
                "html": html(A, B, C, xr, xt, title), "text": terminal(A, B, C),
                "quality": C["quality"], "match": C["match"], "tempo": C["tempo"],
                "syllables": [s["text"] for s in A["syllables"]],
                "selected": [dict(i=s.get("i"), i_end=s.get("i_end"), type=s["type"], dir=s.get("dir"), diff=s.get("diff"),
                                  ratio=round(s["ratio"], 3), say=s["say"]) for s in C["selected"]],
                "minor": [dict(i=s.get("i"), i_end=s.get("i_end"), type=s["type"], dir=s.get("dir"), diff=s.get("diff"),
                               ratio=round(s["ratio"], 3), say=s["say"]) for s in C.get("minor", [])],
                "rows": checks_brief(C)})

    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", action="append", default=[],
                    help="대사 폴더(clips.json) — 여러 번 줄 수 있다(원작 발췌 먼저). 없으면 원본을 직접 올려 쓴다")
    ap.add_argument("--labels", default="", help="지적마다 사람 판단(맞아요/아니에요)을 이 JSONL 에 모은다(녹음은 안 남긴다)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8962)
    a = ap.parse_args()
    from mimicry.pipeline import Coach
    S["coach"] = Coach()
    S["clips"] = load_clips(a.clips)
    S["labels"] = a.labels or None
    print("[mimicry-demo] 견본 %d개 · %s · http://%s:%d" % (len(S["clips"]), S["coach"].al.device, a.host, a.port), flush=True)
    import uvicorn
    uvicorn.run(make_app(os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")),
                host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
