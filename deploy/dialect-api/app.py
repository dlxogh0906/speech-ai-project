# -*- coding: utf-8 -*-
"""사투리 분류 API + 데모 화면 (2026-10-26).

국적 분류 API(NPU nat-api)와 같은 모양이다 — multipart 로 오디오를 올리면 JSON 이 온다.
토큰은 `X-API-Key: <토큰>` 또는 `Authorization: Bearer <토큰>`. 화면과 /health 는 토큰 없이 열린다.
모델은 봉인 test 를 채점한 EAT Large + XLS-R 2B 앙상블(검증셋에서 고른 규칙 그대로)이다.
--backend gpu(기본)는 원래 모델을 GPU 에서, --backend npu 는 NPU 번들(dialect_classifier.npu_ensemble · 12초 고정 .rbln)을 돈다.

  GET  /               데모 화면
  GET  /health         상태 · 모델 · 규칙 · 누적 요청 수
  POST /predict        audio 1개(최대 60초) [+ parts 1~5] → 12초 이하 구간으로 나눠 판정하고 평균한다
  POST /predict_long   /predict 와 같고 parts 기본값이 3 (국적 API 와 같은 이름)
  POST /predict_multi  audio 여러 개(1~5, 문장마다 따로 녹음) → 파일마다 판정하고 평균한다
  선택 필드 self_report(출신 지역 자기 신고) · session · sentence_id 는 판정에 쓰지 않고 저장 기록에만 남는다.

  200  {"label", "label_ko", "prob", "probs": [6개, 확률 순], "standard", "low_confidence", "n", "parts_requested",
        "sec", "voiced_sec", "per_part": [...], "models": {...}, "rule", "encode_ms", "total_ms", "rid"}
  400  {"error", "code"}  too_large · too_many_files · decode_failed · too_short · no_speech
  401  {"error", "code": "unauthorized"}
  500  {"error", "code": "server_decoder"}   ffmpeg 자체가 못 돈다 — 요청이 아니라 서버 설정 문제

Funnel(`tailscale funnel --set-path /dialect`)은 프리픽스를 벗겨서 넘기지만, / 와 /dialect 양쪽에 마운트한다.
SPEECH_CAPTURE_DIR 이 있으면 모든 요청의 업로드 원본과 응답을 남긴다(shared/request_capture.py · NPU 와 같은 형식).
실행은 deploy/dialect-api/start.sh, 연동 안내는 deploy/dialect-api/README.md.
"""
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import FFMPEG
from shared.request_capture import capture_app, capture_context

import argparse, datetime, hmac, os, shutil, subprocess, tempfile, time
from typing import List

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SR = 16000
MAX_BYTES = 8 * 1024 * 1024        # 파일 하나
MAX_SEC = 60.0                     # 파일 하나에서 읽는 길이
MAX_FILES = 5
MIN_PART, MAX_PART = 3.0, 12.0     # 학습 구간이 3~12초였다
GATE_RMS = 1.5e-3                  # 말소리 프레임 기준 — 국적 API 와 같은 값
PART_MIN_VOICED = 0.3              # 말소리가 이보다 적은 구간(긴 쉼)은 건너뛴다
MIN_VOICED = 1.0
LOWCONF = 0.6                      # 검증셋: 1위 확률 0.6 미만(9.4%)의 정답률 0.66, 나머지 0.986
FFMPEG_PATH = shutil.which(FFMPEG) or FFMPEG
FFMPEG_ENV = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}   # conda 함정(ENVIRONMENT.md 함정 10)

MODEL = None
MODEL_DESC = "EAT Large + XLS-R 2B 앙상블 (content_grouped_full_20261015)"
API_KEY = ""
STATS = dict(requests=0, errors=0, started=datetime.datetime.now().astimezone().isoformat(timespec="seconds"))


class AudioError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def decode(raw, max_sec=MAX_SEC):
    # m4a 는 목차(moov)가 파일 끝에 있을 수 있어 파이프가 아니라 임시 파일로 넘긴다(국적 API 와 같은 이유).
    with tempfile.NamedTemporaryFile(suffix=".audio") as tf:
        tf.write(raw)
        tf.flush()
        p = subprocess.run([FFMPEG_PATH, "-hide_banner", "-loglevel", "error", "-i", tf.name,
                            "-f", "f32le", "-ac", "1", "-ar", str(SR), "-t", str(max_sec), "pipe:1"],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=False, env=FFMPEG_ENV)
    if p.returncode != 0:
        err = p.stderr.decode("utf-8", "replace").strip()
        if p.returncode in (126, 127) or "error while loading shared libraries" in err or "symbol lookup error" in err:
            raise RuntimeError("ffmpeg 실행 실패 (rc=%d): %s" % (p.returncode, err[:500]))
        raise AudioError("decode_failed", "오디오 파일을 해석할 수 없다")
    return np.frombuffer(p.stdout, np.float32).copy()


def voiced_sec(wav, frame_sec=0.03):
    f = int(SR * frame_sec)
    n = wav.size // f
    if n == 0:
        return 0.0
    rms = np.sqrt(np.square(wav[: n * f].reshape(n, f), dtype=np.float64).mean(axis=1))
    return float((rms >= GATE_RMS).sum() * frame_sec)


def split(wav, parts):
    """parts 개로 나눈다. 구간이 3초보다 짧아지면 줄이고 12초를 넘으면 늘린다. 말소리가 거의 없는 구간은 뺀다."""
    sec = wav.size / SR
    n = parts if sec >= parts * MIN_PART else max(1, int(sec // MIN_PART))
    n = max(n, int(np.ceil(sec / MAX_PART)))
    cuts = [round(j * wav.size / n) for j in range(n + 1)]
    return [(wav[a:b], round(a / SR, 2), round(b / SR, 2)) for a, b in zip(cuts[:-1], cuts[1:])
            if b - a >= SR and voiced_sec(wav[a:b]) >= PART_MIN_VOICED]


def entry(p):
    from dialect_classifier.ensemble import LABELS, KO
    i = int(np.argmax(p))
    return {"label": LABELS[i], "label_ko": KO[LABELS[i]], "prob": round(float(p[i]), 4)}


def summarize(out, parts):
    from dialect_classifier.ensemble import LABELS, KO
    mean = out["selected"].mean(0)
    order = np.argsort(-mean)
    eat, xlsr = out["eat_large"].mean(0), out["xlsr2b"].mean(0)
    top = entry(mean)
    return {**top,
            "probs": [{"label": LABELS[i], "label_ko": KO[LABELS[i]], "prob": round(float(mean[i]), 4)} for i in order],
            "standard": top["label"] == "Standard",
            "low_confidence": bool(top["prob"] < LOWCONF),
            "n": len(parts),
            "per_part": [{**entry(p), **span} for p, span in zip(out["selected"], parts)],
            "models": {"eat_large": {**entry(eat), "standard_prob": round(float(eat[LABELS.index("Standard")]), 4)},
                       "xlsr2b": entry(xlsr)},
            "rule": MODEL.rule, "encode_ms": round(out["encode_ms"])}


def make_app():
    from fastapi import APIRouter, FastAPI, File, Form, Request, UploadFile
    from fastapi.responses import HTMLResponse, JSONResponse
    from starlette.concurrency import run_in_threadpool

    app = FastAPI(title="dialect-api", docs_url=None, redoc_url=None, openapi_url=None)

    def fail(code, message, status=400):
        STATS["errors"] += 1
        return JSONResponse({"error": message, "code": code, "rid": (capture_context.get() or {}).get("rid")},
                            status_code=status)

    def authorized(request):
        if not API_KEY:
            return True
        given = request.headers.get("x-api-key", "")
        auth = request.headers.get("authorization", "")
        if not given and auth.lower().startswith("bearer "):
            given = auth[7:].strip()
        return hmac.compare_digest(given.encode("utf-8"), API_KEY.encode("utf-8"))

    async def load(upload):
        raw = await upload.read()
        if len(raw) > MAX_BYTES:
            raise AudioError("too_large", "파일이 너무 크다 (8 MB 초과)")
        return await run_in_threadpool(decode, raw)

    async def run(request, uploads, parts, parts_requested):
        STATS["requests"] += 1
        t0 = time.perf_counter()
        if not authorized(request):
            return fail("unauthorized", "토큰이 없거나 맞지 않다", 401)
        if not uploads or len(uploads) > MAX_FILES:
            return fail("too_many_files", "audio 파일은 1~%d개" % MAX_FILES)
        windows, spans, sec, voiced = [], [], 0.0, 0.0
        try:
            for k, upload in enumerate(uploads):
                wav = await load(upload)
                if wav.size < SR:
                    if len(uploads) == 1:
                        raise AudioError("too_short", "오디오가 너무 짧다 (1초 미만)")
                    continue                         # 여러 파일 중 1초도 안 되는 녹음은 건너뛴다
                sec += wav.size / SR
                voiced += voiced_sec(wav)
                for w, a, b in await run_in_threadpool(split, wav, parts):
                    windows.append(w)
                    spans.append({"file": k, "start": a, "end": b} if len(uploads) > 1 else {"start": a, "end": b})
        except AudioError as e:
            return fail(e.code, str(e))
        except RuntimeError as e:
            print("[decode] %s" % e, flush=True)
            return fail("server_decoder", "서버 오디오 디코더 오류", 500)
        if sec == 0:
            return fail("too_short", "1초 이상인 녹음이 없다")
        if not windows or voiced < MIN_VOICED:
            return fail("no_speech", "말소리가 1초 이상 들리지 않는다 (무음이거나 소리가 너무 작다)")
        out = await run_in_threadpool(MODEL.score, windows)
        body = summarize(out, spans)
        body.update(parts_requested=parts_requested, sec=round(sec, 2), voiced_sec=round(voiced, 1),
                    total_ms=round((time.perf_counter() - t0) * 1000),
                    rid=(capture_context.get() or {}).get("rid"))
        print("[%s] files=%d n=%d %s %.2f %dms" % (request.url.path, len(uploads), body["n"], body["label"],
                                                   body["prob"], body["total_ms"]), flush=True)
        return body

    def parse_parts(parts, default):
        try:
            return min(5, max(1, int(parts))) if parts else default
        except ValueError:
            return default

    r = APIRouter()

    @r.get("/", response_class=HTMLResponse)
    async def index():
        with open(os.path.join(HERE, "index.html"), encoding="utf-8") as f:
            return f.read()

    @r.get("/health")
    async def health():
        from dialect_classifier.ensemble import LABELS
        return {"ok": MODEL is not None, "model": MODEL_DESC,
                "rule": MODEL.rule, "labels": LABELS, "device": str(MODEL.device), "low_confidence_below": LOWCONF,
                "max_sec_per_file": MAX_SEC, "max_files": MAX_FILES, "auth": bool(API_KEY),
                "capture": bool(os.environ.get("SPEECH_CAPTURE_DIR")), **STATS}

    # 아래 필드(self_report · session · sentence_id)는 판정에 쓰지 않는다 — 캡처 기록(request.json)에 남기려고 받는다.
    @r.post("/predict")
    async def predict(request: Request, audio: UploadFile = File(...), parts: str = Form(None),
                      self_report: str = Form(None), session: str = Form(None), sentence_id: str = Form(None)):
        n = parse_parts(parts, 1)
        return await run(request, [audio], n, n)

    @r.post("/predict_long")
    async def predict_long(request: Request, audio: UploadFile = File(...), parts: str = Form(None),
                           self_report: str = Form(None), session: str = Form(None), sentence_id: str = Form(None)):
        n = parse_parts(parts, 3)
        return await run(request, [audio], n, n)

    @r.post("/predict_multi")
    async def predict_multi(request: Request, audio: List[UploadFile] = File(...), self_report: str = Form(None),
                            session: str = Form(None), sentence_id: str = Form(None)):
        return await run(request, audio, 1, len(audio))

    app.include_router(r)
    app.include_router(r, prefix="/dialect")
    return capture_app(app, "dialect")


def main():
    global MODEL, MODEL_DESC, API_KEY
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default=os.environ.get("DIALECT_BACKEND", "gpu"), choices=["gpu", "npu"])
    ap.add_argument("--model-dir", default=os.environ.get("DIALECT_MODEL_DIR"),
                    help="gpu: comparison_report.json 과 eat_large/ · xlsr2b/best.pt 가 있는 폴더 · "
                         "npu: 번들의 models 폴더(meta.json · *.rbln) (기본 $DIALECT_MODEL_DIR)")
    ap.add_argument("--npu-devices", default=os.environ.get("DIALECT_NPU_DEVICES", "5,5"),
                    help="npu: rbln 장치 번호 eat_large,xlsr2b")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8987)
    ap.add_argument("--no-auth", action="store_true", help="토큰 검사를 끈다 (로컬 시험용)")
    a = ap.parse_args()
    if not a.model_dir:
        raise SystemExit("--model-dir 또는 DIALECT_MODEL_DIR 가 필요하다")
    API_KEY = "" if a.no_auth else os.environ.get("DIALECT_API_KEY", "")
    if not API_KEY and not a.no_auth:
        raise SystemExit("DIALECT_API_KEY 가 없다 — 토큰 없이 띄우려면 --no-auth")
    p = subprocess.run([FFMPEG_PATH, "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=False, env=FFMPEG_ENV)
    if p.returncode != 0:
        raise SystemExit("ffmpeg 을 실행할 수 없다 (%s)" % FFMPEG_PATH)
    if a.backend == "npu":
        from dialect_classifier.npu_ensemble import NpuEnsemble
        MODEL = NpuEnsemble(a.model_dir, backend="rbln", devices=[int(d) for d in a.npu_devices.split(",")])
        MODEL_DESC += " · NPU 번들 %s" % MODEL.meta["created"]
    else:
        from dialect_classifier.ensemble import DialectEnsemble
        MODEL = DialectEnsemble(a.model_dir)
    MODEL.score([np.zeros(3 * SR, np.float32) + 1e-3 * np.random.default_rng(0).standard_normal(3 * SR).astype(np.float32)])
    print("[ready] rule=%s device=%s capture=%s port=%d" % (MODEL.rule, MODEL.device,
                                                           os.environ.get("SPEECH_CAPTURE_DIR") or "off", a.port), flush=True)
    import uvicorn
    uvicorn.run(make_app(), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
