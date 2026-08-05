# -*- coding: utf-8 -*-
"""국적 분류 KD 인코더 데모 — ONNX CPU 전용 서빙 + 동시접속 부하 테스트.

모델: 인코더 증류 최종본 (test top1 0.8245 / top5 0.9584 / macro5 0.7124)
      기존 배포본(0705, top1 0.7510) 대비 top1 +7.4%p — 같은 크기·같은 지연.

구조: onnxruntime 세션 풀(POOL개, 세션당 스레드 분리 — 동접 경합 방지) + numpy 헤드.
      torch/fairseq 불필요. GPU 불필요.

    $PY deploy/nationality-kd-demo/app.py --port 8905
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r)]
from shared.paths import FFMPEG, NAT_TASK

import argparse, glob, os, queue, subprocess, time
import numpy as np

SR = 16000
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get("KD_MODEL_DIR",
                           str(NAT_TASK / "outputs/kd_encoder/run2/deploy"))
POOL = int(os.environ.get("POOL", "4"))
THREADS = int(os.environ.get("THREADS", str(max(1, (os.cpu_count() or 8) // POOL))))

# ── 모델 로드 ──
import onnxruntime as ort
_h = np.load(os.path.join(MODEL_DIR, "head_np.npz"), allow_pickle=True)
W, B, MU, SD = _h["W"], _h["b"], _h["mu"], _h["sd"]
LABELS = [str(x) for x in _h["labels"]]
_sessions = queue.Queue()
print(f"[load] ONNX 세션 {POOL}개 × {THREADS}스레드 로딩...", flush=True)
_t0 = time.time()
for _ in range(POOL):
    so = ort.SessionOptions(); so.intra_op_num_threads = THREADS
    _sessions.put(ort.InferenceSession(os.path.join(MODEL_DIR, "encoder_kd_l17.onnx"),
                                       so, providers=["CPUExecutionProvider"]))
print(f"[load] 완료 {time.time()-_t0:.0f}s · {len(LABELS)}클래스", flush=True)

_stats = {"requests": 0, "errors": 0, "busy": 0}


def decode(raw_bytes):
    p = subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
                        "-f", "f32le", "-ac", "1", "-ar", str(SR), "-t", "12", "pipe:1"],
                       input=raw_bytes, stdout=subprocess.PIPE)
    return np.frombuffer(p.stdout, np.float32).copy()


def classify(wav):
    sess = _sessions.get()          # 풀에서 세션 하나 (동접 = POOL 까지 병렬)
    _stats["busy"] += 1
    try:
        t = time.perf_counter()
        pooled = sess.run(["pooled"], {"source": wav[None, :]})[0]
        ms = (time.perf_counter() - t) * 1000
    finally:
        _stats["busy"] -= 1
        _sessions.put(sess)
    x = np.clip(np.nan_to_num(pooled), -50, 50)
    z = ((x - MU) / SD) @ W.T + B
    p = np.exp(z - z.max()); p /= p.sum()
    top = np.argsort(-p[0])[:5]
    return [{"label": LABELS[i], "prob": round(float(p[0][i]), 4)} for i in top], ms


# ── 샘플 오디오 (부하 테스트·원클릭 데모용) ──
SAMPLES = sorted(glob.glob("/home/tta/test_samples/*.wav"))[:10]


def make_app():
    from fastapi import FastAPI, UploadFile, File, APIRouter
    from fastapi.responses import HTMLResponse, Response, JSONResponse
    from starlette.concurrency import run_in_threadpool
    app = FastAPI(title="nationality-kd-demo")
    r = APIRouter()

    @r.get("/", response_class=HTMLResponse)
    def index():
        return open(os.path.join(HERE, "index.html"), encoding="utf-8").read()

    @r.get("/health")
    def health():
        return {"ok": True, "model": "encoder_kd_run2 (top1 0.8245)", "pool": POOL,
                "threads": THREADS, "classes": len(LABELS), **_stats}

    @r.post("/predict")
    async def predict(audio: UploadFile = File(...)):
        _stats["requests"] += 1
        t0 = time.perf_counter()
        raw = await audio.read()
        wav = await run_in_threadpool(decode, raw)
        if wav.size < SR // 2:
            _stats["errors"] += 1
            return JSONResponse({"error": "오디오가 너무 짧거나 해석 불가"}, status_code=400)
        top5, enc_ms = await run_in_threadpool(classify, wav)
        return {"top5": top5, "sec": round(wav.size / SR, 2),
                "encode_ms": round(enc_ms), "total_ms": round((time.perf_counter() - t0) * 1000)}

    @r.get("/samples")
    def samples():
        return {"samples": [os.path.basename(s) for s in SAMPLES]}

    @r.get("/sample/{idx}")
    def sample(idx: int):
        return Response(open(SAMPLES[idx % len(SAMPLES)], "rb").read(), media_type="audio/wav")

    app.include_router(r)
    app.include_router(r, prefix="/kd")      # funnel --set-path 는 프리픽스를 벗기지 않는다
    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8905)
    ap.add_argument("--host", default="0.0.0.0")
    a = ap.parse_args()
    import uvicorn
    uvicorn.run(make_app(), host=a.host, port=a.port)
