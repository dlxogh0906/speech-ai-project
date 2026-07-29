"""Dynamic micro-batching inference server (fixes serial-under-load).

Each request: decode audio in a threadpool (non-blocking), enqueue (wave, future),
await result. A single background batcher drains the queue, groups requests arriving
within MAX_WAIT_MS (up to MAX_BATCH) into ONE batched GPU forward, and fans results back.

Why this and not "more uvicorn workers": one GPU serializes forwards anyway; batching turns
N concurrent requests into 1 forward -> real throughput. Handlers never block the event loop.

Run:  DEMO_DEVICE=cuda python serve_batched.py --port 8003
Env:  MAX_BATCH (default 16), MAX_WAIT_MS (default 10)
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import os, asyncio, time
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import batch_infer as B  # reuse load(), decode_audio(), predict_batch(), DEVICE

MAX_BATCH = int(os.environ.get("MAX_BATCH", "16"))
MAX_WAIT_S = float(os.environ.get("MAX_WAIT_MS", "10")) / 1000.0

app = FastAPI(title="XLS-R nationality (dynamic batching)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
Q: asyncio.Queue = None
STATS = {"requests": 0, "batches": 0, "max_batch_seen": 0, "sum_batch": 0}


async def batcher():
    loop = asyncio.get_event_loop()
    while True:
        wave, fut = await Q.get()
        batch = [(wave, fut)]
        deadline = loop.time() + MAX_WAIT_S
        while len(batch) < MAX_BATCH:
            timeout = deadline - loop.time()
            if timeout <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(Q.get(), timeout))
            except asyncio.TimeoutError:
                break
        waves = [b[0] for b in batch]; futs = [b[1] for b in batch]
        try:
            results = await loop.run_in_executor(None, B.predict_batch, waves)  # GPU fwd off-loop
            for f, r in zip(futs, results):
                if not f.done():
                    f.set_result(r)
        except Exception as e:
            for f in futs:
                if not f.done():
                    f.set_exception(e)
        STATS["batches"] += 1; STATS["sum_batch"] += len(batch)
        STATS["max_batch_seen"] = max(STATS["max_batch_seen"], len(batch))


@app.on_event("startup")
async def _startup():
    global Q
    await asyncio.get_event_loop().run_in_executor(None, B.load)  # load model without blocking loop
    Q = asyncio.Queue()
    asyncio.create_task(batcher())
    print(f"[serve] batching ready · MAX_BATCH={MAX_BATCH} MAX_WAIT_MS={MAX_WAIT_S*1000:.0f}", flush=True)


async def _infer_one(raw: bytes):
    loop = asyncio.get_event_loop()
    wave = await loop.run_in_executor(None, B.decode_audio, raw)   # ffmpeg off-loop
    fut = loop.create_future()
    await Q.put((wave, fut))
    STATS["requests"] += 1
    return await fut


@app.get("/health")
def health():
    return {"status": "ok" if Q is not None else "loading", "device": B.DEVICE,
            "mode": "dynamic-batching", "max_batch": MAX_BATCH}


@app.get("/stats")
def stats():
    b = STATS["batches"]
    return {**STATS, "avg_batch": round(STATS["sum_batch"] / b, 2) if b else 0}


@app.post("/predict")
async def predict(audio: UploadFile = File(...)):
    if Q is None:
        raise HTTPException(503, "loading")
    t0 = time.time()
    r = await _infer_one(await audio.read())
    r = dict(r); r["took_sec"] = round(time.time() - t0, 3); r["device"] = B.DEVICE
    return r


if __name__ == "__main__":
    import argparse, uvicorn
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8003)
    a = ap.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=a.port)
