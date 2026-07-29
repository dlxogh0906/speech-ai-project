"""Thin compare proxy: one upload -> fan out to GPU(8000) + CPU(8001) demo backends
server-side, return both results side by side. Loads NO model itself.
Run:  python compare_proxy.py   (serves on :8080)
"""
import os, time, asyncio
from pathlib import Path
import httpx
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse

HERE = Path(__file__).resolve().parent
GPU_URL = os.environ.get("GPU_URL", "http://localhost:8000")
CPU_URL = os.environ.get("CPU_URL", "http://localhost:8001")
PORT = int(os.environ.get("PORT", "8080"))
app = FastAPI(title="CPU vs GPU 비교")


async def call(client, url, raw, filename):
    t0 = time.time()
    try:
        r = await client.post(f"{url}/predict",
                              files={"audio": (filename or "audio", raw, "application/octet-stream")},
                              timeout=120)
        j = r.json()
        if r.status_code != 200:
            return {"ok": False, "error": j.get("detail", r.status_code), "wall_sec": round(time.time()-t0, 2)}
        j["ok"] = True; j["wall_sec"] = round(time.time() - t0, 2)
        return j
    except Exception as e:
        return {"ok": False, "error": str(e), "wall_sec": round(time.time() - t0, 2)}


@app.get("/health")
async def health():
    async with httpx.AsyncClient() as c:
        out = {}
        for name, url in (("gpu", GPU_URL), ("cpu", CPU_URL)):
            try:
                r = await c.get(f"{url}/health", timeout=5); out[name] = r.json()
            except Exception as e:
                out[name] = {"status": "down", "error": str(e)}
        return out


@app.post("/predict_both")
async def predict_both(audio: UploadFile = File(...)):
    raw = await audio.read()
    if len(raw) < 1000:
        raise HTTPException(400, "오디오가 너무 짧거나 비어있어요")
    async with httpx.AsyncClient() as c:
        gpu, cpu = await asyncio.gather(
            call(c, GPU_URL, raw, audio.filename),
            call(c, CPU_URL, raw, audio.filename))
    return {"gpu": gpu, "cpu": cpu, "filename": audio.filename}


@app.get("/", response_class=HTMLResponse)
def index():
    return (HERE / "compare.html").read_text(encoding="utf-8")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
