"""FastAPI web server for the XLS-R nationality classifier (policy31, 31-class).

POST /predict  (multipart file 'audio', WAV/FLAC/OGG 16k mono recommended)
  -> {"top": [{"label": "...", "prob": 0.87}, ...], "device": "cuda", "took_sec": 1.2}
GET  /health   -> {"status": "ok", ...}

Reuses the bundle's run_inference helpers (load_heads, logits_for_audio) so the model /
preprocessing exactly matches the offline script. XLS-R is loaded once at startup.
"""
import io
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from fastapi import FastAPI, File, HTTPException, UploadFile
from transformers import AutoFeatureExtractor, AutoModel

from run_inference import DEFAULT_MODEL_NAME, load_heads, logits_for_audio

BUNDLE = Path(__file__).resolve().parent
CKPT_DIR = BUNDLE / "checkpoints"
SR = 16000
TOP_K = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

app = FastAPI(title="XLS-R nationality classifier")
STATE = {}


@app.on_event("startup")
def _load():
    print(f"[startup] device={DEVICE}; loading {DEFAULT_MODEL_NAME} ...", flush=True)
    t0 = time.time()
    extractor = AutoFeatureExtractor.from_pretrained(DEFAULT_MODEL_NAME)
    encoder = AutoModel.from_pretrained(DEFAULT_MODEL_NAME)
    encoder.eval().to(DEVICE)
    if DTYPE == torch.float16:
        encoder.half()
    heads, labels = load_heads(CKPT_DIR, DEVICE)
    STATE.update(extractor=extractor, encoder=encoder, heads=heads, labels=labels)
    print(f"[startup] ready in {time.time()-t0:.0f}s ({len(labels)} classes)", flush=True)


def _read_audio(raw: bytes) -> np.ndarray:
    try:
        audio, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    except Exception as exc:
        raise HTTPException(400, f"could not read audio (send WAV/FLAC/OGG): {exc}")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != SR:
        audio = torchaudio.functional.resample(
            torch.from_numpy(audio).unsqueeze(0), sr, SR).squeeze(0).numpy()
    return np.asarray(audio, dtype=np.float32)


@app.get("/health")
def health():
    return {"status": "ok" if STATE else "loading", "device": DEVICE,
            "classes": len(STATE.get("labels", []))}


@app.post("/predict")
async def predict(audio: UploadFile = File(...)):
    if not STATE:
        raise HTTPException(503, "model still loading")
    wav = _read_audio(await audio.read())
    if wav.size < SR * 0.5:
        raise HTTPException(400, "audio too short (need >= ~0.5s)")
    t0 = time.time()
    logits, _ = logits_for_audio(wav, SR, STATE["extractor"], STATE["encoder"],
                                 STATE["heads"], STATE["labels"], DEVICE, DTYPE, None)
    probs = torch.softmax(logits, dim=-1)
    top = probs.topk(min(TOP_K, probs.size(1)), dim=-1)
    out = [{"label": STATE["labels"][i], "prob": round(float(v), 4)}
           for i, v in zip(top.indices[0].tolist(), top.values[0].tolist())]
    return {"top": out, "device": DEVICE, "took_sec": round(time.time() - t0, 2)}
