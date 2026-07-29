"""FastAPI server for the XLS-R nationality classifier — fairseq encoder path (61-class).

Encoder: the LOCAL fairseq XLS-R 2B checkpoint (NOT HuggingFace), so features match
exactly what the linear probe head was trained on. Pipeline:

    wav -> mono/16 kHz -> fairseq XLS-R 2B (frozen) -> mean pool (1920-d)
        -> standardize (mu/sd) -> ClassificationHead -> softmax top-k

Endpoints:
  POST /predict   multipart file field 'audio' (wav/flac/ogg, 16k mono recommended)
                  -> {"top":[{"label":"중국어","prob":0.83}, ...], "device":"cuda", "took_sec":1.1}
  GET  /health    -> {"status":"ok", "device":"cuda", "classes":61}

Config via env vars (all have sane local defaults):
  REPO_ROOT   dir that contains country_classifier/ and fairseq/   (default $HOME)
  XLSR_CKPT   fairseq encoder checkpoint  (default $SPEECH_AI_DATA/models/xlsr2_2B_1000k_inference.pt)
  HEAD_PT     probe head .pt {head,mu,sd,labels}  (default $NATIONALITY_RUNS/probe_fine_811/head.pt)
  MAX_SEC     crop each clip to this many seconds before the encoder (default 12)
  TOP_K       number of predictions to return (default 5)

Run:  uvicorn app:app --host 0.0.0.0 --port 8000
"""
import io
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# --- import path: mirror the proven policy32_inference_compact.py ordering ---
# country_classifier.model inserts fairseq_root at the FRONT of sys.path before
# `import fairseq`, so the editable fairseq package wins over the bare repo dir.
REPO_ROOT = Path(os.environ.get("REPO_ROOT", os.path.expandvars("$HOME")))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch.nn as nn  # noqa: E402
from country_classifier.model import ClassificationHead, build_model  # noqa: E402


def _build_head(payload, num_classes):
    """Reconstruct the trained classifier from either supported head format:
      - probe_fine_811 style: state dict keys '1.weight'/'1.bias' (ClassificationHead.net)
      - duration-sweep style:  state dict keys 'weight'/'bias'     (plain nn.Linear)
    """
    sd = payload["head"]
    keys = set(sd.keys())
    if {"weight", "bias"} <= keys:
        head = nn.Linear(1920, num_classes)
        head.load_state_dict(sd)
        return head
    head = ClassificationHead(input_dim=1920, num_classes=num_classes, hidden_dim=0)
    head.net.load_state_dict(sd)
    return head


def _norm_stats(payload):
    """mu/sd (probe_fine_811) or mean/std (duration sweep)."""
    mean = payload.get("mu", payload.get("mean"))
    std = payload.get("sd", payload.get("std"))
    return mean, std

XLSR_CKPT = Path(os.environ.get("XLSR_CKPT", os.path.expandvars("$SPEECH_AI_DATA/models/xlsr2_2B_1000k_inference.pt")))
HEAD_PT = Path(os.environ.get("HEAD_PT", str(Path(__file__).resolve().parent / "head_fine62_16s.pt")))
FAIRSEQ_ROOT = REPO_ROOT / "fairseq"
MAX_SEC = float(os.environ.get("MAX_SEC", "16"))
TOP_K = int(os.environ.get("TOP_K", "5"))
SR = 16000
DEVICE = (
    torch.device("cuda", torch.cuda.current_device())
    if torch.cuda.is_available()
    else torch.device("cpu")
)

from fastapi import FastAPI, File, HTTPException, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import HTMLResponse  # noqa: E402

BUNDLE = Path(__file__).resolve().parent
app = FastAPI(title="XLS-R nationality classifier (fairseq)")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)
STATE = {}


@app.on_event("startup")
def _load():
    t0 = time.time()
    print(f"[startup] device={DEVICE}; head={HEAD_PT}", flush=True)
    payload = torch.load(HEAD_PT, map_location="cpu")
    labels = list(payload["labels"])

    # Reconstruct the encoder (frozen) and graft the trained head onto its classifier.
    print(f"[startup] loading fairseq XLS-R from {XLSR_CKPT} ...", flush=True)
    model = build_model(
        backend="fairseq_xlsr",
        num_classes=len(labels),
        checkpoint_path=XLSR_CKPT,
        fairseq_root=FAIRSEQ_ROOT,
        freeze_encoder=True,
    )
    # head.pt stores a ClassificationHead state dict (keys '1.weight','1.bias');
    # load it into a fresh head of the right shape rather than the encoder's own.
    model.classifier = _build_head(payload, len(labels))
    model = model.to(DEVICE).eval()

    mean, std = _norm_stats(payload)
    mu = torch.as_tensor(mean, dtype=torch.float32, device=DEVICE)
    sd = torch.as_tensor(std, dtype=torch.float32, device=DEVICE)
    STATE.update(model=model, labels=labels, mu=mu, sd=sd)
    print(f"[startup] ready in {time.time() - t0:.0f}s ({len(labels)} classes)", flush=True)


def _read_audio(raw: bytes) -> np.ndarray:
    """Decode an uploaded clip to float32 mono @ 16 kHz."""
    try:
        import soundfile as sf
        audio, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    except Exception as exc:  # webm/opus/mp3 etc. are not readable by libsndfile
        raise HTTPException(
            400, f"could not decode audio (send WAV/FLAC/OGG mono): {exc}"
        )
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != SR:
        import torchaudio
        audio = (
            torchaudio.functional.resample(torch.from_numpy(audio).unsqueeze(0), sr, SR)
            .squeeze(0)
            .numpy()
        )
    return np.asarray(audio, dtype=np.float32)


def _sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize(DEVICE)


def _gpu_snapshot():
    """Device + process GPU memory, in MB. peak_req = peak this request only."""
    if DEVICE.type != "cuda":
        return {"name": "cpu"}
    free, total = torch.cuda.mem_get_info(DEVICE)
    mb = 1024 * 1024
    return {
        "name": torch.cuda.get_device_name(DEVICE),
        "device_used_mb": round((total - free) / mb),       # whole GPU (incl. other procs)
        "device_total_mb": round(total / mb),
        "proc_allocated_mb": round(torch.cuda.memory_allocated(DEVICE) / mb),  # tensors live now
        "proc_reserved_mb": round(torch.cuda.memory_reserved(DEVICE) / mb),    # cached by torch
        "peak_req_mb": round(torch.cuda.max_memory_allocated(DEVICE) / mb),    # peak in this request
    }


@torch.inference_mode()
def _predict(wav: np.ndarray):
    used_sec = min(len(wav), int(MAX_SEC * SR)) / SR
    wav = wav[: int(MAX_SEC * SR)]
    waveforms = torch.from_numpy(np.ascontiguousarray(wav)).unsqueeze(0).to(DEVICE)
    padding_mask = torch.zeros_like(waveforms, dtype=torch.bool)
    model = STATE["model"]

    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats(DEVICE)

    _sync(); t0 = time.perf_counter()
    if DEVICE.type == "cuda":
        with torch.autocast(device_type="cuda"):
            feats, fmask = model._extract_features(waveforms, padding_mask)
            pooled = model._pool(feats, fmask)
    else:
        feats, fmask = model._extract_features(waveforms, padding_mask)
        pooled = model._pool(feats, fmask)
    _sync(); t1 = time.perf_counter()

    logits = model.classifier((pooled.float() - STATE["mu"]) / STATE["sd"])
    probs = torch.softmax(logits, dim=-1)[0]
    k = min(TOP_K, probs.numel())
    scores, idx = probs.topk(k)
    _sync(); t2 = time.perf_counter()

    labels = STATE["labels"]
    top = [
        {"label": labels[int(i)], "prob": round(float(s), 4)}
        for s, i in zip(scores.cpu(), idx.cpu())
    ]
    timing = {
        "encoder_sec": round(t1 - t0, 3),
        "head_sec": round(t2 - t1, 3),
        "infer_sec": round(t2 - t0, 3),
        "used_sec": round(used_sec, 2),
        "realtime_factor": round((t2 - t0) / max(used_sec, 1e-9), 3),
    }
    return top, timing


@app.get("/", response_class=HTMLResponse)
def index():
    return (BUNDLE / "index.html").read_text(encoding="utf-8")


@app.get("/health")
def health():
    return {
        "status": "ok" if STATE else "loading",
        "device": DEVICE.type,
        "classes": len(STATE.get("labels", [])),
        "gpu": _gpu_snapshot(),
    }


@app.post("/predict")
async def predict(audio: UploadFile = File(...)):
    if not STATE:
        raise HTTPException(503, "model still loading")
    wall0 = time.time()
    wav = _read_audio(await audio.read())
    if wav.size < SR * 0.5:
        raise HTTPException(400, "audio too short (need >= ~0.5s)")
    top, timing = _predict(wav)
    timing["wall_sec"] = round(time.time() - wall0, 3)  # incl. decode + transfer
    return {
        "filename": audio.filename,
        "top": top,
        "device": DEVICE.type,
        "timing": timing,
        "gpu": _gpu_snapshot(),
    }
