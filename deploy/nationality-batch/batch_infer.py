"""Batched (parallel) inference for the XLS-R nationality classifier.

Processes MANY audio inputs in a SINGLE padded forward pass instead of one-at-a-time.
Same feature spec as demo_app.py: XLS-R layer-31 mean-pool (mask-aware) -> clamp[-50,50]
-> (x-mu)/sd -> linear head. English nationality names in the response.

Endpoints:
  POST /predict         one file field 'audio'                 -> single result
  POST /predict_batch   repeated file field 'audio' (N files)  -> list of N results
  GET  /health

Benchmark (no server):
  python batch_infer.py --benchmark --n 32 --sec 8
    -> compares serial (N forwards) vs batched (1 forward) throughput.
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists(): _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.fairseq_guard import ensure_fairseq
ensure_fairseq()
from shared.paths import MODEL_ROOT, NAT_OUTPUTS

import os, sys, time, subprocess, argparse
from pathlib import Path
from typing import List
import numpy as np
import torch
import torch.nn as nn
from country_classifier.model import build_model

HERE = Path(__file__).resolve().parent
HEAD_PT = os.environ.get("HEAD_PT", f"{NAT_OUTPUTS}/0702_country41_sent_best/head.pt")
CKPT = os.environ.get("XLSR_CKPT", f"{MODEL_ROOT}/xlsr2_2B_1000k.pt")
FAIRSEQ_ROOT = os.environ["FAIRSEQ_ROOT"]
LAYER = 31
SR = 16000
MAX_SEC = 12
DEVICE = os.environ.get("DEMO_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")

EN = {"네팔":"Nepal","독일":"Germany","러시아":"Russia","루마니아":"Romania","르완다":"Rwanda",
"말레이시아":"Malaysia","몽골":"Mongolia","미국":"USA","미얀마":"Myanmar","방글라데시":"Bangladesh",
"베트남":"Vietnam","불가리아":"Bulgaria","브라질":"Brazil","스페인":"Spain","싱가포르":"Singapore",
"에티오피아":"Ethiopia","영국":"UK","우즈베키스탄":"Uzbekistan","우크라이나":"Ukraine","이란":"Iran",
"이탈리아":"Italy","인도":"India","인도네시아":"Indonesia","일본":"Japan","잠비아":"Zambia","중국":"China",
"칠레":"Chile","카자흐스탄":"Kazakhstan","캄보디아":"Cambodia","콜롬비아":"Colombia","키르기스스탄":"Kyrgyzstan",
"타지키스탄":"Tajikistan","태국":"Thailand","투르크메니스탄":"Turkmenistan","튀르키예":"Turkey","파키스탄":"Pakistan",
"페루":"Peru","프랑스":"France","필리핀":"Philippines","한국어":"Korea (native)","홍콩":"Hong Kong"}

S = {}


def load():
    t0 = time.time()
    model = build_model(backend="fairseq_xlsr", num_classes=61, checkpoint_path=CKPT,
                        fairseq_root=FAIRSEQ_ROOT, freeze_encoder=True).eval().to(DEVICE)
    ck = torch.load(HEAD_PT, map_location="cpu", weights_only=False)
    head = nn.Linear(1920, len(ck["labels"])); head.load_state_dict(ck["head"]); head.eval().to(DEVICE)
    S.update(model=model, head=head, labels=ck["labels"],
             mu=torch.as_tensor(ck["mu"]).float().to(DEVICE),
             sd=torch.as_tensor(ck["sd"]).float().to(DEVICE))
    print(f"[load] {DEVICE} ready in {time.time()-t0:.0f}s ({len(ck['labels'])} classes)", flush=True)


def decode_audio(raw: bytes) -> np.ndarray:
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    p = subprocess.run([ff, "-nostdin", "-i", "pipe:0", "-f", "f32le", "-ac", "1",
                        "-ar", str(SR), "pipe:1"], input=raw, capture_output=True)
    if p.returncode != 0 or not p.stdout:
        raise ValueError("could not decode audio")
    return np.frombuffer(p.stdout, dtype=np.float32).copy()


@torch.no_grad()
def predict_batch(waves):
    """waves: list of 1-D float32 np arrays (16k). Returns list of {top:[...]} dicts."""
    lens = [min(len(w), MAX_SEC * SR) for w in waves]
    Tmax = max(lens)
    B = len(waves)
    x = torch.zeros(B, Tmax, dtype=torch.float32)
    pm = torch.ones(B, Tmax, dtype=torch.bool)          # True = pad (fairseq convention)
    for i, w in enumerate(waves):
        n = lens[i]
        x[i, :n] = torch.from_numpy(np.ascontiguousarray(w[:n]))
        pm[i, :n] = False
    x = x.to(DEVICE); pm = pm.to(DEVICE)

    with torch.autocast("cuda", enabled=(DEVICE == "cuda")):
        res = S["model"].encoder.extract_features(source=x, padding_mask=pm, mask=False)
        feats = res["layer_results"][LAYER][0].transpose(0, 1).float()   # (B, T', C)
        fmask = res.get("padding_mask")                                  # (B, T') True=pad or None
        if fmask is not None:
            valid = (~fmask).unsqueeze(-1).to(feats.dtype)
            pooled = (feats * valid).sum(1) / valid.sum(1).clamp_min(1.0)
        else:
            pooled = feats.mean(1)
        pooled = pooled.clamp(-50, 50)
        logits = S["head"]((pooled - S["mu"]) / S["sd"])
        probs = torch.softmax(logits, -1)

    labels = S["labels"]; out = []
    tk = min(5, probs.size(1))
    top = probs.topk(tk, dim=-1)
    for b in range(B):
        items = []
        for i, pv in zip(top.indices[b].tolist(), top.values[b].tolist()):
            lab = labels[i]
            items.append({"label": lab, "nationality": EN.get(lab, lab), "prob": round(float(pv), 4)})
        out.append({"top": items})
    return out


# ---------------- server ----------------
def make_app():
    from fastapi import FastAPI, File, UploadFile, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    app = FastAPI(title="XLS-R nationality (batched)")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    @app.on_event("startup")
    def _s(): load()

    @app.get("/health")
    def health():
        return {"status": "ok" if S else "loading", "device": DEVICE, "batched": True}

    @app.post("/predict")
    async def predict(audio: UploadFile = File(...)):
        if not S: raise HTTPException(503, "loading")
        wav = decode_audio(await audio.read())
        t0 = time.time()
        r = predict_batch([wav])[0]
        r["took_sec"] = round(time.time() - t0, 3); r["device"] = DEVICE
        return r

    @app.post("/predict_batch")
    async def predict_batch_ep(audio: List[UploadFile] = File(...)):
        if not S: raise HTTPException(503, "loading")
        waves = [decode_audio(await f.read()) for f in audio]
        t0 = time.time()
        res = predict_batch(waves)
        dt = time.time() - t0
        return {"n": len(waves), "batch_sec": round(dt, 3),
                "per_item_sec": round(dt / max(1, len(waves)), 4), "device": DEVICE, "results": res}

    return app


app = None if __name__ == "__main__" else make_app()


def benchmark(n, sec):
    load()
    rng = np.random.default_rng(0)
    waves = [rng.standard_normal(int((sec * (0.6 + 0.4 * rng.random())) * SR)).astype(np.float32) * 0.05
             for _ in range(n)]  # varied lengths to exercise padding
    _ = predict_batch(waves[:2])  # warmup
    if DEVICE == "cuda": torch.cuda.synchronize()

    # serial: one forward per item
    t0 = time.time()
    for w in waves:
        _ = predict_batch([w])
    if DEVICE == "cuda": torch.cuda.synchronize()
    serial = time.time() - t0

    # batched: single forward for all N
    t0 = time.time()
    _ = predict_batch(waves)
    if DEVICE == "cuda": torch.cuda.synchronize()
    batched = time.time() - t0

    print(f"\n=== {DEVICE} | N={n} clips (~{sec}s each) ===")
    print(f"serial  (1-by-1):  {serial:6.2f}s total | {n/serial:6.1f} clips/s | {serial/n*1000:6.0f} ms/clip")
    print(f"batched (1 fwd):   {batched:6.2f}s total | {n/batched:6.1f} clips/s | {batched/n*1000:6.0f} ms/clip")
    print(f"-> batched is {serial/batched:.1f}x higher throughput")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--sec", type=float, default=8)
    ap.add_argument("--port", type=int, default=8002)
    a = ap.parse_args()
    if a.benchmark:
        benchmark(a.n, a.sec)
    else:
        import uvicorn
        app = make_app()
        uvicorn.run(app, host="0.0.0.0", port=a.port)
