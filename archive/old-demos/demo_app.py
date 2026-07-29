"""Web demo: record Korean speech -> predict speaker nationality (top-5).

Backend = our best model: frozen XLS-R 2B (fairseq) + layer-31 mean-pool + linear head.
Audio from the browser (webm/opus) is decoded with a bundled ffmpeg, so any format works.

Run:  python demo_app.py   (then open http://<host>:8000/)
"""
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import imageio_ffmpeg
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

sys.path.insert(0, os.path.expandvars("$HOME"))
from country_classifier.model import build_model

HERE = Path(__file__).resolve().parent
HEAD_PT = os.path.expandvars("$NATIONALITY_OUTPUTS/0702_country41_sent_best/head.pt")   # 41개 국적(한국어+유럽문장병합) layer-31
CKPT = os.path.expandvars("$SPEECH_AI_MODELS/xlsr2_2B_1000k.pt")
FAIRSEQ_ROOT = os.path.expandvars("$FAIRSEQ_ROOT")
LAYER = 31
SR = 16000
# DEMO_DEVICE=cpu 로 강제 가능 (GPU가 다른 작업으로 꽉 찼을 때)
DEVICE = os.environ.get("DEMO_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

# 국적 라벨 -> (표시명, 국기)
FLAG = {
    "네팔": "🇳🇵", "독일": "🇩🇪", "러시아": "🇷🇺", "루마니아": "🇷🇴", "르완다": "🇷🇼",
    "말레이시아": "🇲🇾", "몽골": "🇲🇳", "미국": "🇺🇸", "미얀마": "🇲🇲", "방글라데시": "🇧🇩",
    "베트남": "🇻🇳", "불가리아": "🇧🇬", "브라질": "🇧🇷", "스페인": "🇪🇸", "싱가포르": "🇸🇬",
    "에티오피아": "🇪🇹", "영국": "🇬🇧", "우즈베키스탄": "🇺🇿", "우크라이나": "🇺🇦", "이란": "🇮🇷",
    "이탈리아": "🇮🇹", "인도": "🇮🇳", "인도네시아": "🇮🇩", "일본": "🇯🇵", "잠비아": "🇿🇲",
    "중국": "🇨🇳", "칠레": "🇨🇱", "카자흐스탄": "🇰🇿", "캄보디아": "🇰🇭", "콜롬비아": "🇨🇴",
    "키르기스스탄": "🇰🇬", "타지키스탄": "🇹🇯", "태국": "🇹🇭", "투르크메니스탄": "🇹🇲", "튀르키예": "🇹🇷",
    "파키스탄": "🇵🇰", "페루": "🇵🇪", "프랑스": "🇫🇷", "필리핀": "🇵🇭", "한국어": "🇰🇷", "홍콩": "🇭🇰",
}
DISPLAY = {"한국어": "한국 (내국인)"}   # 보기 좋은 표시명 보정

app = FastAPI(title="국적 분류 데모")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
S = {}


@app.on_event("startup")
def _load():
    print(f"[startup] device={DEVICE}; loading XLS-R 2B ...", flush=True)
    t0 = time.time()
    model = build_model(backend="fairseq_xlsr", num_classes=61, checkpoint_path=CKPT,
                        fairseq_root=FAIRSEQ_ROOT, freeze_encoder=True).to(DEVICE).eval()
    ck = torch.load(HEAD_PT, map_location="cpu", weights_only=False)
    head = nn.Linear(1920, len(ck["labels"]))
    head.load_state_dict(ck["head"]); head.eval().to(DEVICE)
    S.update(model=model, head=head, labels=ck["labels"],
             mu=torch.as_tensor(ck["mu"]).float().to(DEVICE),
             sd=torch.as_tensor(ck["sd"]).float().to(DEVICE))
    print(f"[startup] ready in {time.time()-t0:.0f}s", flush=True)


def decode_audio(raw: bytes) -> np.ndarray:
    p = subprocess.run([FFMPEG, "-nostdin", "-i", "pipe:0", "-f", "f32le", "-ac", "1",
                        "-ar", str(SR), "pipe:1"], input=raw, capture_output=True)
    if p.returncode != 0 or not p.stdout:
        raise HTTPException(400, "오디오를 읽을 수 없어요 (지원: wav/webm/m4a/mp3 등)")
    return np.frombuffer(p.stdout, dtype=np.float32).copy()


@app.get("/health")
def health():
    return {"status": "ok" if S else "loading", "device": DEVICE}


@app.post("/predict")
async def predict(audio: UploadFile = File(...)):
    if not S:
        raise HTTPException(503, "모델 로딩 중")
    wav = decode_audio(await audio.read())
    dur = len(wav) / SR
    if dur < 0.5:
        raise HTTPException(400, "녹음이 너무 짧아요 (최소 0.5초)")
    wav = wav[: SR * 12]                       # 학습과 동일하게 12초로 자름
    x = torch.from_numpy(wav).unsqueeze(0).to(DEVICE)
    pm = torch.zeros_like(x, dtype=torch.bool)
    t0 = time.time()
    with torch.no_grad(), torch.autocast("cuda", enabled=(DEVICE == "cuda")):
        res = S["model"].encoder.extract_features(source=x, padding_mask=pm, mask=False)
        feat = res["layer_results"][LAYER][0].transpose(0, 1).mean(1).float()  # (1,1920)
        feat = feat.clamp(-50, 50)             # ★ 캐시 추출과 동일한 클램프 (스케일 일치)
        logits = S["head"]((feat - S["mu"]) / S["sd"])
        probs = torch.softmax(logits, -1)[0]
    top = probs.topk(5)
    out = []
    for i, pv in zip(top.indices.tolist(), top.values.tolist()):
        lab = S["labels"][i]
        out.append({"label": lab, "nationality": DISPLAY.get(lab, lab),
                    "flag": FLAG.get(lab, "🏳️"), "prob": round(float(pv), 4)})
    return {"top": out, "duration_sec": round(dur, 1), "took_sec": round(time.time() - t0, 2),
            "device": DEVICE}


@app.get("/", response_class=HTMLResponse)
def index():
    return (HERE / "index.html").read_text(encoding="utf-8")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
