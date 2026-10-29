# -*- coding: utf-8 -*-
"""국적 분류 — 만든 모델 전부를 한 오디오로 나란히 비교하는 페이지 (2026-09-24).

  ONNX(300m KD 계열, CPU): 기존 배포 · run5 · koraug · chanaug · zeroth
  2B(torch GPU fp16): 동결+선형 헤드(보고용) · 1 epoch(22~31층) · 2 epoch(전 층)
  무음 정규화(데모 8922 와 같은 `silence_normalize`)를 켜고 끌 수 있다 — 모든 모델에 똑같이 적용.

  GET  /            페이지          GET /health   모델 목록·로딩 상태
  POST /predict?norm=1   audio → 모델별 top-5 · Korea 확률 · 처리 시간

    MULTI_PORT=8924 $PY deploy/nationality-multi-demo/app.py --port 8924        (tmux)
환경: MULTI_MODELS(JSON 목록, 없으면 아래 기본), MULTI_SAVE_DIR(업로드 저장, 선택), SKIP_2B=1(2B 생략)
"""
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src"))), str(_r / "tasks/nationality-classification/experiments")]
from shared.paths import NAT_OUTPUTS

import argparse, json, os, subprocess, tempfile, threading, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np

SR = 16000; MAX_SEC = 12
HERE = os.path.dirname(os.path.abspath(__file__))
W = f"{NAT_OUTPUTS}/webmic_kd"
DEFAULT_MODELS = [
    {"name": "기존 배포 (300m KD · 웹마이크 미학습)", "type": "onnx", "dir": "/data/speech-ai/weights/final/nationality"},
    {"name": "run5 (웹마이크 재학습)", "type": "onnx", "dir": f"{W}/run5/deploy"},
    {"name": "koraug (한국어 채널 증강)", "type": "onnx", "dir": f"{W}/run_koraug/deploy"},
    {"name": "chanaug (전 클래스 채널 증강)", "type": "onnx", "dir": f"{W}/run_chanaug/deploy"},
    {"name": "zeroth ★최종 (Zeroth 원어민 낭독 추가)", "type": "onnx", "dir": f"{W}/run_zeroth/deploy"},
    {"name": "2B 동결 + 선형 헤드 (보고용)", "type": "2b", "ckpt": ""},
    {"name": "2B 1 epoch (22~31층 학습)", "type": "2b", "ckpt": f"{W}/run_2b/ep0.pt"},
    {"name": "2B 2 epoch (전 층 학습)", "type": "2b", "ckpt": f"{W}/run_2b_ep2/ep0.pt"},
]
SAVE_DIR = os.environ.get("MULTI_SAVE_DIR", "")
MODELS = []                     # dict(name, type, run(wav)->top5, device)
STATE = {"loading": True, "error": ""}
_gpu_lock = threading.Lock()
_ffmpeg = None


def ffmpeg():
    """FFMPEG_BIN → PATH 의 ffmpeg → imageio 순. imageio 의 4.2.2 는 iPhone Safari 의 MP4(fragmented)를 첫 조각(≈1초)만 읽는다 — 실측 2026-09-24."""
    global _ffmpeg
    if _ffmpeg is None:
        import shutil
        _ffmpeg = os.environ.get("FFMPEG_BIN") or shutil.which("ffmpeg")
        if not _ffmpeg:
            import imageio_ffmpeg
            _ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            print("[warn] 시스템 ffmpeg 이 없어 imageio 4.2.2 를 쓴다 — iPhone MP4 녹음이 잘릴 수 있다", flush=True)
    return _ffmpeg


def decode(raw):
    with tempfile.NamedTemporaryFile(suffix=".audio") as tf:
        tf.write(raw); tf.flush()
        p = subprocess.run([ffmpeg(), "-hide_banner", "-loglevel", "error", "-i", tf.name, "-f", "f32le", "-ac", "1", "-ar", str(SR), "-t", str(MAX_SEC), "pipe:1"],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise ValueError("오디오를 해석할 수 없다")
    return np.frombuffer(p.stdout, np.float32).copy()


def top5_from(z, labels):
    z = np.asarray(z, np.float64).reshape(-1); p = np.exp(z - z.max()); p /= p.sum()
    o = np.argsort(-p)[:5]
    return [{"label": labels[i], "prob": round(float(p[i]), 4)} for i in o], round(float(p[labels.index("Korea")]), 4)


def load_onnx(m):
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = int(os.environ.get("ONNX_THREADS", "3"))
    sess = ort.InferenceSession(os.path.join(m["dir"], "encoder_kd_l17.onnx"), so, providers=["CPUExecutionProvider"])
    h = np.load(os.path.join(m["dir"], "head_np.npz"), allow_pickle=True)
    Wh, b, mu, sd = h["W"], h["b"], h["mu"], h["sd"]; labels = [str(x) for x in h["labels"]]
    def run(wav):
        x = np.clip(np.nan_to_num(sess.run(["pooled"], {"source": wav[None, :]})[0]), -50, 50)
        return top5_from(((x - mu) / sd) @ Wh.T + b, labels)
    return {"name": m["name"], "type": "onnx", "run": run, "device": "CPU"}


def load_2b_models(ms):
    import torch
    from shared.fairseq_guard import ensure_fairseq
    ensure_fairseq()
    from xlsr2b_finetune import build, LAYER, XLSR2B
    from country_classifier.kd_student import forward_pooled
    labels = [str(x) for x in np.load(f"{W}/run5/deploy/head_np.npz", allow_pickle=True)["labels"]]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = []
    for m in ms:
        t0 = time.time()
        enc, head, mu, sd, lab = build(XLSR2B, m["ckpt"], labels, dev, 0)
        enc.eval(); head.eval()
        for p in enc.parameters(): p.requires_grad = False
        def run(wav, enc=enc, head=head, mu=mu, sd=sd, lab=lab):
            with _gpu_lock, torch.no_grad():
                x = torch.from_numpy(wav)[None].to(dev)
                with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                    pooled = forward_pooled(enc, x, torch.zeros_like(x, dtype=torch.bool), LAYER)
                z = head((pooled - mu) / sd).float().cpu().numpy()
            return top5_from(z, lab)
        out.append({"name": m["name"], "type": "2b", "run": run, "device": dev.type.upper()})
        print(f"[load] {m['name']} {time.time()-t0:.0f}s", flush=True)
    return out


def load_all(cfg, skip_2b):
    try:
        for m in cfg:
            if m["type"] == "onnx":
                t0 = time.time(); MODELS.append(load_onnx(m)); print(f"[load] {m['name']} {time.time()-t0:.0f}s", flush=True)
        if not skip_2b:
            MODELS.extend(load_2b_models([m for m in cfg if m["type"] == "2b"]))
        STATE["loading"] = False
        print(f"[load] 완료 · 모델 {len(MODELS)}개", flush=True)
    except Exception as e:
        STATE["loading"] = False; STATE["error"] = repr(e); print("[load] 실패", repr(e), flush=True)


def run_all(wav):
    res = [None] * len(MODELS)
    def one(i):
        t = time.perf_counter(); top5, pk = MODELS[i]["run"](wav)
        res[i] = {"name": MODELS[i]["name"], "type": MODELS[i]["type"], "device": MODELS[i]["device"], "top5": top5, "korea": pk, "ms": round((time.perf_counter() - t) * 1000)}
    onnx_idx = [i for i, m in enumerate(MODELS) if m["type"] == "onnx"]; other = [i for i, m in enumerate(MODELS) if m["type"] != "onnx"]
    with ThreadPoolExecutor(max_workers=max(1, len(onnx_idx))) as ex:
        list(ex.map(one, onnx_idx))
    for i in other: one(i)
    return res


def make_app():
    from fastapi import FastAPI, File, UploadFile
    from fastapi.responses import HTMLResponse, JSONResponse
    from starlette.concurrency import run_in_threadpool
    from country_classifier.wav_augment import silence_normalize
    app = FastAPI(title="국적 모델 비교")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return open(os.path.join(HERE, "index.html"), encoding="utf-8").read()

    @app.get("/health")
    def health():
        return {"ok": not STATE["loading"] and not STATE["error"], "loading": STATE["loading"], "error": STATE["error"],
                "models": [{"name": m["name"], "type": m["type"], "device": m["device"]} for m in MODELS]}

    @app.post("/predict")
    async def predict(audio: UploadFile = File(...), norm: int = 1):
        if STATE["loading"]:
            return JSONResponse({"error": "모델 로딩 중 — 잠시 뒤 다시"}, status_code=503)
        raw = await audio.read()
        try:
            wav = await run_in_threadpool(decode, raw)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        if wav.size < SR // 2:
            return JSONResponse({"error": "오디오가 너무 짧다 (0.5초 미만)"}, status_code=400)
        if SAVE_DIR:
            os.makedirs(SAVE_DIR, exist_ok=True)
            open(os.path.join(SAVE_DIR, time.strftime("%Y%m%d_%H%M%S") + ".raw"), "wb").write(raw)
        sec_in = round(wav.size / SR, 2)
        if norm:
            wav = silence_normalize(wav)
        t0 = time.perf_counter()
        results = await run_in_threadpool(run_all, np.ascontiguousarray(wav, dtype=np.float32))
        return {"sec": sec_in, "norm": bool(norm), "results": results, "total_ms": round((time.perf_counter() - t0) * 1000)}
    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=int(os.environ.get("MULTI_PORT", "8924"))); a = ap.parse_args()
    cfg = json.loads(os.environ["MULTI_MODELS"]) if os.environ.get("MULTI_MODELS") else DEFAULT_MODELS
    threading.Thread(target=load_all, args=(cfg, os.environ.get("SKIP_2B") == "1"), daemon=True).start()
    import uvicorn
    uvicorn.run(make_app(), host="0.0.0.0", port=a.port)
