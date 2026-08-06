# -*- coding: utf-8 -*-
"""국적 분류 3모델 동시 비교 데모 — 같은 오디오를 KD · 300m · 2B 에 한 번에 태운다.

  KD    encoder_kd_l17.onnx  (2B→300m 지식증류)  top1 0.8245 · top5 0.9584
  300m  encoder_l17.onnx     (현재 배포본)        top1 0.7510 · top5 0.9331
  2B    xlsr2_2B_1000k       (정확도 참고용)      top1 0.7849 · top5 0.9422

세 수치는 **같은 test 셋(135,770 · 043 포함)** 기준이다. 문서에 자주 인용되는 2B
top5 0.950 은 043 이 빠진 다른 셋(133,509)이라 여기 쓰면 안 된다.

라벨 순서가 모델마다 다르다 — KD·300m 은 영문 알파벳순, 2B(0702)는 한글순이다.
각 모델의 자기 라벨로 인덱스를 풀고 영문 이름으로 통일해 표시한다. 이걸 안 맞추면
**에러 없이 조용히 엉뚱한 나라가 나온다.**

2B 는 GPU 가 있으면 fp16 GPU, 없으면 fp32 CPU(6초 발화에 1.8초). ONNX 두 개는
onnxruntime 이 CPU 빌드라 CPU 전용이다.

    $PY deploy/nationality-compare/app.py --port 8906
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.fairseq_guard import ensure_fairseq
ensure_fairseq()
from shared.paths import FFMPEG, DATA_ROOT, NAT_OUTPUTS, FAIRSEQ_ROOT

import argparse, glob, os, subprocess, threading, time
import numpy as np

SR = 16000
MAX_SEC = 12
HERE = os.path.dirname(os.path.abspath(__file__))
THREADS = int(os.environ.get("THREADS", "8"))
LOAD_2B = os.environ.get("SKIP_2B") != "1"

KD_DIR = f"{DATA_ROOT}/kd_encoder/run2/deploy"
D300 = f"{NAT_OUTPUTS}/0705_country41_sent043_300m"
D2B = f"{NAT_OUTPUTS}/0702_country41_sent_best"
CKPT_2B = f"{DATA_ROOT}/models/xlsr2_2B_1000k_inference.pt"

MODELS = {}                      # name -> dict(run=callable, meta=...)
_lock = threading.Lock()         # 인코더 호출 직렬화 (세 모델이 CPU 를 나눠 쓰면 값이 흔들린다)
_stats = {"requests": 0, "errors": 0}


def _softmax_top5(pooled, W, b, mu, sd, labels):
    x = np.clip(np.nan_to_num(pooled), -50, 50)
    z = ((x - mu) / sd) @ W.T + b
    p = np.exp(z - z.max()); p /= p.sum()
    return [{"label": labels[i], "prob": round(float(p[0][i]), 4)}
            for i in np.argsort(-p[0])[:5]]


def _load_onnx(name, path, head_npz=None, head_pt=None, note=""):
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = THREADS
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    if head_npz:
        h = np.load(head_npz, allow_pickle=True)
        W, b, mu, sd = h["W"], h["b"], h["mu"], h["sd"]
        labels = [str(x) for x in h["labels"]]
    else:
        import torch
        d = torch.load(head_pt, map_location="cpu", weights_only=False)
        W = d["head"]["weight"].numpy(); b = d["head"]["bias"].numpy()
        mu = np.asarray(d["mu"], np.float32); sd = np.asarray(d["sd"], np.float32)
        labels = [str(x) for x in d["labels"]]

    def run(wav):
        t = time.perf_counter()
        pooled = sess.run(["pooled"], {"source": wav[None, :]})[0]
        ms = (time.perf_counter() - t) * 1000
        # 헤드 전처리는 서버와 동일해야 한다 — half()->float() 왕복까지 재현
        pooled = pooled.astype(np.float16).astype(np.float32)
        return _softmax_top5(pooled, W, b, mu, sd, labels), ms

    MODELS[name] = {"run": run, "device": "CPU", "note": note}
    print(f"[load] {name}: ONNX CPU · {len(labels)}클래스", flush=True)


def _load_2b():
    import torch
    from country_classifier.model import build_model
    from torch.nn.utils import remove_weight_norm
    enc = build_model(backend="fairseq_xlsr", num_classes=41, checkpoint_path=CKPT_2B,
                      fairseq_root=str(FAIRSEQ_ROOT), freeze_encoder=True).encoder.eval()
    for m in enc.modules():
        try: remove_weight_norm(m)
        except (ValueError, AttributeError): pass
    use_cuda = torch.cuda.is_available()
    enc = (enc.cuda().half() if use_cuda else enc.cpu().float()).eval()

    d = torch.load(f"{D2B}/head.pt", map_location="cpu", weights_only=False)
    W = d["head"]["weight"].numpy(); b = d["head"]["bias"].numpy()
    mu = np.asarray(d["mu"], np.float32); sd = np.asarray(d["sd"], np.float32)
    # ⚠️ 2B(0702) 라벨은 한글 순서다. 영문으로 통일하되 **인덱스는 자기 라벨로 푼다.**
    labels = [str(x) for x in d.get("labels_en", d["labels"])]
    dev = "cuda" if use_cuda else "cpu"
    dt = torch.float16 if use_cuda else torch.float32

    def run(wav):
        s = torch.from_numpy(wav).unsqueeze(0).to(device=dev, dtype=dt)
        pm = torch.zeros(1, wav.size, dtype=torch.bool, device=dev)
        t = time.perf_counter()
        with torch.inference_mode():
            res = enc.extract_features(source=s, padding_mask=pm, mask=False, layer=31)
        f = res["layer_results"][31][0].transpose(0, 1)
        pooled = f.mean(1).float().cpu().numpy()
        if use_cuda:
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t) * 1000
        pooled = pooled.astype(np.float16).astype(np.float32)
        return _softmax_top5(pooled, W, b, mu, sd, labels), ms

    MODELS["2B"] = {"run": run, "device": "GPU" if use_cuda else "CPU",
                    "note": "정확도 참고용 · 백본 25 GB"}
    print(f"[load] 2B: torch {'GPU fp16' if use_cuda else 'CPU fp32'} · {len(labels)}클래스", flush=True)


def decode(raw):
    p = subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
                        "-f", "f32le", "-ac", "1", "-ar", str(SR), "-t", str(MAX_SEC), "pipe:1"],
                       input=raw, stdout=subprocess.PIPE)
    return np.frombuffer(p.stdout, np.float32).copy()


def run_all(wav):
    out = {}
    with _lock:
        for name, m in MODELS.items():
            try:
                top5, ms = m["run"](wav)
                out[name] = {"top5": top5, "ms": round(ms), "device": m["device"]}
            except Exception as e:
                out[name] = {"error": f"{type(e).__name__}: {e}"[:200]}
    return out


SAMPLES = sorted(glob.glob("/home/tta/test_samples/*.wav"))


def make_app():
    from fastapi import FastAPI, UploadFile, File, APIRouter
    from fastapi.responses import HTMLResponse, Response, JSONResponse
    from starlette.concurrency import run_in_threadpool
    app = FastAPI(title="nationality-compare")
    r = APIRouter()

    META = {
        "KD":   {"top1": 0.8245, "top5": 0.9584, "macro5": 0.7124, "size": "916 MB"},
        "300m": {"top1": 0.7510, "top5": 0.9331, "macro5": 0.6170, "size": "915 MB"},
        "2B":   {"top1": 0.7849, "top5": 0.9422, "macro5": 0.6321, "size": "25 GB"},
    }

    @r.get("/", response_class=HTMLResponse)
    def index():
        return open(os.path.join(HERE, "index.html"), encoding="utf-8").read()

    @r.get("/health")
    def health():
        return {"ok": True, "models": {k: {"device": v["device"], **META.get(k, {})}
                                       for k, v in MODELS.items()},
                "threads": THREADS, "테스트셋": "135,770발화 · 41클래스 · 043 포함", **_stats}

    @r.post("/predict")
    async def predict(audio: UploadFile = File(...)):
        _stats["requests"] += 1
        t0 = time.perf_counter()
        wav = await run_in_threadpool(decode, await audio.read())
        if wav.size < SR // 2:
            _stats["errors"] += 1
            return JSONResponse({"error": "오디오가 너무 짧거나 해석 불가"}, status_code=400)
        res = await run_in_threadpool(run_all, wav)
        tops = [v["top5"][0]["label"] for v in res.values() if "top5" in v]
        return {"results": res, "sec": round(wav.size / SR, 2),
                "일치": len(set(tops)) == 1 if tops else None,
                "total_ms": round((time.perf_counter() - t0) * 1000)}

    @r.get("/samples")
    def samples():
        return {"samples": [os.path.basename(s) for s in SAMPLES]}

    @r.get("/sample/{idx}")
    def sample(idx: int):
        return Response(open(SAMPLES[idx % len(SAMPLES)], "rb").read(), media_type="audio/wav")

    app.include_router(r)
    app.include_router(r, prefix="/compare")   # funnel --set-path 는 프리픽스를 안 벗긴다
    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8906)
    ap.add_argument("--host", default="0.0.0.0")
    a = ap.parse_args()
    t0 = time.time()
    _load_onnx("KD", f"{KD_DIR}/encoder_kd_l17.onnx", head_npz=f"{KD_DIR}/head_np.npz",
               note="2B→300m 지식증류 · 신규 최고")
    _load_onnx("300m", f"{D300}/encoder_l17.onnx", head_pt=f"{D300}/head.pt",
               note="현재 배포본")
    if LOAD_2B:
        _load_2b()
    print(f"[load] 전체 {time.time()-t0:.0f}s · 모델 {len(MODELS)}개", flush=True)
    import uvicorn
    uvicorn.run(make_app(), host=a.host, port=a.port)
