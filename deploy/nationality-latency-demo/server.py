#!/usr/bin/env python3
"""XLS-R 2B 국적 분류 — GPU vs CPU 추론 속도 비교 + latency 측정 데모 서버.

frozen XLS-R 2B (fairseq) → layer 31 mean-pool(1920) → linear head(41 국적).
같은 모델을 CPU(fp32)와 GPU(fp16, RTX 5080)에 모두 올려 나란히 비교한다.

동시성 설계 (동접 처리):
  - uvicorn 워커 1개 (모델이 프로세스당 ~9GB라 멀티프로세스 불가; 스레드로 확장)
  - 엔드포인트는 async def, 무거운 작업(디코딩/추론)은 전부 run_in_threadpool로
    감싸서 이벤트 루프를 절대 막지 않는다 → /status 등은 추론 중에도 즉시 응답
  - GPU: threading.Lock (단일 GPU 직렬화 — 어차피 요청당 ~0.1s라 큐가 짧음)
  - CPU: threading.Semaphore(CPU_CONCURRENCY, 기본 2) — 코어를 나눠 동시 2개까지
  - 락/세마포어 대기시간(wait_ms)과 순수 연산시간(compute_ms)을 분리 측정

feature 규격 (head.pt 학습 시와 100% 동일해야 함):
  16kHz mono → 앞 12초 → extract_features(mask=False, layer=31)
  → layer_results[31][0] → 시간축 mean-pool → nan_to_num → clamp(-50,50)
  → fp16 round-trip → (x-mu)/sd → head → softmax → top5
  (layer=31 조기 종료는 전층 forward와 바이트 단위 동일 — break 전에 append됨)

실행:
  conda run --no-capture-output -n fairseq_xlsr_5080 \
      source tools/env.sh && python deploy/nationality-latency-demo/server.py
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists(): _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
if "--help" in _sys.argv or "-h" in _sys.argv:   # 인자 파서가 없는 스크립트다. --help 로 본체가 실행되면 안 된다.
    print(__doc__ or "설명이 없다 — 소스를 열어볼 것"); raise SystemExit(0)


import asyncio
import copy
import os
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# ── 경로/설정 (env var로 override 가능) ─────────────────────────────────────
# 경로는 저장소 루트의 .env 또는 환경변수로 준다 (shared/paths.py 와 같은 키).
FAIRSEQ_ROOT = os.environ["FAIRSEQ_ROOT"]
# 32L = slim(옵티마이저 제거)에서 추가로 미사용 layer 32~47 물리 절단.
# layer_results[31] feature는 48L과 비트 단위 동일 검증됨. 2.16B→1.45B, 9GB→5.8GB.
XLSR_CKPT = os.environ.get(
    "XLSR_CKPT", os.path.join(os.environ["SPEECH_AI_MODELS"], "xlsr2_1p45B_32L.pt"))
HEAD_PT = os.environ.get(
    "HEAD_PT",
    os.path.join(os.environ["NATIONALITY_OUTPUTS"], "0704_country41_sent043_en/head.pt"))
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8777"))
# 부하 테스트 실측(2026-07-05): 4슬롯×5스레드가 단건 지연(2.95s)과 처리량(0.47rps)
# 모두 최적 — 2×14보다 단건도 빠름(스레드 동기화 오버헤드 감소, 대역폭 병목).
CPU_CONCURRENCY = int(os.environ.get("CPU_CONCURRENCY", "4"))
USE_GPU = os.environ.get("USE_GPU", "1") == "1"
USE_INT8 = os.environ.get("USE_INT8", "1") == "1"  # CPU int8 동적 양자화 모델 추가 로드
USE_ONNX = os.environ.get("USE_ONNX", "1") == "1"  # ONNX Runtime CPU 경로 추가 로드
# ONNX 인코더(layer 출력까지). 기본은 head.pt 옆의 encoder_l17.onnx
ONNX_PT = os.environ.get("ONNX_PT", str(Path(HEAD_PT).parent / "encoder_l17.onnx"))

# 로컬 fairseq repo를 CWD의 ./fairseq shadowing보다 먼저 두어야
# `from fairseq import checkpoint_utils`가 제대로 잡힌다.
if FAIRSEQ_ROOT not in sys.path:
    sys.path.insert(0, FAIRSEQ_ROOT)

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

# torch>=2.6은 weights_only=True가 기본이라 fairseq/numpy가 든 로컬 체크포인트
# 로드가 실패한다. 신뢰된 로컬 파일만 다루므로 기본값을 되돌린다.
_torch_load = torch.load


def _torch_load_full(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _torch_load(*args, **kwargs)


torch.load = _torch_load_full

if os.environ.get("THREADS"):
    torch.set_num_threads(int(os.environ["THREADS"]))
else:
    # 슬롯당 코어 균등 분배 (20코어/4슬롯 = 5스레드)
    torch.set_num_threads(max(1, (os.cpu_count() or 20) // CPU_CONCURRENCY))

# GPU 메모리 하드 락: 이 프로세스가 GPU에서 GPU_MEM_GB 이상 할당하려 하면
# 조용히 커지지 않고 OOM으로 즉시 실패한다 (공유 A100 80GB 보호용).
GPU_MEM_GB = float(os.environ.get("GPU_MEM_GB", "40"))
GPU_MEM_FRACTION = None
if USE_GPU and torch.cuda.is_available() and GPU_MEM_GB > 0:
    _total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    GPU_MEM_FRACTION = min(0.99, GPU_MEM_GB / _total_gb)
    torch.cuda.set_per_process_memory_fraction(GPU_MEM_FRACTION, 0)
    print(f"* GPU memory lock: <= {GPU_MEM_GB:.0f} GB "
          f"(fraction {GPU_MEM_FRACTION:.3f} of {_total_gb:.0f} GB)")

SAMPLE_RATE = 16000

# 41개 한국어 라벨 → (영문명, ISO2). 비교 페이지 표기용.
COUNTRY_EN = {
    "네팔": ("Nepal", "NP"), "독일": ("Germany", "DE"), "러시아": ("Russia", "RU"),
    "루마니아": ("Romania", "RO"), "르완다": ("Rwanda", "RW"), "말레이시아": ("Malaysia", "MY"),
    "몽골": ("Mongolia", "MN"), "미국": ("United States", "US"), "미얀마": ("Myanmar", "MM"),
    "방글라데시": ("Bangladesh", "BD"), "베트남": ("Vietnam", "VN"), "불가리아": ("Bulgaria", "BG"),
    "브라질": ("Brazil", "BR"), "스페인": ("Spain", "ES"), "싱가포르": ("Singapore", "SG"),
    "에티오피아": ("Ethiopia", "ET"), "영국": ("United Kingdom", "GB"),
    "우즈베키스탄": ("Uzbekistan", "UZ"), "우크라이나": ("Ukraine", "UA"), "이란": ("Iran", "IR"),
    "이탈리아": ("Italy", "IT"), "인도": ("India", "IN"), "인도네시아": ("Indonesia", "ID"),
    "일본": ("Japan", "JP"), "잠비아": ("Zambia", "ZM"), "중국": ("China", "CN"),
    "칠레": ("Chile", "CL"), "카자흐스탄": ("Kazakhstan", "KZ"), "캄보디아": ("Cambodia", "KH"),
    "콜롬비아": ("Colombia", "CO"), "키르기스스탄": ("Kyrgyzstan", "KG"),
    "타지키스탄": ("Tajikistan", "TJ"), "태국": ("Thailand", "TH"),
    "투르크메니스탄": ("Turkmenistan", "TM"), "튀르키예": ("Türkiye", "TR"),
    "파키스탄": ("Pakistan", "PK"), "페루": ("Peru", "PE"), "프랑스": ("France", "FR"),
    "필리핀": ("Philippines", "PH"), "한국어": ("Korea", "KR"), "홍콩": ("Hong Kong", "HK"),
}
# 영문 라벨 head(0704_en)용: 영문명 → ISO2
ISO_BY_EN = {en: iso for en, iso in COUNTRY_EN.values()}
ISO_BY_EN["Turkey"] = "TR"  # 0704 head는 'Türkiye' 대신 'Turkey' 표기

app = FastAPI(title="XLS-R 2B GPU vs CPU nationality demo")
STATIC_DIR = Path(__file__).resolve().parent / "static"

# ── 모델 상태 (백그라운드 로드) ──────────────────────────────────────────────
STATE = {
    "status": "loading",  # loading | ready | error   (ready = CPU 사용 가능)
    "gpu_status": "loading" if (USE_GPU and torch.cuda.is_available()) else "absent",
    "error": None,
    "load_started_at": None,
    "encoder_load_sec": None,
    "gpu_load_sec": None,
    "int8_status": "loading" if USE_INT8 else "absent",
    "int8_load_sec": None,
    "onnx_status": "loading" if USE_ONNX else "absent",
    "onnx_load_sec": None,
    "gpu_error": None,
    "int8_error": None,
    "onnx_error": None,
    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "torch_version": torch.__version__,
    "num_threads": torch.get_num_threads(),
    "cpu_concurrency": CPU_CONCURRENCY,
    "uvicorn_workers": 1,
    "gpu_mem_lock_gb": GPU_MEM_GB if GPU_MEM_FRACTION is not None else None,
}
ENCODER_CPU = None
ENCODER_CPU_INT8 = None
ENCODER_GPU = None
ONNX_POOL = None          # queue.Queue of ORT 세션 (동접 시 세션당 스레드풀 분리 → 경합 방지)
HEAD = None
MU = None
SD = None
LABELS = None
LAYER = 31
MAX_SEC = 12.0

# GPU_CONCURRENCY=1(기본)이면 직렬화. >1은 실험용: CUDA가 커널을 시분할할 뿐
# 처리량이 늘지 않는다는 것을 보여주기 위한 스위치.
GPU_CONCURRENCY = int(os.environ.get("GPU_CONCURRENCY", "1"))
GPU_LOCK = threading.Semaphore(GPU_CONCURRENCY)
CPU_SEM = threading.Semaphore(CPU_CONCURRENCY)  # CPU 동시 추론 슬롯

# 라이브 카운터 (/status로 노출 — 동접 상황 관찰용)
COUNTERS = {"cpu_active": 0, "cpu_waiting": 0, "gpu_active": 0, "gpu_waiting": 0,
            "requests_done": 0}
CNT_LOCK = threading.Lock()


def _cnt(key, delta=1):
    with CNT_LOCK:
        COUNTERS[key] += delta


def load_models():
    global ENCODER_CPU, ENCODER_CPU_INT8, ENCODER_GPU, ONNX_POOL, HEAD, MU, SD, LABELS, LAYER, MAX_SEC
    try:
        STATE["load_started_at"] = time.time()

        # 1) head.pt (작음 — 규격 파라미터 확보)
        d = torch.load(HEAD_PT, map_location="cpu")
        head = nn.Linear(int(d["feat_dim"]), len(d["labels"]))
        head.load_state_dict(d["head"])
        head.eval()
        for p in head.parameters():
            p.requires_grad = False
        HEAD = head
        MU = torch.from_numpy(np.asarray(d["mu"], dtype=np.float32))
        SD = torch.from_numpy(np.asarray(d["sd"], dtype=np.float32))
        LABELS = list(d["labels"])
        LAYER = int(d.get("layer", 31))
        MAX_SEC = float(d.get("max_sec", 12))

        # 2) XLS-R 2B 인코더 → CPU fp32
        t0 = time.perf_counter()
        from fairseq import checkpoint_utils

        models, _, _ = checkpoint_utils.load_model_ensemble_and_task([XLSR_CKPT])
        encoder = models[0]
        if hasattr(encoder, "remove_pretraining_modules"):
            encoder.remove_pretraining_modules()
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad = False
        # pos_conv의 weight_norm은 deepcopy를 막는다 (torch 제약). frozen 추론 전용이라
        # weight_norm을 fuse해도 forward는 동일 — CPU ready 선언 전에 제거해 레이스 방지.
        from torch.nn.utils import remove_weight_norm
        for m in encoder.modules():
            try:
                remove_weight_norm(m)
            except ValueError:
                pass
        ENCODER_CPU = encoder
        STATE["encoder_load_sec"] = round(time.perf_counter() - t0, 2)
        STATE["status"] = "ready"  # CPU 추론 가능 시점

        # 3) GPU 복사본 (fp16) + warmup — CPU ready 이후 진행
        if STATE["gpu_status"] == "loading":
            try:
                t0 = time.perf_counter()
                gpu = copy.deepcopy(encoder).half().to("cuda")
                gpu.eval()
                ENCODER_GPU = gpu
                with torch.inference_mode():  # 커널 초기화 warmup (첫 요청 지연 방지)
                    src = torch.zeros(1, SAMPLE_RATE, dtype=torch.float16, device="cuda")
                    pm = torch.zeros(1, SAMPLE_RATE, dtype=torch.bool, device="cuda")
                    ENCODER_GPU.extract_features(source=src, padding_mask=pm, mask=False, layer=LAYER)
                    torch.cuda.synchronize()
                STATE["gpu_load_sec"] = round(time.perf_counter() - t0, 2)
                STATE["gpu_status"] = "ready"
            except Exception:
                STATE["gpu_status"] = "error"
                STATE["gpu_error"] = traceback.format_exc()

        # 4) CPU int8 동적 양자화 (Linear 가중치 int8, 메모리 ~4×↓ 속도 ~1.6×↑)
        if STATE["int8_status"] == "loading":
            try:
                t0 = time.perf_counter()
                from torch.ao.quantization import quantize_dynamic

                q = quantize_dynamic(ENCODER_CPU, {nn.Linear}, dtype=torch.qint8)
                q.eval()
                # fairseq MHA fast-path는 q_proj.bias를 직접 참조해 양자화 Linear와
                # 충돌 → skip_embed_dim_check=True로 모듈 호출 경로 강제 (공식 플래그)
                for m in q.modules():
                    if type(m).__name__ == "MultiheadAttention":
                        m.skip_embed_dim_check = True
                ENCODER_CPU_INT8 = q
                STATE["int8_load_sec"] = round(time.perf_counter() - t0, 2)
                STATE["int8_status"] = "ready"
            except Exception:
                STATE["int8_status"] = "error"
                STATE["int8_error"] = traceback.format_exc()

        # 5) ONNX Runtime 세션 풀 (raw wav → layer 출력 pooled feature)
        #    동접 요청이 세션 1개를 공유하면 intra-op 스레드풀 경합으로 느려진다.
        #    → CPU_CONCURRENCY 개의 세션 풀을 만들어 동접 요청이 각자 세션을 쓰게 한다
        #    (CPU_SEM이 동접을 그 수로 제한하므로 pool.get()은 절대 블록되지 않음).
        if STATE["onnx_status"] == "loading":
            try:
                t0 = time.perf_counter()
                import onnxruntime as ort
                import queue as _queue

                intra = torch.get_num_threads()          # 세션당 스레드 (풀로 코어 분리)
                pool = _queue.Queue()
                for _ in range(CPU_CONCURRENCY):
                    so = ort.SessionOptions()
                    so.intra_op_num_threads = intra
                    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    s = ort.InferenceSession(ONNX_PT, so, providers=["CPUExecutionProvider"])
                    s.run(["pooled"], {"source": np.zeros((1, SAMPLE_RATE), np.float32)})  # warmup
                    pool.put(s)
                ONNX_POOL = pool
                STATE["onnx_sessions"] = CPU_CONCURRENCY
                STATE["onnx_load_sec"] = round(time.perf_counter() - t0, 2)
                STATE["onnx_status"] = "ready"
            except Exception:
                STATE["onnx_status"] = "error"
                STATE["onnx_error"] = traceback.format_exc()
    except Exception:
        STATE["status"] = "error"
        STATE["error"] = traceback.format_exc()


threading.Thread(target=load_models, daemon=True).start()


# ── 오디오 디코딩 (ffmpeg: wav/mp3/webm/ogg/m4a → 16k mono f32) ─────────────
def probe_duration(path: str):
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", path],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=10,
        )
        text = out.stdout.decode().strip()
        return float(text) if out.returncode == 0 and text else None
    except Exception:
        return None


def decode_audio(data: bytes):
    """업로드 바이트 → (16k mono f32 파형, 원본 길이 초).

    - stdin 파이프 대신 임시 파일: moov atom이 파일 끝에 있는 m4a/mp4도 디코딩됨.
    - `-t`로 디코딩을 규격 상한(MAX_SEC)+0.5초에서 끊음. 원본 길이는 ffprobe로.
    """
    tmp = tempfile.NamedTemporaryFile(delete=False)
    try:
        tmp.write(data)
        tmp.close()
        orig_duration = probe_duration(tmp.name)
        try:
            proc = subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error",
                    "-i", tmp.name,
                    "-f", "f32le", "-acodec", "pcm_f32le",
                    "-ac", "1", "-ar", str(SAMPLE_RATE),
                    "-t", str(MAX_SEC + 0.5),
                    "pipe:1",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=120,
            )
        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=400, detail="오디오 디코딩 시간 초과 (120초).")
        except FileNotFoundError:
            raise HTTPException(status_code=500, detail="ffmpeg가 설치되어 있지 않습니다.")
        if proc.returncode != 0:
            raise HTTPException(
                status_code=400,
                detail=f"오디오 디코딩 실패: {proc.stderr.decode(errors='replace')[-500:]}",
            )
        # frombuffer는 읽기전용 뷰 → torch.from_numpy 경고 방지 위해 복사
        wav = np.frombuffer(proc.stdout, dtype=np.float32).copy()
        if wav.size < int(0.25 * SAMPLE_RATE):
            raise HTTPException(status_code=400, detail="오디오가 너무 짧습니다 (0.25초 미만).")
        if orig_duration is None:
            orig_duration = wav.size / SAMPLE_RATE
        return wav, orig_duration
    finally:
        os.unlink(tmp.name)


# ── feature 추출 + 예측 (규격 고정, CPU/GPU 공용) ───────────────────────────
@torch.inference_mode()
def _pool(feats: torch.Tensor, feat_padding_mask):
    feats = feats.transpose(0, 1)  # (B, T, C)
    if feat_padding_mask is None:
        pooled = feats.mean(dim=1)
    else:
        valid = (~feat_padding_mask).to(feats.dtype).unsqueeze(-1)
        denom = valid.sum(dim=1).clamp_min(1.0)
        pooled = (feats * valid).sum(dim=1) / denom
    return pooled, int(feats.shape[1])


@torch.inference_mode()
def _classify(pooled_f32_cpu: torch.Tensor):
    """pooled (1,1920) fp32 CPU 텐서 → top5. clamp/fp16/표준화/head는 장치와
    무관하게 이 CPU 경로 하나로 통일 (규격 고정)."""
    x = torch.nan_to_num(pooled_f32_cpu).clamp(-50.0, 50.0)
    x = x.half().float()  # 학습 feature 캐시가 fp16으로 저장됐던 규격 재현
    x = (x - MU) / SD
    logits = HEAD(x)
    probs = logits.softmax(dim=-1)[0]
    topk = probs.topk(5)
    top5 = []
    for p, i in zip(topk.values.tolist(), topk.indices.tolist()):
        raw = LABELS[i]
        if raw in COUNTRY_EN:  # 한글 라벨 head (0702 계열)
            en, iso = COUNTRY_EN[raw]
        else:  # 영문 라벨 head (0704_en 계열)
            en, iso = raw, ISO_BY_EN.get(raw, "??")
        # 표시 라벨은 항상 영어로 통일
        top5.append({"label": en, "label_en": en, "iso": iso, "prob": round(float(p), 6)})
    return top5


@torch.inference_mode()
def infer_cpu(wav: np.ndarray, runs: int = 1, variant: str = "fp32"):
    """CPU 추론 (variant: fp32 | int8). 슬롯 대기(wait_ms)와 연산(compute)을 분리 측정."""
    enc = ENCODER_CPU_INT8 if variant == "int8" else ENCODER_CPU
    if enc is None:
        raise RuntimeError(f"CPU {variant} 모델 미준비")
    _cnt("cpu_waiting")
    t_wait = time.perf_counter()
    with CPU_SEM:
        wait_ms = (time.perf_counter() - t_wait) * 1000
        _cnt("cpu_waiting", -1)
        _cnt("cpu_active")
        try:
            source = torch.from_numpy(wav).unsqueeze(0)
            padding_mask = torch.zeros(1, wav.size, dtype=torch.bool)
            encoder_ms_runs = []
            feats, fpm = None, None
            for _ in range(runs):
                t0 = time.perf_counter()
                res = enc.extract_features(
                    source=source, padding_mask=padding_mask, mask=False, layer=LAYER
                )
                feats, fpm = res["layer_results"][LAYER][0], res.get("padding_mask")
                encoder_ms_runs.append((time.perf_counter() - t0) * 1000)
            t0 = time.perf_counter()
            pooled, num_frames = _pool(feats, fpm)
            top5 = _classify(pooled.float())
            head_ms = (time.perf_counter() - t0) * 1000
        finally:
            _cnt("cpu_active", -1)
    encoder_ms = sum(encoder_ms_runs) / len(encoder_ms_runs)
    return {
        "device": "cpu", "dtype": "int8-dynamic" if variant == "int8" else "fp32",
        "threads": torch.get_num_threads(),
        "top5": top5, "num_frames": num_frames,
        "wait_ms": round(wait_ms, 1),
        "encoder_ms": round(encoder_ms, 1),
        "encoder_ms_runs": [round(v, 1) for v in encoder_ms_runs],
        "head_ms": round(head_ms, 2),
        "compute_ms": round(encoder_ms + head_ms, 1),
    }


def infer_onnx(wav: np.ndarray, runs: int = 1):
    """ONNX Runtime(CPU) 추론: raw wav → pooled feature(ONNX) → _classify(공용).
    세션 풀에서 하나를 빌려 써서 동접 요청 간 스레드풀 경합을 피한다."""
    if ONNX_POOL is None:
        raise RuntimeError("ONNX 모델 미준비")
    _cnt("cpu_waiting")
    t_wait = time.perf_counter()
    with CPU_SEM:
        wait_ms = (time.perf_counter() - t_wait) * 1000
        _cnt("cpu_waiting", -1)
        _cnt("cpu_active")
        sess = ONNX_POOL.get()                       # 전용 세션 확보 (CPU_SEM 덕에 항상 여유)
        try:
            src = wav[None, :].astype(np.float32)
            encoder_ms_runs = []
            pooled = None
            for _ in range(runs):
                t0 = time.perf_counter()
                pooled = sess.run(["pooled"], {"source": src})[0]  # (1, feat_dim) fp32
                encoder_ms_runs.append((time.perf_counter() - t0) * 1000)
            t0 = time.perf_counter()
            top5 = _classify(torch.from_numpy(pooled).float())
            head_ms = (time.perf_counter() - t0) * 1000
        finally:
            ONNX_POOL.put(sess)
            _cnt("cpu_active", -1)
    encoder_ms = sum(encoder_ms_runs) / len(encoder_ms_runs)
    return {
        "device": "cpu-onnx", "dtype": "fp32-onnx",
        "threads": torch.get_num_threads(),
        "top5": top5, "num_frames": 0,
        "wait_ms": round(wait_ms, 1),
        "encoder_ms": round(encoder_ms, 1),
        "encoder_ms_runs": [round(v, 1) for v in encoder_ms_runs],
        "head_ms": round(head_ms, 2),
        "compute_ms": round(encoder_ms + head_ms, 1),
    }


@torch.inference_mode()
def infer_gpu(wav: np.ndarray, runs: int = 1):
    """GPU fp16 추론. H2D 전송 포함해서 측정 (실제 요청 비용). Lock 직렬화."""
    _cnt("gpu_waiting")
    t_wait = time.perf_counter()
    with GPU_LOCK:
        wait_ms = (time.perf_counter() - t_wait) * 1000
        _cnt("gpu_waiting", -1)
        _cnt("gpu_active")
        try:
            encoder_ms_runs = []
            feats, fpm = None, None
            for _ in range(runs):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                source = torch.from_numpy(wav).unsqueeze(0).half().to("cuda")
                padding_mask = torch.zeros(1, wav.size, dtype=torch.bool, device="cuda")
                res = ENCODER_GPU.extract_features(
                    source=source, padding_mask=padding_mask, mask=False, layer=LAYER
                )
                feats, fpm = res["layer_results"][LAYER][0], res.get("padding_mask")
                torch.cuda.synchronize()
                encoder_ms_runs.append((time.perf_counter() - t0) * 1000)
            t0 = time.perf_counter()
            pooled, num_frames = _pool(feats, fpm)
            pooled = pooled.float().cpu()  # 분류는 CPU 공용 경로 (규격 동일 보장)
            top5 = _classify(pooled)
            head_ms = (time.perf_counter() - t0) * 1000
        finally:
            _cnt("gpu_active", -1)
    encoder_ms = sum(encoder_ms_runs) / len(encoder_ms_runs)
    return {
        "device": "cuda", "dtype": "fp16", "gpu_name": STATE["gpu_name"],
        "top5": top5, "num_frames": num_frames,
        "wait_ms": round(wait_ms, 1),
        "encoder_ms": round(encoder_ms, 1),
        "encoder_ms_runs": [round(v, 1) for v in encoder_ms_runs],
        "head_ms": round(head_ms, 2),
        "compute_ms": round(encoder_ms + head_ms, 1),
    }


# ── 공통 요청 처리 ───────────────────────────────────────────────────────────
async def read_and_decode(file: UploadFile, truncate_sec: float):
    if STATE["status"] == "error":
        raise HTTPException(status_code=500, detail="모델 로드 실패 — /status 참조")
    if STATE["status"] != "ready":
        raise HTTPException(status_code=503, detail="모델 로딩 중 — 잠시 후 다시 시도")

    # 크기 검사는 메모리로 읽기 전에 (spool된 임시파일 기준)
    size = file.size
    if size is None:
        file.file.seek(0, os.SEEK_END)
        size = file.file.tell()
        file.file.seek(0)
    if size > 200 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="파일이 너무 큽니다 (200MB 제한).")
    data = await file.read()

    t0 = time.perf_counter()
    wav, orig_duration = await run_in_threadpool(decode_audio, data)
    decode_ms = (time.perf_counter() - t0) * 1000

    # conv stack이 최소 ~0.025초(400샘플)를 요구하므로 하한 0.25초로 클램프
    limit_sec = MAX_SEC if truncate_sec <= 0 else min(max(float(truncate_sec), 0.25), MAX_SEC)
    wav = wav[: int(limit_sec * SAMPLE_RATE)]
    return wav, orig_duration, round(decode_ms, 1)


# ── 브라우저용 부하 테스트 (서버 내부에서 N개 동시 발사) ────────────────────
# 브라우저가 직접 N개를 쏘면 호스트당 6커넥션 제한으로 왜곡되므로, 발사는 서버가
# 하고 프론트는 /loadtest/status를 폴링해 실시간으로 그린다. 발사된 작업은 실제
# 요청과 동일하게 CPU_SEM/GPU_LOCK을 통과한다 (진짜 서버 부하).
SAMPLE_PATH = STATIC_DIR / "sample.wav"
LOADTEST = {"running": False, "cancel": False, "results": [], "summary": None,
            "mode": None, "concurrency": 0, "total": 0, "started_at": None,
            "done": 0, "finished_at": None}
LT_LOCK = threading.Lock()


def _pct_of(vals, p):
    if not vals:
        return None
    vals = sorted(vals)
    return vals[min(len(vals) - 1, max(0, round(p / 100 * (len(vals) - 1))))]


def _run_loadtest(wav, mode, concurrency, total):
    results = LOADTEST["results"]
    t_start = time.perf_counter()

    def task(i):
        r = results[i]
        if LOADTEST["cancel"]:
            r["status"] = "cancelled"
            with LT_LOCK:
                LOADTEST["done"] += 1
            return
        r["status"] = "running"
        t0 = time.perf_counter()
        try:
            g = infer_gpu(wav) if mode in ("gpu", "both") else None
            if mode == "cpu_int8":
                c = infer_cpu(wav, 1, "int8")
            elif mode == "onnx":
                c = infer_onnx(wav)
            elif mode in ("cpu", "both"):
                c = infer_cpu(wav)
            else:
                c = None
            main = c if c is not None else g
            r.update(
                status="done",
                wall_ms=round((time.perf_counter() - t0) * 1000, 1),
                wait_ms=main["wait_ms"],
                compute_ms=main["compute_ms"],
            )
            if g is not None and c is not None:
                r["gpu_ms"] = g["compute_ms"]
        except Exception as e:
            r.update(status="error", error=str(e)[:200],
                     wall_ms=round((time.perf_counter() - t0) * 1000, 1))
        finally:
            with LT_LOCK:
                LOADTEST["done"] += 1

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        list(ex.map(task, range(total)))

    elapsed = time.perf_counter() - t_start
    ok = [r for r in results if r["status"] == "done"]
    walls = [r["wall_ms"] for r in ok]
    LOADTEST["summary"] = {
        "elapsed_s": round(elapsed, 1),
        "ok": len(ok),
        "errors": sum(1 for r in results if r["status"] == "error"),
        "cancelled": sum(1 for r in results if r["status"] == "cancelled"),
        "throughput_rps": round(len(ok) / elapsed, 2) if elapsed > 0 else 0,
        "wall_p50_ms": _pct_of(walls, 50),
        "wall_p95_ms": _pct_of(walls, 95),
        "wall_max_ms": _pct_of(walls, 100),
        "wait_avg_ms": round(sum(r["wait_ms"] for r in ok) / len(ok), 1) if ok else None,
        "compute_avg_ms": round(sum(r["compute_ms"] for r in ok) / len(ok), 1) if ok else None,
    }
    LOADTEST["finished_at"] = time.time()
    LOADTEST["running"] = False


@app.post("/loadtest/start")
async def loadtest_start(
    file: UploadFile = File(None),
    use_sample: int = Form(1),
    mode: str = Form("cpu"),  # cpu | gpu | both
    concurrency: int = Form(4),
    total: int = Form(0),
):
    if STATE["status"] != "ready":
        raise HTTPException(status_code=503, detail="모델 로딩 중")
    if mode in ("gpu", "both") and STATE["gpu_status"] != "ready":
        raise HTTPException(status_code=503, detail="GPU 미준비")
    if mode == "cpu_int8" and STATE["int8_status"] != "ready":
        raise HTTPException(status_code=503, detail="int8 모델 미준비")
    if mode == "onnx" and STATE["onnx_status"] != "ready":
        raise HTTPException(status_code=503, detail="ONNX 모델 미준비")
    if mode not in ("cpu", "cpu_int8", "gpu", "both", "onnx"):
        raise HTTPException(status_code=400, detail="mode는 cpu|cpu_int8|gpu|both|onnx")
    concurrency = max(1, min(int(concurrency), 64))
    # GPU는 건당 ~30ms라 2000건도 ~1분 — 지속 부하 테스트로 의미 있음.
    # CPU가 끼는 모드는 0.47rps라 256건(~9분)을 상한으로.
    max_total = 2000 if mode == "gpu" else 256
    total = max(1, min(int(total) or concurrency, max_total))

    if use_sample:
        data = SAMPLE_PATH.read_bytes()
    else:
        if file is None:
            raise HTTPException(status_code=400, detail="파일이 없습니다 (또는 use_sample=1)")
        data = await file.read()
        if len(data) > 200 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="파일이 너무 큽니다 (200MB 제한).")
    wav, _ = await run_in_threadpool(decode_audio, data)
    wav = wav[: int(MAX_SEC * SAMPLE_RATE)]

    with LT_LOCK:
        if LOADTEST["running"]:
            raise HTTPException(status_code=409, detail="이미 부하 테스트 실행 중 — 중지 후 재시작하세요")
        LOADTEST.update(
            running=True, cancel=False, mode=mode, concurrency=concurrency,
            total=total, done=0, summary=None, started_at=time.time(), finished_at=None,
            results=[{"status": "pending"} for _ in range(total)],
        )
    threading.Thread(target=_run_loadtest, args=(wav, mode, concurrency, total), daemon=True).start()
    return {"ok": True, "concurrency": concurrency, "total": total, "mode": mode}


@app.post("/loadtest/stop")
async def loadtest_stop():
    LOADTEST["cancel"] = True  # 대기 중인 작업만 취소, 실행 중인 것은 마저 끝냄
    return {"ok": True}


@app.get("/loadtest/status")
async def loadtest_status():
    out = {k: LOADTEST[k] for k in
           ("running", "mode", "concurrency", "total", "done", "summary", "cancel")}
    results = LOADTEST["results"]
    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    out["status_counts"] = counts
    # 대량(GPU 2000건) 테스트는 폴링 페이로드/타일 렌더링이 무거워지므로 집계만 전송
    out["results"] = results if len(results) <= 256 else None
    if LOADTEST["started_at"]:
        end = LOADTEST["finished_at"] or time.time()
        out["elapsed_s"] = round(end - LOADTEST["started_at"], 1)
    with CNT_LOCK:
        out["live"] = dict(COUNTERS)
    return out


@app.get("/sample")
async def sample_audio():
    return FileResponse(SAMPLE_PATH, media_type="audio/wav")


# ── 엔드포인트 ───────────────────────────────────────────────────────────────
@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "compare.html")


@app.get("/loadtest")
async def loadtest_page():
    return FileResponse(STATIC_DIR / "loadtest.html")


@app.get("/latency")
async def latency_page():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/status")
async def status():
    out = {k: v for k, v in STATE.items()}
    out["num_threads"] = torch.get_num_threads()
    with CNT_LOCK:
        out["live"] = dict(COUNTERS)
    if STATE["status"] == "loading" and STATE["load_started_at"]:
        out["loading_elapsed_sec"] = round(time.time() - STATE["load_started_at"], 1)
    if STATE["status"] == "ready":
        out["labels"] = LABELS
        out["layer"] = LAYER
        out["max_sec"] = MAX_SEC
    return out


@app.post("/predict_both")
async def predict_both(file: UploadFile = File(...), truncate_sec: float = Form(0.0)):
    """같은 오디오를 GPU와 CPU에서 동시에(병렬) 추론해 나란히 반환."""
    if STATE["gpu_status"] != "ready":
        raise HTTPException(
            status_code=503,
            detail=f"GPU 미준비 (gpu_status={STATE['gpu_status']}) — CPU 전용은 /predict 사용",
        )
    wav, orig_duration, decode_ms = await read_and_decode(file, truncate_sec)

    # GPU는 병렬, CPU 두 변형(fp32→int8)은 순차 — 동시에 돌리면 코어를 나눠 써서
    # 둘 다 느려져 비교 수치가 오염되기 때문.
    def cpu_chain():
        r32 = infer_cpu(wav)
        r8 = infer_cpu(wav, 1, "int8") if STATE["int8_status"] == "ready" else None
        ronnx = infer_onnx(wav) if STATE["onnx_status"] == "ready" else None
        return r32, r8, ronnx

    gpu_task = run_in_threadpool(infer_gpu, wav)
    cpu_task = run_in_threadpool(cpu_chain)
    gpu_res, (cpu_res, int8_res, onnx_res) = await asyncio.gather(gpu_task, cpu_task)
    _cnt("requests_done")

    speedup = cpu_res["compute_ms"] / max(gpu_res["compute_ms"], 1e-6)
    out = {
        "audio": {
            "filename": file.filename,
            "orig_duration_sec": round(orig_duration, 2),
            "used_duration_sec": round(wav.size / SAMPLE_RATE, 2),
            "num_frames": cpu_res["num_frames"],
        },
        "decode_ms": decode_ms,
        "gpu": gpu_res,
        "cpu": cpu_res,
        "cpu_int8": int8_res,
        "onnx": onnx_res,
        "speedup": round(speedup, 1),
        "top1_match": gpu_res["top5"][0]["label"] == cpu_res["top5"][0]["label"],
    }
    if int8_res:
        out["speedup_int8"] = round(cpu_res["compute_ms"] / max(int8_res["compute_ms"], 1e-6), 2)
        out["top1_match_int8"] = int8_res["top5"][0]["label"] == cpu_res["top5"][0]["label"]
    if onnx_res:
        out["speedup_onnx"] = round(cpu_res["compute_ms"] / max(onnx_res["compute_ms"], 1e-6), 2)
        out["top1_match_onnx"] = onnx_res["top5"][0]["label"] == cpu_res["top5"][0]["label"]
    return out


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    truncate_sec: float = Form(0.0),  # 0 = 규격 기본(max_sec=12초)
    runs: int = Form(1),
    device: str = Form("cpu"),  # cpu | cpu_int8 | gpu
):
    """단일 장치 추론 (기존 /latency 페이지 + 부하 테스트용)."""
    runs = max(1, min(int(runs), 20))
    if device not in ("cpu", "cpu_int8", "gpu", "onnx"):
        raise HTTPException(status_code=400, detail="device는 cpu|cpu_int8|gpu|onnx")
    if device == "gpu" and STATE["gpu_status"] != "ready":
        raise HTTPException(status_code=503, detail="GPU 미준비")
    if device == "cpu_int8" and STATE["int8_status"] != "ready":
        raise HTTPException(status_code=503, detail="int8 모델 미준비")
    if device == "onnx" and STATE["onnx_status"] != "ready":
        raise HTTPException(status_code=503, detail="ONNX 모델 미준비")
    wav, orig_duration, decode_ms = await read_and_decode(file, truncate_sec)

    t0 = time.perf_counter()
    if device == "gpu":
        res = await run_in_threadpool(infer_gpu, wav, runs)
    elif device == "cpu_int8":
        res = await run_in_threadpool(infer_cpu, wav, runs, "int8")
    elif device == "onnx":
        res = await run_in_threadpool(infer_onnx, wav, runs)
    else:
        res = await run_in_threadpool(infer_cpu, wav, runs)
    total_ms = decode_ms + (time.perf_counter() - t0) * 1000
    _cnt("requests_done")

    used_duration = wav.size / SAMPLE_RATE
    return {
        "top5": res["top5"],
        "timing": {
            "decode_ms": decode_ms,
            "wait_ms": res["wait_ms"],
            "encoder_ms": res["encoder_ms"],
            "encoder_ms_runs": res["encoder_ms_runs"],
            "head_ms": res["head_ms"],
            "total_ms": round(total_ms, 1),
            "rtf": round((res["encoder_ms"] / 1000) / max(used_duration, 1e-6), 2),
        },
        "audio": {
            "filename": file.filename,
            "orig_duration_sec": round(orig_duration, 2),
            "used_duration_sec": round(used_duration, 2),
            "num_frames": res["num_frames"],
        },
        "model": {
            "device": res["device"],
            "dtype": res["dtype"],
            "num_threads": torch.get_num_threads(),
            "layer": LAYER,
            "runs": runs,
        },
    }


if __name__ == "__main__":
    import uvicorn

    print(f"* XLS-R 2B GPU vs CPU demo: http://{HOST}:{PORT}")
    print(f"* encoder ckpt: {XLSR_CKPT}")
    print(f"* torch {torch.__version__}, cpu_threads={torch.get_num_threads()}, "
          f"cpu_concurrency={CPU_CONCURRENCY}, gpu={STATE['gpu_name']}")
    # workers=1 고정: 워커당 모델 ~9GB(CPU)+4.6GB(GPU)라 멀티프로세스는 비현실적.
    # 동시성은 threadpool + Semaphore/Lock으로 처리한다.
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
