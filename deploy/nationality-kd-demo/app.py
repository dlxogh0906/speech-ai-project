# -*- coding: utf-8 -*-
"""국적 분류 KD 인코더 데모 — ONNX 서빙(기본 CPU · 선택 GPU) + 동시접속 부하 테스트.

모델: 인코더 증류 최종본 (test top1 0.8245 / top5 0.9584 / macro5 0.7124)
      기존 배포본(0705, top1 0.7510) 대비 top1 +7.4%p — 같은 크기·같은 지연.

구조: onnxruntime 세션 풀(POOL개, 세션당 스레드 분리 — 동접 경합 방지) + numpy 헤드.
      torch/fairseq 불필요. 기본은 CPU 라 GPU 도 불필요.

    $PY deploy/nationality-kd-demo/app.py --port 8905

GPU(선택): onnxruntime-gpu(CUDA 11.8 빌드) + CUDA 11.8 라이브러리를 LD_LIBRARY_PATH 로 준다.

    KD_DEVICE=cuda LD_LIBRARY_PATH=<CUDA 11.8 라이브러리 링크 폴더> $PY deploy/nationality-kd-demo/app.py --port 8905

POST /predict_multi (2026-10-03): 같은 사람이 읽은 녹음 1~10개(audio 필드 반복) → 문장마다 판정 + log 확률 평균 합산.
    200  {"n", "top5", "per_sentence": [{"label", "prob", "sec"}], "gate_korea", "native", "low_confidence", ...}
    400  {"error", "code", "index"}   index = 문제가 된 녹음 순번(0부터). 녹음은 저장하지 않는다. 테스트 화면은 GET /multi.

POST /predict_long (2026-10-04): 한 번에 이어 읽은 녹음 1개(audio, 최대 60초) → 같은 길이로 나눠(parts 개, 1~10 · 기본 3 ·
    구간이 3초보다 짧아지면 줄이고 12초를 넘으면 늘림) 구간마다 판정 + log 확률 평균 합산. 말소리 없는 구간은 건너뛴다.
    KD_CAPTURE_DIR 이 있으면(NPU 운영, 2026-10-09~) /predict_long · /predict 로 들어온 녹음 전부와 판정 · 품질 지표를 남기고 응답에 rid 를 붙인다.
    200  {"n", "parts_requested", "sec", "voiced_sec", "top5", "per_part": [{"label", "prob", "start", "end"}], "gate_korea", "native",
          "low_confidence", ...}   n < parts_requested 면 녹음(말소리)이 짧아 구간을 줄인 것이다 — 연동 가이드: 연동_가이드.md
    KD_CORS_ORIGINS=https://a.com,https://b.com 이면 그 도메인의 웹 페이지가 브라우저에서 직접 부를 수 있다(기본 꺼짐).
    테스트 화면은 GET /multi(1·3·5문장 글을 한 번에 읽기 → parts=문장 수) · GET /long(3문장).
    근거: results/23_긴녹음_분할_합산.md (문장을 따로 녹음해 합산한 것과 정확도가 같다).

POST /predict 응답:
    200  {"top5": [{"label", "prob"} ×5], "sec", "encode_ms", "total_ms"}
    400  {"error", "code"}   code = decode_failed(오디오가 아니거나 깨짐) · too_short(0.5초 미만) · no_speech(무음)
    500  {"error", "code": "server_decoder"}   ffmpeg 자체가 못 돈다 — 요청이 아니라 서버 설정 문제
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), str(_r / "tasks" / "nationality-classification" / "src")]   # country_classifier.wav_augment (numpy 전용) 용
from shared.paths import FFMPEG, NAT_TASK
from shared.request_capture import capture_app, capture_context

import argparse, glob, json, os, queue, shutil, subprocess, tempfile, time

# NAT_WORKERS>1 이면 이 프로세스는 모델을 싣지 않고 uvicorn 다중 워커로 교체된다. 각 워커가 이 모듈을 import 해 모델을 싣는다.
# NPU(rebel) 호출은 한 프로세스 안의 스레드끼리 병렬로 돌지 않아서(실측: 스레드 1 → 16회/초, 4 → 9회/초) 프로세스로 늘린다.
if __name__ == "__main__" and int(os.environ.get("NAT_WORKERS", "1")) > 1:
    _a = _sys.argv[1:]
    _port = _a[_a.index("--port") + 1] if "--port" in _a else "8905"
    _host = _a[_a.index("--host") + 1] if "--host" in _a else "0.0.0.0"
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.execvp(_sys.executable, [_sys.executable, "-m", "uvicorn", "app:make_app", "--factory", "--workers", os.environ["NAT_WORKERS"],
                                "--host", _host, "--port", _port])
import numpy as np

SR = 16000
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get("KD_MODEL_DIR",
                           str(NAT_TASK / "outputs/kd_encoder/run2/deploy"))
# 모델 설명(/health · 웹 화면 부제). 모델 폴더에 model_info.json {"name", "summary"} 가 있으면 그걸 쓴다 —
# 모델 파일만 갈아 끼웠는데 화면이 옛 정확도를 보여 주는 일을 막는다.
MODEL_INFO = {"name": "encoder_kd_run2 (top1 0.8245)", "summary": ""}
if os.path.isfile(os.path.join(MODEL_DIR, "model_info.json")):
    MODEL_INFO.update(json.load(open(os.path.join(MODEL_DIR, "model_info.json"), encoding="utf-8")))
POOL = int(os.environ.get("POOL", "4"))
# 입력 음량 정규화(기본 꺼짐 — 운영 중인 인스턴스의 동작을 바꾸지 않는다).
# 모델은 입력 크기에 민감하다: 같은 녹음을 1/10 로 줄이면 1위 국적이 바뀐다(실측).
# 웹 마이크는 기기·거리·OS 설정에 따라 레벨이 제각각이라, 켜면 학습 오디오의 RMS 중앙값(0.058)에 맞춘다.
RMS_NORM = float(os.environ.get("NAT_RMS_NORM", "0") or 0)
INDEX_EXTRA_JS = os.environ.get("NAT_INDEX_EXTRA_JS", "")
THREADS = int(os.environ.get("THREADS", str(max(1, (os.cpu_count() or 8) // POOL))))
# KD_DEVICE=cuda 면 onnxruntime-gpu 의 CUDA provider 를 쓴다. 기본은 CPU(배포 전제가 GPU 없는 서버).
# CUDA 라이브러리 경로는 파이썬 안에서 바꾸면 안 먹는다 — 프로세스 시작 전에 LD_LIBRARY_PATH 로 줄 것.
DEVICE = os.environ.get("KD_DEVICE", "cpu").strip().lower()
if DEVICE == "cuda":
    # 발화 길이가 매번 달라서, 입력 길이마다 conv 알고리즘 전수 탐색(EXHAUSTIVE, 기본값)을 하지 않게 한다
    _cuda_opts = {"device_id": 0, "cudnn_conv_algo_search": "DEFAULT"}
    # KD_CUDA_ARENA=same: 세션마다 필요한 만큼만 GPU 메모리를 잡는다(기본은 2배씩 늘려 세션 8개면 17 GB 까지 커졌다 — 2026-09-26 실측)
    if os.environ.get("KD_CUDA_ARENA", "") == "same":
        _cuda_opts["arena_extend_strategy"] = "kSameAsRequested"
    PROVIDERS = [("CUDAExecutionProvider", _cuda_opts), "CPUExecutionProvider"]
elif DEVICE == "cpu":
    PROVIDERS = ["CPUExecutionProvider"]
else:
    raise SystemExit(f"KD_DEVICE 는 cpu 또는 cuda 여야 한다: {DEVICE!r}")

# ffmpeg 는 절대경로 + close_fds=False 로 부른다. 그래야 Python 3.8 subprocess 가 fork 대신 posix_spawn 을 쓴다.
# CUDA 를 올린 이 프로세스에서 fork 는 1회 ~68 ms 이고 요청끼리 줄을 서서 처리량이 17 req/s 에서 막혔다
# (실측 2026-09-14 — posix_spawn 이면 0.7 ms, 같은 서버에서 약 70 req/s).
FFMPEG_PATH = shutil.which(FFMPEG) or FFMPEG
try:
    _ff = subprocess.run([FFMPEG_PATH, "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=False)
    _ff_err = None if _ff.returncode == 0 else f"rc={_ff.returncode} {_ff.stderr.decode('utf-8', 'replace').strip()[:300]}"
except OSError as e:
    _ff_err = str(e)
if _ff_err:
    raise RuntimeError(f"ffmpeg 을 실행할 수 없다 ({FFMPEG_PATH}): {_ff_err} — "
                       "FFMPEG_BIN 과 conda LD_LIBRARY_PATH 함정(ENVIRONMENT.md 함정 10)을 확인할 것")

# 음성 유무 게이트. 무음·거의 무음이어도 모델은 그럴듯한 국적을 지어낸다(무음 3초 → Colombia 59%, 실측).
# 30 ms 프레임 RMS 가 GATE_RMS 이상인 구간이 GATE_MIN_SEC 이상일 때만 추론한다. GATE_RMS 1.5e-3 은
# 발음 진단(mdd.pipeline.diagnose)이 발화 전체에 쓰는 무음 기준값이다.
# 에너지만 보므로 큰 소음·음악은 못 거른다 — 그건 VAD 가 필요하다.
GATE_RMS = float(os.environ.get("KD_GATE_RMS", "1.5e-3"))
GATE_MIN_SEC = float(os.environ.get("KD_GATE_MIN_SEC", "0.3"))
SILENCE_NORM = os.environ.get("KD_SILENCE_NORM", "0") == "1"   # 앞뒤 무음을 고정 룸톤으로 — 녹음 도구의 무음 처리가 판정에 섞이지 않게 (results/11_채널편향_한국어.md)
if SILENCE_NORM:
    try:
        from country_classifier.wav_augment import silence_normalize
    except ImportError:                                  # NPU/CPU 번들 단독 실행: app.py 옆의 wav_augment.py
        _sys.path.insert(0, HERE); from wav_augment import silence_normalize
SAVE_DIR = os.environ.get("KD_SAVE_DIR")                 # 지정하면 업로드를 저장한다 (디버그 전용 · 기본 꺼짐)
# 지정하면 /predict · /predict_long 으로 들어온 녹음 전부(보낸 파일 그대로)와 판정·품질 지표를 남긴다 — 실사용 오판 분석용(2026-10-09~).
# 날짜 폴더(YYYYMMDD)에 <rid>.<확장자> + captures.jsonl. 응답에 rid 가 붙어 웹·앱이 문제 사례를 번호로 짚을 수 있다.
CAPTURE_DIR = os.environ.get("KD_CAPTURE_DIR")
if SAVE_DIR:
    os.makedirs(SAVE_DIR, exist_ok=True)

# ── 모델 로드 ──
# NAT_BACKEND=onnx(기본): 동적 길이 ONNX 세션 풀(CPU/CUDA). NAT_BACKEND=rbln: Rebellions NPU — 고정 12초 + 프레임 마스크로 컴파일한
# .rbln(KD_RBLN_PATH, 기본 <모델폴더>/encoder_kd_l17_static12s.rbln)을 rebel 런타임으로 돈다(results/13_NPU_서빙_준비.md).
# KD_GATE_DIR: '원어민 게이트' — 두 번째 모델(웹마이크를 안 본 기존 배포 모델)의 Korea 확률이 KD_GATE_TAU 를 넘으면 Korea 를 1위로 둔다.
# 노트북·브라우저 녹음의 한국인 오판을 막는 장치(11_채널편향_한국어.md, 시험지에서 외국인 top-1 −1점 · 한국인 ≥0.997).
BACKEND = os.environ.get("NAT_BACKEND", "onnx").strip().lower()
GATE_DIR = os.environ.get("KD_GATE_DIR", "")
GATE_TAU = float(os.environ.get("KD_GATE_TAU", "0.9"))
# 게이트는 **정규화 전(raw) 파형**을 본다. 정규화된 입력을 주면 기존 배포 모델이 외국인의 짧은 발화를 Korea 로 부른다
# (부스 시험지 1초 발화: 외국인 84% 를 Korea 로 — 2026-09-24 실측, results/14_부스_조건.md).
GATE_INPUT = os.environ.get("KD_GATE_INPUT", "raw").strip().lower()
# Korea 문턱(기본 꺼짐): 주 모델의 Korea 확률이 이 값을 넘으면 1위로. 관객 대부분이 한국인인 부스용 보정.
KOREA_THR = float(os.environ.get("KD_KOREA_THR", "0") or 0)
# 저신뢰 표시(기본 꺼짐): 1위 확률이 이 값보다 낮으면 low_confidence=true — 화면에서 "조금 더 길게 다시" 를 띄운다.
LOWCONF = float(os.environ.get("KD_LOWCONF", "0") or 0)
STATIC_SEC = 12
BUCKETS = sorted(int(x) for x in os.environ.get("RBLN_BUCKETS", "12").split(","))


def _conv_out_len(n):
    """샘플 수 → wav2vec2 conv 프레임 수 (커널 10,3,3,3,3,2,2 · 스트라이드 5,2,2,2,2,2,2)."""
    for k, st in zip((10, 3, 3, 3, 3, 2, 2), (5, 2, 2, 2, 2, 2, 2)):
        n = (n - k) // st + 1
    return n


class Encoder:
    """파형 → pooled(1,1024) + 헤드. 세션 풀(POOL)로 동접을 받는다."""

    def __init__(self, model_dir, backend, rbln_path=""):
        self.backend = backend
        self.model_dir = model_dir
        self.q = queue.Queue()
        h = np.load(os.path.join(model_dir, "head_np.npz"), allow_pickle=True)
        self.W, self.b, self.mu, self.sd = h["W"], h["b"], h["mu"], h["sd"]
        self.labels = [str(x) for x in h["labels"]]
        if backend == "onnx":
            import onnxruntime as ort
            for _ in range(POOL):
                so = ort.SessionOptions(); so.intra_op_num_threads = THREADS
                sess = ort.InferenceSession(os.path.join(model_dir, "encoder_kd_l17.onnx"), so, providers=PROVIDERS)
                # CUDA 를 못 올려도 onnxruntime 은 경고만 찍고 CPU 로 떨어진다. 조용히 느려지지 않게 여기서 멈춘다.
                if DEVICE == "cuda" and "CUDAExecutionProvider" not in sess.get_providers():
                    raise RuntimeError("KD_DEVICE=cuda 인데 CUDA provider 를 못 올렸다 — onnxruntime-gpu 설치와 "
                                       "LD_LIBRARY_PATH(CUDA 11.8 라이브러리 링크 폴더)를 확인할 것")
                self.q.put(sess)
            self.provider = sess.get_providers()[0]
        elif backend == "rbln":
            import rebel                                     # Rebellions SDK (NPU 서버에서만)
            path = rbln_path or os.path.join(model_dir, "encoder_pure_l17_static12s.rbln")
            devs = [int(x) for x in os.environ.get("RBLN_DEVICES", "0").split(",")]
            import fcntl                                     # 워커·인코더마다 장치를 차례로 (여러 프로세스가 같은 카운터를 공유)
            with open(os.environ.get("RBLN_COUNTER", "/tmp/nat_rbln_counter"), "a+") as cf:
                fcntl.flock(cf, fcntl.LOCK_EX); cf.seek(0); k = int(cf.read().strip() or 0)
                cf.seek(0); cf.truncate(); cf.write(str(k + POOL)); fcntl.flock(cf, fcntl.LOCK_UN)
            for i in range(POOL):
                self.q.put(rebel.Runtime(path, device=devs[(k + i) % len(devs)], tensor_type="np"))
            self.provider = f"RBLN{devs}"
        else:
            raise SystemExit(f"NAT_BACKEND 는 onnx 또는 rbln 이어야 한다: {backend!r}")

    def pooled(self, wav):
        s = self.q.get()
        try:
            if self.backend == "onnx":
                return s.run(["pooled"], {"source": wav[None, :]})[0]
            # 길이 버킷(RBLN_BUCKETS, 초): 입력보다 긴 가장 작은 버킷으로 0 채움. 12초 하나면 예전과 같다.
            # 4·8·12초 버킷이면 짧은 발화의 NPU 계산이 1/3 로 준다(실측 20 · 39 · 59 ms).
            n0 = min(wav.size, SR * STATIC_SEC)
            N = next((b * SR for b in BUCKETS if b * SR >= n0), SR * BUCKETS[-1])
            src = np.zeros((1, N), np.float32); valid = np.zeros((1, _conv_out_len(N)), np.float32)
            n = min(wav.size, N); src[0, :n] = wav[:n]; valid[0, :_conv_out_len(n)] = 1.0
            return np.asarray(s.run(src, valid), np.float32).reshape(1, -1)
        finally:
            self.q.put(s)

    def probs(self, wav):
        x = np.clip(np.nan_to_num(self.pooled(wav)), -50, 50)
        z = ((x - self.mu) / self.sd) @ self.W.T + self.b
        p = np.exp(z - z.max()); p /= p.sum()
        context = capture_context.get()
        if context is not None:
            context.setdefault("encoder_outputs", []).append({
                "model_dir": self.model_dir, "labels": self.labels,
                "probabilities": p[0].tolist(), "input_sec": wav.size / SR,
            })
        return p[0]


print(f"[load] {BACKEND} 세션 {POOL}개 × {THREADS}스레드 · {DEVICE} 로딩...", flush=True)
_t0 = time.time()
MAIN = Encoder(MODEL_DIR, BACKEND, os.environ.get("KD_RBLN_PATH", ""))
LABELS = MAIN.labels
GATE = Encoder(GATE_DIR, BACKEND, os.environ.get("KD_GATE_RBLN_PATH", "")) if GATE_DIR else None
print(f"[load] 완료 {time.time()-_t0:.0f}s · {len(LABELS)}클래스 · {MAIN.provider}", flush=True)
print(f"[load] 무음 정규화(KD_SILENCE_NORM) {'켜짐' if SILENCE_NORM else '꺼짐'} · 원어민 게이트 {GATE_DIR or '없음'}"
      + (f" (τ={GATE_TAU}, 입력 {GATE_INPUT})" if GATE_DIR else "") + f" · Korea 문턱 {KOREA_THR or '꺼짐'} · 최소 유성 {GATE_MIN_SEC}s", flush=True)

_stats = {"requests": 0, "errors": 0, "busy": 0}


class AudioError(ValueError):
    """요청 오디오를 쓸 수 없다 — 클라이언트 오류(400)로 돌려준다."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def decode(raw_bytes, max_sec=12):
    # 업로드를 임시 파일로 넘긴다. 파이프로 넘기면 목차(moov)가 파일 끝에 있는 m4a(ffmpeg 기본 출력 구조)를
    # 약 50 KB 부터 에러 없이 0초로 읽는다(실측: 12초 m4a → "너무 짧다" 400).
    with tempfile.NamedTemporaryFile(suffix=".audio") as tf:
        tf.write(raw_bytes)
        tf.flush()
        p = subprocess.run([FFMPEG_PATH, "-hide_banner", "-loglevel", "error", "-i", tf.name,
                            "-f", "f32le", "-ac", "1", "-ar", str(SR), "-t", str(max_sec), "pipe:1"],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=False)
    if p.returncode != 0:
        err = p.stderr.decode("utf-8", "replace").strip()
        # 반환코드를 봐야 한다. ffmpeg 자체가 못 도는 경우는 오디오 탓이 아니다
        # (conda LD_LIBRARY_PATH 함정 — ENVIRONMENT.md 함정 10).
        if p.returncode in (126, 127) or "error while loading shared libraries" in err or "symbol lookup error" in err:
            raise RuntimeError(f"ffmpeg 실행 실패 (rc={p.returncode}): {err[:500]}")
        print(f"[decode] 해석 불가 입력 (rc={p.returncode}): {err[:300]}", flush=True)
        raise AudioError("decode_failed", "오디오 파일을 해석할 수 없다")
    return np.frombuffer(p.stdout, np.float32).copy()


def rms_normalize(wav, target):
    """유성 구간 RMS 를 target 에 맞춘다. 무음·과증폭은 건드리지 않는다."""
    v = wav[np.abs(wav) > 1e-4]
    cur = float(np.sqrt(np.square(v, dtype=np.float64).mean())) if v.size else 0.0
    if cur < 1e-6:
        return wav, 1.0
    g = float(np.clip(target / cur, 0.1, 40.0))
    return np.clip(wav * g, -1.0, 1.0).astype(np.float32), g


def voiced_sec(wav, frame_sec=0.03):
    """프레임 RMS 가 GATE_RMS 이상인 구간의 합(초)."""
    f = int(SR * frame_sec)
    n = wav.size // f
    if n == 0:
        return 0.0
    rms = np.sqrt(np.square(wav[: n * f].reshape(n, f), dtype=np.float64).mean(axis=1))
    return float((rms >= GATE_RMS).sum() * frame_sec)


def audio_stats(wav):
    """녹음 품질 지표 — 음량(dBFS) · 최대값 · 스펙트럼 평탄도(소리 큰 프레임의 중앙값. 잡음은 1 에 가깝고 말소리는 보통 0.1 미만)."""
    if wav is None or wav.size == 0:
        return {}
    rms = float(np.sqrt(np.square(wav, dtype=np.float64).mean()))
    out = {"rms_dbfs": round(20 * float(np.log10(max(rms, 1e-9))), 1), "peak": round(float(np.abs(wav).max()), 3)}
    f = 512; n = wav.size // f
    if n:
        spec = np.abs(np.fft.rfft(wav[: n * f].reshape(n, f) * np.hanning(f), axis=1)) ** 2 + 1e-12
        flat = np.exp(np.log(spec).mean(axis=1)) / spec.mean(axis=1)
        e = spec.sum(axis=1)
        out["flatness"] = round(float(np.median(flat[e >= np.percentile(e, 50)])), 3)
    return out


def capture(kind, raw, filename, req, result, status, wav=None, voiced=None):
    """KD_CAPTURE_DIR 이 있으면 들어온 녹음(보낸 바이트 그대로)과 판정을 남기고 rid 를 돌려준다. 실패해도 API 는 그대로 동작한다."""
    if not CAPTURE_DIR:
        return None
    context = capture_context.get()
    if context is not None:
        context["kind"] = kind
        if wav is not None:
            context["audio_quality"] = {"sec": round(wav.size / SR, 2), **audio_stats(wav)}
        if voiced is not None:
            context["voiced_sec"] = round(float(voiced), 1)
        return context["rid"]
    try:
        rid = time.strftime("%Y%m%d-%H%M%S") + "-" + os.urandom(3).hex()
        day = os.path.join(CAPTURE_DIR, rid[:8])
        os.makedirs(day, mode=0o700, exist_ok=True)
        ext = (os.path.splitext(filename or "")[1].lower() or ".bin")[:6]
        if len(raw) <= 8 * 1024 * 1024:
            fd = os.open(os.path.join(day, rid + ext), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                os.write(fd, raw)
            finally:
                os.close(fd)
        rec = {"rid": rid, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "kind": kind, "status": status, "bytes": len(raw),
               "filename": filename, **req}
        if wav is not None:
            rec["sec"] = round(wav.size / SR, 2)
            rec.update(audio_stats(wav))
        if voiced is not None:
            rec["voiced_sec"] = round(float(voiced), 1)
        rec["result"] = result
        line = (json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8")
        fd = os.open(os.path.join(day, "captures.jsonl"), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, line)                               # 한 줄을 한 번에 쓴다 — 여러 워커가 같이 써도 줄이 섞이지 않는다
        finally:
            os.close(fd)
        return rid
    except Exception as e:                                   # 저장 실패는 판정에 영향 없이 로그만
        print(f"[capture] 저장 실패: {e}", flush=True)
        return None


def _korea_first(p):
    """Korea 를 1위로 올린다(나머지 순서는 유지)."""
    ki = LABELS.index("Korea"); p = p.copy(); p[ki] = p.max() + 1e-6
    return p / p.sum()


def classify(wav, wav_gate=None):
    _stats["busy"] += 1
    try:
        t = time.perf_counter()
        p = MAIN.probs(wav)
        ms = (time.perf_counter() - t) * 1000
        ki = LABELS.index("Korea")
        if KOREA_THR > 0 and p[ki] > KOREA_THR:
            p = _korea_first(p)
        gate_p = None
        if GATE is not None:
            gw = wav_gate if (GATE_INPUT == "raw" and wav_gate is not None) else wav
            pg = GATE.probs(gw)
            if GATE.labels != LABELS:                        # 라벨 순서를 주 모델에 맞춘다
                pg = np.array([pg[GATE.labels.index(l)] for l in LABELS])
            gate_p = float(pg[ki])
            # 한국인으로 판정되면(게이트 통과 또는 주 모델 1위가 Korea) 두 모델 중 Korea 를 더 확신하는 쪽의 분포를 그대로 보여 준다.
            # 예전에는 주 모델 분포에서 Korea 만 겨우 1위로 올려 "Korea 48% · United States 48%" 같은 화면이 나왔다(2026-09-25).
            if gate_p > GATE_TAU or int(np.argmax(p)) == ki:
                if pg[ki] > p[ki]:
                    p = pg
                if int(np.argmax(p)) != ki:
                    p = _korea_first(p)
    finally:
        _stats["busy"] -= 1
    top = np.argsort(-p)[:5]
    return [{"label": LABELS[i], "prob": round(float(p[i]), 4)} for i in top], ms, gate_p


# ── 여러 문장 합산 (/predict_multi) — 같은 사람이 읽은 문장 N개의 판정을 합친다 ──
# results/21_다문장_합산_TTA.md: 시험 화자 1,007명 기준 1위 정답 1문장 77% → 3문장 86% → 5문장 88%. log 확률 평균이 가장 좋았다.
def _gated(p, pg):
    """classify() 와 같은 원어민 게이트 규칙 — p = 주 모델 분포, pg = 게이트 모델 분포(주 모델 라벨 순서)."""
    ki = LABELS.index("Korea")
    if float(pg[ki]) > GATE_TAU or int(np.argmax(p)) == ki:
        if pg[ki] > p[ki]:
            p = pg
        if int(np.argmax(p)) != ki:
            p = _korea_first(p)
    return p


def _mean_log(ps):
    lp = np.log(np.clip(np.stack(ps), 1e-12, 1.0)).mean(axis=0)
    q = np.exp(lp - lp.max())
    return q / q.sum()


def classify_multi(wavs, wavs_raw):
    """반환: (합산 top5, 문장별 1위 [{label, prob}], 인코더 ms, 합산 게이트 Korea 확률)."""
    ki = LABELS.index("Korea")
    pms, pgs, per = [], [], []
    _stats["busy"] += 1
    try:
        t = time.perf_counter()
        for w, wr in zip(wavs, wavs_raw):
            pm = MAIN.probs(w)
            pms.append(pm)
            p1, pg = pm, None
            if KOREA_THR > 0 and p1[ki] > KOREA_THR:
                p1 = _korea_first(p1)
            if GATE is not None:
                pg = GATE.probs(wr if (GATE_INPUT == "raw" and wr is not None) else w)
                if GATE.labels != LABELS:
                    pg = np.array([pg[GATE.labels.index(l)] for l in LABELS])
                pgs.append(pg)
                p1 = _gated(p1, pg)
            i = int(np.argmax(p1))
            per.append({"label": LABELS[i], "prob": round(float(p1[i]), 4)})
        ms = (time.perf_counter() - t) * 1000
    finally:
        _stats["busy"] -= 1
    p = _mean_log(pms)
    if KOREA_THR > 0 and p[ki] > KOREA_THR:
        p = _korea_first(p)
    gate_p = None
    if pgs:
        pg = _mean_log(pgs)
        gate_p = float(pg[ki])
        p = _gated(p, pg)
    return _top5(p), per, ms, gate_p


# ── 긴 녹음 1개 (/predict_long) — 한 번에 이어 읽은 녹음을 서버가 같은 길이로 나눠 판정한 뒤 합친다 ──
# results/23_긴녹음_분할_합산.md: 문장 3개를 따로 녹음해 합산한 것과 정확도가 같다(1위 0.865 vs 0.868). /predict 는 앞 12초만 본다.
LONG_MAX_SEC = float(os.environ.get("KD_LONG_MAX_SEC", "60"))
LONG_PARTS = int(os.environ.get("KD_LONG_PARTS", "3"))


def split_parts(sec, parts=LONG_PARTS, min_part=3.0, max_part=STATIC_SEC):
    """나눌 구간 수 — 기본 parts 개. 구간이 min_part 초보다 짧아지면 줄이고, max_part(NPU 고정 길이 12초)를 넘으면 늘린다."""
    n = parts if sec >= parts * min_part else max(1, int(sec // min_part))
    return max(n, int(np.ceil(sec / max_part)))


# ── 버전 비교 (/compare): 한 번의 추론으로 기존 서비스 · 기준값(τ)별 새 구성의 판정을 나란히 ──
COMPARE_TAUS = [float(x) for x in os.environ.get("KD_COMPARE_TAUS", "0.9,0.7,0.5").split(",")]


def compare_variants(wav, wav_raw):
    """반환: [(키, 확률분포, low_confidence)] — old = 확인 모델(기존 배포 모델) 단독 · tauX = 새 구성."""
    if GATE is None:
        raise RuntimeError("비교에는 KD_GATE_DIR(기존 모델)이 필요하다")
    ki = LABELS.index("Korea")
    pm = MAIN.probs(wav)
    pg = GATE.probs(wav_raw if wav_raw is not None else wav)
    if GATE.labels != LABELS:
        pg = np.array([pg[GATE.labels.index(l)] for l in LABELS])
    out = [("old", pg, False)]
    for tau in COMPARE_TAUS:
        p = pm
        if pg[ki] > tau or int(np.argmax(pm)) == ki:
            p = pg if pg[ki] > pm[ki] else pm
            if int(np.argmax(p)) != ki:
                p = _korea_first(p)
        low = bool(LOWCONF > 0 and int(np.argmax(p)) != ki and float(p.max()) < LOWCONF)
        out.append((f"tau{tau}", p, low))
    return out


def _top5(p):
    return [{"label": LABELS[i], "prob": round(float(p[i]), 4)} for i in np.argsort(-p)[:5]]


# ── 샘플 오디오 (부하 테스트·원클릭 데모용) ──
# 학습 서버의 절대경로가 박혀 있었다. 없으면 그냥 빈 목록이고 데모 본체는 정상 동작한다.
_SD = os.environ.get("SAMPLE_WAV_DIR", "")
SAMPLES = sorted(glob.glob(os.path.join(_SD, "*.wav")))[:10] if _SD else []


def make_app():
    from typing import List
    from fastapi import FastAPI, UploadFile, File, Form, APIRouter, Request
    from fastapi.responses import HTMLResponse, Response, JSONResponse, FileResponse
    from starlette.concurrency import run_in_threadpool
    app = FastAPI(title="nationality-kd-demo")
    _cors = [o.strip() for o in os.environ.get("KD_CORS_ORIGINS", "").split(",") if o.strip()]
    if _cors:                                                # 다른 도메인의 웹 페이지가 브라우저에서 직접 부를 때만 (앱·서버 호출은 필요 없음)
        from fastapi.middleware.cors import CORSMiddleware
        app.add_middleware(CORSMiddleware, allow_origins=_cors, allow_methods=["GET", "POST"], allow_headers=["*"])
    r = APIRouter()

    @r.get("/", response_class=HTMLResponse)
    def index():
        html = open(os.path.join(HERE, "index.html"), encoding="utf-8").read()
        if INDEX_EXTRA_JS and os.path.isfile(INDEX_EXTRA_JS):     # 인스턴스별 덧붙임(기본은 없음)
            html += "\n<script>" + open(INDEX_EXTRA_JS, encoding="utf-8").read() + "</script>"
        return html

    # ── 공개 낭독 페이지 (/read): 문장 10개를 읽고 판정을 본다. 동의한 녹음만 저장한다 ──
    @r.get("/read", response_class=HTMLResponse)
    def read_page():
        return open(os.path.join(HERE, "read", "index.html"), encoding="utf-8").read()

    @r.get("/read/{name}")
    def read_asset(name: str):
        if name not in ("logo.png", "og.jpg"):
            return JSONResponse({"error": "not found"}, status_code=404)
        return FileResponse(os.path.join(HERE, "read", name), headers={"Cache-Control": "public, max-age=86400"})

    # 버전 비교 화면은 compare/ 폴더가 있는 서버(GPU 데모)에서만 연다 — NPU 번들에는 없다(없으면 세 경로 모두 404)
    has_compare = os.path.isfile(os.path.join(HERE, "compare", "index.html"))

    @r.get("/compare", response_class=HTMLResponse)
    def compare_page():
        if not has_compare:
            return JSONResponse({"error": "not found"}, status_code=404)
        return open(os.path.join(HERE, "compare", "index.html"), encoding="utf-8").read()

    @r.post("/compare/predict")
    async def compare_predict(audio: UploadFile = File(...), consent: str = Form(None), sentence: str = Form(None), session: str = Form(None)):
        if not has_compare:
            return JSONResponse({"error": "not found"}, status_code=404)
        _stats["requests"] += 1
        raw = await audio.read()
        if len(raw) > 8 * 1024 * 1024:
            return JSONResponse({"error": "파일이 너무 크다 (8 MB 초과)", "code": "too_large"}, status_code=400)
        try:
            wav = await run_in_threadpool(decode, raw)
        except AudioError as e:
            return JSONResponse({"error": str(e), "code": e.code}, status_code=400)
        if wav.size < SR // 2:
            return JSONResponse({"error": "오디오가 너무 짧다 (0.5초 미만)", "code": "too_short"}, status_code=400)
        if await run_in_threadpool(voiced_sec, wav) < GATE_MIN_SEC:
            return JSONResponse({"error": "음성이 감지되지 않았다 (무음이거나 소리가 너무 작다)", "code": "no_speech"}, status_code=400)
        stem = None
        if SAVE_DIR and consent == "1":
            import wave as _wave
            stem = time.strftime("%Y%m%d_%H%M%S") + f"_cmp{_stats['requests']}"
            open(os.path.join(SAVE_DIR, stem + ".raw"), "wb").write(raw)
            with _wave.open(os.path.join(SAVE_DIR, stem + ".wav"), "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
                w.writeframes((np.clip(wav, -1, 1) * 32767).astype(np.int16).tobytes())
        wav_raw = wav
        if SILENCE_NORM:
            wav = await run_in_threadpool(silence_normalize, wav, GATE_RMS)
        vs = await run_in_threadpool(compare_variants, wav, wav_raw)
        res = [{"key": k, "top5": _top5(p), "native": LABELS[int(np.argmax(p))] == "Korea", "low_confidence": low} for k, p, low in vs]
        if stem:
            with open(os.path.join(SAVE_DIR, "compare.jsonl"), "a", encoding="utf-8") as mf:
                mf.write(json.dumps({"stem": stem, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "session": session, "sentence": sentence,
                                     "variants": res}, ensure_ascii=False) + "\n")
        return {"variants": res, "sec": round(wav_raw.size / SR, 2), "stem": stem}

    @r.post("/compare/vote")
    async def compare_vote(session: str = Form(""), choice: str = Form(""), stem: str = Form(""), letter: str = Form(""), note: str = Form("")):
        if not has_compare:
            return JSONResponse({"error": "not found"}, status_code=404)
        if SAVE_DIR:
            with open(os.path.join(SAVE_DIR, "votes.jsonl"), "a", encoding="utf-8") as vf:
                vf.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "session": session, "choice": choice, "letter": letter,
                                     "stem": stem, "note": note}, ensure_ascii=False) + "\n")
        return {"ok": True}

    @r.get("/health")
    def health():
        return {"ok": True, "model": MODEL_INFO["name"], "summary": MODEL_INFO["summary"], "device": DEVICE, "pool": POOL,
                "threads": THREADS, "classes": len(LABELS), "backend": BACKEND, "provider": MAIN.provider,
                "silence_norm": SILENCE_NORM, "gate": bool(GATE), "gate_tau": GATE_TAU if GATE else None,
                "gate_input": GATE_INPUT if GATE else None, "korea_thr": KOREA_THR or None, "lowconf": LOWCONF or None,
                "min_voiced_sec": GATE_MIN_SEC, **_stats}

    @r.post("/predict")
    async def predict(request: Request, audio: UploadFile = File(...), consent: str = Form(None), sentence_id: str = Form(None),
                      sentence: str = Form(None), self_report: str = Form(None), session: str = Form(None)):
        _stats["requests"] += 1
        t0 = time.perf_counter()
        raw = await audio.read()
        req = {"ua": request.headers.get("user-agent", "")[:80], "host": request.headers.get("host", ""), "consent": consent,
               "sentence_id": sentence_id, "sentence": sentence, "self_report": self_report, "session": session}

        def done(payload, status=200, w=None, v=None):          # 응답 직전에 녹음 · 판정을 남긴다(KD_CAPTURE_DIR)
            rid = capture("predict", raw, audio.filename, req, payload, status, w, v)
            if rid:
                payload = {**payload, "rid": rid}
            return payload if status == 200 else JSONResponse(payload, status_code=status)

        if len(raw) > 8 * 1024 * 1024:
            _stats["errors"] += 1
            return done({"error": "파일이 너무 크다 (8 MB 초과)", "code": "too_large"}, 400)
        stem = None
        try:
            wav = await run_in_threadpool(decode, raw)
            if SAVE_DIR and consent != "0":                # 업로드 저장 (KD_SAVE_DIR). 낭독 페이지는 동의(consent=1)한 경우만 온다
                import wave as _wave
                stem = os.path.join(SAVE_DIR, time.strftime("%Y%m%d_%H%M%S") + f"_{_stats['requests']}")
                open(stem + ".raw", "wb").write(raw)
                with _wave.open(stem + ".wav", "wb") as w:
                    w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
                    w.writeframes((np.clip(wav, -1, 1) * 32767).astype(np.int16).tobytes())
        except AudioError as e:        # 오디오가 아니거나 깨진 파일 — 요청 쪽 문제
            _stats["errors"] += 1
            return done({"error": str(e), "code": e.code}, 400)
        except RuntimeError as e:      # ffmpeg 자체가 못 도는 경우 — 오디오 탓이 아니다. 내부 메시지는 로그에만
            _stats["errors"] += 1
            print(f"[decode] {e}", flush=True)
            return done({"error": "서버 오디오 디코더 오류", "code": "server_decoder"}, 500)
        if wav.size < SR // 2:
            _stats["errors"] += 1
            return done({"error": "오디오가 너무 짧다 (0.5초 미만)", "code": "too_short"}, 400, wav)
        voiced = await run_in_threadpool(voiced_sec, wav)
        if voiced < GATE_MIN_SEC:
            _stats["errors"] += 1
            return done({"error": "음성이 감지되지 않았다 (무음이거나 소리가 너무 작다)", "code": "no_speech"}, 400, wav, voiced)
        wav_raw = wav                                        # 게이트는 정규화 전 파형 (KD_GATE_INPUT=raw)
        if SILENCE_NORM:
            wav = await run_in_threadpool(silence_normalize, wav, GATE_RMS)
        rms = float(np.sqrt(np.square(wav, dtype=np.float64).mean())) if wav.size else 0.0
        gain = 1.0
        if RMS_NORM > 0:
            wav, gain = await run_in_threadpool(rms_normalize, wav, RMS_NORM)
        top5, enc_ms, gate_p = await run_in_threadpool(classify, wav, wav_raw)
        if stem and (consent or sentence_id):              # 낭독 페이지 메타 (문장 · 자기 보고 · 세션 · 판정)
            with open(os.path.join(SAVE_DIR, "meta.jsonl"), "a", encoding="utf-8") as mf:
                mf.write(json.dumps({"stem": os.path.basename(stem), "time": time.strftime("%Y-%m-%d %H:%M:%S"), "consent": consent,
                                     "sentence_id": sentence_id, "sentence": sentence, "self_report": self_report, "session": session,
                                     "top5": top5, "gate_korea": gate_p}, ensure_ascii=False) + "\n")
        return done({"top5": top5, "sec": round(wav.size / SR, 2), "rms": round(rms, 4), "gain": round(gain, 2), "silence_norm": SILENCE_NORM,
                     "gate_korea": gate_p, "native": top5[0]["label"] == "Korea",
                     "low_confidence": bool(LOWCONF > 0 and top5[0]["label"] != "Korea" and top5[0]["prob"] < LOWCONF),   # 외국 국적인데 확신 낮음 → 재요청
                     "encode_ms": round(enc_ms), "total_ms": round((time.perf_counter() - t0) * 1000)}, 200, wav_raw, voiced)

    # ── 여러 문장 합산 테스트 (/multi 화면 · /predict_multi API) — 테스트용이라 녹음을 저장하지 않는다 ──
    @r.get("/multi", response_class=HTMLResponse)
    def multi_page():
        return open(os.path.join(HERE, "multi", "index.html"), encoding="utf-8").read()

    @r.post("/predict_multi")
    async def predict_multi(audio: List[UploadFile] = File(...)):
        _stats["requests"] += 1
        t0 = time.perf_counter()
        if not 1 <= len(audio) <= 10:
            _stats["errors"] += 1
            return JSONResponse({"error": "녹음은 1~10개를 보내야 한다", "code": "bad_count"}, status_code=400)
        wavs, raws, secs = [], [], []
        for k, a in enumerate(audio):
            raw = await a.read()
            if len(raw) > 8 * 1024 * 1024:
                _stats["errors"] += 1
                return JSONResponse({"error": f"{k + 1}번째 파일이 너무 크다 (8 MB 초과)", "code": "too_large", "index": k}, status_code=400)
            try:
                wav = await run_in_threadpool(decode, raw)
            except AudioError as e:
                _stats["errors"] += 1
                return JSONResponse({"error": f"{k + 1}번째: {e}", "code": e.code, "index": k}, status_code=400)
            except RuntimeError as e:
                _stats["errors"] += 1
                print(f"[decode] {e}", flush=True)
                return JSONResponse({"error": "서버 오디오 디코더 오류", "code": "server_decoder"}, status_code=500)
            if wav.size < SR // 2:
                _stats["errors"] += 1
                return JSONResponse({"error": f"{k + 1}번째 녹음이 너무 짧다 (0.5초 미만)", "code": "too_short", "index": k}, status_code=400)
            if await run_in_threadpool(voiced_sec, wav) < GATE_MIN_SEC:
                _stats["errors"] += 1
                return JSONResponse({"error": f"{k + 1}번째 녹음에서 음성이 감지되지 않았다", "code": "no_speech", "index": k}, status_code=400)
            wav_raw = wav
            if SILENCE_NORM:
                wav = await run_in_threadpool(silence_normalize, wav, GATE_RMS)
            if RMS_NORM > 0:
                wav, _ = await run_in_threadpool(rms_normalize, wav, RMS_NORM)
            wavs.append(wav); raws.append(wav_raw); secs.append(round(wav_raw.size / SR, 2))
        top5, per, enc_ms, gate_p = await run_in_threadpool(classify_multi, wavs, raws)
        for d, s in zip(per, secs):
            d["sec"] = s
        return {"n": len(wavs), "top5": top5, "per_sentence": per, "gate_korea": gate_p, "native": top5[0]["label"] == "Korea",
                "low_confidence": bool(LOWCONF > 0 and top5[0]["label"] != "Korea" and top5[0]["prob"] < LOWCONF),
                "encode_ms": round(enc_ms), "total_ms": round((time.perf_counter() - t0) * 1000)}

    @r.get("/long", response_class=HTMLResponse)
    def long_page():
        return open(os.path.join(HERE, "long", "index.html"), encoding="utf-8").read()

    @r.post("/predict_long")
    async def predict_long(request: Request, audio: UploadFile = File(...), parts: str = Form(None)):
        _stats["requests"] += 1
        t0 = time.perf_counter()
        wav = voiced = None
        try:                                                 # 나눌 구간 수 — 화면이 고른 문장 수(1·3·5). 없거나 이상하면 기본값
            n_req = min(10, max(1, int(parts))) if parts else LONG_PARTS
        except ValueError:
            n_req = LONG_PARTS
        raw = await audio.read()
        req = {"ua": request.headers.get("user-agent", "")[:80], "host": request.headers.get("host", ""), "parts": parts}

        def done(payload, status=200):                       # 응답 직전에 녹음 · 판정을 남긴다(KD_CAPTURE_DIR)
            rid = capture("predict_long", raw, audio.filename, req, payload, status, wav, voiced)
            if rid:
                payload = {**payload, "rid": rid}
            return payload if status == 200 else JSONResponse(payload, status_code=status)

        if len(raw) > 8 * 1024 * 1024:
            _stats["errors"] += 1
            return done({"error": "파일이 너무 크다 (8 MB 초과)", "code": "too_large"}, 400)
        try:
            wav = await run_in_threadpool(decode, raw, LONG_MAX_SEC)
        except AudioError as e:
            _stats["errors"] += 1
            return done({"error": str(e), "code": e.code}, 400)
        except RuntimeError as e:
            _stats["errors"] += 1
            print(f"[decode] {e}", flush=True)
            return done({"error": "서버 오디오 디코더 오류", "code": "server_decoder"}, 500)
        if wav.size < SR // 2:
            _stats["errors"] += 1
            return done({"error": "오디오가 너무 짧다 (0.5초 미만)", "code": "too_short"}, 400)
        voiced = await run_in_threadpool(voiced_sec, wav)
        n = split_parts(wav.size / SR, n_req)
        cuts = [round(j * wav.size / n) for j in range(n + 1)]
        wavs, raws, spans = [], [], []
        for a, b in zip(cuts[:-1], cuts[1:]):
            ch = wav[a:b]
            if ch.size < SR // 2 or await run_in_threadpool(voiced_sec, ch) < GATE_MIN_SEC:
                continue                                     # 말소리가 없는 구간(긴 쉼)은 건너뛴다
            w = ch
            if SILENCE_NORM:
                w = await run_in_threadpool(silence_normalize, w, GATE_RMS)
            if RMS_NORM > 0:
                w, _ = await run_in_threadpool(rms_normalize, w, RMS_NORM)
            wavs.append(w); raws.append(ch); spans.append((round(a / SR, 1), round(b / SR, 1)))
        if not wavs:
            _stats["errors"] += 1
            return done({"error": "음성이 감지되지 않았다 (무음이거나 소리가 너무 작다)", "code": "no_speech"}, 400)
        top5, per, enc_ms, gate_p = await run_in_threadpool(classify_multi, wavs, raws)
        for d, (s0, s1) in zip(per, spans):
            d["start"], d["end"] = s0, s1
        return done({"n": len(wavs), "parts_requested": n_req, "sec": round(wav.size / SR, 2), "voiced_sec": round(voiced, 1),
                     "top5": top5, "per_part": per, "gate_korea": gate_p,
                     "native": top5[0]["label"] == "Korea",
                     "low_confidence": bool(LOWCONF > 0 and top5[0]["label"] != "Korea" and top5[0]["prob"] < LOWCONF),
                     "encode_ms": round(enc_ms), "total_ms": round((time.perf_counter() - t0) * 1000)})

    @r.get("/samples")
    def samples():
        return {"samples": [os.path.basename(s) for s in SAMPLES]}

    @r.get("/sample/{idx}")
    def sample(idx: int):
        if not SAMPLES:      # SAMPLE_WAV_DIR 미설정 — 0 나누기로 죽지 않게
            return JSONResponse({"error": "샘플 오디오가 없다. SAMPLE_WAV_DIR 을 설정할 것"},
                                status_code=404)
        return Response(open(SAMPLES[idx % len(SAMPLES)], "rb").read(), media_type="audio/wav")

    app.include_router(r)
    app.include_router(r, prefix="/kd")      # funnel --set-path 는 프리픽스를 벗기지 않는다
    return capture_app(app, "nationality")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8905)
    ap.add_argument("--host", default="0.0.0.0")
    a = ap.parse_args()
    import uvicorn
    uvicorn.run(make_app(), host=a.host, port=a.port)
