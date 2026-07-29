# XLS-R 2B 국적분류 — 속도/동접 데모 (이식 패키지)

GPU(fp16) vs CPU(fp32) vs CPU(int8) 추론 속도 비교 + 동접 부하 테스트 웹 데모.
이 번들에는 **코드·웹페이지·분류 head·41개국 샘플만** 들어 있습니다.
인코더 체크포인트(5.5GB)와 fairseq/파이썬 환경은 대상 컴퓨터에 따로 준비해야 합니다.

## 이 번들에 든 것

```
latency_demo/
  server.py          FastAPI 서버 (GPU/CPU/int8 3경로, async+threadpool)
  load_test.py       CLI 부하 테스트 (Python 표준 라이브러리만 사용)
  run.sh             원래 머신용 실행 스크립트 (conda 경로 하드코딩)
  run_portable.sh    ← 이식용 실행 스크립트 (이걸 쓰세요)
  static/            compare.html(/), loadtest.html(/loadtest), index.html(/latency), sample.wav
  class_samples/     41개국 화자 샘플 wav (데모 테스트용)
0704_country41_sent043_en/
  head.pt            분류기 (Linear 1920→41, 영문 라벨) + mu/sd + labels
```

---

## 사용법 3가지 — 상황에 맞게 고르세요

### ① 가장 간단: 실행 중인 서버에 붙어서 부하만 측정 (설치 불필요)

원본 컴퓨터의 서버가 떠 있고 접근 가능한 URL(예: Cloudflare 터널)이 있으면,
**대상 컴퓨터엔 Python만** 있으면 됩니다. 모델·환경·fairseq 전부 불필요:

```bash
python3 latency_demo/load_test.py --url https://<서버-URL> \
    --wav 아무_음성.wav --mode gpu --levels 1,4,8,16,32 --per-level 16
```

부하는 서버 안에서 발생하고, 이 스크립트는 결과만 수집합니다.
웹 부하 테스트 페이지도 그냥 브라우저에서 `https://<서버-URL>/loadtest` 열면 됩니다.

### ② 대상 컴퓨터에서 서버를 직접 실행 (모델 + 환경 필요)

**필요한 것**
- **인코더 체크포인트** `xlsr2_1p45B_32L.pt` (5.5GB) — 이 번들엔 없음.
  원본 컴퓨터에서 복사: `scp <원본>:$PROJECT_ROOT/models/xlsr2_1p45B_32L.pt .`
  (없으면 원본 `xlsr2_2B_1000k_slim.pt`(9GB)나 `xlsr2_2B_1000k.pt`(26GB)도 동작 —
   feature는 동일, layer 31만 씀. 파일이 클수록 로드만 느림.)
- **fairseq repo** — 원본의 `$PROJECT_ROOT/fairseq` 를 그대로 복사하는 게 가장 안전
  (버전 불일치 시 layer_results 인덱싱이 달라질 수 있음). `.so`는 py3.10 빌드.
- **파이썬 환경** (conda 권장):
  - `torch` (2.x, GPU 쓸 거면 CUDA 빌드) · `numpy` · `fastapi` · `uvicorn` · `python-multipart`
  - **ffmpeg / ffprobe** (시스템 패키지, 오디오 디코딩용)
  - fairseq 실행 의존성: `omegaconf` 등 (fairseq import 시 필요한 것들)

**실행**
```bash
XLSR_CKPT=/경로/xlsr2_1p45B_32L.pt \
FAIRSEQ_ROOT=/경로/fairseq \
bash run_portable.sh
```
- head.pt는 번들 안이라 자동으로 잡힙니다.
- 브라우저: `http://localhost:8777` (비교) · `http://localhost:8777/loadtest` (부하 테스트)
- 로드: 체크포인트 크기에 따라 수십 초~수 분. 페이지 상단 배너에 진행 표시.

### ③ 다른 컴퓨터에서도 외부로 열고 싶으면 (Cloudflare 임시 터널)

서버를 띄운 뒤:
```bash
cloudflared tunnel --url http://localhost:8777
```
발급된 터널 링크를 공유하면 접속 가능 (HTTPS라 마이크 녹음도 됨).
※ 임시 터널이라 재시작하면 주소가 바뀝니다.

---

## 환경변수 (run_portable.sh에서 override)

| 변수 | 기본값 | 설명 |
|---|---|---|
| `XLSR_CKPT` | (필수) | 인코더 체크포인트 경로 |
| `FAIRSEQ_ROOT` | (필수) | fairseq repo 경로 (`sys.path`에 먼저 삽입) |
| `HEAD_PT` | 번들 내 head.pt | 분류기 |
| `USE_GPU` | `1` | `0`이면 CPU 전용 (GPU 없는 컴퓨터) |
| `USE_INT8` | `1` | `0`이면 int8 경로 로드 안 함 |
| `PORT` / `HOST` | `8777` / `0.0.0.0` | |
| `CPU_CONCURRENCY` | `4` | CPU 동시 추론 슬롯 |
| `THREADS` | 코어수÷슬롯 | torch 스레드 수 |

## 동작/성능 참고 (원본 머신 RTX 5080 + i5-14600K 실측)

- GPU fp16 인코더 ~30ms, CPU fp32 ~2.7s, CPU int8 ~1.7s (12초 발화 기준)
- CPU 처리량 상한 ~0.47 req/s (메모리 대역폭 병목) → 동접 늘려도 처리량 불변, 대기만 증가
- GPU ~36 req/s (배칭 없이 batch=1 기준)
- 자세한 내용은 `latency_demo/README.md`

## 트러블슈팅

- `from fairseq import checkpoint_utils` 실패 → `FAIRSEQ_ROOT` 경로 확인. CWD에 `fairseq/`가
  있으면 shadowing될 수 있으니 서버는 `FAIRSEQ_ROOT`를 `sys.path` 맨 앞에 넣어 처리함.
- torch ≥ 2.6에서 체크포인트 로드 실패(weights_only) → server.py가 내부적으로 처리하므로
  torch 버전만 맞으면 됨.
- GPU 없음 → `USE_GPU=0`으로 실행. 비교 페이지는 CPU/int8 카드만 표시.
- int8 로드 실패해도 fp32/GPU는 정상 동작 (상태는 /status의 int8_status/int8_error 참조).
