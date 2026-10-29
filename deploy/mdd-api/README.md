# mdd-api — 발음 진단 + 점수 3종 API (PyTorch · **CPU/GPU ONNX**)

`POST /diagnose` 하나로 음소 진단과 점수 특징을 **같은 forward 한 번**에서 뽑는다.
엔드포인트 설명은 [`deploy/README.md`](../README.md) §mdd-api, 수치의 근거는
[`results/11_경량화.md`](../../tasks/pronunciation-assessment/results/11_경량화.md) 에 있다.

이 문서는 **CPU 배포판**(ONNX int8)에 관한 것을 한곳에 모은 것이다.

---

## 1. CPU 버전이란 무엇인가 — 재학습이 아니다

현역 모델 `mdd_tag_all_native` **그 가중치 그대로** 를 ONNX 로 내보내고 int8 로 양자화한 것이다.
학습 데이터가 들어갈 자리가 없다. 동적 양자화(`quantize_dynamic`)라 **보정 데이터도 0건** 이다
(기각한 정적 QDQ 만 60발화가 필요했다).

| | 현역 PyTorch | **CPU 배포판(int8)** | GPU 배포판(fp16) |
|---|--:|--:|--:|
| 파일 크기 | 1,204 MiB | **339 MiB** | 603 MiB |
| 프로세스 메모리(단독 실행) | 3.3 GB | **1.1 GB** | — |
| 모델 적재 | 7.6~8.2 s | **1~2 s** | 1~2 s |
| 6초 발화 인코더(16코어) | 487 ms | **304 ms** | GPU 7.5 ms |

- **학습을 새로 한 것은 12층 학생 모델뿐이고, 그건 보류했다**(오류 F1 −0.014). 11_경량화.md §4단계.
- 그래서 AIHub 원본 데이터(041·043·001)는 CPU 배포판에 필요 없다. 12층을 되살릴 때만 필요하다.

## 2. 파일

`$SPEECH_AI_DATA/weights/final/pronunciation/` (= `.env` 의 `MDD_DATA_DIR`)

| 파일 | 크기 | 용도 |
|---|--:|---|
| `mdd_tag_all_native.pt` | 1,204 MiB | 기본(PyTorch) 경로 |
| `mdd_tag_all_native_int8.onnx` | 339 MiB | **CPU 서빙** |
| `mdd_tag_all_native_fp16.onnx` | 603 MiB | GPU 서빙 |
| `dict.ltr.txt` | — | 자모 사전(ONNX 경로가 직접 읽는다) |
| `score_models_v2_native.joblib` | — | 점수 회귀기 — **바꾸지 않는다**(보존 검증 완료) |

⚠️ 짝이 정해져 있다: `tag_all_native` ↔ `score_models_v2_native.joblib`.
섞으면 **에러 없이 점수만 틀린다** → [`docs/WEIGHTS.md`](../../docs/WEIGHTS.md).

## 3. 실행

```bash
source tools/env.sh

# (a) 지금까지와 같은 PyTorch 경로 — 기본값, 아무것도 안 바꿔도 된다
bash deploy/serve.sh mdd --port 8914

# (b) CPU · ONNX int8 (권장 구성: 워커 8 × 스레드 2)
export MDD_SERVE_ONNX=$MDD_DATA_DIR/mdd_tag_all_native_int8.onnx
export MDD_SERVE_DEVICE=cpu MDD_SERVE_THREADS=2
$PY deploy/mdd-api/mdd_infer_api.py --port 8914 --workers 8

# (c) GPU · ONNX fp16
export MDD_SERVE_ONNX=$MDD_DATA_DIR/mdd_tag_all_native_fp16.onnx
export MDD_SERVE_DEVICE=cuda MDD_SERVE_THREADS=2
export LD_LIBRARY_PATH=/data/speech-ai/cuda11_shim${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
$PY deploy/mdd-api/mdd_infer_api.py --port 8914 --workers 3
```

| 환경변수 / 옵션 | 기본값 | 뜻 |
|---|---|---|
| `MDD_SERVE_ONNX` | 없음 | 지정하면 인코더를 ONNX 로 교체. **비우면 PyTorch** |
| `MDD_SERVE_DEVICE` | `cpu` | ONNX 실행 장치 (`cpu` \| `cuda`) |
| `MDD_SERVE_THREADS` | `4` | 요청 하나가 쓰는 ONNX 스레드 |
| `--workers` | `1` | 워커 프로세스 수. **처리량은 이걸로 늘린다** |
| `MDD_SERVE_CKPT` / `MDD_SERVE_SCORER` | `.env` | PyTorch 경로의 체크포인트·회귀기 |
| `MDD_FORCE_CPU` | — | PyTorch 경로를 CPU 로 강제 |

- `/health` 가 `backend` 로 무엇이 떠 있는지 알려준다 — 수치를 인용하기 전에 확인할 것.
- ⚠️ `deploy/serve.sh` 는 아직 위 변수를 넘기지 않는다. ONNX 로 띄우려면 `export` 후 위처럼 직접 실행한다.
- ⚠️ `.env` 에 있는 키는 `tools/env.sh` 가 덮어쓴다. `MDD_SERVE_ONNX` 는 `.env` 에 없으므로 export 가 통한다.

## 4. 정확도 — 원본과 같은가 (전부 실측)

**(1) 변환 자체**: `mdd_tag_all_native.pt` → fp32 ONNX 는 200발화에서 **프레임 단위 100% 일치**
(토큰 편집률 0.0). int8 은 fp32 대비 프레임 argmax 99.89% 일치.

**(2) 학습에 쓴 세 영역 전부에서 대조**했다. 기준은 서빙 경로(PyTorch GPU fp16).

| 데이터 | 발화 | int8 결과 | fp16 결과 |
|---|--:|---|---|
| TAG test (040~043) | 10,000 | 오류 F1 **+0.0002** [−0.0036, +0.0042] | +0.0005 [−0.0004, +0.0014] |
| LAR 낭독 (041·043) | 2,000 | PER **−0.00007** [−0.00024, +0.00008] · 출력 동일 99.1% | ΔPER 0 · 동일 100% |
| 원어민 자유대화 (001) | 2,000 | PER **+0.00023** [−0.00044, +0.00097] · 출력 동일 **77.1%** | ΔPER 0 · 동일 98.8% |

**(3) 점수 3종 보존** (원본 대비 상관 · 평균 |Δ|, 1~5점 척도)

| 데이터 | int8 | fp16 |
|---|---|---|
| TAG test | 0.997~0.998 · 0.012~0.016 | 0.9999 · 0.0007~0.0010 |
| LAR | 0.995~0.996 · 0.012 | 0.9997~0.9998 · 0.0007~0.0009 |
| 원어민 | 0.987~0.991 · 0.020~0.026 | 0.9995~0.9996 · 0.0010~0.0012 |

**(4) API 수준 A/B** (`jobs/ab_compare_api.py`, PyTorch 8917 ↔ ONNX 8916):
20발화 전부 **인식 자모열과 지적 목록이 동일**, 세션 점수 차이 ≤0.01점. 워커 6개에서도 같았다.

> ⚠️ **원어민 발화에서 int8 은 출력이 흔들린다** — 발화의 23%에서 한두 토큰이 바뀐다(TAG 2.5%, LAR 0.9%).
> 맞고 틀림이 상쇄돼 오류율은 그대로지만, 원어민·자유발화 비중이 큰 서비스라면 GPU fp16 을 쓴다.

## 5. 속도 (단독 실행 · 6초 발화 · 인코더 forward)

| 코어 | PyTorch fp32 | ONNX fp32 | **ONNX int8** |
|--:|--:|--:|--:|
| 16 | 487 ms | 352 ms | **304 ms** |
| 14 | — | — | **341 ms** |
| 8 | 696 ms | 497 ms | **452 ms** |
| 6 | — | — | **569 ms** |
| 4 | 1,378 ms | 768 ms | **748 ms** |
| 2 | 2,129 ms | 1,418 ms | **1,386 ms** |

int8 · 발화 길이별: 6코어 284 / 569 / 1,174 ms, 14코어 168 / 341 / 710 ms (3 · 6 · 12초).
GPU: PyTorch fp16 17.0 ms → ONNX fp16 **7.5 ms**.

- 속도 이득의 대부분은 ONNX 변환에서 나온다. int8 은 그 위에 3~14% 를 더한다(특징 추출 conv 는 fp32 로 남는다).
  **int8 의 진짜 이득은 메모리 1/3 · 파일 1/3.6 이다.**
- VNNI/AMX 가 있는 최신 인텔 CPU 는 int8 이 더 빠를 수 있다 — 이 서버(EPYC 7H12)는 **AVX2 뿐**이라 못 쟀다.

## 6. 처리량 — HTTP 실측 (16코어 · 평균 6.4초 발화)

| 구성 | 처리량 | 부하중 CPU | 합계 RSS |
|---|--:|--:|--:|
| CPU int8 · 1워커 × 4스레드 | 2.7 req/s | — | 1.2 GB |
| CPU int8 · 6워커 × 2스레드 | 3.4 req/s | — | ~10 GB |
| **CPU int8 · 8워커 × 2스레드 (권장)** | **4.07 req/s** | 700% / 1600% | **13.7 GB** |
| CPU int8 · 12워커 × 1스레드 | 4.13 req/s | 1,200% / 1600% | 21.1 GB |
| **GPU fp16 · 3워커 × 2스레드** | **37.7 req/s** | — | — |

- **처리량 ≈ 워커 수 ÷ 요청당 지연.** 스레드를 늘리는 게 아니라 **워커를 늘려야** 한다.
  한 프로세스 안에서는 GIL 이 막는다 — 인코더(ONNX)는 GIL 을 놓지만 강제정렬·GOP·점수는 파이썬·numpy 다.
- 8→12 워커로 늘려도 4 req/s 근처에서 평평하다(메모리만 1.5배). CPU 도 다 안 쓴다.
- 요청 1건 지연: GPU PyTorch 0.13 s · CPU int8 1.0 s(4스레드) / 1.9 s(2스레드, 7.6초 발화).
  이 중 오디오 디코딩(ffmpeg)은 0.03 s 로 병목이 아니다.
- HTTP 경로는 오프라인 처리량(5.25 req/s)의 약 80% 다.

## 7. 하드웨어 규모 산정

학습자 1명이 10~20초에 문장 1개를 보낸다고 보고 환산한 값이다.

| 장비 | 요청 1건 | 처리량 | 편하게 받는 동시 사용자 | 100건 동시 도착 |
|---|--:|--:|--:|--:|
| 16코어 서버 · CPU int8 (실측) | 0.85~1.5 s | 4.1 req/s | 30~55명 | 25초 |
| **i5-14600K · CPU int8 (추정)** | 0.35~0.5 s | 3.5~5.5 req/s | 30~50명 | 20~30초 |
| **RTX 4060 Ti 16GB · GPU fp16 (추정)** | 0.1~0.15 s | 8~15 req/s | **80~150명** | 7~12초 |
| A100 · GPU fp16 (실측) | 0.08 s | 37.7 req/s | 300명+ | 2.7초 |

- **i5+4060Ti 한 대라면 GPU fp16 을 기본으로 쓴다.** CPU 만으로는 40명 근처가 한계다.
- i5 추정 근거: 코어는 적지만(6 P + 8 E) 코어당 성능이 약 2배이고 **AVX-VNNI 가 있다**. 위 코어별 실측 곡선에
  코어당 성능비를 적용했다. **그 하드웨어로 직접 재지 않았다** — 실측하면 30분 안에 확정된다.
- RAM 32 GB 주의: 현재 워커당 **1.7 GB** 다(§8-2). 8워커면 14 GB.
- 데스크톱 CPU 는 지속 100% 부하에서 클럭이 내려간다 — 위 추정에 반영돼 있지 않다.

## 8. 제약과 남은 일

1. **`--workers` 를 쓰면 `/session` 점수가 갈라진다.** 세션이 프로세스 메모리에 있어서, 같은 `session_id` 라도
   매번 다른 워커에 붙는다. 스티키 라우팅이나 공용 저장소로 옮기기 전까지 **다중 워커는 단발 `/diagnose` 전용** 이다.
   (기동 시 경고를 출력한다.)
2. **워커당 RSS 1.7 GB** — int8 단독 1.1 GB 보다 크다. API 가 여전히 torch·fairseq 를 import 하기 때문이다.
   떼어내려면 사전 로딩과 `ctc_loss` 를 numpy 로 바꿔야 한다.
3. `deploy/serve.sh` 가 ONNX 환경변수를 넘기지 않는다(§3).
4. int8 은 CPU 전용, fp16 은 GPU 전용이다. ONNX GPU 는 CUDA 11 라이브러리가 필요하다
   (이 서버는 `LD_LIBRARY_PATH=/data/speech-ai/cuda11_shim`).
5. 원어민 발화 int8 출력 흔들림(§4).
6. 실서비스 교체 전 확인: 다른 포트로 띄워 응답을 대조한다(CLAUDE.md). `jobs/ab_compare_api.py` 가 그 용도다.

## 9. 재현

```bash
source tools/env.sh
T=tasks/pronunciation-assessment/scripts
C=/data/speech-ai/out/compress/mdd

# 내보내기 · 양자화 (--out 은 폴더다. wav_dir 은 패리티·보정용 16 kHz wav)
$PY $T/export/export_onnx_mdd.py   --ckpt $MDD_SERVE_CKPT --out $C/onnx  --wav_dir $SPEECH_AI_DATA/mdd_gold_wav16k --n 200
$PY $T/export/quantize_onnx_mdd.py --fp32 $C/onnx/mdd_fp32.onnx --out $C/quant \
    --wav_dir $SPEECH_AI_DATA/mdd_gold_wav16k --variants dyn_matmul,fp16

# 도메인 표본 만들기 → 디코드 → 대조
$PY $T/data/build_domain_check_manifest.py --domain lar    --out $C/domain_check
$PY $T/data/build_domain_check_manifest.py --domain native --out $C/domain_check --cap 100
$PY $T/eval/decode_manifest_backend.py --manifest $C/domain_check/lar --split test \
    --backend ort --onnx $C/quant/mdd_dyn_matmul.onnx --workers 4 --threads 4 --preds <preds>
$PY $T/eval/domain_parity_mdd.py --manifest $C/domain_check/lar --ref <torch preds> --cand int8=<preds> --out parity.json

# API 대조 (기준 서버 8917 · 교체본 8916)
$PY /data/speech-ai/out/compress/jobs/ab_compare_api.py --ref 8917 --cand 8916 --n 20
```

## 10. 근거 문서

- [`results/11_경량화.md`](../../tasks/pronunciation-assessment/results/11_경량화.md) — 전체 실험(1~4단계), 이 문서 수치의 원본
- [`results/09_추론속도.md`](../../tasks/pronunciation-assessment/results/09_추론속도.md) — PyTorch 기준 지연·동시 요청
- [`docs/WEIGHTS.md`](../../docs/WEIGHTS.md) — 가중치 짝, 어느 수치가 어느 모델인지
- 산출물 원본: `/data/speech-ai/out/compress/mdd/` (onnx · quant · eval · domain_check), 작업 스크립트 `/data/speech-ai/out/compress/jobs/`
