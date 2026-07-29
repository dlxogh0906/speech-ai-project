# XLS-R 2B 국적 분류 — GPU vs CPU vs int8 비교 + Latency 데모

frozen **XLS-R 2B** (fairseq) → **layer 31** mean-pool(1920) → **linear head**(41개 국적, 영문 라벨).
같은 모델을 **GPU(RTX 5080, fp16) / CPU(fp32) / CPU(int8 양자화)** 세 경로로 올려 나란히 비교.

## 경량화 이력 (모두 head 재학습 없음)

| 단계 | 디스크 | RAM/VRAM | 속도(12s 발화) | 정확도 영향 |
|---|---|---|---|---|
| 원본 ckpt (48층+Adam) | 26 GB | – | – | – |
| ① 옵티마이저 제거 (slim) | 9 GB | fp32 8.7GB | CPU ~3.0s | **없음** (동일 가중치) |
| ② layer 32~47 물리 절단 (32L) | **5.8 GB** | fp32 5.8GB / GPU fp16 **VRAM 4.1GB** | 동일 (이미 조기종료) | **없음** (feature 비트 동일 검증) |
| ③ int8 동적 양자화 (CPU) | (런타임 변환 5s) | **~1.7 GB** | **~1.7s (1.6×↑)** | top1 **-4.9%p** (아래 평가) |

### int8 정확도 실측 (로컬 실발화 61건/30클래스, 2026-07-05)

| | top1 | top5 | 평균 인코더 | 동접4 처리량 |
|---|---|---|---|---|
| CPU fp32 | 65.6% | 88.5% | 2507ms | 0.47 req/s |
| CPU int8 | 60.7% (**-4.9%p**) | 86.9% (-1.6%p) | 1587ms | 0.62 req/s (+32%) |
| GPU fp16 | 65.6% (fp32와 100% 일치) | 88.5% | 27ms | ~36 req/s |

- fp32↔int8 top1 일치율 85.2%, top1 확률 평균 |Δ| 9.1%p — int8은 "정확도 예산"을 실제로 씀.
- top5는 거의 보존(-1.6%p). 서비스가 top5 기반이면 부담이 작고, top1이 중요하면 fp32 유지 권장.
- (절대 정확도가 낮은 건 도메인 차이 — head 학습 데이터와 다른 로컬 131 데이터 기준 상대 비교용)

기본 ckpt: `models/xlsr2_1p45B_32L.pt` (1.45B 파라미터). int8은 로드 시 `quantize_dynamic`으로 생성.
fairseq MHA fast-path가 양자화 Linear와 충돌 → `skip_embed_dim_check=True`로 모듈 경로 강제.

## 실행

```bash
bash $PROJECT_ROOT/latency_demo/run.sh
# http://localhost:8777          → GPU vs CPU 비교 페이지 (영문, 스크린샷 스타일)
# http://localhost:8777/loadtest → 동접 부하 테스트 (브라우저에서 클릭으로 실행)
# http://localhost:8777/latency  → CPU latency 측정 페이지 (한국어, 3/6/12s 비교 등)
```

- env: `fairseq_xlsr_5080` (py3.10, torch 2.11+cu128, 로컬 fairseq repo)
- 로드: CPU 인코더 ~25초(페이지 캐시 시) → GPU fp16 복사+warmup ~2초
- 마이크 녹음은 localhost 또는 https에서만 (원격이면 `ssh -L 8777:localhost:8777`)

## 동시성 설계 (동접 처리)

| 항목 | 값 | 이유 |
|---|---|---|
| uvicorn 워커 | **1개** | 모델이 프로세스당 CPU 9GB + GPU 4.6GB — 워커 증설은 메모리×N. 스레드로 확장 |
| 엔드포인트 | **async def** | 무거운 작업은 전부 `run_in_threadpool` — 이벤트 루프 절대 비차단 |
| GPU 접근 | `threading.Lock` | 단일 GPU 직렬화. 요청당 ~30ms라 큐가 짧음 |
| CPU 추론 | `Semaphore(4)` × torch 5스레드 | 실측 최적: 단건 2.95s + 처리량 0.47rps 동시 달성 (아래 튜닝 표) |
| 측정 분리 | `wait_ms` vs `compute_ms` | 큐 대기와 순수 연산을 응답에 분리 보고 |

부하 중에도 `/status`는 <1ms 응답 (이벤트 루프가 안 막힌다는 실측 증거).

## 부하 테스트

**브라우저: `http://localhost:8777/loadtest`** — 장치(CPU/GPU/양쪽)·동접(1~64)·총 요청 수를
고르고 🚀 발사. 요청별 타일이 대기→실행중→완료로 실시간 갱신되고, 끝나면 처리량/p50/p95/
평균 큐 대기가 요약된다. 발사는 서버 내부 ThreadPoolExecutor에서 실행 (브라우저 직접 발사는
호스트당 6커넥션 제한으로 왜곡됨). 각 작업은 실제 API와 동일하게 세마포어/락을 통과한다.
중복 실행은 409로 거부, 중지 버튼은 대기 중 작업만 취소.

CLI:
```bash
python3 load_test.py --wav sample.wav --mode both   # /predict_both
python3 load_test.py --wav sample.wav --mode gpu --levels 1,2,4,8,16,32
python3 load_test.py --wav sample.wav --mode cpu
```

실측 (12초 발화, RTX 5080 + 20코어, 2026-07-05):

| 모드 | 지속 처리량 | 비고 |
|---|---|---|
| GPU 단독 | **~28–32 req/s** | 동접 32까지 에러 0, p95 1.06s(거의 전부 큐 대기), compute 26–33ms 일정 |
| CPU 단독 | **~0.47 req/s** | 하드웨어 상한(대역폭 병목). **동접 64까지 에러 0** — 큐 대기만 선형 증가 |
| predict_both | CPU에 묶임 | 에러 없이 우아한 큐잉 |

CPU 슬롯×스레드 튜닝 (12초 발화):

| 설정 | 단건 지연 | 최대 처리량 |
|---|---|---|
| 1슬롯×14스레드 | 3.26s | 0.30 rps |
| 2슬롯×14스레드 | 3.33s | 0.39–0.40 rps |
| **4슬롯×5스레드 (기본)** | **2.95s** | **0.47 rps** |
| 6슬롯×3스레드 | 4.51s | 0.47–0.49 rps |

CPU 동접 한계 (4×5 기준, Little's law: 평균 응답 ≈ N/0.47초):
- **안 죽는 한계: 동접 64+ 실측 에러 0** (그 이상은 클라이언트 타임아웃이 실질 한계)
- 10초 이내 응답 원하면 동접 ~4, 30초 이내 ~14, 60초 이내 ~28
- 부하 중에도 /status <1ms, GPU 경로는 CPU 큐와 무관하게 동작

## API

```bash
curl -F "file=@s.wav" http://localhost:8777/predict_both   # GPU+CPU 병렬, speedup/top1_match 포함
curl -F "file=@s.wav" -F "device=gpu" http://localhost:8777/predict
curl -F "file=@s.wav" -F "truncate_sec=6" -F "runs=3" http://localhost:8777/predict
curl http://localhost:8777/status                          # gpu_status, live 카운터 포함
```

## feature 규격 (head 학습 시와 동일 — 변경 금지)

1. 16kHz mono (ffmpeg, [-1,1] float32, 파형 정규화 없음), 앞 **12초**
2. `extract_features(source, padding_mask, mask=False, layer=31)` → `layer_results[31][0]`
   — layer=31 조기 종료는 전층 forward와 바이트 단위 동일 (append가 break보다 먼저)
3. mean-pool → `nan_to_num` → `clamp(-50,50)` → fp16 round-trip → `(x-mu)/sd` → head
4. GPU는 인코더+풀링만 fp16/CUDA로, clamp 이후 분류는 CPU 공용 경로 (규격 통일).
   fp16 오차는 확률 소수 둘째 자리 수준 (top1 일치 확인됨)
5. `pos_conv`의 weight_norm은 로드 시 fuse(remove) — deepcopy 제약 회피, forward 동일

## 경로 (env var로 override)

| 변수 | 기본값 |
|---|---|
| `XLSR_CKPT` | `$PROJECT_ROOT/models/xlsr2_2B_1000k.pt` (26GB) |
| `HEAD_PT` | `$PROJECT_ROOT/0702_country41_sent_best/head.pt` |
| `FAIRSEQ_ROOT` | `$PROJECT_ROOT/fairseq` |
| `PORT` / `HOST` | `8777` / `0.0.0.0` |
| `CPU_CONCURRENCY` | `2` |
| `USE_GPU` | `1` (0이면 CPU 전용) |
| `THREADS` | (미설정 시 torch 기본값) |
