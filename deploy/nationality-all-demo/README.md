# 국적 모델 전부 비교 데모 (2026-10-27)

문장 하나를 읽으면 서버에 있는 국적 모델 전부(60개)가 각각 어느 나라로 듣는지 한 화면에 나란히 보여 준다.
내부 비교용이다. 앱·웹 연동용 API 가 아니다(그건 NPU 국적 API · `docs/NPU_서버_API_안내.md`).

| 항목 | 내용 |
|---|---|
| 주소 | `https://<GPU-funnel-주소>/nat-all` (GPU 서버 · Tailscale Funnel) |
| 키 | `~/.config/nat-all-demo/demo.env` 의 `NAT_ALL_KEY`. 처음 한 번 화면에 넣거나 `…/nat-all/?key=<키>` 링크로 받는다(페이지가 기억하고 주소창에서 지운다) |
| 모델 목록 | `models.json` — 그룹 · 이름 · 체크포인트 · test 채점 폴더. 이름은 대시보드 · results 문서의 이름을 따른다 |
| 계산 | torch 계열(전체학습 · LoRA · 학습 안 함 · 9월 KD · 9월 2B)은 GPU 에서 차례로, 서비스 구성(run_zeroth · 기존 배포 ONNX)은 CPU 스레드에서 동시에 |
| 저장 | 모든 요청의 업로드 원본과 응답(`SPEECH_CAPTURE_DIR`, `shared/request_capture.py`). 화면에 저장 안내가 있다 |

## 무엇을 돌리나

| 그룹 | 개수 | 형식 |
|---|--:|---|
| 서비스 (NPU 앱·웹) | 1 | run_zeroth ONNX + 무음 정규화 + 원어민 게이트(기존 배포 KD, τ 0.9, 원본 입력) — NPU `run_npu.sh` 설정 그대로 |
| 기준 · 백본 (화자 분할 8:1:1) | 8 | `finetune_full_encoder.py` 체크포인트 |
| 데이터 · 불균형 조정 (전체학습) | 7 | 같음 |
| LoRA 이어 학습 | 22 | 같음 (LoRA 는 합친 가중치) · 풀링 · 운율 실험 포함 |
| 9월 KD 모델 (옛 분할) | 15 | run_zeroth · 기존 배포는 배포 ONNX(CPU). 나머지 13개는 배포 ONNX 를 뽑은 `ep0.pt` 를 GPU 에서 — CPU 16코어로 ONNX 15개를 동시에 돌리면 요청마다 몇 초씩 걸려서다. 짧은 test 녹음 2개에서 ONNX 와의 확률 차이 최대 0.0035 |
| 9월 2B 모델 (옛 분할) | 3 | `xlsr2b_finetune.py` 부분 체크포인트 · 보고용 2B 동결 + 0704 헤드 |

빠진 것: 언어식별 모델 그대로(107개 언어 출력) · 2B 첫 시도 `run_2b_lr5e-6_l24`(학습 정체) · 운율 점수 사후 결합(모델 없음).

입력은 NPU 국적 API `/predict` 와 같다: ffmpeg 로 앞 12초 → 0.5초 미만 `too_short` · 말소리 1초 미만 `no_speech` →
무음 정규화(화면 스위치, 기본 켬) → 앞 12초. 서비스 행은 스위치와 상관없이 서비스 설정대로 계산한다.

화면의 test 수치는 화자 분할 8:1:1 test 의 웹마이크 문장 발화(`eval_*/report.json` 의 `web_sent`)다. 9월 모델은 다른 분할로
학습해 같은 잣대가 없다. run_zeroth 의 수치는 누수 의심(화자 분할 전 목록으로 학습).

## 평가와 같은 답인가

GPU 모델은 인코더의 `nn.Linear` 가중치만 bf16 으로 두고 bf16 autocast 로 돈다(메모리 절약, 행렬곱은 평가와 같은 정밀도).
평가는 길이 묶음 배치 · fp32 가중치 + bf16 autocast 였다. `tasks/nationality-classification/scripts/eval/check_zoo_parity.py` 로
저장된 test 로짓과 대조한다 — 결과는 `$NATIONALITY_OUTPUTS/demo_all_20261027/parity_torch_n60.json`.
2026-10-27: 전체학습 · LoRA 계열 36개 × test 60발화에서 1위 일치 2,135/2,160(98.8%), 모델별 57~60/60. 운율 궤적 모델은
대조 스크립트에 궤적 함수가 없어 따로 쟀다(데모의 궤적 함수로 60/60, 확률 차이 최대 0.038). 이 차이는 정밀도 탓이다 — 200발화에서 원래 평가의 bf16 판과 fp32 판끼리도
300m 1.5% · 2B 3.0% 가 달랐고, 데모와 bf16 평가의 차이는 300m 1.0% · 2B 2.0% 였다.

## 운영

| 항목 | 내용 |
|---|---|
| 프로세스 | tmux `nat-all-demo` → `start.sh` → `app.py` (127.0.0.1:8988). systemd 가 아니라 재부팅하면 다시 띄워야 한다 |
| 설정 | `~/.config/nat-all-demo/demo.env` (키 · 포트 · 로그 · 캡처 폴더, 권한 600) |
| 로그 · 캡처 | `$NATIONALITY_OUTPUTS/demo_all_20261027/{serve.log, captures/}` |
| 적재 | 모델 60개 몇 분. `/health` 의 `loaded/total` 로 진행을 본다. 못 올린 모델은 `failed` 에 남고 나머지는 돈다 |
| 공개 경로 | `tailscale funnel --bg --set-path /nat-all http://127.0.0.1:8988` · 내릴 때 `tailscale funnel --bg --https=443 --set-path /nat-all off` (`--https=443 off` 만 쓰면 443 전체가 내려간다) |
| 다시 띄우기 | `tmux kill-session -t nat-all-demo; tmux new-session -d -s nat-all-demo bash deploy/nationality-all-demo/start.sh` |
| 시험 | `$PY deploy/nationality-all-demo/app.py --no-auth --port 8989 --only service,zeroth,deployed,full300m` (일부만 · 저장 안 함) |
