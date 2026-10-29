# 사투리 분류 API — NPU 번들 (2026-10-28)

GPU 서버의 사투리 API(`deploy/dialect-api`)와 같은 모델·규칙·요청/응답을 NPU 서버(RBLN-CA25)에서 돈다.
모델은 봉인 test 를 채점한 EAT Large + XLS-R 2B 앙상블(`eat_standard_gate_regional_eat_0.5`) 그대로다.

## 무엇이 다른가

| | GPU 판 | NPU 판 |
|---|---|---|
| 인코더 | fairseq XLS-R 2B · HF EAT (가변 길이) | 같은 가중치를 순수 PyTorch 로 다시 쓴 모듈(`dialect_classifier/npu_models.py`)을 12초 고정 입력 · 실수 마스크로 컴파일한 `.rbln` |
| 정밀도 | fp32 가중치 + fp16 autocast | 가중치 fp16 으로 옮겨 컴파일 |
| fbank · 헤드 · 규칙 | torchaudio · torch | 호스트 torch(CPU) · numpy — NPU 서버에 torchaudio 가 없어 kaldi fbank 를 torch 연산으로 다시 썼다 |
| 실행 | `--backend gpu` | `--backend npu --model-dir <번들>/models --npu-devices <EAT>,<2B>` |

12초보다 짧은 구간은 0 으로 채우고 마스크로 가린다. 합성곱 프레임·패치가 채운 부분과 겹치지 않고, 위치 합성곱 앞에서
채움 프레임을 0 으로 만들고, 어텐션이 채움 키를 가려서 답이 바뀌지 않는다(대조 결과 아래).

## 번들 만들기 (GPU 서버)

```bash
source tools/env.sh
export PYTHONPATH=/data/speech-ai/tools/dialect_compare_py38:/data/speech-ai/tools/hf_py38
B=/dev/shm/dialect_npu_bundle                 # /data 가 빠듯하면 RAM 디스크에서 만들고 보낸 뒤 지운다
bash deploy/dialect-api/npu/make_bundle.sh $B
$PY tasks/dialect-classification/scripts/export/export_npu_bundle.py \
    --root /data/speech-ai/out/dialect/content_grouped_full_20261015 --out $B/models
rsync -a --partial $B/ npu:/root/dialect_serve/bundle/
```

`export_npu_bundle.py` 는 쓰기 전에 봉인 test 200구간(고정 시드, `check_serving_parity.py` 와 같은 표본)으로 순수 모듈을 GPU 에서 대조한다.
`models/parity.npz` 에 그 구간(int16 PCM · 라벨 번호 · 저장된 test 확률 · GPU 특징)을 넣는다 — 경로·화자 필드는 없다.

## NPU 서버에서

```bash
cd /root/dialect_serve/bundle
PY=/root/nat_serve/venv/bin/python           # rebel 0.10.2 · torch 2.9.1+cpu (국적 서비스 venv 를 빌려 쓴다 — 설치하지 않는다)
nice -n 10 $PY tasks/dialect-classification/scripts/export/compile_rbln.py models
$PY tasks/dialect-classification/scripts/eval/check_npu_parity.py models --backend rbln --devices 5,5 [--parity parity_n1000.npz]
```

## 운영 (2026-10-28 17:00 적용)

| 항목 | 내용 |
|---|---|
| 서비스 | systemd `dialect-api` (자동 시작) → `deploy/dialect-api/npu/run_npu.sh` → `app.py --backend npu`, 127.0.0.1:8907, NPU 5번에 EAT · 2B 둘 다 |
| 설정 | `/root/dialect_serve/api.env` (600): `DIALECT_API_KEY` · `DIALECT_PORT=8907` · `DIALECT_NPU_DEVICES=5,5`. 키는 출력하거나 문서에 적지 않는다 |
| 저장 | drop-in `/etc/systemd/system/dialect-api.service.d/capture.conf` → `/data/speech-logs/captures/dialect/<날짜>/<요청번호>/` (`request.json` = 입력 필드 · 파일 정보 · 상태 · 응답 전체, `audio-00.*` = 올린 원본) + 날짜별 `captures.jsonl` |
| 로그 · 배포 기록 | `/data/speech-logs/runtime/dialect-api.log` (`/root/dialect_serve/serve.log` 가 가리킨다) · `/data/speech-logs/deployments/dialect-npu-20261028.json` (모델 파일 SHA256 · 설정) |
| 외부 입구 | Caddy(`/root/caddy/Caddyfile`)의 사투리 블록 → 127.0.0.1:8907. 토큰 검사는 앱이 한다(발음과 같은 방식). 주소 값은 Caddyfile 에만 둔다 |
| 다시 띄우기 | `systemctl restart dialect-api` — 모델 적재가 끝나면 로그에 `[ready]` 가 찍힌다 |
| 되돌리기 | 입구: `cp -p /root/caddy/Caddyfile.bak-20261028-1700 /root/caddy/Caddyfile && systemctl reload caddy-npu` · 서비스: `systemctl disable --now dialect-api` |

설치할 때 한 일(다시 만들 때):

```bash
install -m 644 deploy/dialect-api/npu/dialect-api.service /etc/systemd/system/
install -D -m 644 deploy/dialect-api/npu/capture.conf /etc/systemd/system/dialect-api.service.d/capture.conf
# api.env 는 secrets.token_urlsafe(24) 로 새 키를 만들어 600 으로
systemctl daemon-reload && systemctl enable --now dialect-api
# Caddyfile 은 백업 → 블록 추가한 새 파일을 caddy validate → 교체 → systemctl reload caddy-npu (끊김 없음)
```

확인(10-28): 키 없는 요청 401 · 키 넣은 요청 오디오 · 입력 필드 · 결과 저장, 키 값은 기록 · 로그 어디에도 없음.
GPU 서버에서 공개 입구로 8.1초 녹음 `/predict_long` 왕복 0.55초(서버 처리 0.49초). 확인용 요청 기록은 지웠다.
입구를 연 직후부터 인터넷 스캐너 요청(`/.env` 등, 404)이 들어와 기록된다 — 국적 · 발음과 같다.

## 대조 기록

| 단계 | 봉인 test 200구간 · 저장된 test 채점과 1위 일치 | 확률 최대 차이 | 이 200구간 정답률 |
|---|--:|--:|--:|
| GPU 서빙 경로(구간 1개씩) | 200/200 | 0.008 | 0.965 |
| 순수 모듈 · 12초 고정 · fp16 (GPU, 10-28) | EAT 200/200 · 2B 200/200 · 최종 200/200 | 0.010 · 0.018 · 0.009 | 0.965 |
| 순수 모듈 · fp32 (GPU, 10-28) | 최종 199/200 (경계 1건) | 0.012 | 0.970 |
| 순수 모듈 · fp16 (GPU, 1,000구간) | EAT · 2B · 최종 1,000/1,000 | 0.022 · 0.081 · 0.038 | 0.966 |
| 순수 모듈 · fp32 (GPU, 1,000구간) | 2B 997/1,000 · 최종 997/1,000 | 2B 0.68 | |
| **NPU `.rbln` (rbln4 · rbln5, 1,000구간, 10-28)** | **EAT 996 · 2B 996 · 최종 991 / 1,000** | 0.22 · 0.31 · 0.17 | **0.964** (저장된 채점 0.966) |

fbank 다시 쓴 것과 torchaudio 의 최대 차이 1.9e-6.

NPU 의 2B 차이는 층이 깊을수록 커진다(유효 프레임 평균의 상대 차이 중앙값 31층 1.0% · 47층 4.8%, 길이와 무관).
잔차 흐름을 1/8 · 1/32 로 줄여 컴파일해도 그대로였다 — LayerNorm 넘침이 아니라 fp16 계산이 48층 쌓인 오차다.
같은 모델을 GPU 에서 fp32 로 돌려도 저장된 fp16 채점과 2B 1위가 3/1,000 달라서, NPU 의 4/1,000 은 이 모델이 원래 가진 정밀도 민감도 수준이다.

## 속도 · 메모리 (10-28 실측, 12초 구간 하나)

| | 혼자 | 같은 장치에서 국적 모델과 동시에 (둘 다 쉬지 않고) | 다른 장치에서 동시에 |
|---|--:|--:|--:|
| 2B | 191 ms | 206 ms | 204 ms |
| 국적 KD (`zeroth`) | 59 ms | 206 ms — 2B 계산이 끝나길 기다린다 | 63 ms |
| EAT | 20 ms | | |

- 장치 하나는 한 번에 한 모델만 계산한다. 그래서 EAT · 2B 를 장치 하나(`DIALECT_NPU_DEVICES=5,5`)에 몬다. 국적 8905 워커 8개 중 그 장치를 쓰는 하나만, 동시에 요청이 올 때 최대 약 0.2초 기다린다.
- 메모리: 2B 4.4 GiB + EAT 0.6 GiB. 장치 5 가 6.4 → 11.4 / 15.7 GiB.
- API 전체(NPU 서버 안에서 잰 값): 17.6초 녹음 `/predict`(2구간) 0.53초 · `/predict_long`(3구간) 0.71초.
- 컴파일: EAT 16초(614 MB) · 2B 58초(4.4 GB). 2B 의 위치 합성곱은 그룹 합성곱 그대로면 호스트 CPU 로 넘어가 92 ms 가 들어서 그룹별 일반 합성곱 16개로 나눴다(4.5 ms).
