# 13 — NPU(Rebellions ATOM-Max) 서빙 준비 (2026-09-23)

## 대상 칩 (공개 자료 기준)

ATOM-Max(RBLN-CA25): ATOM 4칩, FP16 128 TFLOPS · INT8 512 TOPS, GDDR6 64 GB(1 TB/s), PCIe Gen5, 350 W.
SDK = RBLN 컴파일러 + optimum-rbln(HuggingFace 연동, 모델 zoo 에 Wav2Vec2·Whisper 포함). 컴파일은 입력 shape 고정.
출처: rebellions.ai/rebellions-product/atom-max · White-Paper Issue5 ATOM-Max · github.com/rebellions-sw/optimum-rbln · docs.rbln.ai

## 300m 국적 모델 — 고정 길이 ONNX (완료)

`scripts/export/export_onnx_kd_static.py --ckpt <run>/ep0.pt --xlsr300 <xlsr2_300m.pt> --sec 12 --opset 17`
→ `<run>/deploy/encoder_kd_l17_static12s.onnx` (960 MB fp32)

| 항목 | 값 |
|---|---|
| 입력 | `source` f32 [1, 192000] (12 s, 짧으면 0 채움) · `frame_mask` bool [1, 599] (True = 채움) |
| 출력 | `pooled` f32 [1, 1024] — 이후 `(x−mu)/sd · W + b` · softmax 는 호스트(numpy, `head_np.npz`) |
| 그래프 | 노드 1,590 (동적판 3,390) · Shape 18 (148) · LayerNormalization 44 융합 · ScatterND/CumSum 없음 |
| 비표준 연산 | Where 37(어텐션 키 마스크) · Erf 26(GELU) · Cast 81 · Equal/Expand/ConstantOfShape 18 — 컴파일러가 못 받으면 Erf→Tanh 근사, Cast 제거 |
| 정확도 | 동적 모델(배포 규격) 대비 pooled 최대 차 4.3e-5 (2·4·8·12 s 입력) · 실발화 top-1 12/12 일치 (run5 기준) |
| 만든 것 | run5 · **run_zeroth**(현 데모 모델) 두 벌 |

호스트가 만드는 프레임 마스크: 유효 샘플 n → 유효 프레임 `conv_out_len(n)` (커널 10,3,3,3,3,2,2 · 스트라이드 5,2,2,2,2,2,2, 각 단계 `(n−k)//s+1`). 192,000 → 599.
fairseq 의 샘플→프레임 마스크(cumsum·scatter)는 ONNX 로 깨져서 그래프 밖으로 뺐다. 데모의 무음 정규화(`KD_SILENCE_NORM`)는 파형 단계라 NPU 와 무관하게 호스트에서 그대로 쓴다.

정밀도: fp16 이면 검증만 하면 되고(MDD 모델은 fp16 동등 확인됨, `pronunciation-assessment/results/11_경량화.md`), int8 만 되는 경우 보정셋(505·웹마이크·Zeroth)으로 정적 양자화 후 top-5 재측정.

## 2B 국적 모델 (진행 중)

- 백본 `xlsr2_2B_1000k.pt`(26 GB) 확보, 헤드 0704(41×1920, 31층) 재현 검증: valid 505 top-1 0.884 / top-5 0.986.
- 미세조정 `experiments/xlsr2b_finetune.py`: 31층까지 자른 2B(14.6억 파라미터), 데이터 lab131full_zeroth 774,819. A100 단독 70 utt/s, GPU 9.5 GB.
  - 1차(`run_2b_lr5e-6_l24`, lr 5e-6/5e-5 · 24~31층 3.5억): step 7,500(30%)에 valid 웹마이크 top-5 0.654 · 505 0.959 · 한국어 0.887 로 정체, 학습 loss 2.2 → **과소적합, 중단**.
    (300m KD 는 같은 시점에 0.86 — 헤드가 웹마이크를 처음 보는데 lr 이 너무 낮았다.)
  - 2차(`run_2b`, 23:00~02:12): lr 2e-5/2e-4 · warmup 300 · 22~31층(4.4억) + 헤드. valid 웹마이크 top-5 0.27 → 0.73 (step 10,500 이후 정체).

### 2B 1 epoch 결과 (`eval_2b/`, 02:12) — 300m(zeroth)에 전 구간에서 진다

| 구간 | 300m zeroth | **2B 1 epoch** |
|---|---|---|
| 웹마이크 외국인 top-1 / top-5 (15,608) | 0.641 / 0.869 | 0.463 / 0.738 |
| 웹마이크 문장 top-5 | 0.960 | 0.846 |
| 505 외국인 (1,585) | 0.902 / 0.984 | 0.761 / 0.936 |
| 한국어 스튜디오 top-1 (670) | 0.994 | 0.970 |
| Zeroth 원어민 낭독(시험 화자) | 0.993 / 1.000 | 0.956 / 0.996 |
| 사용자 노트북 녹음 | 2/6 | 2/6 |

- 300m 학생은 KD run2 → run3 → run4 → run5 → zeroth 로 웹마이크를 누적 3 epoch 이상 봤고 헤드도 적응된 상태에서 시작했다. 2B 는 1 epoch · 상위 10층뿐.
  같은 1 epoch 로는 공정한 비교가 아니지만, **같은 GPU 시간이면 300m 이 압도적으로 낫다.**
- 3차(`run_2b_ep2`, 02:12~05:29): run_2b/ep0.pt 이어서 **전 층 0~31 학습**(14.2억), lr 1e-5/1e-4, 76 utt/s, GPU 15 GB. valid 웹마이크 top-5 0.729 → 0.741 로 사실상 정체
  (학습 loss 는 1.45 → 1.2 대로 계속 내려감 = 학습 화자에 맞추고 미지 화자로 일반화되지 않는다).

### 2B 최종 (`eval_2b_ep2/`, 05:29) — **NPU 후보에서 제외**

| 구간 | 300m zeroth | 2B 1 epoch | 2B 2 epoch(전 층) |
|---|---|---|---|
| 웹마이크 외국인 top-1 / top-5 (15,608) | 0.641 / 0.869 | 0.463 / 0.738 | 0.474 / 0.749 |
| 웹마이크 문장 top-5 | 0.960 | 0.846 | 0.858 |
| 505 외국인 (1,585) | 0.902 / 0.984 | 0.761 / 0.936 | 0.750 / 0.930 |
| 한국어 스튜디오 top-1 (670) | 0.994 | 0.970 | 0.979 |
| Zeroth 원어민 낭독(시험 화자) | 0.993 / 1.000 | 0.956 / 0.996 | 0.943 / 0.980 |
| 사용자 노트북 녹음 | 2/6 | 2/6 | 2/6 |

- 2 epoch·전 층으로도 300m 에 13점(웹마이크 top-5) 뒤진다. **결론: 이 과제에서는 크기보다 데이터·학습량이 성능을 정하고, 같은 GPU 시간이면 300m 이 낫다.**
  2B 는 NPU 후보에서 뺀다. ATOM-Max 에는 300m zeroth(정적 ONNX 준비됨)를 올린다.
- 2 epoch 평가는 Zeroth 복구 후 재실행한 값(19,686행 · kor001 10행만 빠짐). 평가 직전(04:35) 다른 세션의 디스크 정리로 `corpus/zeroth` 와
  `corpus/aihub/001.자유대화(일반남여)` 가 삭제됐고 Zeroth 는 재다운로드했다(→ `11_채널편향_한국어.md` 끝 "데이터 삭제 사고").
- 체인 `jobs/nat_2b_chain.sh` → `run_2b/ep0.pt`(학습 층 + 헤드만 1.4 GB) → `eval_2b/logits.npz`.
- NPU 에 올릴 크기: fp16 약 3 GB. 고정 길이 수출은 300m 과 같은 방식(`export_onnx_kd_static.py` 를 2B 로더로 확장)으로 한다 — 2B 가 300m 을 이길 때만.

## 배포 번들 (2026-09-24 19:20) — `/data/speech-ai/out/novelty/npu_bundle/` (tar: `npu_bundle.tar`, 5.8 GB)

서빙 조합 = **zeroth(국적) + 기존 배포 모델 게이트(τ 0.8) + 무음 정규화**. 데모 8922 도 이 조합으로 교체(`KD_GATE_DIR`).
- `models/{zeroth,deployed_gate}/`: 동적 ONNX(CPU) · 정적 ONNX(패리티) · **TorchScript `encoder_kd_l17_static12s.ts.pt`(NPU 컴파일용, fairseq 불필요)** · head_np.npz
- `app/`(서버 `NAT_BACKEND=onnx|rbln` · 게이트 · 무음 정규화) · `samples/` 16개 · `compile_rbln.py` · `validate_parity.py` · `run_cpu.sh` · `run_npu.sh` · `README_NPU.md`
- RBLN SDK 프런트엔드는 torch/TorchScript/TF(ONNX 직접 지원 없음) → `compile_from_torchscript` 1순위, onnx2torch 2순위. SDK 없는 이 서버에서는 컴파일을 검증하지 못했다.
- 번들 단독 실행(CPU, 저장소 밖) 검증: /health OK · 사용자·Zeroth 샘플 Korea · 게이트 0.999 · 12초 발화 1.1~2.5 s.
- NPU 서버(rbln0~7, 각 15.7 GiB, SDK 미설치, 디스크 100 GB): 1단계 CPU 서빙 → 2단계 컴파일·패리티 → `run_npu.sh`.

## 실제 NPU 배포 (2026-09-26)

NPU 서버(RBLN-CA25 ×8, SDK rebel_compiler 0.10.2)에 Tailscale 로 접속해 배포했다. 서빙 구성은 zeroth + 확인 모델(τ0.9, raw) + 무음 정규화 그대로.
- fairseq TorchScript·ONNX 경로는 불리언 마스크에서 컴파일 실패 → **순수 PyTorch 재구현**(`country_classifier/npu_encoder.py`, 실수 마스크, fairseq 와 max|Δ| 5e-5) 을 `compile_from_torch`.
- NPU vs CPU: 시험 발화 300개 정답률 0.883 vs 0.880 · 1위 일치 298/300 · 확률 차 평균 0.008. 폰 녹음 17개 동일.
- 지연 0.24 s/요청(NPU 호출 58 ms). 8 워커로 32.7건/초(동시 32명 중앙값 0.87 s). 단일 프로세스는 4.4건/초에서 막힌다(NPU 호출이 스레드 병렬 불가).

## 운영 8905 앱 갱신 (2026-10-05 15:31)

- **적용한 것**: GPU 데모와 같은 `deploy/nationality-kd-demo/app.py` 와 `multi/` · `long/` 화면을 `/root/nat_serve/bundle/app` 에 넣었다.
  - 모델 파일과 실행 설정(12초 고정 rbln, 워커 8, 확인 모델 τ0.9 원본 입력, 무음 정규화)은 그대로다.
  - 이전 파일은 `*.bak-20261005-1531` 로 남겼다.
- **새 기능**
  - `POST /predict_long` 을 추가했다. 한 번에 읽은 녹음을 `parts` 구간으로 나눠 합산하고, 저장하지 않는다(결과 23).
  - `GET /multi` · `GET /long` 테스트 화면을 추가했다.
  - CORS 를 붙였다. 허용 도메인은 `/root/nat_serve/cors_origins` 에서 읽는다. 지금은 로컬 개발 주소 localhost·127.0.0.1 의 3000·5173 이다.
  - `/compare` 는 NPU 에 화면 파일이 없어 404 를 돌려준다.
  - 기존 `/predict` 는 그대로다.
- **`restart.sh` 를 고쳤다.**
  - 이제 8905 서버만 끈다. 예전 `pkill -f "uvicorn app:make_app"` 은 8906 실험 서버와 다른 국적 서버까지 껐다.
  - `cors_origins` 를 읽어 `KD_CORS_ORIGINS` 로 넘긴다.
  - 운영 웹 도메인을 추가할 때는 그 파일에 한 줄을 더하고 `bash /root/nat_serve/restart.sh` 를 돌린다(약 1분 끊김).
- **확인한 것** (공개 주소 https://<NPU-funnel-주소>)
  - 워커 8개가 모두 올라왔다.
  - `/predict_long` 은 GPU 데모와 1위·구간별 판정이 6/6 같았고, 확률 차는 0.05 이하였다.
  - 서버 처리 시간은 3구간 0.6초, 5구간 1.0초다.
  - 기존 `/predict` 는 같은 응답 형식이었다.
  - CORS 는 등록 도메인만 허용됐다.
  - 휴대폰 크기 브라우저(가짜 마이크)로 1·3·5문장 판정, 짧은 녹음 막기, 구간 축소 안내를 확인했다.
  - 업로드 폴더 파일 수는 그대로였다(9개).
  - 발음 평가 8932 와 8906 은 영향이 없었다.
- **정리**: 예전 테스트 서버 8907(`bundle_m`)은 껐다.

## 재부팅 뒤 자동 시작 (2026-10-05 21:30)

웹 요청서 「국적 분류 API 서울 직통 입구 추가 요청」의 자동 시작 항목이다. 국적 API 와 공개 입구(Caddy)를 systemd 서비스로 바꿨다. 둘 다 `enabled` 라 NPU 서버가 재부팅되면 자동으로 다시 뜬다. 비정상 종료 때도 다시 뜬다.

| 서비스 | 하는 일 | 실행 | 다시 시작 |
|---|---|---|---|
| `nat-api.service` | 국적 API :8905 (워커 8) | `/root/nat_serve/run_8905.sh` (`rbln_daemon` 과 NPU 장치 준비 뒤) | `bash /root/nat_serve/restart.sh` 또는 `systemctl restart nat-api` |
| `caddy-npu.service` | 공개 입구 443 (지금은 발음 API 만) | `/root/caddy/caddy run --config /root/caddy/Caddyfile` | 설정만 바꿨으면 `systemctl reload caddy-npu`(끊김 없음) · `bash /root/caddy/start_caddy.sh` |

- **로그**: 국적 API 는 `/root/nat_serve/serve.log`, Caddy 는 `/root/caddy/caddy.log` 에 이어 쓴다.
- **스크립트**: 이전 수동 실행 스크립트는 `restart.sh.bak-nohup`, `start_caddy.sh.bak-nohup` 으로 남겼다.
- **확인한 것**
  - 전환한 뒤 공개 주소에서 `/health`, `/predict_long`, `/predict`, CORS 가 그대로 동작했다.
  - 수동으로 띄운 프로세스는 남지 않았다.
  - 발음 직통 입구 `/health` 는 200 이었다.
- **남은 것**
  - 발음 API(:8932)는 아직 수동 실행(`/root/pron_serve/start_pron.sh`)이다. 재부팅하면 발음 입구가 그 API 를 다시 띄울 때까지 502 를 낸다.
  - 국적 API 용 직통 주소와 토큰은 아직 붙이지 않았다. 도메인·인증서 변경은 사용자 승인 대기 중이다.
- **발음 API 도 자동 시작 (2026-10-05 22:00)**: `pron-api.service` 로 바꿨다.
  - 실행: `/root/pron_serve/run_8932.sh`. tailnet IP 와 NPU 장치가 준비된 뒤 시작하고, 토큰은 `.token` 에서 읽는다.
  - 재시작: `bash /root/pron_serve/start_pron.sh`. 예전 스크립트는 `.bak-nohup` 으로 남겼다.
  - 전환은 6초 걸렸다. 직통 입구와 Funnel :8443 의 `/health` 200, 토큰 없는 `/session` 401 을 확인했다.
  - 이제 NPU 의 세 서비스(국적 · 발음 · Caddy)가 모두 재부팅 뒤 자동으로 뜬다.
