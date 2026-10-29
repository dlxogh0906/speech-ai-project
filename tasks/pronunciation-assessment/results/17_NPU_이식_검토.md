# 17 — 발음 평가 모델(w2v-BERT CTC) NPU 이식 검토 (2026-09-26)

**결론: 된다.** 서비스 모델 `w2vbert_stage2/step_38000`(w2v-BERT 2.0 CTC, 8932 발음 데모가 쓰는 것)을
Rebellions NPU(RBLN-CA25, rebel 0.10.2)로 컴파일해 돌렸다. 실제 앱 녹음 161건에서 서비스(GPU fp16)와
인식·진단·점수가 거의 같고, 모델 1회 지연은 중앙값 **28 ms**(GPU 34 ms)다.
단 NPU 쪽 차이는 GPU fp16 의 자체 오차보다 크다(아래 2절) — "완전히 같다"가 아니라 "제품에서 구분 안 되는 수준"이다.

---

## 1. 막힌 곳과 해결

| 문제 | 증상 | 해결 |
|---|---|---|
| 불리언 마스크 | `x and y must have the same dtype: float32 vs bool` (국적 인코더에서 먼저 확인) | 마스크 세 곳을 곱셈·가산 편향(−1e4)으로 바꾼 감싼 모듈 `src/mdd/npu_w2vbert.py` |
| **입력 길이** | 층 1개는 컴파일되지만 **2층 이상 쌓으면** `Graph Generation: [DEVICE_GRAPH_CONVERSION]` | **프레임 수를 64 의 배수로.** 2층 실측: 128·192·256·320·384·512·640·768 통과, **200·800 실패**. 24층 전체도 256 에서 통과 |
| 상대위치 어텐션 | HF `relative_key` 는 층마다 거리 임베딩 표를 [T,T,64] 상수로 펼친다 | q·Eᵀ([T,73])를 구해 양끝 복제·flatten·오프셋 슬라이스로 배치(rel-shift). 상수·gather 없음, 값 동일 |
| 가중치 정밀도 | fp16 사본만으로도 원본 대비 프레임 0.2% 가 바뀐다(클립 000: 494프레임 중 1개) | 원본 fp32 가중치로 컴파일 |

- 길이 실패는 상대위치 어텐션에서 난다. T=200·2층 기준으로 FFN·합성곱 모듈만 쌓으면 통과했다.
  rel-shift 어텐션만 쌓은 것과 HF 원본 어텐션이 든 전체 2층은 실패했다. 위치항 없는 평범한 어텐션은 통과했다.
- 감싼 모듈 = 원본 HF: CPU fp32 에서 40클립(패딩 포함) 프레임 1위 **100%**, 확률 차 **0**.

## 2. 정확도

### 2-1. 기준 클립 40개 — 원본(fp32, CPU) 대비

웹마이크 외국인 12 · 505 외국인 8 · 한국인 대화(Kspon) 5 · 한국인 낭독(Zeroth) 5 · 한국인 아이폰(사내) 10.

| | 프레임 1위 일치 | 그리디 자모열 오차율 (평균/최대) | 자모열 완전 일치 |
|---|--:|--:|--:|
| GPU fp16 (현 서비스 설정) | 99.98% | 0.19% / 6.67% | 38/40 |
| **NPU** (6버킷) | **99.74%** | **0.45% / 9.09%** | **34/40** |
| NPU (3버킷) | 99.74% | 0.43% / 9.09% | 35/40 |

한국인 20클립은 NPU 도 20/20 완전 일치다. 차이는 외국인 발화(웹마이크 10/12, 505 4~5/8)에서만 났다.

### 2-2. 실제 앱 녹음 161건 — 서비스 `analyze()` 전체 경로

`/data/speech-ai/out/mdd/webmic_takes` 178건 중 서비스가 "말소리 부족"으로 거절하는 19건을 뺐다.
서비스 코드(`deploy/pronunciation-demo/app.py` 의 `analyze`)를 그대로 쓰고 로그확률만 바꿔 넣었다.

| 비교 | 인식 자모열 완전 일치 | 자리 판정 일치 | 판정이 바뀐 발화 | 점수 \|Δ\| 평균 (발음/이해/유창) | 점수 \|Δ\| 최대 |
|---|--:|--:|--:|--:|--:|
| **NPU vs 서비스 GPU fp16** | **157/161 (97.5%)** | **99.93%** | **3건 (1.9%)** | 0.007 / 0.005 / 0.007 | 0.155 / 0.169 / 0.243 |
| GPU fp16 vs fp32 (기준 잡음) | 161/161 | 100% | 0건 | 0.001 / 0.001 / 0.001 | 0.040 / 0.072 / 0.061 |

- 달라진 4건은 전부 **모호한 모음 1개**다: ㅏ→ㅘ, ㅣ→ㅡ, ㅣ→ㅕ ×2(뒤 둘은 같은 녹음을 두 번 낸 것).
  그중 3건은 오류 자리 1개가 생기거나 사라졌다.
- 점수(1~5) 차이가 0.1 을 넘은 건 위의 같은 녹음 2건뿐이다. 점수 상관은 세 점수 모두 0.999 이상.

## 3. 속도 — 모델 1회, 배치 1

NPU 는 버킷(패딩한 길이)마다 따로 컴파일된 그래프를 쓴다. GPU 는 A100-SXM4-80GB, 서비스와 같은 fp16·패딩 없음.

| 입력 길이 (버킷) | NPU | GPU fp16 |
|---|--:|--:|
| 3.84초 (192) | 29 ms | 33 ms |
| 6.40초 (320) | 59 ms | 33 ms |
| 8.96초 (448) | 94 ms | 34 ms |
| 12.80초 (640) | 178 ms | 33 ms |
| 16.64초 (832) | 284 ms | 34 ms |
| 20.48초 (1024) | 429 ms | 45 ms |

- NPU 는 길이에 따라 크게 늘고, GPU 는 20초까지 거의 일정하다.
- 그러나 서비스는 앞뒤·중간 무음을 잘라서 넣는다. 실제 녹음은 자른 뒤 중앙값 **2.4초**, p90 4.1초, 최대 7.1초라
  161건 중 141건이 192 버킷에 들어간다.
- 실제 녹음 161건의 모델 1회 지연:

| | 중앙값 | p90 | 최대 |
|---|--:|--:|--:|
| NPU | 28 ms | 53 ms | 94 ms |
| GPU fp16 | 34 ms | 34 ms | 36 ms |

- 모델 밖 처리(무음 자르기·G2P·강제정렬·점수, 특징 추출 제외)는 중앙값 23 ms 로 어느 쪽이든 같다.

## 4. NPU 메모리

버킷마다 가중치가 따로 들어간다(버킷당 약 1.5 GiB).

| 버킷 | .rbln 크기 | NPU 1장 점유 | 국적 서비스와 같은 장에서 |
|---|--:|--:|--:|
| 6개 (192·320·448·640·832·1024) | 7.2 GB | 9.0 GiB | 15.4 / 15.7 GiB — 빠듯 |
| **3개 (192·448·1024)** | **3.6 GB** | **5.0 GiB** | **11.4 / 15.7 GiB** |

**권장: 3버킷.** 정확도는 같고(2-1절), 실제 녹음 대부분이 192 버킷이다.
대가는 3.84~6.4초 발화가 320 대신 448 버킷으로 가서 59 → 94 ms 가 되는 것이다(실제 녹음 161건 중 16건).
서비스가 입력을 20초에서 자르므로 1024(20.48초)가 상한으로 충분하다.

## 5. 서비스로 올리기 — 2026-09-27 NPU 서버에서 운영 시작

앱 전체(`deploy/pronunciation-demo/app.py`)를 NPU 서버로 옮겨 띄웠다. 막혔던 두 가지와 해결:

| 문제 | 해결 |
|---|---|
| NPU 서버 파이썬 3.12 에 fairseq 가 없는데, `mdd_eval` 이 fairseq 를 모듈 첫머리에서 import 했다 | fairseq import 를 `mdd_eval.Decoder()` 안으로 옮겼다(직전에 `ensure_fairseq()` 재호출). 기존 fairseq 디코더 적재 확인 |
| 점수 모델 joblib 이 sklearn 1.3.2 pickle 이라 numpy 2 환경의 sklearn 1.9.1 에서 안 열린다(`__pyx_unpickle_CyHalfSquaredError`) | 트리 노드를 .npz 로 뽑아 numpy 로 예측(`mdd.hgbr_np`, `scripts/export/export_scorer_npz.py`). 무작위 5,000행(결측 포함)에서 원본과 **차이 0** |

- 모델 쪽은 `mdd.npu_w2vbert.RBLNDecoder` 가 HFDecoder 자리에 들어가고, 앱은 `--rbln` 을 주면 이걸 쓴다. 기본값(GPU)은 그대로다.
- 서버 앱 경로로 161건을 다시 돌린 결과는 2-2절과 같다: 자모열 완전 일치 157/161, 판정이 모두 같은 발화 158/161, 자리 판정 99.93%.
- 처리 시간(서버 안, `/speak` 1건: 디코드·검사·특징·NPU·정렬·점수·저장): 중앙값 **64 ms**, 최대 204 ms. `analyze()` 만은 중앙값 41 ms.
- GPU 서버에서 tailnet 으로 부르면 왕복 중앙값 581 ms 다. 두 서버 사이가 직접 연결이 아니라 Tailscale 중계(DERP 도쿄, 핑 97 ms)를 타기 때문이다.

| 항목 | 값 |
|---|---|
| 주소 | 공개 `https://<NPU-funnel-주소>:8443`(funnel) · tailnet `http://<NPU-tailscale-IP>:8932`. 공인 IP 포트는 닫혀 있다 |
| 인증 | 켬(2026-09-27). NPU 전용 토큰 `/root/pron_serve/.token`(권한 600) — GPU 서비스 토큰과 다르다. `/session` · `/reset` 도 토큰 필요. 데모 페이지는 토큰을 넣지 않고 브라우저에서 입력받는다(`--ask-token`) |
| NPU | 7번 장치 5.0 GiB (국적 서비스 포함 11.4 / 15.7 GiB) |
| 녹음 저장 | `/root/pron_serve/uploads` (GPU 서비스와 같은 동작) |
| 재시작 | `bash /root/pron_serve/start_pron.sh [장치]` · 로그 `/root/pron_serve/serve.log` |
| 번들 | `/root/pron_serve/app`(저장소 일부 + `.env`) · `data/`(점수·보정·L1 파일) · `g2p_tools/KoG2P` · `model_fp32/` |

남은 일:
1. **출처 표기.** AI 허브 이용정책상 원본 데이터는 외부 제공 금지이고, 학습한 모델의 결과를 주는 API 는 한국지능정보사회진흥원·AI 허브 출처를 밝히면 공개할 수 있다(2026-09-27 확인). 결과를 쓰는 앱 화면에 출처를 넣어야 한다 — 문구는 `docs/NPU_서버_API_안내.md` 4절(출처 표기). 서버 관리 정보는 `docs/NPU_서버_운영.md`.
2. 데모 페이지는 원래 토큰을 HTML 에 넣어 보내서, 공개 서버에서는 `--ask-token` 으로 토큰을 브라우저에서 입력받게 했다(`https://<NPU-funnel-주소>:8443/`).
3. 자동 재시작(systemd)이 없다. 서버가 재부팅되면 `start_pron.sh` 를 다시 돌려야 한다.
4. **정밀도 여지.** 공개된 컴파일 옵션에 계산 정밀도 설정이 없다. NPU 차이(2절)를 더 줄이려면 SDK 문서 확인이 필요하다.

## 6. 재현

| 무엇 | 어디 |
|---|---|
| 감싼 모듈 (마스크 제거·rel-shift) | `tasks/pronunciation-assessment/src/mdd/npu_w2vbert.py` |
| 컴파일 + 40클립 비교 + 버킷 지연 | `/data/speech-ai/out/novelty/pron_npu/npu_compile_check.py` (NPU 서버 `/root/pron_serve/`) |
| 원인 찾기 (조각·층 수·블록·길이) | 같은 폴더 `npu_part_check.py` · `npu_full_probe.py` · `npu_block_probe.py`, 로그는 `npu_logs/` |
| 실제 녹음 기준 만들기 (GPU fp16·fp32) | `e2e/e2e_prep.py` |
| NPU 로그확률 | `e2e/e2e_npu_run.py` (NPU 서버) |
| 서비스 경로 비교 | `e2e/e2e_compare.py` → `e2e/res_lp_npu.json` |
| GPU 지연 | `e2e/gpu_latency.py` → `e2e/gpu_latency.json` |
| GPU fp16 기준 잡음 (40클립) | `e2e/gpu16_ref_check.py` |

```bash
# NPU 서버 (/root/pron_serve, venv = /root/nat_serve/venv)
/root/nat_serve/venv/bin/python npu_compile_check.py both 192,448,1024     # 컴파일 약 3분 + 비교
/root/nat_serve/venv/bin/python e2e/e2e_npu_run.py /root/pron_serve/w2vbert_ctc_f192_448_1024.rbln
# GPU 서버 (저장소 루트, source tools/env.sh 뒤)
MDD_FORCE_CPU=1 PYTHONPATH=/data/speech-ai/tools/hf_py38:$PYTHONPATH $PY /data/speech-ai/out/novelty/pron_npu/e2e/e2e_compare.py lp_npu
```

2-2절 표와 3절 실제 녹음 지연은 6버킷 파일로 쟀다. 3버킷은 192·448·1024 그래프가 6버킷의 같은 길이 그래프와 지연이 같았다(29·94·429 ms).

## 재부팅 뒤 자동 시작 (2026-10-05)

NPU 발음 API(:8932)를 systemd `pron-api.service` 로 바꿨다(enabled).

- **실행**: `/root/pron_serve/run_8932.sh` 가 예전 `start_pron.sh` 와 같은 인자·토큰 파일로 포그라운드 실행한다.
- **기동 조건**: `rbln_daemon` 다음, `/dev/rbln7` 과 tailnet IP <NPU-tailscale-IP> 이 준비된 뒤 뜬다.
- **재시작**: `bash /root/pron_serve/start_pron.sh`(= `systemctl restart pron-api`)
- **로그**: `/root/pron_serve/serve.log` 에 이어 쓴다.
- **공개 입구**: Caddy(443 직통)도 `caddy-npu.service` 로 자동 시작한다. 자세한 건 국적 결과 13 끝에 있다.
