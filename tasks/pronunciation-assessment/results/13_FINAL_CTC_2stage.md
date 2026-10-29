# FINAL CTC — 2-stage (KsponSpeech → 외국인 TAG) (2026-09-15)

목표: 논문용 ablation 이 아니라 **서비스용 실제 발음 CTC 체크포인트 1개**.
구조·사전은 그대로(XLS-R 300M + 46자모 CTC, `labels=ltr`, 출력 50), 디코딩은 greedy CTC(LM·lexicon·prompt 없음).

## 결론

- FINAL = Stage 2 **2,000 스텝** 체크포인트 (valid_unseen JER 최소).
- **안 본 문장**에서는 지금까지 만든 모델 중 가장 좋다: E2 대비 JER 25.5% → **18.9%**, 정상 자리 오류 23.0% → **15.9%**,
  보존 45.7% → **49.9%**, MDD F1 0.204 → **0.264**. 되돌림은 E2 와 비슷하다(24.6% → 25.5%).
- 대신 **본 문장**(서비스 고정 문항)에서는 크게 나빠졌다: JER 5.4%(E2)·4.6%(OLD) → **14.2%**.
  문항을 외우지 않아 본 문장 되돌림은 60%(E2)·56%(OLD) → **26%** 로 떨어졌지만, 정상 자리 오류가 약 11% 라
  서비스에서 **억울한 지적이 OLD(1.8%)보다 6배 많이** 나올 수 있다.
- **OLD 를 그대로 대체하는 모델은 아니다.** 고정 문항 서비스에서는 OLD 가 정확도(JER·정상 자리)에서 앞서고,
  FINAL 은 표준발음으로 교정하지 않는 성질(되돌림 26%)과 새 문장 일반화에서 앞선다. 배포 여부는 결정이 필요하다.

## 1. 학습

| | Stage 1 | Stage 2 |
|---|---|---|
| 초기화 | `xlsr2_300m.pt` | Stage 1 `checkpoint_best.pt` (가중치만. 옵티마이저·LR·스텝 새로 시작 — 로그 `epoch 2 @ 0 updates`) |
| 데이터 | KsponSpeech clean train **531,162발화 / 691.2h** (발음 전사 → KoG2P → 46자모, ㅙ·ㅞ 분해) | `manifest_foreign_clean_v2` TAG 실제 발음 · cap1000 · LAR 없음 · **195,001발화 / 200h** |
| 검증 | Kspon dev 2,149 | valid_unseen 5,000 · valid 5,000 |
| lr · 스케줄 | 3e-5 · tri_stage · 인코더 동결 2k | **1e-5** · tri_stage · 동결 없음 |
| 스텝 | 10,000 (**1.3 epoch**, 1 epoch ≈ 7,700) | 5,000 (1k 마다 저장) |
| 공통 | max_tokens 1.6M × update_freq 4 · fp16 · mask 0.5 · seed 42 | 같음 |
| 시간 | 18:25 → 20:17 (A100 단독) | 20:18 → 21:04 |
| 설정 | `configs/final_stage1_kspon.yaml` | `configs/final_stage2_foreign.yaml` |

Stage 1 원어민 dev JER: 3k 19.3% → 5k 12.6% → 7k 11.3% → **10k 10.9% (best)**.

### Stage 2 체크포인트 선택 (valid_unseen, greedy 디코드)

규칙: JER 최소 → 동률(소수 4자리)이면 되돌림 낮은 것 → 보존 높은 것 (`scripts/eval/select_final_ckpt.py`).

| 스텝 | JER | 되돌림 | 보존 | 정상 자리 오류 | MDD F1 | fairseq valid(본 문장) UER |
|--:|--:|--:|--:|--:|--:|--:|
| 1,000 | 20.11% | 24.22% | 49.64% | 16.76% | 0.275 | 18.0% |
| **2,000 ★** | **20.07%** | 26.29% | 47.90% | 16.51% | 0.272 | 15.4% |
| 3,000 | 20.42% | 26.19% | 46.70% | 16.98% | 0.267 | 13.2% |
| 4,000 | 20.63% | 26.58% | 46.79% | 17.28% | 0.262 | 12.4% |
| 5,000 | 20.63% | 26.97% | 46.31% | 17.14% | 0.263 | 12.2% |

외국인 학습을 더 할수록 **본 문장** UER 은 18.0% → 12.2% 로 계속 좋아지지만 **안 본 문장**은 20.1% → 20.6% 로 나빠진다.
Stage 2 에서 새로 배우는 것의 대부분이 학습 문항 적응(암기)이라는 뜻이다. 1k 와 2k 의 JER 차이는 0.04%p 로 사실상 같다.

## 2. 결과 — OLD / E2 / FINAL

| Metric | OLD | E2 | FINAL |
|---|--:|--:|--:|
| Foreign Seen JER | **4.55%** | 5.44% | 14.19% |
| **Foreign Unseen JER** | 4.48%* | 25.48% | **18.93%** |
| ├ Substitution | 3.21% | 18.47% | 13.88% |
| ├ Deletion | 0.74% | 5.46% | 3.00% |
| └ Insertion | 0.53% | 1.55% | 2.04% |
| **Reversion (unseen)** | 58.43%* | **24.56%** | 25.50% |
| **Preservation (unseen)** | 35.48%* | 45.67% | **49.88%** |
| Other error (unseen) | 6.09% | 29.76% | 24.63% |
| **Normal-site error (unseen)** | **1.65%*** | 22.96% | 15.85% |
| MDD Precision / Recall (unseen) | 0.506 / 0.416 | 0.118 / 0.754 | 0.161 / 0.745 |
| MDD F1 (unseen) | **0.457*** | 0.204 | 0.264 |
| FAR / FRR (unseen) | 58.4% / 1.7% | 24.6% / 23.0% | 25.5% / 15.9% |
| Strict-text-unseen JER | 4.52% | 25.04% | 19.06% |
| **Reversion (seen)** | 55.72% | 60.22% | **26.08%** |
| 기존 Seen Weighted F1 | 0.9557 | 0.9490 | 0.9016 |
| Native JER (Kspon dev) | 18.59% | 12.64% | 15.82% |

\* OLD 는 287문항 전부로 학습해 test_unseen 문항도 **본 문장**이다(암기 이득 포함, 공정 비교 아님).
Strict-text-unseen 기준 문장 목록이 다르다: E2 = 혼합 학습 원어민 3.7만 문장, FINAL = Stage 1 원어민 49.6만 문장.

### 2-1. 과제별

| split | 과제 | OLD JER | E2 JER | FINAL JER | FINAL 되돌림 | FINAL 정상 자리 오류 | 완전 일치 (OLD / E2 / FINAL) |
|---|---|--:|--:|--:|--:|--:|---|
| unseen | RW 단어 | 5.6% | 29.0% | **21.7%** | 26.6% | 17.9% | 78.3% / 20.2% / **33.9%** |
| unseen | RS 문장 | 3.7% | 23.0% | **16.9%** | 24.0% | 14.4% | 49.3% / 0.1% / 1.0% |
| seen | RW 단어 | 5.4% | 6.7% | 15.2% | **25.3%** | 11.2% | 79.6% / 76.5% / 53.0% |
| seen | RS 문장 | 4.0% | 4.7% | 13.6% | **27.0%** | 10.2% | 52.9% / 50.4% / 12.8% |

## 3. 실제 예측 사례

OLD 대비 사례 수 (사례 원본: `experiments/final_ctc/compare_{OLD,E2}_vs_FINAL_{test_unseen,test_seen}/`, 유형별 100개 · 오디오 경로 포함)

| | A: OLD 되돌림 → FINAL 보존 | B: 둘 다 되돌림 | C: FINAL other | D: 정상 자리를 FINAL 이 틀림 |
|---|--:|--:|--:|--:|
| test_unseen | 2,288 | 2,065 | 2,268 | 32,525 |
| test_seen | 13,832 | 14,559 | 12,129 | 139,782 |

**A — 원했던 행동** (OLD 는 표준으로 교정, FINAL 은 들린 대로)

| 문항 | 자리 | LAB 표준 | TAG 실제 | OLD | FINAL |
|---|---|---|---|---|---|
| 일본 (unseen) | 모음 | ㅗ | ㅜ | ㅗ | **ㅜ** |
| 합법적으로 인정된… (unseen) | 초성 | ㅈ | ㅊ | ㅈ | **ㅊ** |
| 지금 (seen) | 초성 | ㄱ | ㄲ | ㄱ | **ㄲ** |
| 발 편한 운동화를… (seen) | 초성 | ㅂ | ㅍ | ㅂ | **ㅍ** |
| 은행이 … 텐데요 (seen) | 초성 | ㅌ | ㄸ | ㅌ | **ㄸ** |

**D — 서비스 위험** (정상 발음을 FINAL 이 틀리게 들음)

| 문항 | TAG(=표준) | OLD | FINAL |
|---|---|---|---|
| 김치 (unseen) | ㄱㅣㅁㅊㅣ | ㄱㅣㅁㅊㅣ | ㅋㅡㅁㅊㅣ |
| 배고파요 (unseen) | ㅂㅐㄱㅗㅍㅏㅛ | ㅂㅐㄱㅗㅍㅏㅛ | ㅂㅐㄱㅜㅃㅏㅛ |
| 흙마루 (seen) | ㅎㅡㅇㅁㅏㄹㅜ | ㅎㅡㅇㅁㅏㄹㅜ | ㅎㅡㅁㅁㅏㄹㅣ |
| 죄송하지만 못하겠습니다 (seen) | …ㅁㅗㅌㅏㄱㅔㄷㅆㅡㅁ… | 같음 | …ㅁㅗㄷㅎㅏㄱㅔㅆㅡㅁ… |

**B — 여전히 되돌림**: "더 드실 분" TAG `ㅂ` → OLD·FINAL 모두 `ㅃ` · "방송" TAG `ㅃ` → 둘 다 `ㅂ`.

## 4. 서비스 관점 판단 (결정 필요)

| | OLD | FINAL |
|---|---|---|
| 고정 문항에서 정상 발음을 틀리게 듣는 비율 | **1.8%** | 10~11% |
| 고정 문항에서 오발음을 표준으로 교정 | 56% | **26%** |
| 새 문장·단어 | 문항을 외운 이득이 없으면 약함(E1/E2 실험으로 추정) | **JER 18.9%** |
| MDD 지적 정밀도(unseen) | 0.51* | 0.16 |

- 지금 서비스처럼 **정해진 문항**을 읽게 하면, FINAL 로 바꾸는 순간 정상 발음에 대한 오탐이 크게 늘어난다.
- **자유 문장·새 교재 문장**으로 넓힐 때는 FINAL 이 유일하게 쓸 만한 후보다.
- 체크포인트 파일은 옵티마이저 상태가 들어 있는 3.8 GB 원본이다(서빙용 축소 전).

## 5. 산출물

| | 경로 |
|---|---|
| FINAL 체크포인트 | `$SPEECH_AI_DATA/checkpoints/FINAL_ctc/checkpoint_final.pt` (= `final_stage2_foreign/checkpoint_1_2000.pt`) · `dict.ltr.txt` · `SHA256SUMS`(`d337e8f2…`) · `SOURCE.txt` |
| 예측 | `$MDD_OUTPUTS/experiments/final_ctc/final_{test_seen,test_unseen,native_dev}.preds.jsonl` |
| 지표 | 같은 폴더 `*.metrics.json` · 발화별 `*.utt.jsonl` · 선택 근거 `final_selection.json` · `sel/` |
| 비교표 원본 | `$MDD_OUTPUTS/experiments/results_final_ctc.{md,json}` (저장소 사본 `13_FINAL_CTC_2stage_표.*`) |
| 학습 기록 | `$MDD_OUTPUTS/experiments/final_stage{1_kspon,2_foreign}/` (run_info · train.log · config) |

## 6. 재현

```bash
source tools/env.sh; T=tasks/pronunciation-assessment
# manifest_kspon_stage1 = manifest_kspon_native_v1 의 train + dev(→valid) 심볼릭 링크 + dict.ltr.txt
bash $T/scripts/train/run_ctc_experiment.sh final_stage1_kspon final_stage1_kspon 0
bash $T/scripts/train/run_ctc_experiment.sh final_stage2_foreign final_stage2_foreign 0 \
     checkpoint.restore_file=$SPEECH_AI_DATA/checkpoints/final_stage1_kspon/checkpoint_best.pt
for ck in $SPEECH_AI_DATA/checkpoints/final_stage2_foreign/checkpoint_*_*.pt; do
  $PY $T/scripts/eval/eval_actual_ctc.py score --manifest $MDD_OUTPUTS/manifest_foreign_clean_v2 --split valid_unseen \
      --ckpt $ck --out <dir>/$(basename $ck .pt)_valid_unseen; done
$PY $T/scripts/eval/select_final_ckpt.py --metrics <dir>/*_valid_unseen.metrics.json --out <dir>/final_selection.json
```
전체 자동화: `$MDD_OUTPUTS/experiments/pipeline_final.sh`. git `cb63efd`(+ 미커밋 스크립트) · fairseq `3d262bb` · torch 2.1.2+cu118 · A100 80GB.
