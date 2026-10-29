# 실제 발음 CTC — 데이터 구성 실험 E1 / E2 / E2-F (2026-09-15)

**질문**: 모델·사전·레시피는 그대로 두고 **데이터 구성만** 바꾸면, CTC 가 "사용자가 실제로 낸 소리"를 더 잘 받아쓰는가
(문장 암기 → 표준발음으로 되돌림 감소 + 안 본 문장 일반화).

**결론**: **세 모델 모두 성공 기준을 통과하지 못했다.** 되돌림은 58% → 20~25% 로 줄었지만, 이유가 "더 잘 들어서"가
아니라 **안 본 단어·문장에서 전반적으로 못 받아써서**다 — 정상 자리 오류가 1.7% → 23~31% 로 폭증했다.
원어민 보조 데이터(E2)는 일반화를 분명히 도왔고(unseen JER 33.1% → 25.5%), 마지막 5k 외국인 적응(E2-F)은 효과가 없었다.
본 문장(test_seen)에서는 문항당 상한 1000 을 걸어도 되돌림이 OLD 보다 오히려 높았다(53~69%).

## 1. 설정

| | OLD | E1 | E2 | E2-F |
|---|---|---|---|---|
| 모델 | 서비스 `mdd_tag_all_native` (재학습 안 함) | XLS-R 300M + 46자모 CTC | 같음 | 같음 |
| 초기화 | – | `xlsr2_300m.pt` | `xlsr2_300m.pt` | E2 의 15k 체크포인트(옵티마이저·LR 포함) |
| 학습 데이터 | 287문항 전체 + LAR + 원어민 001 | 외국인 TAG 만 · 문항당 1000 · LAR 없음 | E1 + KsponSpeech 50h (시간 20%) | 15k 혼합 → 5k 외국인만 |
| 학습 발화 / 시간 | – | 195,001 / 200h | 232,495 / 250h | – |
| 고유 문장 | 287 | 196 | 37,690 | – |
| 스텝 | – | 20k | 20k | 20k (15k+5k) |

공통: lr 3e-5 · tri_stage(0.1/0.4/0.5) · max_tokens 1.6M × update_freq 4 · freeze 2k · mask 0.5 · fp16 · seed 42 ·
평가 = `checkpoint_last.pt`, greedy CTC(LM·lexicon·prompt 없음). 레시피 = `configs/e1_foreign_clean.yaml` 등.

평가 split (`manifest_foreign_clean_v2`, 화자 md5 분할은 기존 test 와 동일):
- **test_unseen (PRIMARY)** 23,562발화 — 새 화자 + 학습에서 뺀 30문항(RW 20 · RS 10). 누수 assert 통과
- test_seen 160,512발화 — 새 화자 + 학습 문항 (연속성 참고용)

⚠️ **OLD 는 287문항 전체로 학습됐다.** OLD 에게 test_unseen 문항은 본 문장이라 OLD 의 unseen 수치는 암기 이득이 섞여 있다.
공정한 unseen 비교 대상이 아니다(재학습 금지 조건이라 같은 홀드아웃의 OLD 레시피는 없다).

## 2. 결과

| Metric | OLD | E1 | E2 | E2-F |
|---|--:|--:|--:|--:|
| Foreign Seen JER | 4.55% | 4.93% | 5.44% | 5.38% |
| **Foreign Unseen JER** | 4.48%* | 33.11% | **25.48%** | 26.07% |
| ├ Substitution | 3.21% | 21.03% | 18.47% | 18.82% |
| ├ Deletion | 0.74% | 11.04% | 5.46% | 5.66% |
| └ Insertion | 0.53% | 1.04% | 1.55% | 1.59% |
| **Reversion (unseen)** | 58.43%* | **19.94%** | 24.56% | 23.87% |
| **Preservation (unseen)** | 35.48%* | 44.12% | 45.67% | **46.39%** |
| Other error (unseen) | 6.09% | 35.94% | 29.76% | 29.74% |
| **Normal-site error (unseen)** | **1.65%*** | 31.36% | 22.96% | 23.54% |
| MDD Precision / Recall (unseen) | 0.506 / 0.416 | 0.094 / 0.801 | 0.118 / 0.754 | 0.116 / 0.761 |
| MDD F1 (unseen) | 0.457 | 0.169 | 0.204 | 0.202 |
| FAR / FRR (unseen) | 58.4% / 1.7% | 19.9% / 31.4% | 24.6% / 23.0% | 23.9% / 23.5% |
| Strict-text-unseen JER | 4.52% | (=unseen) | 25.04% | 25.63% |
| Reversion (seen) | 55.72% | 60.21% | 60.22% | 60.88% |
| 기존 weighted F1 — seen / unseen | 0.9557 / 0.9554 | 0.9521 / 0.7783 | 0.9490 / 0.8305 | 0.9492 / 0.8268 |
| Native JER (Kspon dev) | 18.59% | 59.86% | **12.64%** | 13.09% |

\* OLD 는 이 문항들을 학습에서 봤다.

채점기 검증: OLD 의 기존 weighted F1 0.9557 은 문서의 `tag_all_native` 값(0.9558)과 일치한다.

### 2-1. 과제별 (단어 RW / 문장 RS)

| | split | OLD JER | E1 JER | E2 JER | E2-F JER | 예측이 TAG 와 완전 일치한 발화 (OLD / E1 / E2) |
|---|---|--:|--:|--:|--:|---|
| RW 단어 | unseen | 5.6% | 34.0% | 29.0% | 29.7% | 78.3% / 11.8% / 20.2% |
| RS 문장 | unseen | 3.7% | 32.5% | 23.0% | 23.5% | 49.3% / 0.0% / 0.1% |
| RW 단어 | seen | 5.4% | 5.9% | 6.7% | 6.6% | 79.6% / 78.0% / 76.5% |
| RS 문장 | seen | 4.0% | 4.3% | 4.7% | 4.7% | 52.9% / 52.0% / 50.4% |

본 문항에서는 세 모델 모두 OLD 수준인데, 안 본 문항에서는 한 단어짜리도 5개 중 4개를 틀린다.
모국어별(영어·인니·중국어·베트남어) unseen JER 도 E1 32~35% · E2 23~28% 로 고르게 높다 — 특정 L1 문제가 아니다.

### 2-2. OLD 대비 사례 (test_unseen, 오발음/정상 자리 수)

| | A: OLD 되돌림 → 새 모델 보존 | B: 둘 다 되돌림 | C: 새 모델 other | D: 정상 자리를 새 모델이 망침 |
|---|--:|--:|--:|--:|
| E1 | 1,842 | 1,599 | 3,310 | **67,523** |
| E2 | 1,920 | 1,911 | 2,741 | **48,598** |
| E2-F | 1,954 | 1,864 | 2,739 | **49,887** |

A 는 실제로 원했던 행동이다(예: "바꾸다" TAG `ㅂㅏㄲㅡㄷㅏ` — OLD `ㄲㅜ` 로 교정, 새 모델 `ㄲㅡ` 보존).
그러나 D 가 A 의 25~37배다(예: "법원" 정상 발음을 E1 이 `ㅂㅓㅝㄴ`, "노력" 을 E2 가 `ㄴㅜㅕㄱ`).
사례 원본: `$MDD_OUTPUTS/experiments/<실험>/compare_vs_old_test_unseen/{A,B,C,D}_*.jsonl` (유형별 100개, 오디오 경로 포함).

## 3. 해석

1. **되돌림 감소는 일반화의 결과가 아니라 인식 실패의 부산물이다.** 되돌림 = "예측 == 표준발음" 인데, 모델이 그 자리를
   아예 틀리면 되돌림도 보존도 아닌 other 로 간다. unseen 에서 other 가 6% → 30~36% 로 뛰었다.
2. **외국인 데이터 196문항으로는 일반 한국어 자모 인식이 학습되지 않는다.** 200시간이라도 음절 조합이 196개 문항에 갇혀
   있어, 처음 보는 단어는 음향만으로 받아쓰지 못한다. 원어민 50h(문장 3.7만)를 섞은 E2 가 unseen JER 을 7.6%p,
   정상 자리 오류를 8.4%p 줄였고, 원어민 dev JER 은 59.9% → 12.6% 로 좋아졌다 — 방향은 맞지만 양이 모자란다.
3. **문항당 상한 1000 은 본 문장 암기를 줄이지 못했다.** test_seen 되돌림 55.7%(OLD) → 60~61%. 1000 번 반복도 외우기엔 충분하다.
4. **E2-F(외국인 5k 재적응)는 E2 와 차이가 없다.** unseen JER 25.5% vs 26.1%, 보존 +0.7%p 뿐.
5. 과거 홀드아웃 실험(`08_문장암기_일반화.md`, 다른 30문항)의 unseen weighted F1 0.8218(전체 데이터)·0.8734(상한 1000+원어민 001)
   과 같은 계열이다 — E1 0.7783 · E2 0.8305. 이번 홀드아웃은 짧은 단어(RW 20개) 비중이 커 더 어렵다.

## 4. 다음 실험 제안 (이번 조건에서 벗어나는 것이라 결정 필요)

| 후보 | 근거 | 바꾸는 변수 |
|---|---|---|
| **원어민 비중 확대** (예: 50% · 원어민 수백 h) | E2 에서 원어민이 가장 큰 개선원. 20% 는 부족 | 데이터 비율만 |
| **2단계: KsponSpeech 로 한국어 자모 CTC 를 먼저 학습 → 외국인 TAG 로 적응** | 모델이 일반 한국어 음향-자모 대응을 먼저 갖춰야 안 본 단어를 받아쓴다 | 초기화(아키텍처·사전은 동일) |
| 상한 더 낮추기 (300·100) | 본 문장 되돌림이 1000 에서 안 줄었다 | cap (매니페스트 CLI 로 바로 가능) |
| 체크포인트 선택을 valid_unseen 기준으로 | 지금은 본 문장 valid loss 기준 best / last | 선택 기준만 (GPU 디코드 필요) |

## 5. 재현

```bash
source tools/env.sh; T=tasks/pronunciation-assessment
$PY $T/scripts/data/build_foreign_clean_v2.py --dict $MDD_DATA_DIR/dict.ltr.txt --out $MDD_OUTPUTS/manifest_foreign_clean_v2 --cap-per-prompt 1000 --seed 42
$PY $T/scripts/data/build_kspon_native_v1.py --dict $MDD_DATA_DIR/dict.ltr.txt --audit 500 --out-audit <md>
$PY $T/scripts/data/build_kspon_native_v1.py --dict $MDD_DATA_DIR/dict.ltr.txt --out $MDD_OUTPUTS/manifest_kspon_native_v1
$PY $T/scripts/data/build_mix_manifest.py --foreign $MDD_OUTPUTS/manifest_foreign_clean_v2 --native $MDD_OUTPUTS/manifest_kspon_native_v1 --native-ratio 0.2 --seed 42 --out $MDD_OUTPUTS/manifest_mix_f80_n20
bash $T/scripts/train/run_ctc_experiment.sh e1_foreign_clean e1_foreign_clean 0
bash $T/scripts/train/run_ctc_experiment.sh e2_native20 e2_native20 0
bash $T/scripts/train/run_ctc_experiment.sh e2f_foreign_adapt e2f_foreign_adapt 0 checkpoint.restore_file=<e2 checkpoint_*_15000.pt>
$PY $T/scripts/eval/eval_actual_ctc.py score --manifest $MDD_OUTPUTS/manifest_foreign_clean_v2 --split test_unseen --ckpt <pt> --out <prefix> [--exclude-texts $MDD_OUTPUTS/manifest_mix_f80_n20/native_train_texts.txt]
$PY $T/scripts/eval/summarize_actual_ctc.py --out <md>
```

산출물: `$MDD_OUTPUTS/experiments/{old_reference,e1_foreign_clean,e2_native20,e2f_foreign_adapt}/`
(run_info.json · train.log · config.yaml · *.metrics.json · *.preds.jsonl · *.utt.jsonl · compare_vs_old_test_unseen/),
체크포인트 `$SPEECH_AI_DATA/checkpoints/<실험>/`, 원본 결과표 `$MDD_OUTPUTS/experiments/results_actual_ctc.{md,json}`,
KsponSpeech 변환 audit `$MDD_OUTPUTS/experiments/kspon_native_v1/kspon_jamo_audit.md`.

재현 정보: git `cb63efd`(+ 미커밋 스크립트) · fairseq `3d262bb` · torch 2.1.2+cu118 · A100 80GB(KD 학습과 GPU 공유) ·
E1 03:43→12:34 · E2 03:54→13:23 · E2-F 10:40→13:23 · 매니페스트 sha256 은 각 `stats.json`.
