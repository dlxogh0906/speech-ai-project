# 외국인 한국어 발화 국적(모국어) 분류 — 실험 기록 및 분석

XLS-R 2B 인코더 기반, AIHub "외국인 한국어 발화 음성"(datasetkey 505)로 화자의
**모국어(=국적 프록시)** 를 분류. 라벨 = `skill_info.motherTongue`(JSON), 61개 클래스.
평가는 **화자 분리(speaker-disjoint)** test로 진행 — 실서비스(처음 보는 화자)를 반영.

이 문서는 지금까지의 모든 실험을 **목적 / 방법 / 결과 / 성공·실패 원인**으로 정리한다.

---

## 0. 핵심 결론 요약 (먼저)

| 구분 | micro top-5 | macro top-5 | top-1 | macro-F1 |
|---|--:|--:|--:|--:|
| **Frozen probe (최고 성능, 채택)** | **0.937** | **0.498** | 0.731 | 0.197 |
| 불균형 트릭(reweight/oversample/logit-adj) | ≤0.937 | ≤0.498 | — | ↓ |
| 균등 샘플링 | 0.902 | 0.484 | — | — |
| 부분 unfreeze FT (cap 2000) | 0.600 | 0.420 | — | — |
| 부분 unfreeze FT (cap 8000) | 0.694 | 0.439 | — | — |

- **micro top-5 0.937 = 코퍼스 분포 가중치(다수 4개국이 73%)** → 과대평가. **실제 배포 체감 = macro top-5 0.469~0.498**.
- **지금까지 시도한 어떤 기법도 frozen probe를 못 이김.** macro top-5 ~0.50이 **이 데이터의 천장**.
- 병목은 일관되게 **소수 국적의 화자 수**(발화 수 아님). 평가가 화자 분리라 화자 1~4명 클래스는 원천적으로 일반화 불가.

---

## 1. 데이터 파이프라인 (실험 전제)

### 1-1. 오디오 추출
- **목적**: 매니페스트가 참조하는 wav를 디스크에 확보.
- **방법**: `scripts/extract_audio.py`로 `원천데이터/*.zip`(Training 300GB + Validation 38GB) 해제. 총 wav 1,059,743개(중복 17,474 포함, 고유 1,042,269).
- **함정/해결**: zip 내부 한글 경로가 **CP949 + UTF-8 플래그 없음** → py3.8 zipfile이 CP437로 오독해 `1. ╟╤▒╣└╧╣▌` 모지바케 폴더 생성. `cp437→cp949` 재디코딩으로 복원. 또한 베트남어 Training은 `TS_1.베트남어/TS_1. 베트남어/...` 이중중첩 등 폴더구조 불일치 → **basename 인덱스**(파일명 전역 유일)로 해석해 100% 매칭.

### 1-2. 라벨 추출
- **목적**: 발화별 모국어 라벨.
- **방법**: `scripts/build_fine_labels.py`가 `라벨링데이터/*.zip`의 JSON을 **디스크에 풀지 않고 스트리밍**으로 읽어 `data/fine_labels.csv`(basename, speaker_id, mother_tongue, country) 생성. 1,042,269행 = 전체 발화와 일치.
- **라벨 필드 검증**: JSON에 깨끗한 "국적" 필드 없음. `skill_info.motherTongue`(모국어)가 출신 신호로 일관되고, `residence_info.country`는 **거주국**이라 230명(12%)이 KR로 오염(예: 베트남어 화자가 한국 5년+ 거주 → country=KR). → **motherTongue 채택**.

### 1-3. 화자 분리 8:1:1 split
- **목적**: 실서비스(미지 화자) 반영 + 공정 비교용 고정 기준선.
- **방법**: `scripts/make_fine_split.py` — 화자 단위로 train/valid/test 배정, 라벨 stratified, seed 42. → `data/aihub_nationality_fine_811/`.
- **결과**: train 837,122(화자 1,066) / valid 103,329(420) / test 101,818(424). **pairwise 화자 교집합 전부 0**(누수 없음). train엔 61클래스, **test엔 29클래스만**(나머지 32개는 화자 부족으로 test 미도달).

### 1-4. 클래스 화자수 분포 (이후 모든 분석의 핵심)
| tier | 화자수 | 클래스 수 | 비고 |
|---|---|--:|---|
| A | ≥20 | 13 | 실제 학습됨 (중/일/베/태/몽/러/영/미얀마/인니/광동/우즈벡/말레이/스페인) |
| B | 5~19 | 11 | 경계 |
| C | 2~4 | 17 | 거의 불가 |
| D | 1 | 20 | 원천 불가 (발화 수 무관) |

상위 4개(중/일/베/태)가 전체 발화의 **~73%**, 상위 13개(A)가 **~95%**.

---

## 2. 실험 E1 — Frozen XLS-R + linear probe (베이스라인)

- **목적**: 사전학습 XLS-R 표현이 이 태스크에 담은 정보량의 하한선 측정. 잡기술 없는 baseline.
- **방법**:
  1. `scripts/extract_features.py` — 인코더 freeze, mean-pool 1920-d feature를 train/valid/test 전부 캐싱(`runs/features_fine_811/*.npz`, max_sec 12, ~7h, ~40 utt/s).
  2. `scripts/train_probe.py` — 표준화 후 linear head(CE) 60 epoch, valid top-5로 best 선택.
  - frozen이므로 feature가 불변 → end-to-end frozen 학습과 **수학적으로 동일**, 비용만 ~100배 절감.
- **결과 (test)**: **top-5 0.937 / top-1 0.731 / macro-F1 0.191** (test 등장 29클래스).
- **분석 (왜 이런 결과)**:
  - top-5/top-1이 높아 보이는 건 **다수 4개국이 test의 73%**라 집계지표를 떠받쳐서. 이들은 인코더가 잘 가름.
  - macro-F1 0.19로 낮은 건 **꼬리 클래스 다수가 0점**(화자 부족) → 클래스 평균이 깔림. 불균형의 정직한 신호.
  - **성공/실패 판정**: baseline으로서 **성공**(기준점 확보). 다만 이 숫자는 다수클래스 편향이라 이후 macro 지표로 재평가 필요(E4).

---

## 3. 실험 E2 — 불균형 대응 트릭 (reweight / oversample / logit-adjust)

- **목적**: 소수 클래스를 살려 macro 성능을 올릴 수 있나? (캐싱 feature 위에서 수 분 실험)
- **방법**: `scripts/train_probe_imbalance.py` — 동일 frozen feature에 6개 변형, valid macro-F1로 best 선택.
  - class-balanced(effective-number, Cui), oversampling(클래스 균등), logit-adjustment(Menon, 사후/학습), 조합.
- **결과 (test)**:
  | 변형 | top1 | top5 | macro-F1 |
  |---|--:|--:|--:|
  | baseline | 0.734 | 0.937 | **0.197** |
  | class_balanced | 0.732 | 0.937 | 0.183 |
  | oversample | 0.697 | 0.933 | 0.154 |
  | posthoc_LA(τ=1) | 0.628 | 0.835 | 0.124 |
  | la_train(τ=1) | 0.257 | 0.597 | 0.089 |
  | cb+la | 0.142 | 0.433 | 0.054 |
  - τ 스윕(사후 LA): τ=0.2에서 B티어 F1 0.177→0.197(거의 공짜), τ=0.5에서 0.210(top1/5 소폭 손해), C티어는 어떤 τ에서도 ~0.01 고정.
- **분석 (왜 실패)**:
  - **전체 macro-F1을 올린 변형이 하나도 없음.** 중간(B)티어만 +0.02~0.03, 꼬리(C/D)는 불변.
  - LA τ=1·oversample·cb+la는 다수클래스를 망가뜨려 **오히려 크게 하락**(과보정/소수화자 암기).
  - **근본 원인**: 재가중/복제는 *새 화자*를 못 만든다. 화자 1~4명 클래스는 sampling을 어떻게 바꿔도 그 목소리만 외워 미지화자 test에서 0점. → 불균형은 분포 문제가 아니라 **화자 수 문제**.

---

## 4. 실험 E3 — 균등 샘플링 vs 인스턴스 샘플링 (macro top-5 기준)

- **목적**: "학습 시 클래스 균등 샘플링하면 macro가 오르나?"를 배포 지표(macro top-5)로 직접 검증.
- **방법**: 동일 feature, valid macro top-5로 best 선택. 인스턴스 랜덤(현재) vs 클래스 균등(`multinomial(1/cnt)`) vs logit-adjust(τ=0.5).
- **결과 (test)**:
  | 방식 | micro top-5 | macro top-5 |
  |---|--:|--:|
  | 인스턴스 균등(현재) | 0.931 | **0.498** |
  | 클래스 균등 샘플링 | 0.902 | 0.484 |
  | logit adjust(τ=0.5) | 0.829 | 0.466 |
- **분석 (왜 실패)**: 균등 샘플링이 macro도 못 올리고 micro까지 깎음. E2와 동일 원인 — 소수 클래스에 다양한 화자가 없어 오버샘플=같은 목소리 반복 → 일반화 0. **현재 전체데이터 인스턴스 샘플링이 frozen 안에선 이미 최선.**

---

## 5. 실험 E4 — micro vs macro top-5 진단 (지표의 진실)

- **목적**: 0.937 top-5가 실제 배포 성능을 반영하나? 클래스 동일가중(macro)으로 재평가.
- **방법**: baseline 로짓에서 per-class top-5 계산 후 평균.
- **결과 (test)**:
  - **micro top-5 = 0.938** (코퍼스 분포 가중)
  - **MACRO top-5 = 0.469** (국적마다 동일 = 실제 배포 체감)
  - top-5 ≥ 0.8 (믿을만): **8개** — 중국어0.998/베트남0.994/일본0.986/태국0.985/몽골0.984/러시아0.981/미얀마0.870/영어0.868
  - top-5 < 0.5 (못맞춤): **15개** — 광동0.49 … 키르기스/카자흐/힌디/타지크 = **0.00**
- **분석 (성공 — 중요한 진단)**: **0.937은 다수국적이 떠받친 허수.** 임의 국적 사용자가 왔을 때 기대 top-5는 **0.47**. 실제로 신뢰 가능한 국적은 ~8개뿐. → 이후 모든 개선의 목표 지표를 **macro top-5**로 전환.

---

## 6. 실험 E5 — country(거주국) 라벨로 진짜 "국적" 분류 시도

- **목적**: motherTongue 대신 실제 국적(country)으로 더 세밀하게 가면?
- **방법**: 동일 feature, 라벨만 `residence_info.country`(화자→대표국적)로 교체. 71개국 클래스, linear probe.
- **결과 (test)**: top-1 0.592 / **top-5 0.872** / top-10 0.926.
- **분석 (왜 실패)**:
  1. **top-5가 0.9 미만으로 하락** — 같은 모국어를 쓰는 여러 국가(중국/대만/홍콩/싱가포르, 미국/영국, 페루/콜롬비아/스페인)는 **한국어 악센트가 동일** → 모델이 원리적으로 못 가름.
  2. country 필드가 **거주국 오염**(KR 230명) → 라벨 자체가 노이즈.
  - **결론**: 진짜 country 단위 국적은 (a) 물리적으로 불가 + (b) 라벨 부정확 → **motherTongue이 악센트가 허용하는 최대 granularity.**

---

## 7. 실험 E6 — 클래스 병합 (학습되는 것만 + "기타")

- **목적**: 못 살리는 소수 클래스를 하나의 "기타"로 묶으면 깔끔해지나?
- **방법**: 동일 feature, train 화자 ≥5인 클래스(17개)만 유지 + 나머지 44개 → "기타(희소)" = **18클래스**. linear probe.
- **결과 (test)**: top-1 0.727 / **top-5 0.942** / **macro-F1 0.415**(61클래스 0.197 대비 2배+). 단 **"기타" 클래스 자체 F1 = 0.144**(precision 0.10, recall 0.28).
- **분석 (부분 성공 + 한계)**:
  - macro-F1이 오른 건 **0점 클래스 30여 개를 분모에서 제거**한 효과(정직화) — 모델링상 옳음.
  - 그러나 **"기타"는 44개 무관한 악센트 잡탕**이라 음향적으로 일관된 클래스가 아님 → frozen feature에서 깔끔히 못 가름(F1 0.14). 희귀 화자는 여전히 다수클래스로 샘. → "기타"는 *학습 클래스*보다 *추론 시 confidence abstain*이 적합.

---

## 8. 실험 E7 — 부분 unfreeze fine-tuning (cap 2000)

- **목적**: frozen 천장(macro 0.50)을 인코더 학습으로 뚫기.
- **방법**: `scripts/finetune_partial.py` — XLS-R **상위 8레이어 unfreeze + head**, layer-wise LR(enc 1e-5/head 1e-3), 오디오 end-to-end. 속도 위해 **cap 2000/클래스(≈67k)**, batch 8, 4 epoch, max_sec 8.
  - 기술 이슈 해결: AMP × fairseq activation-checkpointing 재계산 dtype 충돌 → **체크포인팅 끄고 AMP**(127s→16s/240샘플, 8배 가속).
- **결과 (test)**: best macro 0.420 / micro ~0.60. train_loss 0.054.
- **분석 (왜 실패)**:
  1. **데이터 cap 2000이 다수클래스를 굶김**(frozen은 837k 사용) → micro 0.93→0.60 붕괴.
  2. **과적합**(train_loss 0.05) — 작은 set의 화자 음색 암기 → 미지화자 test 역효과.
  3. best를 test로 선택한 경미한 누수도 있었음. → E8에서 두 결함 수정.

---

## 9. 실험 E8 — 부분 unfreeze fine-tuning (cap 8000, 결함 수정판)

- **목적**: E7의 두 결함(데이터 부족·test 누수)을 고쳐 FT를 공정하게 재검증.
- **방법**: 동일 스크립트, **cap 8000/클래스(train 150,502)**, batch 16, 3 epoch, max_sec 8, **valid로 best 선택 / test는 보고만**. ~6h(22.4 utt/s).
- **결과 (test)**:
  | epoch | train_loss | valid macro | test macro | test micro |
  |--:|--:|--:|--:|--:|
  | 1 | 0.435 | 0.357 | 0.432 | 0.703 |
  | 2 | 0.084 | 0.351 | 0.419 | 0.689 |
  | 3 (best) | 0.047 | 0.387 | **0.439** | **0.694** |
- **분석 (왜 또 실패 — 결정적)**:
  - cap을 4배(8000)로 늘렸는데도 **micro 0.93→0.69, macro 0.50→0.44 둘 다 하락.** 즉 **데이터량 문제가 아님.**
  - train_loss 0.047 = 또 과적합. **인코더를 학습 화자에 적응시키는 것 자체가 speaker-disjoint 일반화를 해침.**
  - frozen 다국어 표현이 더 robust("heavy fine-tuning hurts"의 전형). → **FT는 이 데이터에서 막다른 길로 확정**(2회 재현).

---

## 10. 종합 분석 — 왜 모든 개선이 막혔나

1. **단일 병목 = 소수 국적의 화자 수.** 평가가 화자 분리라 일반화의 단위는 *발화*가 아니라 *화자*. 화자 1~4명(37/61 클래스)은:
   - oversample/reweight/balanced-sampling → 같은 목소리 반복 → 일반화 0 (E2,E3)
   - fine-tuning → 그 화자에 과적합 → 오히려 하락 (E7,E8)
   - TTS 증강 → L1-악센트 한국어를 못 만듦(별도 리서치) → 불가
2. **macro top-5 ~0.50이 천장.** frozen이 최고. 다수 8개국은 잘 되고(≥0.8), 나머지는 데이터가 허락 안 함.
3. **micro 0.937은 분포 가중 허수** — 실서비스에선 macro 0.47이 진짜.
4. **country 단위 국적은 원리적 불가**(같은 언어권 국가 = 같은 악센트). motherTongue이 상한.

## 11. 남은 레버 (정직한 우선순위)

| 레버 | 비용 | 기대 | 상태 |
|---|---|---|---|
| **A. 중간 레이어 feature / 레이어 가중합** | 중(재추출 ~7h, 학습 0) | frozen 0.50 소폭 상회 가능(과적합 없음) | 미시도 (다음) |
| **B. LoRA** | 고(~6h) | 학습 파라미터 ↓로 과적합 덜함, 단 기대 낮음 | 미시도 |
| **C. 소수 국적 화자 수집** | 매우 고 | macro를 실제로 올리는 유일 확실 | — |
| **D. frozen 확정 + 신뢰 ~8-14클래스 + abstain** | 0 | 정직한 배포본(micro 0.93 충족) | 권장 종착 |

**계획: A 시도 → 안 되면 B → 그래도 안 되면 C/D.**
A·B가 frozen을 못 넘으면 macro 0.50이 데이터 천장임이 최종 확정되고, 개선은 데이터(C)로만 가능.

---

## 부록 — 재현 경로

```
PY=python
# 데이터
$PY scripts/extract_audio.py --split 1.Training --only-ext .wav   # + 2.Validation
$PY scripts/build_fine_labels.py                                  # -> data/fine_labels.csv
$PY scripts/make_fine_split.py                                    # -> data/aihub_nationality_fine_811
# E1 baseline
$PY scripts/extract_features.py --manifest_dir data/aihub_nationality_fine_811 \
    --out_dir runs/features_fine_811 --splits test valid train --batch_size 32 --max_sec 12
$PY scripts/train_probe.py --features_dir runs/features_fine_811 --out_dir runs/probe_fine_811 --epochs 60
# E2 불균형 / E3 샘플링 : scripts/train_probe_imbalance.py (+ 인라인 비교)
# E7,E8 FT : scripts/finetune_partial.py --cap_per_class 8000 --unfreeze_top_layers 8 ...
```

주요 산출물: `runs/probe_fine_811/`(baseline), `runs/probe_imbalance_fine_811/`,
`runs/ft_top8_fine811/`(FT), `runs/features_fine_811/*.npz`(캐싱 feature, 재학습용).
