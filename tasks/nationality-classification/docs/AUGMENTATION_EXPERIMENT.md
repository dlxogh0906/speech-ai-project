# 오디오 증강 실험 — 목적 / 방법 / 결과 / 해석

소수 국적(모국어) 클래스의 성능을 **오디오 증강**으로 끌어올릴 수 있는지 실제로 검증한 기록.
(상위 맥락은 `EXPERIMENTS.md` 참조. baseline = frozen XLS-R 2B + linear probe, 평가 = 화자 분리 test.)

---

## 1. 목적 / 의도

- **문제**: 61개 모국어 클래스 중 **~37개가 화자 1~4명**이라, 화자 분리 test에서 거의 0점.
  실제 배포 체감 지표인 **macro top-5 = 0.50**(=국적마다 동일가중)이 천장에 막혀 있음.
- **가설(검증 대상)**: 소수 클래스의 train 오디오를 **증강해 데이터를 늘리면**,
  그 클래스의 표현이 풍부해져 macro top-5가 오를 것이다.
- **의도**: "증강/생성으로 불균형을 풀어보겠다"는 팀 약속을, **추측이 아니라 실측 + 시각적 근거**로
  검증·설명한다. (성공이면 채택, 실패면 *왜* 실패하는지까지 시각화)

## 2. 방법 (정확히)

### 2-1. 어떤 증강을 썼나 (`scripts/augment_extract.py`)
소수 47개 클래스(train 발화 < 6,000개)의 train 오디오 각각에 대해 **4종 증강**을 적용,
원본 1개 → 증강 4개 생성:

| 증강 | 구현 | 음향 효과 / 의도 |
|---|---|---|
| **speed 0.9** | `resample(w, 16000, 16000/0.9)` 후 16kHz로 취급 | 느리고 낮게, **포먼트 하향** → 가짜화자에 가장 근접(ASR 표준) |
| **speed 1.1** | `resample(w, 16000, 16000/1.1)` | 빠르고 높게, **포먼트 상향** |
| **noise** | 가우시안 노이즈 **SNR 15dB** 가산 | 녹음환경 강건성(화자 정체성 불변) |
| **time-mask** | 0.3초 구간 2개 0으로 마스킹 | SpecAugment식, 구간손실 강건성 |

- **안 쓴 것**: pitch-shift(phase vocoder, 속도 이슈 — 단 speed-perturb가 피치도 이동), VTLP,
  RIR 리버브/MUSAN/코덱. (speed-perturb가 "가짜화자"에 가장 근접한 증강이라 핵심으로 채택)

### 2-2. 파이프라인
1. 소수 클래스 train 오디오 **29,050개**(클래스당 ≤1,000 cap) → 4종 증강 → **116,200개** 증강 오디오
2. 각 증강 오디오를 **frozen XLS-R 2B로 재추출**(mean-pool 1920-d) → `runs/features_aug/train_aug.npz`
   (증강은 GPU에서 처리, ~8 src/s, 약 1시간)
3. **원본 train(837,122) + 증강(116,200)** 으로 linear probe 재학습, **valid macro top-5로 best 선택**, test 평가
4. baseline(원본만) vs +증강 비교 (`scripts/train_probe_aug.py` → `runs/aug_result.json`)

## 3. 결과

### 3-1. 전체 (test top-5)
| | micro | **MACRO** |
|---|--:|--:|
| baseline (원본만) | 0.9308 | **0.4975** |
| **+ 오디오 증강** | 0.9296 | **0.4901** |
| **Δ** | −0.0012 | **−0.0074** |

→ **macro top-5가 오히려 0.007 하락. 개선 없음.**

### 3-2. 증강한 소수 클래스별 top-5 변화
- **상승**: 스와힐리 0.00→0.20(단 test n=5, 통계적 무의미), 네팔 +0.03, 프랑스 +0.02
- **하락**: 독일 −0.10, 타갈로그 −0.08, 포르투갈 −0.03, 폴란드 −0.03
- **화자 1~2명(카자흐/힌디/타지크) = 0.00 → 0.00 불변**
- → 변화는 전부 **노이즈 수준, 일관된 상승 없음, 합계 음수**.

## 4. 해석 — *왜* 안 됐나 (시각화로 증명)

핵심: **평가가 화자 분리라, 일반화의 단위는 "발화"가 아니라 "화자"다.**
그런데 증강은 **기존 화자의 변형**일 뿐 → 새 화자를 못 만든다.

### 시각적 증거
- **`runs/viz/B_by_speaker.png`** — XLS-R 특징공간이 **국적이 아니라 화자별로** 뭉침.
  소수 클래스 = 화자 몇 명의 작은 덩어리. 처음 보는 test 화자는 그 밖에 떨어짐.
- **`runs/viz/C_augmentation.png`** (핵심) — 한 소수 클래스(예: Kazakh)에서:
  - 색 = 화자. `o` = 원본 train, `+` = 증강, 빨간 `X` = test(처음 보는 화자)
  - **증강점(+)이 *같은 색 원본 화자 덩어리 위에만* 쌓임** → 밀도만 증가
  - **test 화자(빨간 X)는 멀리 떨어진 별개 영역** → 증강이 그 영역엔 점을 못 만듦
  - = "증강해도 처음 보는 화자를 커버 못 한다"를 그림으로 증명

### 동일 원인의 반복
이번 실패는 앞선 실패들과 **근본 원인이 같다**:
| 시도 | 결과 | 공통 원인 |
|---|---|---|
| 오버샘플링 | macro ↓ | 같은 목소리 복제 |
| 균등 샘플링 | macro ↓ | 같은 목소리 반복 |
| logit-adjustment | ↓ | 화자 다양성 X |
| 부분 fine-tuning ×2 | ↓ | 학습 화자 과적합 |
| **오디오 증강** | **macro −0.007** | **기존 화자 변형** |
| (리서치) TTS 생성 | 불가 | L1악센트 못 만듦 + 합성 갭 |

→ **"기존 화자에서 파생되는 모든 기법"은 화자 분리 test에서 막힌다.**

## 5. 결론

- **오디오 증강(speed/noise/time-mask, 116k 샘플)을 실제로 돌렸으나 macro top-5 0.498→0.490, 개선 없음.**
- 이유는 알고리즘이 약해서가 아니라 **구조적**: 소수 국적의 진짜 병목은 **화자 수**이고,
  증강은 새 화자를 만들지 못한다(시각화 C로 증명).
- **소수 국적을 실제로 살리는 유일한 확실한 길 = 그 국적의 진짜 화자를 더 수집.**
- 단기적으로는 **신뢰 가능한 ~8개 국적 + "불확실/기타" abstain**으로 정직한 배포본을 구성하는 것이 현실적.

## 6. 재현

```
PY=python
# 증강 + 재추출 (GPU)
$PY scripts/augment_extract.py --out runs/features_aug/train_aug.npz \
    --cap_per_class 1000 --max_train_per_class 6000 --batch_size 24 --max_sec 12
# 증강 추가 학습 + 평가
$PY scripts/train_probe_aug.py            # -> runs/aug_result.json
# 시각화
$PY scripts/viz_feature_space.py          # -> runs/viz/{A_by_class,B_by_speaker}.png
$PY scripts/viz_augmentation.py           # -> runs/viz/C_augmentation.png
```

---

## 7. 다음으로 해볼만한 기법 (멀티에이전트 리서치 + 검증)

문헌이 우리 진단을 **확증**: 화자 분리 평가에서 일반화의 단위는 **distinct speaker 수**이고,
speed/VTLP/VC-to-new-voice/오버샘플/부분FT/오디오증강은 전부 **기존 화자의 변형/재가중**이라
새 화자를 못 만들어 막힌다(speaker-recognition 학계 합의 + 우리 실측 일치).
→ **"기존 화자에서 파생되지 않는" 레버만 의미 있음.** 병목 해결 여부로 분류:

### 🟢 Tier 1 — 싸고 지금 바로 (표현/진단, 데이터 불필요)
1. **XLS-R 레이어 가중합 / 악센트 레이어 선택** *(병목: partly, cheap)* — **최우선 추천**
   - 악센트 정보는 **중상위 레이어(~9, 11-12)**, 화자 정체성은 **초기 레이어(1-7)**에 몰림.
     지금 우리는 최상위 레이어만 씀 → **학습형 레이어 가중합(SUPERB식)**으로 화자-과다 레이어를 피하면
     새 데이터 없이 화자 shortcut 의존을 줄임. (근거: arXiv 2306.06524, 2509.00230)
   - 비용 거의 0(재추출 1회), 우리가 앞서 말한 "중간 레이어 실험"이 이걸로 정당화됨.
2. **Non-timbral / speaker-disentangled 특징** *(병목: yes, medium)*
   - 인코더 표현에서 **화자 음색을 적대적으로 제거**(speaker-adversarial + KL)하고 악센트만 남겨 probe에 투입.
     "악센트-화자 entanglement"가 실패의 근원이라, **음색 제거**가 정공법. (근거: GenAID arXiv 2604.25332)
   - ⚠️ 검증자 주의: GenAID의 *VC 증강* 부분은 기존화자 파생(=막다른 길)이고, **non-timbral embedding 부분만** 실제 레버.
3. **NCM/프로토타입 헤드 + 더 나은 confidence (진단·캐 libration)** *(병목: no, cheap)*
   - 천장을 뚫진 못하지만 (a) 더 잘 보정된 baseline, (b) **진단**: kNN/NCM이 "꼬리 정확도가
     특정 1~4명 음색 암기인지"를 직접 폭로. 캐싱 feature로 즉시.

### 🟡 Tier 2 — 배포 계약 자체를 바꾸기 (높은 실효, 추천)
4. **계층적 coarse→fine 라우팅** *(promising)* — **배포의 정답**
   - 서비스 가능한 fine 국적은 fine으로, 아니면 **언어권/대분류(6-class)로 백오프**.
     대분류는 이 코퍼스에서 **이미 잘 됨**(`data/aihub_korean_nationality_speaker_811`).
     → 꼬리 화자에게 *기권*이 아니라 **맞는 상위 답**을 줌.
5. **margin/MaxLogit 기반 abstain (max-softmax 아님)** *(promising, cheapest)*
   - max-softmax confidence는 speaker-disjoint shift에서 순위가 뒤집힘 → **MaxLogit-pNorm/margin**으로 교체.
     (근거: arXiv 2405.05160, 2305.15508) 순수 post-hoc.
6. **Mahalanobis / kNN OOD 거부기**로 정직한 "불확실/기타" *(promising)*
   - 1920-d pooled feature에 비지도 Mahalanobis → 미지 방언/국적 거부가 softmax보다 우수.
     (근거: Das et al. Interspeech 2023, arXiv 2308.04886)

### 🟠 Tier 3 — 단발성 검증 트라이얼
7. **VC-as-NORMALIZER** *(병목: partly, medium)*
   - 증강(다양화)의 **반대**: 모든 발화를 **1~3개 고정 목소리로 변환**해 음색을 *제거* → 모델이 악센트에 집중.
     long-tail 화자분리에서 **유일하게 양(+)의 결과**(F1 0.404→0.420, arXiv 2507.03641). 단 이득 작고,
     변환이 **L1 악센트를 보존하는지 반드시 검증** 후 신뢰.

### 🔴 유일한 천장 돌파 — 진짜 새 화자 (데이터 중심, 비싸지만 확실)
8. **외부 L2-한국어 코퍼스 흡수로 새 *실제* 화자 추가** *(병목: yes)* — **최고 천장**
   - 다른 AIHub L2-한국어 코퍼스(예: **71479** 비중·일 아시아권 L1 1,500h, 71469 영어, 71490)를 흡수,
     L1 메타데이터를 61-class로 매핑, **전 소스 화자분리 강제**. 꼬리 국적에 **새 실화자** 공급.
   - 수집 시 **클래스당 distinct 화자 ≥10(train)+~20(test)** 을 목표로(화자 *수*가 핵심, 발화량/다양성 트릭 아님; 근거 arXiv 2506.04364).

### ⛔ 확정 막다른 길 (추가 검증으로 재확인)
- VC로 *새 목소리* 생성 / pseudo-speaker (기존 화자 파생), speed·VTLP(이미 실패),
  distribution-balanced loss·LWS·τ-norm(재가중만), accent-conversion-TTS 합성.

### 권장 순서
**Tier 1-1(레이어 가중합) → Tier 1-2(disentangle) 로 macro 천장 시도 → Tier 2(계층+abstain)로 정직한 배포본 → (예산 있으면) Tier 4(외부 코퍼스)로 진짜 돌파.**
Tier 1·2는 캐싱 feature/post-hoc이라 **수 시간 내** 검증 가능. Tier 4만이 꼬리를 근본적으로 살림.
