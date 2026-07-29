# 화자 변형(Speaker-Perturbation) 증강 실험 정리

작성: 2026-06-26 · 대상 split: `feature_resplit_61_merged6_seed42_target_counts` (62-class = 외국어 61 + 한국어, speaker-disjoint) · duration: **12s** (16s는 미실행, 맨 아래 TODO)

---

## 1. 목적

frozen XLS-R + 분류기 헤드만으로는 소수(롱테일) 클래스의 macro-F1이 ~0.19에서 막혔다.
원인 분석 결과, 발화 수보다 **학습 화자 수 부족**이 핵심(예: 투르크멘어 발화 1346개인데 화자 1명 → test top1 0%).
그래서 **원본 wav의 음색만 바꿔 "가상 화자"를 만들어** 화자 다양성을 늘리면 소수 클래스가 개선되는지 검증.

> 핵심 가설: 인코더는 안 건드리고(동결), 화자 변형 증강으로 train만 늘려 분류기를 더 강건하게.

---

## 2. 어떻게 증강했나 (방법)

스크립트: `augment_tail_wavs.py` · 백엔드: **Praat (parselmouth)** · 논문: accent-ID perturbation (Yang et al., arXiv:2306.06524)

원본 wav 1개에 아래 3가지를 **무작위 파라미터로** 적용해 변형본을 생성:

| # | 변형 | 내용 | 파라미터 |
|---|---|---|---|
| 1 | **포먼트 스케일링** | 성도(목·입) 길이 변형 → 체격감/굵기 변화 | β1 |
| 2 | **F0(피치) 스케일링** | 목소리 높낮이 변화 | β2 |
| 3 | **랜덤 피킹 EQ** | 3밴드 EQ로 음색 미세 변형 | f0∈[200,6000]Hz, Q∈[0.7,3.0], gain∈[−6,+6]dB |

- 비율: `β1, β2 ~ U(1.0, 1.4)`, 각각 50% 확률로 역수로 뒤집음 → 실효 범위 **[0.71, 1.4]**
- 변형본마다 β·EQ를 **새로 추출** → 같은 원본도 서로 다른 "가상 화자"가 됨
- **말 내용·발음·억양은 보존**(외국인 억양 단서 유지), **목소리(음색)만** 변경
- `speaker_id`는 **원본 그대로**(같은 사람) + `is_augmented=1` 표시

### 중요한 한계 (설계상)
증강 가짜 화자는 **기존 화자에서 파생**된다. 스페인어(화자 19명)를 아무리 증강해도 **진짜 새 화자가 19명을 넘지 않는다.** 효과는 "그 19명 안에서 음색 불변성 학습 → 새 화자에 약간 더 강건"까지가 한계.

---

## 3. 대상 클래스 & 설정

증강 타깃 = **mid 구간(화자 13~36명, top1 낮음)** 5개. 화자 1~3명짜리는 만들 다양성이 없어 제외.

| 타깃 | 61-class id | train 화자 | 증강 전 top1 |
|---|--:|--:|--:|
| 인도네시아어 | 8 | 36 | 26.5% |
| 광동어 | 9 | 26 | 10.4% |
| 말레이어 | 11 | 20 | 45.7% |
| 스페인어 | 13 | 19 | 10.4% |
| 타갈로그어 | 15 | 13 | 0.0% |

증강량: `--n_aug 8 --max_per_class 2000` → **클래스당 2,000개 × 5 = 10,000개**

---

## 4. 파이프라인 (단계 + 산출물)

| 단계 | 스크립트 | 산출물 |
|---|---|---|
| 1. 타깃 train 매니페스트 | `build_tail_targets_manifest.py` | `data/train_tail_targets.csv` (50,685행, **target_counts train 화자만**, 셔플) |
| 2. 증강 | `augment_tail_wavs.py` | `$SPEECH_AI_DATA/taeho/aug_wavs/` (10,000 wav), `data/train_augmented.csv` |
| 3. 청취 확인 | — | (말 내용 유지·음색 변화 확인) |
| 4. feature 추출 (12s, frozen) | `scripts/extract_features.py --max_sec 12` | `runs/features_aug_12s/train.npz` (10,000×1920) |
| 5. train에 append → 재학습/평가 | `decouple_eval_62class.py --extra_train_npz`, `perclass_table.py` | `runs/decouple62_12s_aug.json`, `runs/perclass_aug_12s.csv` |

- **valid/test는 절대 미변경** → speaker-disjoint 유지, leakage 없음.
- 인코더 동결, **분류기 헤드만 학습** (frozen feature 위 linear).
- 평가: foreign 61-way (한국어 제외) macro-F1 / top1 / top5 + head/mid/few 버킷.

---

## 5. 실험 결과 (12s, linear)

### 5-1. 집계 — 증강 전 → 후 (foreign 61-way)
| 지표 | 증강 전 | 증강 후 | 변화 |
|---|--:|--:|--:|
| macro-F1 | 0.1896 | 0.1879 | **−0.0017** |
| macro-F1 (mid 버킷) | 0.1630 | 0.1603 | **−0.0027** |
| top1 | 0.7551 | 0.7538 | ≈0 |
| top5 | 0.9194 | 0.9195 | ≈0 |
| head 버킷 | 0.7389 | 0.7342 | ≈0 |

→ 타깃들이 속한 **mid 버킷 F1이 오히려 소폭 하락.** 전체 macro도 미세 하락.

### 5-2. 타깃 5개 클래스 — 증강 전 → 후 (top1)
| 클래스 | 전 | 후 | Δ |
|---|--:|--:|--:|
| 인도네시아어 | 26.5 | 29.8 | +3.3 |
| 광동어 | 10.4 | 11.9 | +1.5 |
| 말레이어 | 45.7 | 40.7 | **−5.0** |
| 스페인어 | 10.4 | 9.1 | −1.3 |
| 타갈로그어 | 0.0 | 0.1 | ~0 |

→ 2개 소폭↑, 2개↓, 1개 무변화 = **net 무의미.**

### 5-3. recall vs F1 (왜 "오른 것처럼" 보이나)
일부 top1(=recall)이 오른 건 **"그 클래스로 더 자주 예측"**해서다. 그러면 오답(precision↓)이 같이 늘어 **균형 지표 F1은 안 오른다**(mid F1 오히려 ↓). 즉 catch율은 찔끔 늘었지만 실질 분류 품질은 개선 없음.

### 5-4. 분류기 헤드 5종 (증강 후, foreign macro-F1) — 참고
| head | top1 | macro-F1 |
|---|--:|--:|
| **linear** | 0.7538 | **0.1879** |
| tau_norm | 0.5952 | 0.1623 |
| cosine | 0.5231 | 0.1525 |
| crt | 0.7071 | 0.1805 |
| ncm | 0.0291 | 0.0152 |

→ 증강 후에도 **linear이 여전히 최고**, decoupling 헤드는 전부 더 나쁨. (tau_norm/cosine의 타깃 top1이 높아 보이는 건 head/전체 정확도를 깎아 tail로 예측을 몰아준 재조정 효과이지 증강 효과 아님.)

---

## 6. 결론

- **화자 변형 증강은 이 setup에서 효과 없음** (집계 macro 미세 하락, 타깃 net 무변화).
- 원인: **frozen 인코더 + 같은 화자 파생 증강 = 새 화자 정보가 안 생김.** 이미 이 클래스들을 못 가르는 인코더에 같은 화자의 음색 변형본을 넣어봐야 표현이 안 바뀐다.
- 앞선 결론(병목 = frozen 표현 + 화자 다양성 부족, 머리 교체로는 macro ~0.19 벽 못 넘음)과 **정확히 일치.**

### 의미
frozen 위에서의 모든 시도(불균형 처리 · classifier decoupling · 화자 변형 증강)가 macro ~0.19에서 막혔다. 이 벽을 넘으려면 **인코더 자체를 학습(finetune)** 하는 길뿐.

---

## 7. TODO / 다음

- [ ] **16s 버전**: 증강 wav는 그대로 두고 `extract_features.py --max_sec 16`로 재추출(`runs/features_aug_16s`) → `--duration 16s`로 동일 비교. (12s에서 무효였으니 16s도 비슷할 전망)
- [ ] 진짜 개선은 **인코더 partial finetune** — 단, 현재 `finetune_partial.py`는 ① 평가가 macro/micro **top-5**(F1 아님), ② `best_head.pt`에 **헤드만 저장**(인코더 미저장 → 오프라인 재평가 불가)라 패치 필요(전체모델 저장 + uncapped 풀test + 과적합 억제).
- [ ] (선택) recall만 중요하면 tau_norm 재조정이 타깃 catch율은 올려주지만 전체 정확도를 희생함 — 제품 목표에 따라 판단.

### 관련 파일
- 스크립트: `augment_tail_wavs.py`, `build_tail_targets_manifest.py`, `decouple_eval_62class.py`, `perclass_table.py`, `scripts/extract_features.py`
- 결과: `runs/decouple62_12s.json`(before), `runs/decouple62_12s_aug.json`(after), `runs/perclass_aug_12s.csv`, `runs/features_aug_12s/`
