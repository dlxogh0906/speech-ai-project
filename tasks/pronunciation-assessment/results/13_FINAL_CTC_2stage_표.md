# 실제 발음 CTC — OLD / E1 / E2 / E2-F

생성 2026-09-15 21:35 · greedy CTC (LM 없음)

| Metric | OLD | E2 | FINAL |
|---|--:|--:|--:|
| Foreign Seen JER | 4.55% | 5.44% | 14.19% |
| Foreign Unseen JER | 4.48% | 25.48% | 18.93% |
|   Substitution | 3.21% | 18.47% | 13.88% |
|   Deletion | 0.74% | 5.46% | 3.00% |
|   Insertion | 0.53% | 1.55% | 2.04% |
| Reversion Rate (unseen) | 58.43% | 24.56% | 25.50% |
| Preservation Rate (unseen) | 35.48% | 45.67% | 49.88% |
| Other Error Rate (unseen) | 6.09% | 29.76% | 24.63% |
| Normal-site Error Rate (unseen) | 1.65% | 22.96% | 15.85% |
| MDD Precision (unseen) | 0.5061 | 0.1180 | 0.1607 |
| MDD Recall (unseen) | 0.4157 | 0.7544 | 0.7450 |
| MDD F1 (unseen) | 0.4565 | 0.2040 | 0.2643 |
| FAR (unseen) | 58.43% | 24.56% | 25.50% |
| FRR (unseen) | 1.65% | 22.96% | 15.85% |
| Strict-text-unseen JER | 4.52% | 25.04% | 19.06% |
| Reversion (seen) | 55.72% | 60.22% | 26.08% |
| Existing Seen Weighted F1 | 0.9557 | 0.9490 | 0.9016 |
| Native JER (Kspon dev) | 18.59% | 12.64% | 15.82% |
| Train utterances | 기존 서비스 모델 tag_all_native (287문항 전부 학습 · 재학습 안 함) | 232,495 | Stage1 KsponSpeech 531,162발화·691h (10k) → Stage2 외국인 TAG cap1000 195,001발화·200h (≤5k, valid_unseen JER 선택) |
| Train hours | 기존 서비스 모델 tag_all_native (287문항 전부 학습 · 재학습 안 함) | 250.0 | Stage1 KsponSpeech 531,162발화·691h (10k) → Stage2 외국인 TAG cap1000 195,001발화·200h (≤5k, valid_unseen JER 선택) |
| Unique texts | 기존 서비스 모델 tag_all_native (287문항 전부 학습 · 재학습 안 함) | 37,690 | Stage1 KsponSpeech 531,162발화·691h (10k) → Stage2 외국인 TAG cap1000 195,001발화·200h (≤5k, valid_unseen JER 선택) |
| Unique speakers | 기존 서비스 모델 tag_all_native (287문항 전부 학습 · 재학습 안 함) | 9,369 + 원어민(ID 없음) | Stage1 KsponSpeech 531,162발화·691h (10k) → Stage2 외국인 TAG cap1000 195,001발화·200h (≤5k, valid_unseen JER 선택) |
| Foreign/native ratio | 기존 서비스 모델 tag_all_native (287문항 전부 학습 · 재학습 안 함) | 80 / 20 (시간) | Stage1 KsponSpeech 531,162발화·691h (10k) → Stage2 외국인 TAG cap1000 195,001발화·200h (≤5k, valid_unseen JER 선택) |

## 읽는 법

- JER = (S+D+I)/N, 기준 = TAG 실제 발음. 자리 지표는 LAB 표준발음 축(치환·삭제형 오발음) — 삽입형은 JER 에만 들어간다.
- 보존 + 되돌림 + other = 100% (오발음 자리). FAR = 되돌림률, FRR = 정상 자리에서 표준과 다르게 낸 비율.
- ⚠️ OLD 는 287문항 전체로 학습됐다 — OLD 에게 test_unseen 문항은 **본 문장**이다. OLD 의 unseen 수치는 암기 이득이 섞인 참고값.
- Strict-text-unseen = test_unseen 중 원어민 학습 문장에도 없는 문항만 (E2·E2-F 는 원어민 문장 1개 '배고파요' 제외).
- Native JER 은 sanity check 이며 모델 선택 기준이 아니다.
