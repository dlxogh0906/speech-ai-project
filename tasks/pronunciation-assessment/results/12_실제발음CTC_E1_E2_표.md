# 실제 발음 CTC — OLD / E1 / E2 / E2-F

생성 2026-09-15 14:21 · greedy CTC (LM 없음)

| Metric | OLD | E1 | E2 | E2-F |
|---|--:|--:|--:|--:|
| Foreign Seen JER | 4.55% | 4.93% | 5.44% | 5.38% |
| Foreign Unseen JER | 4.48% | 33.11% | 25.48% | 26.07% |
|   Substitution | 3.21% | 21.03% | 18.47% | 18.82% |
|   Deletion | 0.74% | 11.04% | 5.46% | 5.66% |
|   Insertion | 0.53% | 1.04% | 1.55% | 1.59% |
| Reversion Rate (unseen) | 58.43% | 19.94% | 24.56% | 23.87% |
| Preservation Rate (unseen) | 35.48% | 44.12% | 45.67% | 46.39% |
| Other Error Rate (unseen) | 6.09% | 35.94% | 29.76% | 29.74% |
| Normal-site Error Rate (unseen) | 1.65% | 31.36% | 22.96% | 23.54% |
| MDD Precision (unseen) | 0.5061 | 0.0942 | 0.1180 | 0.1163 |
| MDD Recall (unseen) | 0.4157 | 0.8006 | 0.7544 | 0.7613 |
| MDD F1 (unseen) | 0.4565 | 0.1685 | 0.2040 | 0.2018 |
| FAR (unseen) | 58.43% | 19.94% | 24.56% | 23.87% |
| FRR (unseen) | 1.65% | 31.36% | 22.96% | 23.54% |
| Strict-text-unseen JER | 4.52% | - | 25.04% | 25.63% |
| Reversion (seen) | 55.72% | 60.21% | 60.22% | 60.88% |
| Existing Seen Weighted F1 | 0.9557 | 0.9521 | 0.9490 | 0.9492 |
| Native JER (Kspon dev) | 18.59% | 59.86% | 12.64% | 13.09% |
| Train utterances | 기존 서비스 모델 tag_all_native (재학습 안 함) | 195,001 | 232,495 | 15k 스텝 E2 혼합 데이터 → 5k 스텝 E1 외국인 데이터 |
| Train hours | 기존 서비스 모델 tag_all_native (재학습 안 함) | 200.0 | 250.0 | 15k 스텝 E2 혼합 데이터 → 5k 스텝 E1 외국인 데이터 |
| Unique texts | 기존 서비스 모델 tag_all_native (재학습 안 함) | 196 | 37,690 | 15k 스텝 E2 혼합 데이터 → 5k 스텝 E1 외국인 데이터 |
| Unique speakers | 기존 서비스 모델 tag_all_native (재학습 안 함) | 9,369 | 9,369 + 원어민(ID 없음) | 15k 스텝 E2 혼합 데이터 → 5k 스텝 E1 외국인 데이터 |
| Foreign/native ratio | 기존 서비스 모델 tag_all_native (재학습 안 함) | 100 / 0 | 80 / 20 (시간) | 15k 스텝 E2 혼합 데이터 → 5k 스텝 E1 외국인 데이터 |

## 읽는 법

- JER = (S+D+I)/N, 기준 = TAG 실제 발음. 자리 지표는 LAB 표준발음 축(치환·삭제형 오발음) — 삽입형은 JER 에만 들어간다.
- 보존 + 되돌림 + other = 100% (오발음 자리). FAR = 되돌림률, FRR = 정상 자리에서 표준과 다르게 낸 비율.
- ⚠️ OLD 는 287문항 전체로 학습됐다 — OLD 에게 test_unseen 문항은 **본 문장**이다. OLD 의 unseen 수치는 암기 이득이 섞인 참고값.
- Strict-text-unseen = test_unseen 중 원어민 학습 문장에도 없는 문항만 (E2·E2-F 는 원어민 문장 1개 '배고파요' 제외).
- Native JER 은 sanity check 이며 모델 선택 기준이 아니다.
