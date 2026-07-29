# 41클래스 국적 분류 — 문장-병합 best 모델 (2026-07-02)

505 + 한국어 + **041(문장 ≥4단어만)** 병합 41-class linear head.
성능: **micro top5 0.9496 / macro top5 0.6754** (본 실험 best).
출처별: 505 top5 0.9615 / 041 top5 0.8743 / 한국어 top5 1.0. (metrics.json 참조)

## 이 디렉토리 파일
| 파일 | 내용 |
|---|---|
| `head.pt` | **분류기 가중치** (`head` state_dict [41,1920]) + `mu`/`sd`(표준화) + `labels`(41) + `c2i` + `layer=31`/`max_sec=12`/`euro_sent_only=True` |
| `metrics.json` | overall micro/macro top1/top5 + 출처별(505/041/한국어) |
| `label2id.json` | 클래스명 → 인덱스 |
| `per_class.csv` | 클래스별 test수/화자/top1/top5 |
| `per_class_domain.csv` | 클래스별 505 vs 041 출처 분해 |

## 의존 (이 head만으로는 추론 불가 — 아래 필요)
- **인코더(frozen)**: `$SPEECH_AI_MODELS/xlsr2_2B_1000k.pt` (26GB)
- **feature 규격(반드시 동일)**: XLS-R → `layer 31` mean-pool(padding mask) → **max_sec 12s 절단** → clamp[-50,50] → fp16 → `(x-mu)/sd`.
- 추출 코드 참고: `scripts/extract_layer_features.py`, `country_classifier/model.py`.

## 추론 레시피
```python
import torch, torch.nn as nn
d = torch.load(".../head.pt", map_location="cpu")
head = nn.Linear(d["feat_dim"], len(d["labels"])); head.load_state_dict(d["head"]); head.eval()
# feat = XLS-R layer31 mean-pool (max_sec 12, clamp[-50,50]) 로 뽑은 1920-d
x = (feat - d["mu"]) / d["sd"]               # 표준화 (mu/sd는 numpy, torch tensor로 변환)
logits = head(torch.from_numpy(x).float())
probs = logits.softmax(-1)
pred = d["labels"][int(probs.argmax())]
top5 = [d["labels"][i] for i in probs.topk(5).indices.tolist()]
```

## 재현/학습 근거
- 학습 스크립트: `scripts/merge_country41_euro.py --euro 1 --euro_sent_only 1 --save_head <dir>`
- feature: `$SPEECH_AI_DATA/taeho_layers_full/*.npz`(505) + `$SPEECH_AI_DATA/taeho_layers_korean/korean.npz` + `$SPEECH_AI_DATA/taeho_layers_euro/train.npz`(041)
- 문장 필터 마스크: `$SPEECH_AI_DATA/taeho_layers_euro/sent_mask_ge4.npy` (≥4단어)
- 보고서: `$HOME/docs/country41_euro_sentence_merge_experiment.md`

## ⚠️ caveat
- baseline/merged split 미고정, 노이즈 플로어 ±0.05 (보고서 8장).
- 505=모바일16k/문장, 041=웹48k → 배포(웹) 실측치는 041 블록(top5 ~0.87) 수준이 현실적.
- 극소수 화자 클래스(칠레·타지키 등)는 top5 ~0, 평가 무의미 → drop 권장.
