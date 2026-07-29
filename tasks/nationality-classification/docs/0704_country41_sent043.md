# 41클래스 국적 분류 — 043 선별 병합 best 모델 (2026-07-04)

505 + 한국어 + 041(문장 ≥4단어) + **043(선별 14클래스, 문장 prompt≥4단어)** 병합 41-class linear head.
성능: 자체 test **micro top5 0.9422 / macro 0.6321 (37/41 평가가능)**.
**동일 시험지(⑧ best의 test) 비교 시 micro 0.9589 / macro 0.7574 — 이전 best 전면 우세.**
출처별: 505 top5 0.9640 / 041 0.8714 / 043 0.5178 / 한국어 0.9998.

## 이 디렉토리 파일
| 파일 | 내용 |
|---|---|
| `head.pt` | 분류기 가중치 (`head` [41,1920]) + `mu`/`sd` + `labels`(41) + `c2i` + `layer=31`/`max_sec=12`/`euro_sent_only=True`/`asia043=True` |
| `metrics.json` | overall micro/macro + 출처별(505/041/043/한국어) |
| `label2id.json` | 클래스명 → 인덱스 |

## 의존 (이 head만으로는 추론 불가)
- **인코더(frozen)**: `$SPEECH_AI_MODELS/xlsr2_2B_1000k.pt` (26GB) 또는 `$SPEECH_AI_DATA/models/xlsr2_2B_1000k_inference.pt` (8.1GB)
- **feature 규격(반드시 동일)**: XLS-R → `layer 31` mean-pool(padding mask) → **max_sec 12s 절단** → clamp[-50,50] → `(x-mu)/sd`.

## 추론 레시피 (0702와 동일)
```python
import torch, torch.nn as nn
d = torch.load(".../head.pt", map_location="cpu")
head = nn.Linear(d["feat_dim"], len(d["labels"])); head.load_state_dict(d["head"]); head.eval()
x = (feat - d["mu"]) / d["sd"]
probs = head(torch.from_numpy(x).float()).softmax(-1)
top5 = [d["labels_en"][i] for i in probs.topk(5).indices.tolist()]   # 영어 표기
# d["labels"] = 한글(내부 기준) / d["labels_en"] = 영어(표시용, 같은 순서)
```

## 재현/근거
- 학습: `scripts/merge_country41_euro.py --euro 1 --euro_sent_only 1 --asia043 1 --save_head <dir>`
- feature: 기존 3종 + `$SPEECH_AI_DATA/taeho_layers_asia/train.npz` (043, 19,689발화/227화자/14클래스)
- 보고서: `$HOME/docs/country41_asia043_merge_experiment.md` (⑨단계)
- 검증: `$NATIONALITY_RUNS/diag_043/{frozen,anatomy}.{json,log}`

## ⚠️ caveat
- 진짜 하락은 칠레(−0.086)·필리핀(−0.055)뿐 (극소화자 클래스, 새 이웃에 확률 잠식).
- 이란 0.25 / 파키스탄 0.16 / 방글라·네팔 0.0 = 화자수 한계(각 14/10/2/4명) — 배포 시 화자 <5명 클래스는 drop 또는 "불확실" 처리 권장.
- 평가불가 4클래스: 에티오피아·르완다·잠비아·싱가포르.
