# 실험 기록 — 국적/모국어 분류

문서에 쓰인 수치가 **어느 실험에서 나왔는지** 여기서 확인한다.
흩어져 있던 `metrics.json` 들을 **한 파일로 병합**했고, 각 항목은 `source` 키에 원본 경로를 남겼다.

**모든 수치는 화자 분리(speaker-disjoint) 테스트다** — train 에서 한 번도 안 본 화자.
제품이 마주칠 미지의 화자를 반영하기 위해서다.

---

## 한눈에 — 주요 실험

| 실험 | 클래스 | top-1 | top-5 | macro top-5 | 비고 |
|---|--:|--:|--:|--:|---|
| frozen 2B + linear probe | 61 | 0.731 | 0.937 | – | 기준선 |
| country41 layer-31 12s | 41 | 0.833 | 0.963 | 0.578 | |
| **country41 + 유럽문장 병합** | 41 | 0.798 | **0.950** | **0.675** | ★현역★ 데모가 쓰는 헤드 |
| + 043 아시아 병합 | 41 | 0.785 | 0.942 | 0.632 | |
| 300m 백본 layer-17 | 41 | 0.751 | 0.933 | 0.617 | CPU ONNX 배포용 |

> **top-5 는 높은데 macro-F1 이 낮은 이유.** 상위 4개 클래스(중국어 26만 · 일본어 19만 ·
> 베트남어 17만 · 태국어 12만)가 데이터 대부분이고, 하위 수십 클래스는 발화 수백 이하다.
> 화자가 1명뿐인 클래스 20개는 speaker-disjoint 테스트에 아예 들어가지 못한다.

> **왜 현역이 top-1 최고가 아닌가.** `0629_country41_12s` 가 top-1 0.833 으로 더 높지만,
> **macro top-5 가 0.578 → 0.675** 로 오르는 쪽을 택했다. 소수 클래스를 버리고 얻은
> 전체 정확도는 제품에서 의미가 없다.

---

## 이 트랙의 결론 (중요)

frozen XLS-R 위에서 시도한 **모든 것**(decoupling · 불균형 처리 · 증강 · 다운샘플링)이
macro-F1 ~0.19 에서 막혔고, 부분 파인튜닝도 top-5 기준 frozen 을 못 넘었다.

**병목은 알고리즘이 아니라 소수 클래스의 학습 화자 수다.**
- LTAS/포먼트 분석 — 국적 음향 신호 자체가 약하다
- t-SNE — 표현이 **화자 지배적**이다. 화자 1명짜리 클래스는 0%

근거는 `../notebooks/` 의 4개 노트북(스펙트로그램·t-SNE·occlusion·GradCAM).
유일하게 효과 있는 방향은 **소수 클래스의 실제 화자를 늘리는 것**이다.

🚫 어족/권역 병합(31-class, 13-class)은 **중단**했다. `taxonomy13_중단` 항목은 참고용이다.
macro-F1 은 오르지만 제품이 답해야 할 질문이 바뀐다("이 사람 국적은?" → "어느 권역?").

---

## 파일별 안내

### `01_실험별_성능.json`

10개 실험의 테스트 성능. 병합 순서는 시간순이 아니라 **읽는 순서**다 —
기준선 → 불균형 처리 → country41 계보 → 배포용 → 중단된 방향.

`불균형처리_비교` 에 LA / class-balanced / oversample 변형이 다 들어 있다.
결론은 "**전부 macro-F1 을 못 올렸다**"이다.

### `02_클래스별_성능.csv`

현역 포함 4개 실험의 클래스별 top1/top5 (135행). `_실험` 열로 구분한다.
`0702_country41_sent_best` 는 도메인별(505 / 041) 성능도 함께 들어 있다 —
같은 모델이 코퍼스에 따라 얼마나 다른지 볼 수 있다.

---

## 재생성

```bash
cd <저장소 루트> && source tools/env.sh
T=tasks/nationality-classification

# 특징 캐시 (GPU, 수시간 → tmux)
python $T/scripts/data/extract_features.py \
    --manifest_dir $NATIONALITY_DATA/aihub_nationality_fine_811 \
    --out_dir $T/outputs/features_fine_811 \
    --splits test valid train --batch_size 32 --max_sec 12

# 캐시 위에 probe (CPU 몇 분)
python $T/scripts/train/train_probe.py \
    --features_dir $T/outputs/features_fine_811 \
    --out_dir $T/outputs/probe_fine_811 --epochs 60

python $T/scripts/eval/country41_per_class.py    # 클래스별 표
```

원본 `metrics.json` 들은 `$NATIONALITY_OUTPUTS/<실험>/` 과 `$NATIONALITY_RUNS/<실험>/` 에
그대로 있다. **여기 2개는 그중 현역만 추린 사본이다.**
특징 캐시(`*.npz`, 7.6 GB)와 헤드(`head.pt`)는 저장소에 넣지 않는다.
