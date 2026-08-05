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

> **크기 비교로 인용할 때는 `0704`(2B) vs `0705`(300m) 쌍을 쓸 것.** 위 표의 현역
> `0.950` 은 043 아시아가 빠진 test 셋(133,509)이고 300m 의 `0.933` 은 043 포함
> (135,770)이라, 나란히 놓으면 크기 손실이 과장된다. 같은 셋에서는 0.941 vs 0.933 이다.
> 자세한 건 [`07_크기별_재평가.md`](07_크기별_재평가.md).

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

### `06_지식증류_2B_to_300m.json`

2B(교사) → 300m(학생) 선형 헤드 지식증류. 두 인코더가 동결이고 특징 캐시가 발화 단위로
정렬돼 있어 **GPU 없이** 수행했다 (유럽 041 은 행 순서가 달라 6조각 순열 복구 —
`scripts/data/recover_euro_sent_perm.py`, 발화 지문 cos 0.9993 로 검증).

| | top1 | top5 | macro5 |
|---|--:|--:|--:|
| baseline (출하본 재현) | 0.7512 | 0.9333 | 0.6181 |
| **KD (043제외, α=0.3 T=2)** | **0.7696** | **0.9412** | **0.6356** |
| clean Δ (2차 누수 제거) | +1.6%p | +0.4%p | **−1.7%p** |

> **이득과 비용이 갈린다.** 교사가 아는 도메인(505·041)에선 clean top1 +1.8%p 의 진짜
> 이득. 대신 꼬리 클래스 top5 가 내려간다 — ① 교사 미학습(043) 클래스로 무지가 새고
> ② 교사 자신이 약한 꼬리를 soft label 이 뭉갠다(KD 의 고전적 부작용).
> **트래픽 가중 지표가 목표면 KD 헤드, 소수 국적 공정성이 목표면 baseline.**
> 학습된 헤드: `outputs/kd_no043/kd_heads/kd_best_head.pt` (재학습 불필요).

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

# 지식증류 (CPU ~43분: 교사 로짓 + baseline + KD 스윕)
python $T/experiments/kd_2b_to_300m.py --kd_domains asia,korean,euro
```

원본 `metrics.json` 들은 `$NATIONALITY_OUTPUTS/<실험>/` 과 `$NATIONALITY_RUNS/<실험>/` 에
그대로 있다. **여기 2개는 그중 현역만 추린 사본이다.**
특징 캐시(`*.npz`, 7.6 GB)와 헤드(`head.pt`)는 저장소에 넣지 않는다.
