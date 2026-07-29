# 국적/모국어 분류 (nationality-classification)

외국인이 한국어로 말한 음성을 듣고 **화자의 모국어**를 맞힌다.
최종 목표가 "음성을 업로드하면 국적을 예측하는 웹 서비스"이므로 모든 분할은
**화자 단위로 겹치지 않게(speaker-disjoint)** 나눴다 — test 화자는 학습에서 한 번도 안 본 화자다.

## 성능 (실측)

| 실험 | 클래스 | top-1 | top-5 | macro top-5 | 산출물 |
|---|--:|--:|--:|--:|---|
| frozen 2B + linear probe (기준선) | 61 | 0.731 | 0.937 | – | `runs/probe_fine_811/` |
| country41 layer-31 12s | 41 | 0.833 | 0.963 | 0.578 | `0629_country41_12s/` |
| **country41 + 유럽문장 병합 ★현역★** | 41 | 0.798 | **0.950** | **0.675** | `0702_country41_sent_best/` |
| + 043 아시아 병합 | 41 | 0.785 | 0.942 | 0.632 | `0704_country41_sent043/` |
| 300m 백본(layer 17, ONNX 배포용) | 41 | 0.751 | 0.933 | 0.617 | `0705_country41_sent043_300m/` |

경로는 `$NATIONALITY_OUTPUTS` (기본 `$NATIONALITY_OUTPUTS`) 기준.

> **top-5 는 높은데 macro-F1 이 낮은 이유.** 상위 4개 클래스(중국어 26만·일본어 19만·베트남어
> 17만·태국어 12만)가 데이터 대부분을 차지하고, 하위 수십 클래스는 발화 수백 이하다.
> 화자가 1명뿐인 클래스 20개는 speaker-disjoint test 에 아예 들어가지 못한다.

> **핵심 결론(중요).** frozen XLS-R 위에서 시도한 모든 것(decoupling·불균형 처리·증강·다운샘플링)이
> macro-F1 ~0.19 에서 막혔고 부분 파인튜닝도 top-5 기준 frozen 을 못 넘었다.
> 병목은 알고리즘이 아니라 **소수 클래스의 학습 화자 수**다 — LTAS/포먼트 분석에서 국적 음향
> 신호 자체가 약하고, t-SNE 에서 표현이 화자 지배적이다. 근거는 `notebooks/` 참고.

> 위 수치의 **근거 파일은 [`results/`](results/)** 에 있다. 흩어져 있던 `metrics.json` 을
> 한 파일로 병합했고, 클래스별 성능은 CSV 로 정리했다.

## 라벨 체계

- **61-class** — 화자 모국어(`skill_info.motherTongue`). 캐노니컬 분할은 `aihub_nationality_fine_811`.
- **62-class** — 61 + `한국어_모국어`. "한국어인가 외국어인가" 판별을 겸한다 (한국어 F1 0.9995).
- **41-class** — 62-class 중 speaker-disjoint test 에 실제로 등장 가능한 것만. 현재 주력.
- 🚫 **어족/권역 병합(31-class, 13-class)은 하지 않는다.** 과거 실험은 중단됐고 `archive/` 로 갔다.

## 실행

모든 명령은 **저장소 루트에서** 실행한다. 아래 두 줄을 먼저 치면 `$PY`·`$T`·경로 변수가 잡힌다.

```bash
cd $HOME/speech-ai-project
source tools/env.sh                       # $PY 와 경로 변수 설정 (파이썬 쪽은 없어도 돌아감)
T=tasks/nationality-classification
```

> 어떤 스크립트든 **`--help` 를 먼저 쳐 보면 안전하다.** 인자 파서가 없는 스크립트에도
> `--help` 가드를 넣어 뒀으니, 실수로 본체가 실행되는 일은 없다.

### 0. 저장소가 멀쩡한지 (30초, GPU 불필요)

```bash
$PY tests/test_imports.py
$PY $T/tests/smoke_test.py --manifest_dir $NATIONALITY_DATA/aihub_nationality_fine_811 \
    --backend tiny --batch_size 2 --synthetic
```

### 1. 데이터 준비

```bash
# 코퍼스 zip 해제 — 반드시 이 스크립트로. plain unzip 은 CP949 때문에 폴더명이 깨진다.
$PY $T/scripts/data/extract_audio.py --split 2.Validation --only-ext .wav

# 라벨링 JSON → fine_labels.csv (모국어·국가)
$PY $T/scripts/data/build_fine_labels.py

# 61-class 화자분리 8:1:1 분할 (seed 42, 캐노니컬)   ⚠️ 기존 분할을 덮어쓴다
$PY $T/scripts/data/make_fine_split.py --out_dir $NATIONALITY_DATA/aihub_nationality_fine_811

# 분할 검증 — 발화 수·오디오 존재·라벨 분포·화자 중복
$PY $T/scripts/data/inspect_manifests.py --data_root $NATIONALITY_DATA \
    --output $T/outputs/manifest_inspection.json
```

### 2. 학습 — 특징 캐시 → probe (권장)

인코더가 동결이므로 end-to-end 와 **수학적으로 동일**하면서 100배 싸다.
한 번 캐시해 두면 헤드·lr·배치 스윕을 몇 분 만에 돌린다.

```bash
# 2-1) 평균풀링 1920-d 특징을 split 별 .npz 로 캐시   ⚠️ GPU · 수시간 → tmux 에서
tmux new -s feat
$PY $T/scripts/data/extract_features.py \
    --manifest_dir $NATIONALITY_DATA/aihub_nationality_fine_811 \
    --out_dir $T/outputs/features_fine_811 \
    --splits test valid train --batch_size 32 --max_sec 12

# 2-2) 캐시 위에 선형 헤드 학습 → test top1/top5/macro-F1  (CPU 몇 분)
$PY $T/scripts/train/train_probe.py \
    --features_dir $T/outputs/features_fine_811 \
    --out_dir $T/outputs/probe_fine_811 --epochs 60
```

부분 파인튜닝이 필요하면 (⚠️ GPU 장시간):

```bash
$PY $T/scripts/train/finetune_partial.py \
    --manifest_dir $NATIONALITY_DATA/aihub_nationality_fine_811 \
    --out_dir $T/outputs/ft_top8 \
    --unfreeze_top_layers 8 --cap_per_class 8000 \
    --batch_size 16 --epochs 10 --max_sec 12 --enc_lr 1e-5 --head_lr 1e-3
```

### 3. 평가

```bash
$PY $T/scripts/eval/eval_country_classifier.py \
    --manifest_dir $NATIONALITY_DATA/aihub_nationality_fine_811 \
    --split test --ckpt <head.pt> --output_dir $T/outputs/eval

$PY $T/scripts/eval/country41_per_class.py        # 클래스별 top1/top5 표
$PY $T/scripts/eval/plot_confusion_country41.py   # 혼동행렬
```

### 4. 배포용 내보내기

```bash
# 26 GB 체크포인트에서 optimizer state 제거 (추론용 축소)
$PY $T/scripts/export/create_xlsr_inference_checkpoint.py --help

# ONNX (300m layer-17) + int8 양자화 → CPU 서빙용   ⚠️ GPU 필요, 인자 없이 바로 실행됨
$PY $T/scripts/export/export_onnx_300m.py
$PY $T/scripts/export/quantize_onnx_300m.py
```

배포용 추론 서버는 [`deploy/`](../../deploy/README.md) 참고.

## 함정

1. **오디오는 `data/` 에 없다.** 매니페스트 1행의 root 는 다른 머신 경로(`/home/<타 사용자>/...`)다.
   모든 스크립트가 `--audio_root` 로 실제 위치에 remap 한다.
2. **디스크 폴더 구조가 매니페스트 상대경로와 항상 일치하지 않는다** (예: 베트남어 이중 중첩).
   발화 basename 이 전역 고유라서 **basename 인덱스**(`baseline_common.build_audio_index`,
   약 100만 파일 3초)로 해석한다. 이 덕분에 매니페스트가 100% resolve 된다.
3. **`collate_audio_batch` 가 `None` 을 반환할 수 있다** — 배치 전체가 스킵된 경우.
   호출부에서 반드시 처리할 것. `padding_mask` 규약은 `True = pad`.
4. **매니페스트 포맷이 2종이다** — fairseq(`.tsv/.lang/.spk`)와 CSV.
   `country_classifier/baseline_common.py` 가 둘 사이의 브리지다.
5. `data/` 안의 `*_from_validation*`, `*_train_quarter_*` 등은 **초기 실험용**이라 캐노니컬이 아니다.

## 디렉터리

| | |
|---|---|
| `src/country_classifier/` | 라이브러리. `audio`(torchaudio 없는 WAV 로더) · `dataset` · `model`(`build_model`) · `metrics` · `baseline_common`(브리지) · `make_speaker_split`(화자분리) |
| `scripts/data/` | 코퍼스 해제, 라벨 추출, 분할 생성, 특징 캐시 |
| `scripts/train/` | probe · 불균형 처리 · 부분 파인튜닝 · 스윕 |
| `scripts/eval/` | 평가 · 클래스별 표 · 혼동행렬 · 도메인 진단 · 시각화 |
| `scripts/export/` | ONNX · 체크포인트 축소 |
| `experiments/` | 일회성 스윕(cap/weighted sampling/decoupling) |
| `notebooks/` | 스펙트로그램·LTAS·t-SNE·occlusion·GradCAM — "왜 안 되는가"의 근거 |
| `docs/` | `HANDOFF.md` 부터 읽을 것 |
