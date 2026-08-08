# XLS-R 2B 외국인 한국어 발화 국적 분류 — Baseline

AIHub *"인공지능 학습을 위한 외국인 한국어 발화 음성 데이터"* (datasetkey 505)로 화자
국적을 분류한다. 백본은 로컬 XLS-R 2B 체크포인트(`models/xlsr2_2B_1000k.pt`)이며,
인코더를 freeze → mean pooling → linear head만 학습하는 가벼운 baseline이다.

라벨(6 coarse class): `chinese / english / japanese / thai / vietnamese / etc`
평가지표: Top-1 / Top-5(=`min(5,num_classes)`) / Macro-F1 / per-class acc / confusion matrix.

기존 `country_classifier/` 패키지(dataset/model/metrics/utils)는 **그대로 재사용**하고,
`scripts/`에 실행 진입점만 추가했다.

---

## 0. 데이터 위치 / 경로 리맵 (중요)

매니페스트(`data/<set>/*.tsv|.lang|.spk`)는 다른 머신에서 만들어져 TSV 첫 줄(root)이
`$PROJECT_ROOT/131.인공지능 학습을 위한 ...`(공백) 을 가리킨다.
이 서버의 실제 코퍼스는 여기에 있다(언더스코어):

```
$SPEECH_AI_DATA/aihub/foreign_korean_speech/131.인공지능_학습을_위한_외국인_한국어_발화_음성_데이터/01.데이터_new_20220719/
├── 1.Training/   (원천데이터/, 원천데이터_230113_add/  ← 약 300GB, zip 상태)
└── 2.Validation/ (원천데이터/, 원천데이터_230113_add/  ← 약 38GB,  zip 상태)
```

모든 스크립트는 `--audio_root`(기본값 = 위 `01.데이터_new_20220719`)로 매니페스트 root를
실제 경로로 리맵한다. root 끝의 `1.Training`/`2.Validation` 토큰은 자동 보존된다.

### 경로 해석: basename 인덱스 (기본 ON)
추출된 폴더 구조가 매니페스트의 리터럴 rel-경로와 **항상 일치하지는 않는다**. 예:
- Training 베트남어는 `원천데이터/TS_1.베트남어/TS_1. 베트남어/<카테고리>/...`로 **이중 중첩**
- 일부 폴더는 인코딩 변종 이름을 가짐

발화 파일명(`VN16RC139_VN0019_20210806.wav`)은 코퍼스 전역에서 유일하므로, 모든 스크립트는
기본적으로 코퍼스를 1회 스캔(`build_audio_index`, ~1M 파일 ≈ 3초)해 **basename→실제경로**로
해석한다. 덕분에 위 구조 quirk를 전부 흡수하며 5개 set 모두 train/valid/test 100% 해석된다.
리터럴 경로로 강제하려면 `--no_audio_index`. (검증: `inspect_manifests.py`에서 audio 20/20 present)

### 오디오는 아직 zip 상태 → 추출 필요

원천데이터 폴더에는 `.wav`가 아니라 `VS_1._베트남어.zip` / `TS_6._기타_add.zip` 같은
zip이 들어있다. zip **파일명**(`._`, 언더스코어)과 매니페스트 **rel 경로**(`VS_1. 베트남어`,
공백)가 다르므로, 추출 시 폴더명을 매니페스트 규칙으로 정규화해야 한다(`._` → `. `).
`scripts/extract_audio.py`가 이를 처리한다.

> ⚠️ **인코딩 함정(중요): 반드시 `extract_audio.py`로 풀 것.** 이 zip들은 내부 한글 경로
> (`1. 한국일반` 등)를 **CP949**로 저장하고 UTF-8 플래그가 없다. 일반 `unzip`이나 Python
> `zipfile`(특히 py3.8/3.10)로 그냥 풀면 `1. ╟╤▒╣└╧╣▌` 같은 **모지바케 폴더명**이 생겨
> 매니페스트(정상 한글)와 **0개 매칭**된다. `extract_audio.py`는 `cp437→cp949`로 이름을
> 복원하므로 인터프리터와 무관하게 올바른 한글 경로를 만든다.

```bash
# Validation 전체 wav만 추출(권장 시작점, 압축 38GB)
python scripts/extract_audio.py --split 2.Validation --only-ext .wav

# 한 언어만 (파이프라인 검증용, 영어 1.1GB)
python scripts/extract_audio.py --split 2.Validation --zip-glob 'VS_2._영어.zip' --only-ext .wav

# Training 전체(대용량 ~300GB, 디스크/시간 확인 후)
python scripts/extract_audio.py --split 1.Training --only-ext .wav
```

`scripts/inspect_manifests.py`의 `audio_probe`로 추출 성공 여부(`present/checked`)를 확인할 수 있다.

---

## 1. Conda 환경

`base`에는 torch가 없고 fairseq는 Python 3.13과 비호환이므로 `fairseq_xlsr`(py3.8) 환경을 만든다.

```bash
cd $HOME
bash tools/create_env.sh
conda activate fairseq_xlsr
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import fairseq; print('fairseq ok')"
```

설치 내용: torch 2.1.2 + torchaudio 2.1.2 (cu118), numpy<2, pandas, scipy, scikit-learn,
tqdm, pyyaml, soundfile, 그리고 `pip install -e $FAIRSEQ_ROOT`.

### fairseq 설치 트러블슈팅
- fairseq editable 설치는 **non-fatal**(실패해도 환경은 사용 가능). `tiny` 백엔드는 fairseq가
  필요 없고, **XLS-R(`fairseq_xlsr`) 백엔드에서만** fairseq가 필요하다.
- 실패 시:
  - `pip install -e $FAIRSEQ_ROOT --no-build-isolation`
  - 또는 빌드 없이 경로만: `export PYTHONPATH=$FAIRSEQ_ROOT:$PYTHONPATH`
- `conda create`가 ToS 에러로 실패하면(아래 명령은 스크립트에 이미 포함):
  `conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main` (그리고 `/pkgs/r`).

---

## 2. 전체 실행 순서

```bash
cd $HOME
conda activate fairseq_xlsr

# 1) 환경
bash tools/create_env.sh

# 2) 매니페스트 점검 (오디오 없어도 동작; 라벨분포/화자누수/경로 확인)
python scripts/inspect_manifests.py \
  --data_root $NATIONALITY_DATA \
  --output $NATIONALITY_RUNS/manifest_inspection.json

# 3) tiny 백엔드 smoke (fairseq/오디오 불필요 — 합성 wav 자동 사용)
python scripts/smoke_test.py \
  --manifest_dir $NATIONALITY_DATA/aihub_korean_nationality_train_quarter_valid_test \
  --backend tiny --batch_size 2 --max_sec 5

# 4) XLS-R 백엔드 smoke (체크포인트 로딩까지 검증)
python scripts/smoke_test.py \
  --manifest_dir $NATIONALITY_DATA/aihub_korean_nationality_train_quarter_valid_test \
  --backend fairseq_xlsr --xlsr_ckpt $SPEECH_AI_MODELS/xlsr2_2B_1000k.pt \
  --batch_size 1 --max_sec 5

# 5) baseline 학습 smoke (오디오 미추출이면 --synthetic 사용)
python scripts/train_country_classifier.py \
  --manifest_dir $NATIONALITY_DATA/aihub_korean_nationality_train_quarter_valid_test \
  --xlsr_ckpt $SPEECH_AI_MODELS/xlsr2_2B_1000k.pt \
  --output_dir $NATIONALITY_RUNS/xlsr_country_baseline_smoke \
  --batch_size 1 --grad_accum_steps 8 --epochs 1 --lr 1e-4 --max_sec 10 \
  --freeze_encoder true --amp --num_workers 2 --limit_train 200 --limit_valid 100

# 6) 평가 smoke
python scripts/eval_country_classifier.py \
  --manifest_dir $NATIONALITY_DATA/aihub_korean_nationality_train_quarter_valid_test \
  --split test --ckpt $NATIONALITY_RUNS/xlsr_country_baseline_smoke/best.pt \
  --output_dir $NATIONALITY_RUNS/xlsr_country_baseline_smoke_eval \
  --batch_size 1 --max_sec 10 --limit_samples 100

# 7) 전체 학습 (먼저 1.Training/2.Validation 오디오 추출 필요)
python scripts/train_country_classifier.py \
  --manifest_dir $NATIONALITY_DATA/aihub_korean_nationality_train_full_valid_test_no_train_speaker_test \
  --xlsr_ckpt $SPEECH_AI_MODELS/xlsr2_2B_1000k.pt \
  --output_dir $NATIONALITY_RUNS/xlsr_country_baseline_full \
  --batch_size 1 --grad_accum_steps 16 --epochs 5 --lr 1e-4 --max_sec 15 \
  --freeze_encoder true --amp --num_workers 2

# 8) 전체 test 평가
python scripts/eval_country_classifier.py \
  --manifest_dir $NATIONALITY_DATA/aihub_korean_nationality_train_full_valid_test_no_train_speaker_test \
  --split test --ckpt $NATIONALITY_RUNS/xlsr_country_baseline_full/best.pt \
  --output_dir $NATIONALITY_RUNS/xlsr_country_baseline_full_eval \
  --batch_size 1 --max_sec 15
```

> 5–6단계에서 해당 split 오디오가 아직 추출되지 않았다면 train/eval/smoke에 `--synthetic`을
> 추가해 코드 경로 전체를 합성 wav로 검증할 수 있다(국적 학습은 안 되지만 파이프라인은 검증됨).

---

## 3. 데이터셋 split 가이드

`inspect_manifests.py` 실측 결과(화자 누수는 `train↔test` 기준이 핵심):

| set | train | valid | test | train↔test 화자공유 | 비고 |
|---|---|---|---|---|---|
| `..._from_validation` | 104,207 | 11,578 | – | – | Validation에서만 생성, test 없음 |
| `..._from_validation_speaker_split` | 103,817 | 11,968 | – | train↔valid=0 | 가장 깔끔한 소규모 split |
| `..._train_full_valid_test` | 831,638 | 94,846 | 115,785 | **1,651 (누수)** | test에 train 화자 다수 |
| `..._train_full_valid_test_no_train_speaker_test` | 831,638 | 94,846 | 11,997 | **0 (권장)** | test가 train 화자와 분리 |
| `..._train_quarter_valid_test` | 207,009 | 26,331 | 115,785 | 389 | 빠른 실험용 1/4 train |

- **권장 baseline**: `..._train_full_valid_test_no_train_speaker_test` (train↔test 화자 누수 0).
  단, 이 set도 `valid↔test`는 199명 공유 → test 지표 해석 시 train-독립성만 보장됨.
- **빠른 실험**: `..._train_quarter_valid_test`.
- 라벨 불균형이 큼(`english`가 최소, `chinese/etc`가 최대) → **Macro-F1**을 주지표로 볼 것.

---

## 4. 산출물 레이아웃

```
runs/
├── manifest_inspection.json
├── xlsr_country_baseline_full/
│   ├── best.pt          # valid top1 기준 best (head + 메타; encoder는 체크포인트에서 재로딩)
│   ├── last.pt
│   ├── train_log.csv    # epoch,train_loss,val_loss,val_top1,val_top5,val_macro_f1,sec
│   ├── config.json
│   ├── label2id.json
│   └── id2label.json
└── xlsr_country_baseline_full_eval/
    ├── metrics.json         # top1/top5/macro_f1/per_class_acc
    ├── predictions.csv      # utt_id,audio_path,speaker_id,true_label,pred_label,top5_labels,top5_scores
    └── confusion_matrix.csv
```

> `best.pt`는 frozen 인코더 가중치를 저장하지 않고 **분류 헤드 + 메타데이터**만 저장한다
> (`model.checkpoint_state()`). eval 시 `--xlsr_ckpt`로 인코더를 다시 로딩한다.

---

## 5. OOM 대응

XLS-R 2B는 매우 크다. OOM 발생 시(스크립트가 안내 메시지 출력):
- `--batch_size 1` 유지(이미 최소면) → `--grad_accum_steps` 증가
- `--max_sec` 축소(예: 15 → 8)
- `--freeze_encoder true` 유지(기본값; 인코더 학습 시 메모리 폭증)
- `--amp` 사용

이 서버는 A100 80GB라 freeze + `max_sec≤15`, batch 1에서는 충분히 여유롭다.

---

## 6. 학습 방식 B: fairseq `audio_classification` task (참고)

`fairseq/fairseq/tasks/audio_classification.py`(upstream MMS task)는 `data/<set>`의
`.tsv/.lang/.spk/dict.lang.txt`를 **그대로** 사용한다(방식 A의 CSV 변환 불필요).
task가 기대하는 것:
- `--task audio_classification`, `task.data=<manifest_dir>`, `task.labels=lang`
- `dict.lang.txt`를 label dictionary로 로드(특수심볼 없이), `{split}.lang`을 타깃으로 사용
- TSV는 표준 fairseq `FileAudioDataset` 포맷(첫 줄 root + `상대경로<TAB>프레임수`)

**실행 전 체크리스트**
1. TSV root가 실제 오디오 경로를 가리키는가? (방식 B는 자동 리맵을 안 하므로, 추출된 오디오에
   맞게 각 `*.tsv` 첫 줄을 `$SPEECH_AI_DATA/.../{1.Training|2.Validation}`로 수정하거나 심볼릭 링크 구성)
2. `dict.lang.txt`와 `labels.txt`의 라벨 집합이 `.lang` 값과 일치하는가?
3. 분류용 head를 붙인 wav2vec2 모델 config가 필요(예: `examples/wav2vec`의 finetune config를
   classification용으로 수정). 본 baseline은 방식 A를 우선하며 B는 아직 학습 구성 미완성.

명령 초안(구성 확정 후 검증 필요):
```bash
fairseq-hydra-train -m \
  task=audio_classification task.data=$NATIONALITY_DATA/aihub_korean_nationality_train_full_valid_test_no_train_speaker_test \
  task.labels=lang \
  ... (model=wav2vec2 classification head, optimization, dataset.batch_size 등)
```

---

## 7. 생성/수정 파일

- 신규: `scripts/{create_env.sh, extract_audio.py, inspect_manifests.py, smoke_test.py,
  train_country_classifier.py, eval_country_classifier.py, baseline_common.py}`, `README_BASELINE.md`
- 기존 `country_classifier/` 코드는 **수정하지 않음**(import만). `baseline_common.py`가
  fairseq 매니페스트 → CSV 브리지 + 경로 리맵 + 합성 오디오 + 체크포인트 입출력을 담당.
