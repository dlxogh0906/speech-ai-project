# HANDOFF — 외국인 한국어 발화 기반 국적/모국어 분류

> 이 문서는 이 프로젝트를 처음 보는 협업자가 빠르게 따라잡을 수 있도록,
> 실제 코드/데이터/결과 파일을 직접 확인하고 정리한 인수인계 문서입니다.
> 작성 2026-06-26, **갱신 2026-06-27**. 작업 서버: 단일 호스트(`$HOME`), A100 80GB 1장.
> **이 디렉터리는 git 저장소가 아닙니다.** (버전 관리 미적용 — 7번 항목 참고)
>
> 📌 **실험 상세 종합본은 [`taeho/experiments_overall_summary.md`](experiments_overall_summary.md)** (모든 실험 전체 표).
> 노션용 요약 [`taeho/notion_summary.md`](notion_summary.md), 증강 상세 [`taeho/augmentation_experiment_summary.md`](augmentation_experiment_summary.md).
>
> ### 🔔 이 문서는 2026-06-27 기준이다 — 그 뒤로 최고 모델이 두 번 바뀌었다
>
> | 시점 | 최고 모델 | top-1 | top-5 | macro5 | 근거 |
> |---|---|--:|--:|--:|---|
> | 06-27 (이 문서) | frozen 2B + linear probe | 0.798 | 0.950¹ | 0.675 | `results/01` |
> | 07-05 | 300m ONNX (배포) | 0.751 | 0.933 | 0.617 | `results/01` |
> | **08-06** | **KD 인코더 (2B→300m 증류)** | **0.8245** | **0.9584** | **0.7124** | `results/07_인코더증류.json` |
>
> ¹ `0702` 의 0.950 은 **043 아시아가 빠진 다른 test 셋**(133,509)이다. 나머지와 같은 셋
> (135,770 · 043 포함)에서는 **0.9422** 다. 크기·성능 비교에 0.950 을 쓰면 2B 가 과대평가된다.
>
> **아래 "frozen 위에서는 macro-F1 0.19 가 한계" 라는 결론은 그 시점 기준으로는 옳았고,
> 인코더를 열어(층 12–17 학습) 그 한계를 넘었다.** 자세한 건
> [`지식증류_인수인계.md`](지식증류_인수인계.md) — KD 는 그 문서가 정본이다.
>
> 배포 자산: `/data/tta/kd_encoder/run2/deploy/{encoder_kd_l17.onnx, head_np.npz}`
> ⚠️ **인코더와 헤드는 짝이다.** 구버전 `head.pt` 와 섞으면 에러 없이 조용히 틀린다.
>
> ---
>
> ### ⭐ 당시(2026-06-27) 결론 — 아래는 그 시점의 기록이다
> **전제: 클래스 병합 안 함 — 61-class(+한국어) 그대로 유지.**
> frozen XLS-R 위에서 시도한 **모든 것**(decoupling·불균형처리·증강·다운샘플링)이 **macro-F1 ~0.19 / 한계**에 막히고,
> 부분 파인튜닝도 (top-5 기준) frozen을 못 넘었다. **근본 병목 = 소수 클래스의 "학습 화자 수" 부족** —
> 음향적으로 국적 신호 자체가 약하고(LTAS/포먼트 거의 동일), 표현이 **화자 지배적**(t-SNE)이라 화자 1명 클래스는 0%.
> → **유일하게 효과 있는 방향: ① 소수(유럽) 클래스 실제 화자 추가, ② weighted-sampling β≈0.75(top-5 소폭↑), ③ XLS-R 중간 layer feature(미시도).**

---

## 1. 개요

외국인이 한국어로 말한 음성을 듣고 **화자의 국적/모국어를 맞히는** 분류기를 만드는 작업입니다.
음성 인코더는 다국어 사전학습 모델 **XLS-R (wav2vec 2.0) 2B** 를 사용하고, 그 위에 분류 헤드를
얹습니다. 학습 데이터는 AIHub "외국인 한국어 발화 음성"(datasetkey 505)입니다.

최종 목표는 "외국어 억양이 섞인 한국어 음성을 업로드하면 국적을 예측해 주는 웹 서비스"이므로,
모든 데이터 분할은 **화자 단위로 겹치지 않게(speaker-disjoint)** 나눴습니다. 즉 테스트셋은
학습에서 한 번도 본 적 없는 화자만 포함합니다(서비스에서 마주칠 미지의 화자를 반영).

라벨 체계:
- **coarse 6-class** (AIHub 원본 manifest 라벨): `chinese/english/japanese/thai/vietnamese/etc`
- **fine 61-class** (화자 모국어 `skill_info.motherTongue`): 중국어/일본어/베트남어/… → **심한 롱테일**
  (중국어 ~263k 발화 vs 타밀어 4발화)
- **62-class** (61개 외국어 + `한국어_모국어`): 한국어 원어민까지 포함해 "한국어 vs 외국어" 판별 + 국적 분류 ← **현재 타깃**

> 🚫 **클래스 병합(어족/권역으로 묶는 31-class 같은 방식)은 하지 않는다.** 라벨은 **61-class(+한국어=62) 원본 그대로 유지.**
> (과거 권역 병합 실험이 있었으나 **중단**. 본 문서에서 병합 관련 내용은 제외함.)

---

## 2. 전체 구조

```
$HOME/
├── CLAUDE.md                     # 프로젝트 운영 가이드(환경·데이터 위치·gotcha). 필독.
├── README_BASELINE.md            # 베이스라인 파이프라인 운영 런북. 필독.
├── HANDOFF.md                    # (이 문서)
│
├── country_classifier/           # 라이브러리 코드(재사용용, 직접 재작성 X). import해서 쓰는 부분
│   ├── audio.py                  #   torchaudio 없이 stdlib wave + scipy로 16kHz mono WAV 로더
│   ├── dataset.py                #   AudioCountryDataset (CSV manifest 읽음). 읽기 실패 clip은 조용히 skip
│   ├── model.py                  #   build_model(): fairseq_xlsr 백엔드(체크포인트 로드·인코더 freeze·
│   │                             #     mask-aware mean-pool·ClassificationHead) / tiny 백엔드(스모크용)
│   ├── metrics.py                #   TopKAccuracy (top-1/top-5)
│   └── utils.py                  #   seed/json/labels 헬퍼
│
├── scripts/                      # 베이스라인 파이프라인(이 repo가 추가한 실행 스크립트). 전부 fairseq_xlsr 환경
│   ├── create_env.sh             #   conda 환경 fairseq_xlsr 빌드
│   ├── extract_audio.py          #   corpus zip 해제(CP949 안전). 절대 plain unzip 쓰지 말 것
│   ├── build_fine_labels.py      #   라벨링 JSON 스트리밍 → data/fine_labels.csv (모국어/국가)
│   ├── make_fine_split.py        #   61-class 화자 분리 8:1:1 split 생성(canonical, seed 42)
│   ├── make_speaker_split.py     #   coarse 6-class용 화자 분리 split
│   ├── inspect_manifests.py      #   split별 카운트·오디오 존재·라벨 분포·화자 중복 검사 → JSON
│   ├── smoke_test.py             #   1배치 data→model→loss→metric 점검(테스트 대용)
│   ├── extract_features.py       #   mean-pool 1920-d XLS-R feature를 split별 .npz로 캐시
│   ├── train_probe.py            #   캐시된 feature 위에 linear head 학습 → test top1/top5/macroF1
│   ├── train_probe_imbalance.py  #   불균형 처리 변형 실험(LA/class-balanced/oversample 등)
│   ├── train_country_classifier.py / eval_country_classifier.py  # end-to-end 학습/평가
│   ├── finetune_partial.py       #   XLS-R 상위 N개 layer 언프리즈 + 부분 파인튜닝
│   └── create_xlsr_inference_checkpoint.py  # 26GB 체크포인트에서 optimizer state 제거(추론용 축소)
│
├── configs/
│   ├── fine_grained_label_alias_merged6.json     # 모국어 표기 정규화(별칭 병합)
│   └── fine_grained_label_alias_family_sparse.json
│
├── data/                         # manifest(CSV/TSV)와 라벨 — 실제 오디오는 여기 없음
│   ├── fine_labels.csv           # (basename, speaker_id, mother_tongue, country) 전체 라벨 테이블(54MB)
│   ├── fine_labels.csv.dist.json # 61-class 발화 수 분포
│   ├── aihub_nationality_fine_811/        # ★ canonical 61-class 화자분리 split (train/valid/test.csv)
│   ├── aihub_korean_nationality_speaker_811/   # coarse 6-class 화자분리 split
│   └── aihub_korean_nationality_*          # 초기/실험용 split들(아래 "주의사항" 참고)
│
├── runs/                         # 베이스라인 실험 산출물(feature 캐시·probe·로그)
│   ├── features_fine_811/        # 61-class mean-pool feature 캐시(train/valid/test.npz, 총 ~4GB)
│   ├── probe_fine_811/           # ★ 61-class linear probe 결과(test_metrics.json) — 현재 기준 베이스라인
│   ├── probe_imbalance_fine_811/ # 불균형 처리 변형 결과(imbalance_results.json)
│   ├── ft_top8_fine811/          # 61-class 상위8 layer 파인튜닝 결과(ft_log.csv)
│   └── *.log, run_*.sh, watch_*.sh
│
├── taeho/                        # ★★ 주 작업/실험 디렉터리 (대부분의 실제 연구 작업이 여기 있음)
│   ├── (아래 4·5번 항목에서 상세 설명)
│   └── xlsr_nationality_inference_bundle/  # (구버전·31-class 병합 기반 — 현재 방향 아님, 참고용)
│
├── fairseq/                      # editable 설치된 fairseq 소스(수정 가능 사본). 직접 건드리지 말 것
├── models/                       # xlsr2_2B_1000k.pt (~26GB, 미커밋) 등 대형 체크포인트
├── datasets/aihub/               # AIHub 다운로드 도구(aihubshell) 위치
└── src/btop/                     # ★ 무관 — 시스템 모니터(btop) 빌드. 이 프로젝트와 관계 없음
```

> **대형 산출물은 트리 밖(`$SPEECH_AI_DATA/...`)에 있습니다.** `taeho/` 안의 여러 항목은
> `$SPEECH_AI_DATA/taeho/...` 를 가리키는 **심볼릭 링크**입니다(feature 캐시·manifest·인덱스). `ls -la taeho/` 로 확인.

### 2-1. 이번 세션(2026-06-27) 추가된 파일 — 루트(`$HOME/`)
```
decouple_eval.py / decouple_eval_62class.py   # 분류기 헤드 5종 decoupling 비교 (62class판이 정본)
cap_sweep.py                                  # train 다운샘플링 cap sweep (결론: 해로움)
weight_sample_sweep.py                        # 가중 샘플링 β sweep (β0.75 → top-5 소폭↑)
perclass_table.py / perclass_beta.py          # 클래스별 top1/top5 표 (β 적용 전후)
build_tail_targets_manifest.py                # 증강 타깃 train manifest (target_counts 화자만)
augment_tail_wavs.py                          # 화자 변형 증강(Praat: 포먼트·F0·EQ)
data/reading_index.csv                        # 낭독 발화 80만 (fileName/spk/모국어/문장) — 같은문장 비교용
data/class_distribution.xlsx                  # 클래스 분포+화자수 (아프리카계영어 분리)
data/train_tail_targets.csv / train_augmented.csv / aug_feat_manifest/   # 증강 매니페스트
runs/features_aug_12s/                         # 증강 wav의 12s feature
runs/decouple62_12s*.json / cap_sweep_12s.json / wsample_sweep_12s.json   # 결과 JSON
```
### 2-2. 이번 세션 추가 — `taeho/` (해석 노트북 + 요약 문서)
```
experiments_overall_summary.md        # ★ 모든 실험 전체표 종합 (먼저 읽기)
notion_summary.md / notion_weighted_sampling.md / augmentation_experiment_summary.md
spectrogram_by_nationality.ipynb      # 같은 문장 국적별 스펙트로그램 + LTAS/포먼트
tsne_features.ipynb                   # 클래스/화자 t-SNE (화자 지배 확인)
gradcam_saliency.ipynb               # 입력 saliency
interpretability_occlusion_probing.ipynb   # occlusion + probing + 다국적 히트맵
ft_top8_fine811_12s/ (61class 10ep) / ft62_top8_12s/ (62class 2ep)   # 파인튜닝 로그
```
> 새 스크립트/노트북은 전부 **frozen feature 위 또는 캐시 사용** → 인코더 재학습 없음(파인튜닝 제외).
> decouple/probing 노트북은 **전체 XLS-R 로드(~67s, GPU)** 필요. (activation checkpointing은 `disable` 후 backward — autocast 충돌 회피)

---

## 3. 데이터셋 설명

### 3-1. 원천 데이터 (AIHub 505 "외국인 한국어 발화 음성")
- **실제 오디오 위치(중요): `data/` 가 아님.**
  `$SPEECH_AI_DATA/aihub/foreign_korean_speech/131.인공지능_학습을_위한_..._데이터/01.데이터_new_20220719/{1.Training,2.Validation}/`
  (각 split 아래 `원천데이터/`=오디오 zip, `라벨링데이터/`=JSON zip)
- `data/<set>/*.tsv` 의 첫 줄 root는 **다른 머신 경로**(`/home/<타 사용자>/...`)라서, 모든 스크립트가
  `--audio_root` 로 위 실제 경로로 remap. base 상수는 `baseline_common.py:DEFAULT_AUDIO_ROOT`.
- **추출 gotcha (반드시 `extract_audio.py` 사용)**:
  zip 내부 파일명이 CP949인데 UTF-8 플래그가 없어 plain unzip 하면 폴더명이 깨짐(mojibake).
  `extract_audio.py` 가 cp437→cp949 재디코딩 + zip-stem 정규화 처리.
- 디스크 폴더 구조가 manifest 상대경로와 항상 일치하지 않음(예: 베트남어 이중 중첩). 해결책:
  발화 basename이 전역 고유 → 모든 스크립트가 **basename 인덱스**(`bc.build_audio_index`, 약 1M 파일 ~3s)로
  오디오를 해석. 이 덕분에 manifest가 100% resolve 됨.

### 3-2. 라벨
- fine 라벨은 `라벨링데이터/*.zip` JSON의 `skill_info.motherTongue` 키에서 추출, `configs/fine_grained_label_alias_merged6.json` 로 정규화.
- **61-class 발화 분포(전체, `data/fine_labels.csv.dist.json`)** — 극단적 롱테일:

  | 라벨 | 발화수 | 라벨 | 발화수 |
  |---|---:|---|---:|
  | 중국어 | 262,964 | 일본어 | 194,943 |
  | 베트남어 | 174,522 | 태국어 | 125,543 |
  | 몽골어 | 53,438 | 러시아어 | 39,699 |
  | … | … | 나이지리아어 | 12 |
  | | | 타밀어 | 4 |

  → 상위 4개 클래스(중/일/베/태)가 데이터의 대부분. 하위 수십 클래스는 발화 수백 이하.

### 3-3. manifest 두 포맷 + 브리지
1. **fairseq 포맷** (`{split}.tsv/.lang/.spk`, `dict.lang.txt`, `labels.txt`): tsv 1줄=audio root, 이후 `relpath\tframes`.
2. **CSV 포맷** (`audio_path, json_path, speaker_id, label_name, label_id, duration_sec`):
   `country_classifier/dataset.py` 가 직접 읽는 형식. 커스텀 화자분리 split은 전부 이 CSV.
- `scripts/baseline_common.py` 가 둘 사이의 브리지(fairseq→CSV 변환, audio_root remap, basename 인덱스,
  체크포인트 저장/복원, smoke용 합성 WAV 생성).

### 3-4. split 카운트 (두 종류 — test 등장 클래스 수가 다름!)

**(A) 원본 `aihub_nationality_fine_811` (61-class, 한국어 없음)**
| | 발화 | test 등장 클래스 |
|---|---:|---:|
| train | 837,122 | — |
| valid | 103,329 | — |
| test | 101,818 | **29 / 61** |

**(B) `feature_resplit_61_merged6_seed42_target_counts` (62-class = 외국어 61 + 한국어) ← 현재 사용**
| | 외국어 발화 | +한국어 | test 평가가능 클래스 |
|---|---:|---:|---:|
| train | 825,409 | +160,000 | — |
| valid | 98,883 | +20,000 | — |
| test | 117,977 | +20,000 | **42** (외국어 41 + 한국어) |

> **왜 29 vs 42?** target_counts resplit은 희소 클래스(화자 2명)를 일부러 train/test로 쪼개 **test에 넣도록 재분할** → 평가 가능 클래스가 29→**42**로 늘었음. 최근 분석·실험은 전부 (B) 기준.

> **주의사항**: 그래도 화자 1명짜리 클래스(20개)는 여전히 화자분리 test에 못 들어가 train-only. (전체 분류: 평가가능 42 + 평가불가 21, [`data/class_distribution.xlsx`]($NATIONALITY_DATA/class_distribution.xlsx) 참고)
> `data/` 안의 `aihub_korean_nationality_from_validation*`, `*_train_full_valid_test*`, `*_train_quarter_*` 등은
> **초기 실험용 split**으로 보입니다(현재 canonical 아님). 새로 시작한다면 `aihub_nationality_fine_811`(61-class) 또는
> `taeho/feature_resplit_61_merged6_seed42_target_counts`(현재 실험 split, 62-class)를 사용하세요. (구 split들의 정확한 용도는 **확인 필요**.)

### 3-5. (제외) 클래스 병합은 하지 않음
과거 어족/권역 병합(31-class) 실험이 있었으나 **중단**. 라벨은 **61-class(+한국어) 원본 유지**.
관련 산출물(`taeho/xlsr_nationality_inference_bundle/`)은 병합 기반 구버전이라 **현재 방향과 무관**(참고용으로만 존재).

---

## 4. 작업 내용 / 진행 상황 (단계별)

> 큰 흐름: **frozen 베이스라인 → 불균형 처리 → 한국어 판별 추가·입력 길이(62-class) →
> 부분 파인튜닝 → (이번 세션) decoupling·샘플링·증강·해석.** 핵심 발견은 매 단계에서
> "단순 linear probe가 가장 강하다" + "병목은 화자 다양성". **(클래스 병합은 하지 않음)**

### 단계 1 — 환경·데이터 파이프라인 구축
- conda 환경 `fairseq_xlsr` 빌드(`create_env.sh`), AIHub corpus를 CP949 안전하게 추출(`extract_audio.py`).
- 라벨링 JSON → `data/fine_labels.csv` 생성, 61-class 화자분리 8:1:1 split 생성(seed 42).
- 결정: 오디오/manifest 경로 불일치는 **basename 인덱스**로 해결(폴더 구조에 의존하지 않음).

### 단계 2 — Frozen XLS-R 베이스라인 (현재 기준점)
- 인코더는 frozen이므로 **feature를 한 번 캐시(`extract_features.py`)하면 linear probe 학습은
  ~100× 저렴**하고 end-to-end frozen 학습과 수학적으로 동일. → head/lr/batch 스윕을 분 단위로.
- 결과: `runs/probe_fine_811/test_metrics.json` (5번 표).
- 발견: top-5는 높지만(0.94) macro-F1은 0.19로 낮음. 롱테일 클래스가 baseline에서 학습 불가.

### 단계 3 — 불균형 처리 실험 (`train_probe_imbalance.py`)
- 변형: baseline / post-hoc logit adjustment(LA) / train-time LA / class-balanced loss / oversample / CB+LA.
- 결과(`runs/probe_imbalance_fine_811/imbalance_results.json`): **어떤 변형도 baseline macro-F1(0.197)을 못 넘음.**
  LA 계열은 오히려 top1을 크게 떨어뜨림(la_train top1 0.26, cb_plus_la 0.14). class_balanced가 그나마 비슷.
- 결론: 단순 재가중/리샘플링으로는 한계. 근본 원인은 희소 클래스의 **화자 다양성 부족**.

### 단계 4 — (중단) 클래스 병합 실험
과거 어족/권역 병합(31-class) 실험을 했으나 **방침상 병합은 하지 않기로 → 중단.** 본 문서에서 제외.
(`linear > mlp512`라 분류기는 linear 유지가 낫다는 점만 참고로 남김.)

### 단계 5 — 한국어 원어민 판별 추가 (62-class = 외국어 61 + 한국어) + 입력 길이 스윕
- 한국어 원어민 인덱스 구축(`build_korean_native_index.py`, `$SPEECH_AI_DATA/taeho/korean_native_index`),
  외국어 manifest + 한국어 manifest 결합(`build_62class_manifest.py` → `한국어_모국어` 라벨 추가). **병합 없이 61+한국어.**
- **입력 길이 스윕(8s/12s/16s)** (62-class overall):
  모든 지표가 8s<12s<16s로 단조 증가. **16s 최고**(top1 0.795 / top5 0.932), 12s가 지연/정확도 균형점.
  한국어 vs 외국어 판별은 매우 정확(Korean precision/recall ≈ 0.999).
- 화자 불변(speaker-invariant) 실험(`train_62class_speaker_invariant.py`): balanced CE / SupCon(교차화자 대조) /
  조건부 화자 적대 학습 시도. 결과(`62class_speaker_invariant_experiment_summary.md`):
  **여전히 linear baseline이 최고.** 1-화자 클래스는 교차화자 신호 자체가 없어 일반화 불가.
- 결론: "다음으로 의미 있는 실험은 **XLS-R 상위 layer 언프리즈**" → 단계 6으로.

### 단계 6 — 부분 파인튜닝 (`finetune_partial.py`) — **완료, frozen 못 이김**
- XLS-R 상위 8 layer 언프리즈 + 헤드(enc_lr 1e-5 / head_lr 1e-3, cap 8000/cls, 12s).
- **지표 주의**: 이 스크립트의 평가는 **macro/micro top-5**(F1·top1 아님). frozen top-5 기준 = macro 0.498 / micro 0.93.
- **61-class run (`taeho/ft_top8_fine811_12s`, 10 epoch 완료)**: best=epoch5 → **test macro top-5 0.4996 ≈ frozen 0.498(동급)**,
  micro top-5 0.754 ≪ 0.93(하락). epoch5 이후 valid 정체 + train_loss 0.009 = **과적합**.
- **62-class run (`taeho/ft62_top8_12s`, epoch 2에서 중단)**: macro top-5 0.496 / micro 0.765 — 동일하게 못 넘음.
- **결함**: ① 지표가 top-5라 다른 실험(top1/F1)과 직접 비교 불가, ② `best_head.pt`는 **헤드만 저장**(인코더 미저장) → 학습된 모델 오프라인 재평가 불가.
- 결론: **이 하이퍼파라미터의 partial finetune은 frozen을 못 이김.** 제대로 하려면 전체모델 저장 + uncapped 풀test(F1) + enc_lr↓/early-stop 패치 필요.

### 단계 7 — 이번 세션 추가 실험 (2026-06-27): decoupling · 샘플링 · 증강 · 해석
모두 **62-class / target_counts split / 12s / frozen + linear** 기준. 상세·전체표는 `taeho/experiments_overall_summary.md`.

1. **Classifier decoupling** (`decouple_eval_62class.py`): 헤드 5종(linear/tau_norm/cosine/cRT/NCM) 비교 → **linear 최고, 전부 개선 ❌** (few 버킷 ≈0).
2. **다운샘플링 sweep** (`cap_sweep.py`, cap 1000~5000): **전 지표 단조 하락** → 데이터 버리면 손해. ❌
3. **Weighted-sampling sweep** (`weight_sample_sweep.py`, β 0.25~1.0): **β 0.5~0.75에서 top-5 소폭↑** (O.top5 0.931→0.932, F.top5 0.919→0.921, mid 버킷 F1 0.163→0.174). β=1은 과해서 하락. **지금까지 유일한 top-5 플러스**(단 head top1은 깎임, head top5는 거의 무손실).
4. **화자 변형 증강** (`augment_tail_wavs.py` + `build_tail_targets_manifest.py`): 5클래스(인니·광동·말레이·스페인·타갈로그)에 음색 변형 가짜화자 +10,000 → train만 추가. **효과 없음**(macro 미세↓). 원인: frozen + 같은 화자 파생 → 새 화자 정보 0. 상세 `taeho/augmentation_experiment_summary.md`.
5. **해석(interpretability) 노트북 3종** — "왜 안 되나"를 시각적으로 확정:
   - `taeho/tsne_features.ipynb`: feature가 **화자로 뭉침**(국적으론 안 뭉침).
   - `taeho/spectrogram_by_nationality.ipynb`: 같은 문장 국적별 스펙트로그램 + LTAS/포먼트 → 화자 평균하면 국적 차이 거의 없음.
   - `taeho/interpretability_occlusion_probing.ipynb`: occlusion(국적마다 다른 주파수 의존: 영어=저음, 중국어=중음) + probing(임베딩 R²: 에너지0.84·F0 0.62·포먼트0.4 **높음** vs 국적 변별 eta² **<0.1**) → "단순 음향은 국적과 무관, 화자/목소리가 표현 지배".
6. **데이터 정리**: `data/class_distribution.xlsx`(62→63클래스 분포+화자수, **아프리카계영어 분리**), `data/reading_index.csv`(낭독 발화 80만 인덱스, 같은문장 비교용).

### (구버전·병합 기반) 추론 번들 (`taeho/xlsr_nationality_inference_bundle/`)
- 🚫 **31-class 병합 라벨 기반 = 현재 방향(병합 안 함)과 무관.** 과거 산출물이라 참고용으로만 둠.
- HuggingFace `facebook/wav2vec2-xls-r-2b` 기반(별도 env `xlsr_hf`), 첫 4초 feature 조건. 본 repo(fairseq, 12s)와 다름.
- 새로 배포 패키지를 만들 땐 **61-class(+한국어) 모델로 다시** 빌드해야 함.

---

## 5. 결과 (핵심 지표)

### 5-1. 61-class fine 베이스라인 (frozen XLS-R + linear probe, no tricks)
`runs/probe_fine_811/test_metrics.json` (test n=101,818, 존재 클래스 29/61)

| 지표 | 값 |
|---|---:|
| test top-1 | **0.731** |
| test top-5 | **0.937** |
| test macro-F1 | **0.191** |

- 클래스별 top-1: 중국어 0.83 / 일본어 0.87 / 베트남어 0.82 / 태국어 0.77 / 몽골어 0.76 / 러시아어 0.68
  → 상위 클래스는 잘 맞히지만, 독일어·힌디어·아랍어 등 다수 희소 클래스는 0.00.
- 해석: 높은 top-k는 4대 우세 클래스 덕분. 낮은 macro-F1은 학습 불가능한 롱테일 때문.

### 5-2. 불균형 처리 변형 (61-class, `runs/probe_imbalance_fine_811`)
| variant | top1 | top5 | macro-F1 |
|---|---:|---:|---:|
| **baseline** | **0.734** | **0.937** | **0.197** |
| posthoc_la | 0.628 | 0.835 | 0.124 |
| la_train | 0.257 | 0.597 | 0.089 |
| class_balanced | 0.732 | 0.937 | 0.183 |
| oversample | 0.697 | 0.933 | 0.154 |
| cb_plus_la | 0.142 | 0.433 | 0.054 |

→ **baseline이 모든 변형보다 우수.**

### 5-4. 입력 길이 스윕 (62-class overall = 외국어 61 + 한국어, 병합 없음)
| duration | top1 | top5 | macro-F1 |
|---|---:|---:|---:|
| 8s | 0.7595 | 0.9254 | 0.1913 |
| 12s | 0.7879 | 0.9298 | 0.2035 |
| **16s** | **0.7949** | **0.9316** | **0.2040** |

→ 길수록 좋음(16s 최고, 12s가 지연 대비 균형점). 한국어 vs 외국어 판별은 거의 완벽(P/R≈0.999).

### 5-5. 화자 불변 실험 (62-class) — `62class_speaker_invariant_experiment_summary.md`
linear baseline(overall top1 0.792 / macro-F1 0.201)이 SupCon·적대학습 등 모든 변형보다 우세.

### 5-6. 부분 파인튜닝 (완료, top-5 지표) — frozen 못 이김
| run | best | test macro top-5 | test micro top-5 |
|---|---|---:|---:|
| frozen 기준 | — | 0.498 | 0.931 |
| 61-class (10ep) | epoch5 | **0.4996** | 0.754 |
| 62-class (2ep 중단) | epoch2 | 0.496 | 0.765 |

→ macro top-5 동급, micro top-5 하락(과적합). **이 설정으론 개선 ❌.**

### 5-7. Classifier decoupling (62-class overall, 12s) — `runs/decouple62_12s.json`
| head | top1 | top5 | macro-F1 |
|---|---:|---:|---:|
| **linear** | **0.7905** | **0.9310** | **0.2088** |
| crt | 0.7505 | 0.9298 | 0.2028 |
| tau_norm | 0.6489 | 0.9009 | 0.1823 |
| cosine | 0.5679 | 0.8787 | 0.1720 |
| ncm | 0.1575 | 0.2755 | 0.0337 |

→ linear 최고. 머리 교체로 개선 ❌.

### 5-8. Weighted-sampling sweep (62-class overall, 12s) — `runs/wsample_sweep_12s.json`
| β | top1 | top5 | foreign top5 | foreign mid-F1 |
|---|---:|---:|---:|---:|
| none(자연) | 0.7905 | 0.9310 | 0.9194 | 0.1630 |
| 0.5 | 0.7814 | **0.9324** | **0.9211** | 0.1684 |
| 0.75 | 0.7725 | 0.9324 | 0.9210 | **0.1740** |
| 1.0 | 0.7505 | 0.9298 | 0.9181 | 0.1636 |

→ **β 0.5~0.75에서 top-5 +0.0014~0.0017, mid 버킷 F1 +0.011.** 지금까지 **유일한 top-5 플러스**(대가: head top1↓, head top5는 무손실).

### 5-9. 화자 변형 증강 (62-class overall, 12s, linear) — 효과 없음
| 지표 | 증강 전 | 증강 후 |
|---|---:|---:|
| top1 | 0.7905 | 0.7893 |
| top5 | 0.9310 | 0.9311 |
| macro-F1 | 0.2088 | 0.2071 |

→ 타깃 5클래스 top1도 2↑·2↓·1= = **개선 없음.**

### 5-10. 해석 핵심 수치 (왜 안 되나)
- **probing**(임베딩→음향 R² / 국적 변별 eta²): 에너지 0.84/0.01, F0 0.62/0.10, F1 0.40/0.03, F2 0.38/0.06 → **임베딩은 목소리 음향을 풍부히 담지만 그게 국적을 안 가름.**
- **occlusion**: 영어=0–1kHz 결정적, 중국어=2–4kHz → 국적마다 다른 대역 의존(단순 규칙 아님).
- **t-SNE**: 한 클래스 안에서 **화자별로 뚜렷이 군집** = 표현이 화자 지배.

> **한 줄 요약**: 가장 신뢰할 모델은 여전히 **frozen XLS-R + linear**. top-5만 보면 **weighted-sampling β≈0.75**가 미세한 첫 개선.
> frozen 위 모든 트릭이 막힌 근본 원인은 **소수 클래스 화자 다양성 부족 + 음향적으로 약한 국적 신호**(해석 실험으로 확정).

---

## 6. 실행 방법

> 모든 명령은 전용 conda 환경 `fairseq_xlsr`에서 실행. 작업 디렉터리는 `$HOME`.
> **주의**: `import fairseq` 는 CWD가 `$HOME`이면 깨질 수 있음(repo 루트가 네임스페이스 패키지로
> 진짜 패키지를 가림). `country_classifier/model.py` 가 `sys.path` 앞에 `fairseq_root`를 넣어 회피하므로,
> 새 코드에서 체크포인트를 로드할 땐 model.py에 맡기고, **CWD가 `$HOME`가 아닌 곳**에서 import 검증.

### 6-0. 사전 준비 (한 번만)
```bash
# 환경 빌드 (conda ToS accept + activate 가드 포함)
bash tools/create_env.sh
export PY=python

# (필요 시) AIHub corpus 추출 — plain unzip 금지, 반드시 이 스크립트로
$PY scripts/extract_audio.py --split 2.Validation --only-ext .wav
$PY scripts/extract_audio.py --split 1.Training   --only-ext .wav
```
- 대형 자산은 트리 밖: 인코더 체크포인트 `models/xlsr2_2B_1000k.pt`(~26GB), feature 캐시 `$SPEECH_AI_DATA/...`.

### 6-1. 빠른 동작 점검 (실제 오디오 불필요)
```bash
$PY scripts/inspect_manifests.py --data_root $NATIONALITY_DATA --output runs/manifest_inspection.json
$PY scripts/smoke_test.py --manifest_dir data/aihub_nationality_fine_811 --backend tiny --batch_size 2
```

### 6-2. 61-class frozen 베이스라인 재현 (장시간 — tmux 권장)
```bash
# (1) feature 캐시 (test→valid→train 순; train.npz ~3.2GB)
$PY scripts/extract_features.py \
    --manifest_dir data/aihub_nationality_fine_811 \
    --out_dir runs/features_fine_811 \
    --splits test valid train --batch_size 32 --max_sec 12

# (2) linear probe 학습 → runs/probe_fine_811/test_metrics.json
$PY scripts/train_probe.py \
    --features_dir runs/features_fine_811 \
    --out_dir runs/probe_fine_811 --epochs 60
```
- 헬퍼: `runs/run_baseline.sh`, 진행 모니터 `runs/watch_baseline.sh`.

### 6-3. 불균형 처리 변형 재현
```bash
$PY scripts/train_probe_imbalance.py --features_dir runs/features_fine_811 \
    --out_dir runs/probe_imbalance_fine_811   # (정확한 플래그는 스크립트 상단 argparse 확인)
```

### 6-4. 부분 파인튜닝 (현재 진행 중인 실험)
```bash
# taeho/run_ft62_top8_12s.sh 와 동일한 호출
$PY taeho/finetune_partial.py \
  --manifest_dir taeho/manifest_62class_ft \
  --out_dir      taeho/ft62_top8_12s \
  --unfreeze_top_layers 8 \
  --cap_per_class 8000 --valid_cap 600 --test_cap 1500 \
  --batch_size 16 --grad_accum_steps 1 --epochs 10 \
  --max_sec 12 --enc_lr 1e-5 --head_lr 1e-3 --num_workers 8
# 모니터: tail -f taeho/ft62_top8_12s.log  /  taeho/watch_ft.sh
```
> 참고: 루트 `scripts/finetune_partial.py` 와 `taeho/finetune_partial.py` 가 **둘 다 존재**.
> 현재 실행 중인 건 `taeho/` 버전. 둘의 차이는 **확인 필요**(taeho 버전이 62-class용으로 갱신된 것으로 보임).

### 6-5. (구버전 추론 번들 — 병합 기반, 현재 방향 아님)
`taeho/xlsr_nationality_inference_bundle/`는 31-class 병합 라벨 기반 구버전이라 **새 작업엔 사용하지 않음.**
배포가 필요하면 **61-class(+한국어) 모델**로 추론 패키지를 새로 만들 것. (이 번들은 참고용으로만 보존)

### 6-6. 이번 세션 실험 재현 (전부 `$PY` = fairseq_xlsr, `$HOME`에서)
```bash
# (a) 분류기 헤드 5종 비교 (62-class, foreign 61-way)
$PY decouple_eval_62class.py --durations 12s --out runs/decouple62_12s.json --few_max 2000 --mid_max 20000

# (b) weighted-sampling β sweep  ← top-5 소폭 개선 확인됨(β0.5~0.75)
$PY weight_sample_sweep.py --betas 0.25 0.5 0.75 1.0 --out runs/wsample_sweep_12s.json

# (c) train 다운샘플링 sweep (참고: 해로움)
$PY cap_sweep.py --caps 1000 2000 3000 4000 5000 --out runs/cap_sweep_12s.json

# (d) 클래스별 top1/top5 (β 적용)
$PY perclass_beta.py --beta 0.75 --out_csv runs/perclass_beta075_12s.csv

# (e) 화자 변형 증강 파이프라인  (parselmouth 필요: pip install praat-parselmouth)
$PY build_tail_targets_manifest.py                       # data/train_tail_targets.csv
$PY augment_tail_wavs.py --in_csv data/train_tail_targets.csv \
   --out_dir $SPEECH_AI_DATA/taeho/aug_wavs --out_csv data/train_augmented.csv \
   --n_aug 8 --max_per_class 2000 --backend praat
#  → 증강 feature 추출 후 decouple에 --extra_train_npz 로 주입:
$PY scripts/extract_features.py --manifest_dir data/aug_feat_manifest --out_dir runs/features_aug_12s --splits train --max_sec 12
$PY decouple_eval_62class.py --durations 12s --extra_train_npz runs/features_aug_12s/train.npz --out runs/decouple62_12s_aug.json

# (f) 해석 노트북: taeho/*.ipynb (위→아래 실행, occlusion/probing은 GPU+모델 로드)
```
> 데이터/모델 무거운 작업은 직접 실행 권장(이 repo 관례). 위 (a)(b)(c)(d)는 캐시 feature만 써서 각 ~2–15분.

---

## 7. 다음 할 일 / 미완성 부분 / 인계받는 사람이 손댈 것

### 우선순위 (top-5/tail 올리기 — 검증된 결론 기반)
- [ ] **(최우선) 소수 클래스 실제 화자 추가** — 유일하게 근본 해결. 현재 화자 1~6명이라 0%인 **유럽 꼬리**부터:
      루마니아어(화자1)·포르투갈어(2)·폴란드어(2)·이탈리아어(3)·프랑스어(6)·독일어(6)·터키어(11)·스페인어(19).
      다른 데이터셋으로 보충 가능하나 **도메인 시프트 주의**(다른 녹음환경 = "데이터셋 단서"를 외울 위험; test는 AIHub 원본 화자 유지).
      목표 클래스당 ≥10명(가능하면 20+). 증강으론 대체 불가(검증됨).
- [ ] **(저비용) XLS-R 중간 layer feature** — 지금은 마지막 layer만 mean-pool. layer sweep으로 더 좋은 표현 찾기(미시도, 싸고 효과 기대).
- [ ] **weighted-sampling β≈0.75 채택** — top-5 소폭이지만 공짜 개선. β 0.6~0.8 미세 sweep으로 최적점.
- [ ] **finetune 제대로** — 전체모델 저장 + uncapped 풀test(F1) 평가 + enc_lr↓/LoRA/early-stop. (현재 best_head.pt는 헤드만 저장이라 재평가 불가)
- [ ] **duration 16s + 앙상블** — 16s가 12s보다 약간 좋음(top-5↑). 8/12/16s logit 평균도 시도 가능.

### 정리·검증 필요 (확인 필요 항목)
- [ ] **입력 길이 정책 확정**: 12s vs 16s (16s가 약간 우세). 배포 시 지연 고려.
- [ ] **루트 `scripts/` vs `taeho/`의 중복 스크립트**(`finetune_partial.py` 등) 정본 정리.
- [ ] **`data/`의 구 split 디렉터리들** deprecated 표기.
- [ ] **버전 관리 없음**: 코드(.py·.md·소형 csv·ipynb)를 git으로. 대형 .npz/.pt/wav/오디오는 `.gitignore`.

### 정리·검증 필요 (확인 필요 항목)
- [ ] **추론 번들의 feature 조건(첫 4초) vs 학습 실험의 길이(8/12/16s)** 불일치 — 번들이 구버전 체크포인트인지,
      어떤 체크포인트가 어떤 실험에서 나왔는지 매핑 문서화 필요.
- [ ] **루트 `scripts/` vs `taeho/`의 중복 스크립트**(`finetune_partial.py` 등) 차이/정본 정리.
- [ ] **`data/`의 구 split 디렉터리들**(`*_from_validation*`, `*_train_full_valid_test*`, `*_train_quarter_*`)의
      현재 유효성 — 사용 안 한다면 README에 deprecated 표기.
- [ ] **버전 관리 없음**: `$HOME`는 git repo가 아님. 코드(`scripts/`, `country_classifier/`, `taeho/`의 .py·.md·소형 csv)를
      git으로 올리는 것을 권장(대형 .npz/.pt/오디오는 `.gitignore`). 인계 시 코드와 산출물 경로가 분리되도록.

### 환경/운영 주의
- [ ] fairseq 설치가 fragile(editable `--no-deps` + `omegaconf==2.0.6`/`hydra-core==1.0.7`, `pip<24.1` 필요).
      재구축은 `tools/create_env.sh`로만. 임의 pip upgrade 금지.
- [ ] 테스트 스위트 없음 — `scripts/smoke_test.py` 가 사실상 유일한 통합 점검 수단.
- [ ] 장시간 잡(추출·feature 캐시·파인튜닝)은 반드시 `tmux`에서.

---

### 참고 문서 (먼저 읽을 순서)
1. `CLAUDE.md` — 환경/데이터 위치/gotcha 총정리
2. **`taeho/experiments_overall_summary.md`** — ★ 모든 실험 전체표 종합 (이번 세션 결과 포함)
3. `README_BASELINE.md` — 베이스라인 파이프라인 런북
4. `taeho/augmentation_experiment_summary.md` / `notion_summary.md` — 증강 상세 / 노션용 요약
5. `taeho/*.ipynb` (tsne / spectrogram / occlusion_probing) — "왜 안 되나" 해석 그림
6. `taeho/62class_speaker_invariant_experiment_summary.md` — 단계 5(화자 불변) 상세

> 🚫 병합(31-class/policy) 관련 문서·산출물(`policy31_*`, `xlsr_nationality_inference_bundle/LABEL_POLICY.md` 등)은
> 현재 방향이 아니므로 참고 목록에서 제외.
