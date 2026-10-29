# 발음 진단·평가 (pronunciation-assessment)

한국어를 배우는 **외국인의 발음을 진단**한다. 두 가지를 낸다.

## 현재 서비스 모델

**NPU 운영 모델은 M3 `w2vbert_synth_tense_v2/step_3000`(w2v-BERT 2.0)이다.** 10월 21일 추가 학습 모델 100%를 반영했고, 마지막 원격 확인은 **2026-10-25 KST**다. 문서 갱신일은 10월 29일이며 오늘 서버를 재확인한 것은 아니다.

- 운영: `pron-api` · 포트 `8932` · NPU 7, `/root/pron_serve/ft3000_2026-10-19/model_hf`.
- 기존 로컬 CLI / `deploy/mdd-api`: `tag_all_native`와 해당 점수 회귀기를 사용하는 별도 경로.
- H0·P3·λ=.3 fold0 파일럿 등: 연구용이며 운영 채택 기록 없음.

배포 파일·인자·이력은 **[현재 서비스 모델](../../docs/발음평가_현재_서비스_모델.md)**, 학습 계보는 [모델별 학습 명세](../../docs/발음평가_모델별_데이터셋_학습명세_2026-10-26.md), 최근 연구는 [진단 감사·후속 실험](../../docs/발음평가_진단누락정보_감사_2026-10-26.md)을 참고한다.

## 현재 성능과 과거 수치의 차이

**M3의 보존된 화면 지적 평가에서 오류 재현율은 31.95%, 정밀도는 46.29%, 이 둘의 F1은 0.3781이다.** 과거의 **0.9547은 정상까지 포함한 weighted F1**이고, 그때 오류 클래스 F1은 **0.4319**였다. M3를 과거와 동일한 평가셋·채점기로 평가한 weighted F1은 확인된 비교 결과가 없다.

아래는 **서로 다른 평가에서 나온 기록값의 병기**다. 마지막 열은 산술 차이이며 모델 교체의 개선·악화 폭으로 해석하지 않는다.

| 지표 | 과거 XLS-R `tag_all_lar` | M3 운영 설정 재현 | 기록값 차이(M3−과거) |
|---|---:|---:|---:|
| 정상 포함 weighted F1 | 0.9547 | 동일 조건 비교값 미확인 | 계산 불가 |
| 오류 검출 F1 | 0.4319 | **0.3781** | **−0.0538** |
| 오류 재현율 | 40.22% | **31.95%** | **−8.27%p** |
| 오류 정밀도 | 46.63% | **46.29%** | **−0.34%p** |
| 발음숙달 PCC · 화자 단위 | 0.6622 | 0.683 | 약 +0.0208 |
| 이해가능 PCC · 화자 단위 | 0.6825 | 0.713 | 약 +0.0305 |
| 유창성 PCC · 화자 단위 | 0.7618 | 0.752 | 약 −0.0098 |

- **오류 검출 조건:** 과거는 TAG test 178,049발화의 인식열을 3열 정렬한 정상/오류 이진 평가다. M3는 10월 22일 표시 수정까지 반영한 평가이며, 학습에서 본 224문항의 테스트 6,720발화 중 당시 말소리 1초 조건을 통과한 **2,281발화**의 화면 지적을 센다. M3 F1은 저장 카운트 TP 631·FP 732·FN 1,344에서 계산했다. 평가셋·입력 검사·기준열·정렬·표시 정책이 달라 같은 조건의 모델 비교가 아니다.
- **점수 조건:** 과거 PCC는 화자의 전체 가용 발화(약 150개)를 평균한 값이다. M3 PCC는 입력 검사를 통과한 26,390발화·테스트 화자 800명의 별도 비교다. 여기의 1~5점 PCC와 화면 0~100점 PCC도 다르다.
- **지표의 뜻:** 위 오류 F1은 틀린 자리를 지적했는지를 센다. 실제로 어떤 소리를 냈는지까지 맞히는 정확 진단 F1이나 전체 음소 정확도가 아니다. 0.9547→0.3781을 성능 하락으로 계산하거나, 옛 weighted F1로 오류 검출 F1 0.70 달성을 선언하지 않는다.

근거: [과거 오류 검출 JSON](results/01_mdd_모델별성능.json), [과거 점수 JSON](results/03_점수_3종.json), [M3 배포 검증 §3·§10](../../docs/발음평가_개선_반영_2026-10-21.md). [현재 성능의 분모·계산·한계](../../docs/발음평가_현재_서비스_모델.md#성능과-과거-xls-r-수치의-비교)에 원시 집계 경로와 비교 공백을 정리했다.

## 과거 XLS-R `tag_all_lar` 성능과 재현

이하 성능표와 학습·평가 명령은 **기존 XLS-R 계열 기록**이다. 실행 예제 중 `mdd-api`는 `tag_all_native`, `mdd-demo` 기본값은 `tag_all_lar`를 사용한다. 이후 `tag_all_native` 교체 기록은 [당시 인수인계](docs/00_인수인계.md)에 있다.

## 과거 성능 (실측, 근거 파일 포함)

### MDD

| 지표 | 값 | 의미 | 근거(`$MDD_OUTPUTS/`) |
|---|--:|---|---|
| **weighted F1** | **0.9547** | 정상/오류 이진 F1의 지원 수 가중 평균 | `eval_tag71_all_lar.json` |
| 오류 클래스 F1 | 0.4319 | precision 0.466 / recall 0.402 — "틀렸다고 짚는" 능력 | 동일 |
| 전사 PER | 0.0471 | 전문가가 들은 음소의 95.3%를 그대로 받아씀 | `check_b2_preservation.json` |
| 오발음 보존율 | 0.514 | ⚠️ 오발음의 절반만 보존, 나머지는 표준발음으로 되돌림 | 동일 |
| 단어누락 검출 | F1 0.682 | record LAR 267,662발화 전수 | `eval_word_skip_lar_ALL268k.json` |

> **weighted F1 0.9547과 오류 F1 0.4319는 함께 보고한다.** weighted F1은 정상 위치
> (전체의 **95.86%**)를 포함해 계산한다. 그래서 전부 "정상"으로 찍기만 해도 weighted F1
> **0.9384** 가 나오고, **0.9547 의 실제 개선폭은 +0.0163 이다.**
> 오류 자리 지적의 F1은 0.4319다. 실제음·오류 유형까지 맞힌 비율은 이 이진 지표로 확인할 수 없다.
> 기존 3열 정렬에서 평가 위치가 예측에 의존하는 한계도 [감사 기록](results/06_실험감사.md)에 있다.

### 점수 3종 (PCC)

| 차원 | 화자 단위(전체 ~150발화) | 30발화 집계 | 발화 1개 |
|---|--:|--:|--:|
| 발음숙달 | **0.6622** | 0.6455 | 0.4251 |
| 이해가능 | **0.6825** | 0.6707 | 0.4266 |
| 유창성 | **0.7618** | 0.7615 | 0.6070 |

> **점수는 화자 단위로만 유효하다.** 라벨 분산의 74~94%가 "누가 말했나"로 설명된다.
> 발화 1개로 점수를 주면 안 되고 **20~30문장 모으면 포화**한다. 제품 UI 설계에 직결.

동결된 MDD 모델의 **부산물 특징 41종 + 보정기(GBM)만** 학습했다. 추가 모델 0개.
공식 점수 팩은 재현은 되나(0.725 ≒ 저자 보고 0.7228) 학습 화자 미공개라 검증이 불가능하고,
검증 가능한 조건(교차언어)에서 0.33~0.63 으로 떨어져 **자체 방식을 채택**했다.

> 아래 수치의 **근거 파일은 [`results/`](results/)** 에 있다. 원본 JSON 87개 중
> 당시 보고에 사용한 결과를 주제별 5개로 병합했고, 각 항목에 원본 파일명(`source`)을 남겼다.

## 과거 XLS-R 모델 계보

```
xlsr2_300m.pt (공개 사전학습, 43.6만 시간)
      │
      ├─ tag_all          밑바닥 학습 50,000 스텝   ← TAG 라벨, 040~043 전 언어권
      │       │
      │       └─ tag_all_lar   LAR 이어학습 15,000 스텝   ← 아래 평가의 기준 모델
      │
      └─ (폐기) soupwarm  공식 수프 warm-start 50,000 스텝
```

- 과거 평가 재현 체크포인트: `$SPEECH_AI_DATA/mdd_ckpt_tag_all_lar/checkpoint_best.pt`
- **공식 모델 가중치는 이 XLS-R 모델에 안 들어갔다.** 공식팀도 같은 `xlsr2_300m` 에서 출발했으므로
  **같은 출발선에서 독립 재현**한 것이다 (공식 0.96 / 모델 4개 vs 우리 0.9547 / 모델 1개).
- 출력 = **자모 46종** CTC. 71음소는 `jamo_to_kog2p.py` 로 후처리 변환(무손실 검증 완료).

## ⚠️ 라벨이 3종류다 (최대 혼란원)

같은 AIHub 코퍼스(040 영어 / 041 유럽 / 042 중·일 / 043 아시아)에 라벨이 세 겹으로 붙어 있다.

| 라벨 | 정체 | 쓰임 | 판정 |
|---|---|---|---|
| **TAG** | 전문가가 실제 발음을 음소 전사(TextGrid) | **학습·평가의 유일한 잣대** | ✅ |
| **record** | 맞춤법 전사 → G2P 자동변환 (실제 발음과 어긋남) | 단어누락 평가, LAR 도메인 적응 | 🟡 약라벨 |
| **gold(LAB)** | 오류 위치·유형 태그 + 사람 1~5점. **실현형 없음** | **점수 라벨 출처** | ❌ MDD / ✅ 점수 |

> **gold 를 MDD 잣대로 쓰지 말 것.** 실측으로 확인됐다 — TAG 에서 0.965 인 공식 모델도
> gold 잣대에선 **F1 0.42** 로 떨어진다
> (`$MDD_OUTPUTS/eval_official2_041.json`, `checkpoint_mdd_eu.pt`, tol0 `.4196`).
> 같은 조건에서 우리 모델은 `.2235` 다. TAG 기준 수치를 인용할 때는 **weighted F1과 오류 F1, 평가셋·채점 정의를 함께** 적는다.
> ⚠️ 이 공식 수치는 `04_공식팩_검증.json` 에 **아직 병합되지 않았다** — 번들의
> `gold잣대_대조` 는 공식이 아니라 우리 모델이니 근거로 인용하지 말 것.

### 분할 (`manifest_tag_all_lar`)

| split | 발화 | 화자 |
|---|--:|--:|
| train | 1,549,180 | 10,734 |
| valid | 188,570 | 1,279 |
| test | 178,049 | 1,196 |

valid↔test 화자 겹침 0. train↔test 502명 겹침(LAR 적응 시 섞인 낭독분)이지만
**영향 실측 = +0.0000** — 누출 없는 `tag_all` 에서도 같은 격차가 나오므로 "원래 쉬운 화자"일 뿐.

## 기존 XLS-R 경로 실행

모든 명령은 **저장소 루트에서** 실행한다.

```bash
cd $HOME/speech-ai-project
source tools/env.sh                       # $PY 와 경로 변수 설정
T=tasks/pronunciation-assessment
```

> 어떤 스크립트든 **`--help` 를 먼저 쳐 보면 안전하다.** 인자 파서가 없는 스크립트에도
> `--help` 가드를 넣어 뒀다. 아래 ⚠️ 표시는 **실행하면 파일을 덮어쓰는** 것들이다.

### 0. 저장소가 멀쩡한지 (30초, GPU 불필요)

```bash
$PY tests/test_imports.py
```

### 1. 데이터 준비 ⚠️ 전부 덮어쓰기

```bash
$PY $T/scripts/data/mdd_extract_speech.py --help    # 코퍼스 해제
$PY $T/scripts/data/mdd_resample_16k.py --help      # 48k → 16k
$PY $T/scripts/data/build_tag_manifest.py --help    # TAG(TextGrid) → fairseq manifest

# ⚠️ 이건 manifest_tag_all_lar 를 통째로 재생성한다 (약 10분, WAV 리샘플 포함)
$PY $T/scripts/data/build_lar_mix.py
```

**이미 만들어진 매니페스트가 있으면 다시 돌릴 이유가 없다.** 과거 재현 기록의 구성:

```
$MDD_OUTPUTS/manifest_tag_all_lar/   train 1,549,180 · valid 188,570 · test 178,049
```

### 2. 학습 (fairseq hydra) ⚠️ GPU 장시간

```bash
# 설정은 configs/ 에 있다. 이게 학습 재현의 유일한 기록이다.
tmux new -s mddtrain
fairseq-hydra-train --config-dir $T/configs --config-name mdd_ctc_300m_tag_all_lar
```

### 3. 평가 — 예측 캐시가 있으면 CPU 몇 분

```bash
# ① MDD 주 평가 (자모/71음소 이중 채점)
#    --preds 파일이 있으면 그걸 읽어 CPU 로 재채점하고, 없으면 GPU 로 디코드한다(~15분)
$PY $T/scripts/eval/mdd_eval_tag71.py \
    --manifest $MDD_OUTPUTS/manifest_tag_all_lar \
    --ckpt $SPEECH_AI_DATA/mdd_ckpt_tag_all_lar/checkpoint_best.pt \
    --preds $MDD_OUTPUTS/eval_tag_all_lar_preds.jsonl \
    --out   $T/outputs/eval_tag71_all_lar.json
# → jamo.f1_weighted 0.9547 / err_f1 0.4319 가 나와야 정상

# ② 오발음 보존 검증 (B-2)
$PY $T/scripts/eval/check_b2_preservation.py

# ③ 단어누락 검출
$PY $T/scripts/eval/eval_word_skip_record.py --help
```

**`*_preds.jsonl` 은 재채점의 원천이다. 지우지 말 것** — 지우면 GPU 재디코드에 수시간이 든다.

### 4. 점수 3종

```bash
# ① 점수 특징 41종 추출   ⚠️ GPU, ~200발화/초. 캐시가 이미 있으면 생략
$PY $T/scripts/score/extract_gop_feats.py --split test --n 200000

# ② 보정기 학습·평가 (CPU, 화자 분리 assert 내장)
$PY $T/scripts/score/gop_calibrate_v2.py
# → 발음숙달 0.6622 / 이해가능 0.6825 / 유창성 0.7618 이 나와야 정상

# ③ 보정기 저장 → score_models_v2.joblib
$PY $T/scripts/score/train_score_models.py --help
```

### 5. 추론 / 데모

```bash
# MDD 진단 + 점수 3종을 한 JSON 으로 (GOP 특징이 MDD forward 의 부산물 → 모델 1회로 둘 다)
$PY deploy/mdd-api/mdd_infer_api.py --port 8913
curl -s localhost:8913/health

# 진단 웹 UI
$PY deploy/mdd-demo/app.py --port 8911
```

데모·API 상세는 [`deploy/README.md`](../../deploy/README.md).

## 함정

1. **라벨 3종 혼동** — TAG 만 잣대. gold 는 점수용, record 는 약라벨.
2. **weighted F1 ≠ 오류 F1** — 0.9547 과 0.432. 둘 다 사실.
3. **점수는 화자 단위** — 발화 1개 PCC 는 0.43.
4. **`quality_flag != ok` 를 제외하지 말 것** — `usable_for_aggregation=false` 는 "통계 집계에서
   빼라"는 뜻이지 "평가에서 빼라"가 아니다. 실측: 제외했던 25,146발화에 **진짜 단어누락의 69.6%**가
   들어 있었다.
5. **화자 ID = 파일명 앞 5자리** (`00025-F-1-DE-B-RS025-...` → `00025`).
6. **CTC blank 는 무음이 아니다** — 조음속도·쉼은 강제정렬 구간 간격으로 재야 한다(임계 0.25s).
7. **공식 점수 팩 수치(0.72/0.94)를 신뢰하지 말 것** — 학습 화자 미공개.
8. **KoG2P 는 pip 패키지가 아니다** — 클론이고, `g2p.py` 가 import 시 optparse 로 argv 를 파싱한다.
   `shared.paths.ensure_kog2p()` 호출 + argv 임시 비우기가 필요하다(이미 적용됨).

## 디렉터리

| | |
|---|---|
| `src/mdd/mdd_eval.py` | CTC 디코더(`Decoder`). **20개 스크립트가 import 하는 중심 모듈** |
| `src/mdd/pipeline.py` | 문장→표준음소열, NW 정렬, 오류 유형 판정 (구 `mdd_demo.py` 의 라이브러리 부분) |
| `src/mdd/gop_feats_v2.py` | 점수 특징 41종 (강제정렬 기반) |
| `src/mdd/jamo_to_kog2p.py` | 자모 46 ↔ KoG2P 71음소 무손실 변환 |
| `src/mdd/mdd_reproduce_official.py` | 공식 3-way 정렬(`align`/`fixup`) 이식본 |
| `src/mdd/l1prior/` | 모국어별 오류 사전 파이프라인 (s0 → s9 순서) |
| `scripts/{data,train,eval,score}/` | 실행 진입점 |
| `configs/` | fairseq hydra yaml 10종 — 학습 재현의 유일한 기록 |
| `docs/` | `00_인수인계.md` 부터. `04_데이터셋_사전.md` 에 함정 77개 |
