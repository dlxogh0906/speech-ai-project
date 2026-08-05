# 발음 진단·평가 (pronunciation-assessment)

한국어를 배우는 **외국인의 발음을 진단**한다. 두 가지를 낸다.

| | 무엇 | 목표 | 실측 |
|---|---|--:|--:|
| **MDD** | 몇 번째 음소를 어떻게 틀렸는지 위치·유형 지적 | F1 0.70 | ✅ weighted F1 **0.9547** |
| **점수 3종** | 발음숙달 / 이해가능 / 유창성 (1~5점) | PCC 0.60 | ✅ **0.662 / 0.683 / 0.762** (전체 발화)<br>30발화 집계 0.646 / 0.671 / 0.762 |

**과제 성능 요건은 둘 다 충족했다.** 자세한 근거와 배경은 [`docs/00_인수인계.md`](docs/00_인수인계.md).

## 성능 (실측, 근거 파일 포함)

### MDD

| 지표 | 값 | 의미 | 근거(`$MDD_OUTPUTS/`) |
|---|--:|---|---|
| **weighted F1** | **0.9547** | **공식 채점 지표** | `eval_tag71_all_lar.json` |
| 오류 클래스 F1 | 0.4319 | precision 0.466 / recall 0.402 — "틀렸다고 짚는" 능력 | 동일 |
| 전사 PER | 0.0471 | 전문가가 들은 음소의 95.3%를 그대로 받아씀 | `check_b2_preservation.json` |
| 오발음 보존율 | 0.514 | ⚠️ 오발음의 절반만 보존, 나머지는 표준발음으로 되돌림 | 동일 |
| 단어누락 검출 | F1 0.682 | record LAR 267,662발화 전수 | `eval_word_skip_lar_ALL268k.json` |

> **weighted F1 0.9547 과 오류 F1 0.432 의 차이를 반드시 이해할 것.** 공식 지표는 정상 위치
> (전체의 **95.86%**)를 포함해 계산한다. 그래서 전부 "정상"으로 찍기만 해도 weighted F1
> **0.9384** 가 나오고, **0.9547 의 실제 개선폭은 +0.0163 이다.**
> 제품 관점의 "지적 능력"은 0.43 이다. 둘 다 사실이고,
> 보고 시엔 공식 지표를 쓰되 질문이 들어오면 0.43 과 이 베이스라인을 숨기지 말 것.

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
> 현역만 주제별 5개로 병합했고, 각 항목에 원본 파일명(`source`)을 남겼다.

## 모델 계보

```
xlsr2_300m.pt (공개 사전학습, 43.6만 시간)
      │
      ├─ tag_all          밑바닥 학습 50,000 스텝   ← TAG 라벨, 040~043 전 언어권
      │       │
      │       └─ tag_all_lar   LAR 이어학습 15,000 스텝   ★현역★
      │
      └─ (폐기) soupwarm  공식 수프 warm-start 50,000 스텝
```

- 현역 체크포인트: `$SPEECH_AI_DATA/mdd_ckpt_tag_all_lar/checkpoint_best.pt`
- **공식 모델 가중치는 최종 모델에 안 들어갔다.** 공식팀도 같은 `xlsr2_300m` 에서 출발했으므로
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
> 같은 조건에서 우리 모델은 `.2235` 다. 보고 수치는 **TAG F1 0.9547만** 쓴다.
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

## 실행

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

**이미 만들어진 매니페스트가 있으면 다시 돌릴 이유가 없다.** 현재 상태:

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
