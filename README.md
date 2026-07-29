# speech-ai-project

한국어 음성 AI 두 가지를 한 저장소에서 관리한다. 백본은 둘 다 **XLS-R (wav2vec 2.0)** 이지만
모델·데이터·평가 잣대가 완전히 달라 **task 단위로 분리**되어 있다.

| task | 무엇을 하는가 | 모델 | 성능 |
|---|---|---|---|
| [`nationality-classification`](tasks/nationality-classification/) | 외국인이 한국어로 말한 음성 → **화자의 모국어/국적** 예측 | XLS-R **2B** 동결 + 분류 헤드 | top-5 **0.950** · top-1 0.798 (41클래스) |
| [`pronunciation-assessment`](tasks/pronunciation-assessment/) | 외국인의 한국어 발음 → **어느 음소를 어떻게 틀렸는지** + 1~5점 | XLS-R **300m** + CTC (직접 학습) | weighted F1 **0.9547** · PCC **0.662/0.683/0.762** |

두 task 는 **코드상 완전히 독립**이다 (교차 import 0건). 공유하는 것은 경로 설정과
fairseq import 가드뿐이며 [`shared/`](shared/) 에 있다. 오디오 로더는 요구사항이 달라
일부러 합치지 않았다 → [`docs/DECISIONS.md`](docs/DECISIONS.md).

---

## 설치

이 저장소에는 **코드와 문서만** 있다(약 12 MB). 코퍼스·모델·산출물은 전부 저장소 밖에 있고
위치는 환경마다 다르므로 `.env` 로 지정한다.

```bash
git clone <this-repo> && cd speech-ai-project

# 1) 경로 설정 — 자기 환경의 코퍼스·모델·fairseq 소스 위치를 채운다
#    (2단계가 여기서 FAIRSEQ_ROOT 를 읽으므로 순서를 지킬 것)
cp .env.example .env && ${EDITOR:-vi} .env

# 2) 파이썬 환경 (python 3.8 · torch 2.1.2+cu118 · fairseq editable --no-deps)
bash tools/create_env.sh

# 3) 의존성
pip install -r requirements-nationality.txt      # 국적 분류
pip install -r requirements-pronunciation.txt    # 발음 진단

# 4) 확인 (30초, GPU 불필요)
python tests/test_imports.py
```

`pip install -e .` 도 지원하지만 필수는 아니다. 모든 스크립트 상단에 5줄짜리 부트스트랩이
있어 설치 없이도 어디서 실행하든 저장소 패키지를 찾는다.

> `.env` 없이도 **import 와 문법 점검은 전부 통과한다.** 미설정 경로는
> `<SPEECH_AI_DATA 미설정>/...` 같은 값으로 조립되다가, 실제로 파일을 열려는 순간
> 무엇을 채워야 하는지 알려주며 멈춘다. 덕분에 데이터가 없는 머신에서도 코드 상태를
> 검증할 수 있고, `tests/test_imports.py` 는 '환경 미구성'과 '코드 결함'을 구분해 보고한다.

---

## 구조

```
speech-ai-project/
├── shared/              경로 단일 소스 · fairseq import 가드 (두 task 공용은 이게 전부)
├── tasks/
│   ├── nationality-classification/
│   │   ├── src/country_classifier/          라이브러리 (import 해서 쓴다)
│   │   ├── scripts/{data,train,eval,export}/  실행 진입점
│   │   ├── experiments/  notebooks/  tests/  docs/
│   │   └── outputs/                          새 실행 산출물
│   └── pronunciation-assessment/
│       ├── src/mdd/                          라이브러리 (+ l1prior/ 하위 파이프라인)
│       ├── scripts/{data,train,eval,score}/
│       ├── configs/                          fairseq hydra yaml — 학습 재현의 유일한 기록
│       ├── tests/  docs/
│       └── outputs/
├── deploy/              데모·추론 API (학습 코드와 라이프사이클이 다르다)
├── archive/             중단된 실험. 삭제하지 않고 보관
├── tools/               환경 구축
├── tests/               저장소 전체 import 스모크
└── docs/                환경 · 설계 결정 · 남은 문제
```

### 데이터와 모델은 저장소에 없다

| 무엇 | `.env` 키 | 규모 |
|---|---|--:|
| AIHub 코퍼스 · MDD 체크포인트 | `SPEECH_AI_DATA` | ~1.8 TB |
| XLS-R 사전학습 가중치 | `SPEECH_AI_MODELS` | 26 GB (2B) |
| 국적 특징 캐시 · 헤드 · ONNX | `NATIONALITY_OUTPUTS` | 9.8 GB |
| MDD 매니페스트 · 예측 캐시 | `MDD_OUTPUTS` | 2.6 GB |

**예측 캐시(`*_preds.jsonl`)는 지우지 말 것** — 지우면 GPU 재디코드에 수시간이 든다.

---

## 실행

각 task 의 README 에 학습·평가·추론 명령이 있다.

- [국적 분류 — 데이터 준비 · 학습 · 평가 · 내보내기](tasks/nationality-classification/README.md#실행)
- [발음 진단 — 데이터 준비 · 학습 · 평가 · 점수 · 추론](tasks/pronunciation-assessment/README.md#실행)
- [데모·API 실행](deploy/README.md)

```bash
source tools/env.sh          # .env 를 셸에도 반영 (파이썬은 없어도 됨)
python tests/test_imports.py # 문법 166 · 경로 상수 · 라이브러리 import 22
```

> 처음 보는 스크립트는 **`--help` 부터** 쳐 볼 것. 인자 파서가 없는 스크립트에도
> `--help` 가드를 넣어 뒀으니 실수로 본체가 실행되지 않는다. 반대로 **인자 없이 그냥
> 실행하면 본체가 돈다** — 매니페스트를 재생성하는 스크립트가 몇 개 있다.

---

## 개발 시 알아야 할 것

1. **환경은 하나뿐이다** — python 3.8.20 · torch 2.1.2+cu118 · fairseq editable.
   버전 조합이 fairseq 0.12.2 + omegaconf 2.0.6 에 묶여 있다.
2. **`import fairseq` 함정** — fairseq 저장소 루트가 namespace package 로 진짜 패키지를
   가린다. `shared.fairseq_guard.ensure_fairseq()` 를 fairseq import 앞에서 호출한다
   (저장소 전체에 이미 적용됨).
3. **KoG2P 는 pip 패키지가 아니다** — 클론이고, `g2p.py` 가 import 시점에 argv 를 파싱한다.
   `shared.paths.ensure_kog2p()` + argv 임시 비우기가 필요하다.
4. **라벨 체계가 여럿이다** — 국적은 61/62/41 클래스, 발음은 TAG/record/gold 세 종류.
   각 task README 의 "라벨" 절을 먼저 읽을 것. 잘못된 잣대로 재면 수치가 통째로 달라진다.

자세한 내용은 [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md).

---

## 문서

| | |
|---|---|
| [`tasks/nationality-classification/README.md`](tasks/nationality-classification/README.md) | 국적 분류 — 성능·라벨·실행·함정 |
| [`tasks/pronunciation-assessment/README.md`](tasks/pronunciation-assessment/README.md) | 발음 진단 — 성능·모델 계보·라벨 3종·실행·함정 |
| [`tasks/*/results/`](tasks/pronunciation-assessment/results/) | **실험 기록** — 문서 수치의 근거 파일 |
| [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md) | 환경 구축과 함정 6가지 |
| [`docs/DECISIONS.md`](docs/DECISIONS.md) | 왜 이렇게 설계했는가 (다시 뒤집지 않도록) |
| [`docs/REMAINING_ISSUES.md`](docs/REMAINING_ISSUES.md) | 알려진 한계와 미해결 과제 |
| [`MIGRATION.md`](MIGRATION.md) | 이전 디렉터리 구조에서 옮겨온 내역 |

---

## 라이선스 / 데이터 출처

AIHub 공개 데이터셋 기반 (외국인 한국어 발화 음성, 교육용 한국어 학습자 음성 040·041·042·043).
**데이터 재배포 금지.** 코퍼스·라벨·매니페스트·모델 가중치는 저장소에 포함되지 않는다.
