# speech-ai-project

한국어 음성 AI 두 가지를 한 저장소에서 관리한다. 백본은 둘 다 **XLS-R (wav2vec 2.0)** 이지만
모델·데이터·평가 잣대가 완전히 달라 **task 단위로 분리**되어 있다.

| task | 무엇을 하는가 | 모델 | 성능 |
|---|---|---|---|
| [`nationality-classification`](tasks/nationality-classification/) | 외국인이 한국어로 말한 음성 → **화자의 모국어/국적** 예측 | XLS-R **2B** 동결 + 분류 헤드 | top-5 **0.950** · top-1 0.798 (41클래스) |
| [`pronunciation-assessment`](tasks/pronunciation-assessment/) | 외국인의 한국어 발음 → **어느 음소를 어떻게 틀렸는지** + 1~5점 | XLS-R **300m** + CTC (직접 학습) | weighted F1 **0.9547** · PCC **0.662/0.683/0.762** (전체 발화 · 30발화 집계 0.646/0.671/0.762) |

두 task 는 **코드상 완전히 독립**이다 (교차 import 0건). 공유하는 것은 경로 설정과
fairseq import 가드뿐이며 [`shared/`](shared/) 에 있다. 오디오 로더는 요구사항이 달라
일부러 합치지 않았다 → [`docs/DECISIONS.md`](docs/DECISIONS.md).

---

## 설치

이 저장소에는 **코드와 문서만** 있다(약 12 MB). 코퍼스·모델·산출물은 전부 저장소 밖에 있고
위치는 환경마다 다르므로 `.env` 로 지정한다.

**미리 있어야 하는 것**

| | |
|---|---|
| `conda` | miniconda / anaconda 아무거나 |
| `git` · `ffmpeg` | `sudo apt install ffmpeg` |
| **가중치 폴더** | 저장소에 없다. 별도로 받아야 한다 (아래 접기 참고) |
| GPU | **발음 진단에만** 필요. 국적 분류는 CPU 로 돈다 |

```bash
git clone https://github.com/dlxogh0906/speech-ai-project.git
cd speech-ai-project
bash tools/setup.sh <가중치폴더>
```

**이 한 줄이 전부다.** 손으로 채울 설정 파일이 없다. 10~15분 걸린다(torch 2.3 GB 등 내려받는다).
하는 일:

| | |
|---|---|
| 1 | 가중치 폴더 검사 + SHA256 대조 — 빠진 파일이 있으면 여기서 멈춘다 |
| 2 | conda env(python 3.8.20) + torch 2.1.2+cu118 + `requirements.txt` 전부 |
| 3 | fairseq(upstream 커밋 고정, editable) + KoG2P 클론 — 둘 다 pip 로는 못 받는다 |
| 4 | `.env` 자동 작성 |
| 5 | **자체검증** — 두 추론 스크립트를 실제로 돌려 JSON 이 나오는지 확인한다 |

**두 task 가 한 환경에서 돈다.** `requirements.txt` 가 둘의 합집합이고 버전 핀이 서로
충돌하지 않는다. 마지막에 이게 나오면 성공이다:

```
✓ 국적 분류 동작 — 1순위 ...
✓ 발음 진단 동작 — 음소 30개 판정
━━━ 완료 — 두 task 모두 동작한다 ━━━
```

<details>
<summary>필요한 가중치 (배포 번들의 <code>final/</code> 2.1 GB 면 충분)</summary>

```
<가중치폴더>/final/pronunciation/mdd_tag_all_native.pt          발음 진단 모델 (1.18 GB)
                              /dict.ltr.txt                   자모 사전 — 없으면 디코딩 불가
                              /score_models_v2_native.joblib  점수 3종 회귀기
              /nationality/encoder_kd_l17.onnx                국적 인코더 (916 MB)
                          /head_np.npz                        국적 헤드 (41클래스)
```

XLS-R 사전학습 백본(`xlsr2_300m.pt` · `xlsr2_2B_1000k.pt`)은 **추론에 필요 없다.**
체크포인트에 인코더가 통째로 들어 있고 ONNX 는 자체 완결이다. 재학습이나 국적 2B
헤드를 쓸 때만 받으면 된다.

⚠️ 파일명이 서로 비슷해서 **잘못 골라도 에러 없이 답만 틀리는** 조합이 있다
(`encoder_kd_l17.onnx` vs `encoder_l17.onnx`, `_native` 접미사 등).
가중치를 옮기거나 교체하기 전에 **`docs/WEIGHTS.md`** 를 볼 것.
</details>

<details>
<summary>수동 설치 / 부분 설치</summary>

```bash
bash tools/create_env.sh          # 환경만 (가중치 없이도 된다)
cp .env.example .env && vi .env   # 경로를 직접 채운다
python tests/test_imports.py      # 30초, GPU 불필요
```

**국적 분류만 쓸 거면 훨씬 가볍다.** 배포 경로가 ONNX·CPU라 torch·fairseq·GPU 가 전부
불필요하다 — `onnxruntime`·`fastapi`(`requirements-nationality.txt`)와 `ffmpeg` 면 된다.
발음 진단은 fairseq 가 필요하다.
</details>

`pip install -e .` 도 지원하지만 필수는 아니다. 모든 스크립트 상단에 5줄짜리 부트스트랩이
있어 설치 없이도 어디서 실행하든 저장소 패키지를 찾는다.

---

## 실행

### 파일 하나씩 — CLI

```bash
bash deploy/infer_nationality.sh 음성.wav                    # 국적 top-5 (ONNX·CPU, ~1초)
bash deploy/infer_mdd.sh 음성.wav "화자가 읽은 문장"           # 발음 진단 + 점수 (GPU)
```

오디오는 ffmpeg 이 읽는 포맷이면 전부 된다(국적은 앞 12초만 쓴다).
두 스크립트 모두 **API 서버를 잠깐 띄웠다 끄는 방식**이라 모델 적재(수십 초~수 분)를
매번 낸다. 여러 파일을 돌릴 거면 `--keep` 으로 서버를 남기고 `--url` 로 재사용한다:

```bash
bash deploy/infer_mdd.sh a.wav "문장1" --keep --session spk1      # 서버·세션 유지
bash deploy/infer_mdd.sh b.wav "문장2" --url http://127.0.0.1:8931 --session spk1
```

옵션은 각 스크립트의 `--help` 를 볼 것.

### 서비스로 — API 서버

긴 세션은 **tmux 안에서** 띄운다. 포그라운드로 돈다.

```bash
bash deploy/serve.sh nat --port 8905     # POST /predict   오디오 → 국적 top-5
bash deploy/serve.sh mdd --port 8914     # POST /diagnose  오디오 + 문장 → 진단·점수
curl -s localhost:8905/health
```

엔드포인트 상세는 [`deploy/README.md`](deploy/README.md).
**인증이 없다.** 외부에 열 때는 앞단에 접근 제어를 두어야 한다.

### 실행 전에 알아야 할 것

- **발음 점수는 화자 단위로만 유효하다.** 발화 1개 PCC 는 0.43 이고 20~30문장을 모아야
  포화한다. 같은 `--session` 으로 누적해야 하며, **5발화 미만이면 점수 없이 진단만** 나온다.
- **발음 진단은 학습에 쓴 287문장 안에서 강하다.** 안 본 문장은 weighted F1 0.9509 → 0.8218
  로 떨어진다. 정해진 문장을 읽게 하는 서비스면 문제없고, 자유 발화로 넓힐 때 걸린다
  → [`08_문장암기_일반화.md`](tasks/pronunciation-assessment/results/08_문장암기_일반화.md).
- **국적 정확도 수치가 문서마다 다르다** — 0.798(2B) / 0.751(300m 구버전) /
  **0.8245**(현재 배포 KD). 셋 다 사실이지만 다른 모델이다 → [`docs/WEIGHTS.md`](docs/WEIGHTS.md).

### 안 될 때

| 증상 | 원인 |
|---|---|
| `ffmpeg 실패 (rc=127)` | conda 의 `LD_LIBRARY_PATH` 가 시스템 ffmpeg 를 깨뜨린다. `env -u LD_LIBRARY_PATH` 로 우회 → [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md) 함정 10 |
| `포트 N 이 이미 사용 중이다` | 쓰는 포트를 뺏지 않는다. `--port <빈 포트>` 또는 `--url` 로 기존 서버 재사용 |
| `ConfigKeyError: multi_corpus_keys` | fairseq 을 v0.12.2 **태그**로 깔았다. main 커밋을 써야 한다 → `tools/create_env.sh` |
| `ModuleNotFoundError: g2p` | KoG2P 클론이 없다. `.env` 의 `G2P_TOOLS` 확인 |

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
└── docs/                환경 · 가중치 · 설계 결정 · 남은 문제
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
| [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md) | 환경 구축과 함정 10가지 |
| [`docs/WEIGHTS.md`](docs/WEIGHTS.md) | **어느 가중치 파일이 무엇인가** — 짝을 섞으면 조용히 틀린다 |
| [`docs/DECISIONS.md`](docs/DECISIONS.md) | 왜 이렇게 설계했는가 (다시 뒤집지 않도록) |
| [`docs/REMAINING_ISSUES.md`](docs/REMAINING_ISSUES.md) | 알려진 한계와 미해결 과제 |

---

## 라이선스 / 데이터 출처

AIHub 공개 데이터셋 기반 (외국인 한국어 발화 음성, 교육용 한국어 학습자 음성 040·041·042·043).
**데이터 재배포 금지.** 코퍼스·라벨·매니페스트·모델 가중치는 저장소에 포함되지 않는다.
