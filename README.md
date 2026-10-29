# speech-ai-project

한국어 음성 AI를 한 저장소에서 관리한다. 모델·데이터·평가 잣대에 따라 **task 단위로 분리**되어 있다.

| task | 무엇을 하는가 | 모델 | 성능 |
|---|---|---|---|
| [`nationality-classification`](tasks/nationality-classification/) | 외국인이 한국어로 말한 음성 → **화자의 모국어/국적** 예측 | XLS-R **2B** 동결 + 분류 헤드 | top-5 **0.950** · top-1 0.798 (41클래스) |
| [`pronunciation-assessment`](tasks/pronunciation-assessment/) | 외국인의 한국어 발음 → **어느 음소를 어떻게 틀렸는지** + 점수 | **NPU 운영: w2v-BERT 2.0 기반 M3** `w2vbert_synth_tense_v2/step_3000` | 화면 오류 검출 F1 **0.3781** · 재현율 **31.95%** · 정밀도 **46.29%** (당시 입력 검사 통과분) · [평가 조건·과거와의 비교](docs/발음평가_현재_서비스_모델.md#성능과-과거-xls-r-수치의-비교) |

**발음평가의 마지막 원격 운영 확인은 2026-10-25 KST**다(문서 갱신 10월 29일). `pron-api` · 포트 `8932`에서 M3 100% 모델을 사용한다. 기존 XLS-R `tag_all_lar`의 weighted F1 0.9547 · PCC 0.662/0.683/0.762는 과거 평가값이며, H0·P3·λ=.3 파일럿은 연구용이다. 아래 설치·CLI의 `tag_all_native` 번들은 NPU 운영과 별도 경로다.

**발음 성능 비교:** 과거 XLS-R의 오류 F1은 **0.4319**, M3의 보존된 화면 지적 F1은 **0.3781**이다. 기록값 차이는 −0.0538이지만 평가셋·정렬·표시 정책이 달라 모델 교체의 하락폭으로 해석할 수 없다. 옛 **weighted F1 0.9547**과 동일 조건의 M3 비교값은 미확인이다. M3 화자 단위 1~5점 PCC는 **0.683/0.713/0.752**이며 옛 PCC와도 평가 화자·발화 구성이 다르다. 위 M3 오류 지표는 본 224문항의 테스트 6,720발화 중 당시 통과한 2,281발화에서 측정했다.

두 task 는 **코드상 완전히 독립**이다 (교차 import 0건). 공유하는 것은 경로 설정과
fairseq import 가드뿐이며 [`shared/`](shared/) 에 있다. 오디오 로더는 요구사항이 달라
일부러 합치지 않았다 → [`docs/DECISIONS.md`](docs/DECISIONS.md).

---

## 설치

아래는 기존 `tag_all_native` 발음 모델·국적 모델 번들의 설치 절차다. M3 NPU 서비스의 구성은 [현재 서비스 모델](docs/발음평가_현재_서비스_모델.md)을 따른다.

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

아래 발음 명령은 기존 `deploy/mdd-api` 경로를 실행한다. NPU `pron-api`의 요청 형식·인증은 [NPU API 안내](docs/NPU_서버_API_안내.md)를 참고한다.

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
**위 로컬 API 예제에는 인증이 없다.** NPU 발음평가 API는 토큰을 요구한다.

### 실행 전에 알아야 할 것

- **기존 `mdd-api` 점수는 화자 단위 집계 기준이다.** 발화 1개 PCC 는 0.43 이고 20~30문장을 모아야
  포화한다. 같은 `--session` 으로 누적해야 하며, **5발화 미만이면 점수 없이 진단만** 나온다.
- **과거 XLS-R 문항 홀드아웃 실험에서는 문항 일반화 문제가 관찰됐다.** 별도로 축약 학습한 모델의 seen/unseen weighted F1이 0.9509 / 0.8218
  이었던 기록이며, 현재 M3의 서비스 성능이나 자유 문장 지원 검증으로 인용하지 않는다
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
- [발음평가 — 현재 M3 서비스 모델](docs/발음평가_현재_서비스_모델.md)
- [기존 XLS-R 발음 진단 — 데이터 준비 · 학습 · 평가 · 점수 · 추론](tasks/pronunciation-assessment/README.md#기존-xls-r-경로-실행)
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

## 사용한 데이터셋 · 라이선스

**데이터 재배포 금지.** 코퍼스·라벨·매니페스트·모델 가중치는 저장소에 넣지 않는다.
AI 허브 데이터로 만든 모델·서비스는 쓸 수 있지만 원본 음성·라벨을 제3자에게 넘기면 안 되고,
결과를 보여 주는 앱·웹은 AI 허브 출처 문구를 표시해야 한다 → [`docs/NPU_서버_API_안내.md`](docs/NPU_서버_API_안내.md) 4절.
번호는 저장소에서 부르는 AI 허브 번호다 — 내려받기 폴더 번호(040 · 131 등) 또는 datasetkey(71517 등). 괄호 안은 그 데이터의 datasetkey.

### 국적 분류

| 데이터 | 번호 | 쓴 곳 |
|---|---|---|
| 교육용 영어·유럽어·중·일어·아시아어(중·일어 제외) 모국어 사용자의 한국어 음성 (4종) | 040·041·042·043 (71469·71489·71490·71479) | 학습 · 검증 · 테스트 (웹 마이크 녹음) |
| 인공지능 학습을 위한 외국인 한국어 발화 음성 | 131 (505) | 학습 · 검증 · 테스트 (61/62클래스 원래 분할) |
| 자유대화 음성(일반남여) | 001 | 한국인 클래스 학습 · 테스트 (승인 만료로 원본 삭제) |
| 한국어 음성 (KsponSpeech) | — | 한국인 클래스 학습, 채널 증강 배경음, 잡음 시험 |
| Zeroth-Korean (OpenSLR 40, CC BY 4.0) | — | 서비스 모델 `run_zeroth` 의 한국인 클래스 학습 |
| Speech Accent Archive (CC BY-NC-SA 2.0) | — | 영어 억양 전이 실험만 (비상업 · 내부 연구용) |
| MUSAN · FMA | — | 잡음 강건성 평가만 (학습엔 안 씀) |
| 합성 음성: Qwen3-TTS (Apache-2.0) · Chatterbox (MIT) · OpenVoice V2 음성 변환 (MIT) | — | 증강 실험만 — 채택 안 함 |
| CC BY 표시 유튜브 영상 · 사용자가 준 유튜브 영상 | — | 헤드 재학습 파일럿(미채택) · 평가만 |

### 발음 평가 (한국어)

| 데이터 | 번호 | 쓴 곳 |
|---|---|---|
| 040·041·042·043 발음 파트 (TAG 전문가 음소 전사 · 1~5점 LAB · 읽기 문장 LAR) | 040~043 | CTC 학습 · 평가 (TAG), 점수 회귀기 라벨 (LAB), LAR 적응 |
| 한국어 음성 (KsponSpeech) | — | Stage-1 한국인 CTC 학습 (691시간) · 한국인 헛지적 평가 |
| 인공지능 학습을 위한 외국인 한국어 발화 음성 | 131 (505) | 약지도 · 의사 라벨 실험만 |
| 자유대화 음성(일반남여) | 001 | 경량화 평가 (2,000발화) |
| Zeroth-Korean (CC BY 4.0) | — | 합성 데이터 참조 목소리 · 한국인 낭독 평가 |
| 합성 정상/오류 쌍: Chatterbox · Qwen3-TTS | — | 10-21 서비스 모델 학습 (5,452쌍) |
| SpeechSuper `sent.eval.kr` (유료 API) | — | 비교만 |

### 발음 평가 (영어, 한국인 학습자)

| 데이터 | 번호 | 쓴 곳 |
|---|---|---|
| 교육용 한국인의 영어 음성 | 037 (71463) | 미세조정 · 문장 점수 보정 · 잠근 테스트. **외부 API 로 보내지 않는다** |
| speechocean762 (CC BY 4.0) | — | 음소·문장 점수기 학습 · 평가 |
| L2-ARCTIC 한국인 4명 (CC BY-NC) | — | 모델·규칙 고르기만 (학습 안 함) |
| Kokoro-82M 합성 오류 음성 (Apache-2.0) | — | 서버 기본 모델 `synth-d` 학습 |

### 사투리 분류

| 데이터 | 번호 | 쓴 곳 |
|---|---|---|
| 중·노년층 방언 — 1인 발화 질문에 답하기 (강원·경상) | 71517 | 학습 · 검증 · 봉인 테스트 (라벨 = 거주 지역) |
| 중·노년층 방언 — 1인 발화 질문에 답하기 (충청·전라·제주) | 71558 | 〃 |
| 화자 인식용 짧은 명령어 낭독의 Dialect=01 부분 | 537 | 표준어 클래스 학습 · 테스트 (코드 01 의 뜻은 확인 안 됨) |
| 무감정 낭독 | 71349 | 표준어 외부 대조 (학습 · 채점에서 제외) |
| 130 · 118~122 | 130 · 118~122 | 초기 파일럿만 (최종 모델엔 안 씀) |

### 성대모사 교정 (학습 없음)

Zeroth-Korean(합성 검증 · 실제 화자 쌍), Chatterbox 견본 대사, 영화 장면 클립 8개(유튜브 제3자 업로드에서 목소리만 분리 —
**권리 확인 전 내부 시험용**, 목록·음성은 저장소에 없다).

### 사전학습 모델

| 모델 | 쓴 곳 |
|---|---|
| XLS-R 300m · 2B (`xlsr2_300m.pt` · `xlsr2_2B_1000k.pt`) | 국적(2B 동결 · KD 교사, 300m KD · 전체 학습), 사투리(2B 마지막 8층 학습), 이전 발음 모델 |
| XLS-R 1B · omniASR-W2V-300M | 발음 비교 실험 |
| facebook/w2v-bert-2.0 | 현재 발음 서비스 모델 백본 |
| EAT Large (AudioSet-2M 사전학습) | 사투리 (마지막 4블록 학습) |
| XLS-R 300m VoxLingua107 언어 식별 체크포인트 | 국적 초기값 비교 (VoxLingua107 데이터 자체는 안 씀) |
| facebook/wav2vec2-lv-60-espeak-cv-ft | 영어 발음 인식기 출발점 |
| Whisper large-v3-turbo · WavLM · wav2vec2-large-lv60 · HuBERT(한국어) · SPARC · Hybrid Demucs · CREPE | 비교 · 점검 · 분리 도구 (채택 모델에는 안 들어감) |

### 사내 · 서비스 녹음

사내 녹음(일부러 틀린 발음, 노트북·휴대폰, 부스)과 서비스에 저장된 녹음은 **평가에만** 썼다.
개인정보라 저장소와 외부 서비스에는 넣지 않는다.

이름이 저장소에 안 남은 것: KsponSpeech · 537 · 71349 · 130 · 118~122 의 공식 이름과 이용 조건은 각 AI 허브 데이터셋 페이지에서 확인한다.
