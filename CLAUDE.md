# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

이 저장소에서 작업할 때 참고할 것. **읽는 순서: 이 파일 → `README.md` → 해당 task 의 `README.md`.**

## 이게 뭔가

한국어 음성 AI. 아래 표는 기존 보고용 XLS-R 모델 기준이며 **모델·데이터·평가 잣대가 완전히 다르다.**

**발음평가 NPU 운영은 w2v-BERT 2.0 기반 M3 `w2vbert_synth_tense_v2/step_3000`이다.** 마지막 원격 확인은 2026-10-25 KST, 문서 갱신은 10월 29일이다. 운영 식별자·설정·연구와의 구분은 [현재 서비스 모델](docs/발음평가_현재_서비스_모델.md)을 먼저 읽는다. H0·P3·λ=.3 fold0 파일럿은 연구용이며 운영 채택 기록이 없다.

| task | 하는 일 | 모델 | 성능 |
|---|---|---|---|
| `tasks/nationality-classification/` | 외국인의 한국어 발화 → 화자 모국어/국적 | XLS-R **2B** 동결 + 헤드 | top-5 0.950 / top-1 0.798 (41클래스) |
| `tasks/pronunciation-assessment/` | 외국인의 한국어 발음 → 음소 오류 + 1~5점 | 과거 XLS-R **300m** `tag_all_lar` | weighted F1 0.9547 **및 오류 F1 0.4319** · 화자 단위 PCC 0.662/0.683/0.762 (화자별 전체 발화 집계) |

위 수치는 **보고용 모델** 기준이다. `deploy/serve.sh`·`deploy/infer_*.sh` 가 올리는 모델은 다르다 —
국적은 300m KD 인코더(ONNX·CPU, top-1 0.8245 / top-5 0.9584), 발음은 `tag_all_native`(weighted F1 0.9558).
어느 모델의 수치인지 밝히지 않고 섞어 인용하지 말 것 → `docs/WEIGHTS.md` §5.

**M3 보존 화면 평가:** 재현율 31.95%·정밀도 46.29%·오류 검출 F1 0.3781(TP 631/FP 732/FN 1,344), 당시 입력 검사 통과 2,281발화 기준이다. 과거 오류 F1과의 기록값 차이는 −0.0538이지만 같은 평가 조건의 모델 변화량이 아니다. M3 화자 단위 PCC는 0.683/0.713/0.752이며 옛 PCC와 화자·발화 구성도 다르다. M3의 옛 0.9547과 동일 조건 weighted F1은 미확인이다. [성능 비교·분모](docs/발음평가_현재_서비스_모델.md#성능과-과거-xls-r-수치의-비교)를 함께 인용한다.

두 task 의 라이브러리(`src/`) 사이 교차 import 는 **0건**이다. 스크립트 예외가 1곳 있다 —
`tasks/pronunciation-assessment/scripts/eval/eval_word_skip_record.py` 가 디코드 경로에서
`country_classifier.audio.load_wav_mono` 를 쓴다. 그 함수를 고치면 이 스크립트도 확인할 것.
공유하는 건 `shared/`(경로 설정 + fairseq 가드)뿐이고, **오디오 로더는 일부러 안 합쳤다**
(요구사항이 실제로 다르다 — `docs/DECISIONS.md` #1).

## 환경 (이거 모르면 아무것도 안 돌아간다)

```bash
source tools/env.sh      # 저장소 루트 .env 를 export → $PY 와 경로 변수가 잡힌다
```

- **파이썬은 항상 `$PY`** — `.env` 의 `PY=`(`tools/setup.sh` 가 자동 작성). 맨 `python` 은 conda base 라
  torch 가 없다. env 이름·위치가 머신마다 달라서(`create_env.sh` 기본 `speech-ai`, 학습 서버는 `fairseq_xlsr`)
  경로를 박으면 안 된다.
- python 3.8.20 · torch 2.1.2+cu118 · omegaconf 2.0.6 · hydra-core 1.0.7 · `pip<24.1`.
  버전 조합이 서로 묶여 있어 하나만 올려도 깨진다. 재구축은 `bash tools/create_env.sh`, 자세한 건 `docs/ENVIRONMENT.md`.
- **fairseq** = upstream **main 커밋 `3d262bb`** 을 editable 설치(`$FAIRSEQ_ROOT`, 기본은 저장소 옆 `../fairseq`).
  **소스 수정은 없다** — 예전에 "수정 사본"이라 부르던 건 v0.12.2 태그보다 앞선 main 체크아웃이었다.
  PyPI 판이나 v0.12.2 태그로 깔면 체크포인트 로드가 `ConfigKeyError: multi_corpus_keys` 로 죽는다.
- **KoG2P** = 클론 `cbc69bb`(`$G2P_TOOLS/KoG2P`, 기본 `../g2p_tools/KoG2P`). pip 패키지가 아니다.
  `g2p.py` 가 **import 시점에 argv 를 파싱**하므로 `shared.paths.ensure_kog2p()` 뒤 `sys.argv` 를 잠시
  비우고 import 한다(`src/mdd/pipeline.py` 첫머리 패턴). 안 그러면 호출한 스크립트의 인자에 죽는다.
- GPU 는 발음 진단과 학습·특징 추출에 필요하다. 국적 추론은 ONNX·CPU 로 돈다.
- `ffmpeg 실패 (rc=127)` 는 오디오 탓이 아니다 — conda 의 `LD_LIBRARY_PATH` 가 시스템 ffmpeg 를 깨뜨린 것.
  `env -u LD_LIBRARY_PATH` 로 우회(`docs/ENVIRONMENT.md` 함정 10).

## 명령

저장소 루트에서, `source tools/env.sh` 뒤에.

```bash
$PY tests/test_imports.py            # 전 .py 문법 + 경로 상수 + 라이브러리 import 22개. 수 초, GPU·데이터 불필요
$PY tests/test_imports.py --quick    # torch/fairseq 가 필요한 import 는 건너뛴다
$PY <스크립트> --help                 # 스크립트 하나만 점검 — argparse 나 --help 가드가 있어 본체는 안 돈다

bash deploy/infer_nationality.sh a.wav              # 국적 top-5 (임시 서버 기동 → 요청 → 종료)
bash deploy/infer_mdd.sh a.wav "읽은 문장"          # 발음 진단. 같은 --session 으로 5발화부터 점수
bash deploy/serve.sh nat --port 8905                # 포그라운드 서버(mdd 는 8914) — tmux 에서
```

- **pytest 스위트·린터·포매터는 없다.** 테스트는 위 스모크와
  `tasks/nationality-classification/tests/smoke_test.py`(매니페스트 필요, `--backend tiny --synthetic` 이면
  오디오·XLS-R 없이 돈다)뿐이다. 학습·평가·데이터 준비 명령은 각 task README 의 "실행" 절.
- 골든 수치 회귀 테스트도 없다. 평가 로직을 바꿨다면 캐시된 예측(`--preds`)으로 재채점해 `results/` 기록값과
  대조한다(발음 README "실행" 절에 기대값이 적혀 있다).
- `test_imports.py` 의 문법 검사는 **돌린 인터프리터의 문법**으로 판정하므로 `$PY`(3.8)로 돌려야 의미가 있다.
  맨 `python` 으로 돌리면 패키지가 없어 실패가 쏟아진다 — 코드 결함이 아니다.
- 스크립트(`scripts/`·`experiments/`·`deploy/`)는 **compile 만** 되고, 실제 import 는 지정된 모듈 22개뿐이다.
  함수·분기 안의 import 오류나 정의 시점 TypeError 는 스모크를 통과한다.

## 경로 규칙 — 하드코딩 금지

코퍼스·모델·산출물은 저장소 밖에 있고 위치는 환경마다 다르다. **절대경로를 코드에 쓰지 말 것.**

```python
from shared.paths import DATA_ROOT, MDD_OUTPUTS, NAT_OUTPUTS   # .env 에서 읽는다
```

- 값은 저장소 루트 `.env`(git 제외) 또는 환경변수. 셸에서 쓰려면 `source tools/env.sh`.
- **우선순위가 셸과 파이썬에서 반대다.** 파이썬(`shared/paths.py`)은 이미 설정된 환경변수가 이기지만
  `tools/env.sh` 는 `.env` 를 그대로 source 해서 **`.env` 가 이긴다.** `deploy/*.sh` 가 전부 이걸 source 하므로
  `KD_MODEL_DIR=... bash deploy/infer_nationality.sh` 같은 일회성 지정은 `.env` 에 그 키가 있으면 조용히 무시된다.
- 미설정 경로는 `<SPEECH_AI_DATA 미설정>/...` 로 조립되다가 **파일을 열 때** 멈춘다.
  덕분에 데이터 없는 머신에서도 import·문법 점검이 된다. 이 성질을 깨지 말 것 —
  `shared/paths.py` 최상단에서 파생 상수를 계산하면 import 자체가 죽는다(실제로 그랬다).
- fairseq 는 반드시 `ensure_fairseq()` 뒤에 import. sys.path 에 **있기만** 해선 부족하고
  **맨 앞**이어야 namespace package 그림자를 이긴다.

## 코드 흐름 (여러 파일을 읽어야 보이는 것)

- **부트스트랩** — 진입점 맨 위 블록이 `pyproject.toml` 을 찾아 저장소 루트와 `tasks/*/src` 를 `sys.path`
  앞에 넣는다(fairseq 를 쓰는 곳은 이어서 `ensure_fairseq()`). 그래서 `country_classifier`·`mdd` 가 설치 없이
  패키지로 잡힌다. **`import mdd_eval` 처럼 패키지를 빼고 쓰면 안 잡힌다** — `from mdd import mdd_eval`.
- **기존 XLS-R 발음(MDD)** — 문장 →KoG2P→ 표준 발음 / 오디오 →CTC(자모 46종)→ 실현 자모열 → NW 정렬 → 오류 유형
  (`src/mdd/pipeline.py`). 71음소 표기는 `jamo_to_kog2p.py` 후처리다. 점수 3종은 같은 forward 에서 나온
  GOP 특징 41종(`gop_feats_v2.py`) → GBM 회귀기(joblib). `deploy/mdd-api` 는 사용자용 지적에서
  삭제·삽입형을 숨긴다(`suppress.py`, DECISIONS #10).
- **MDD 체크포인트 지정** — `mdd.mdd_eval.Decoder()` 는 **모듈 전역 `mdd_eval.CKPT`** 를 읽는다.
  `mdd.pipeline` 이 import 될 때 이 값을 `tag_all_lar` 로 덮어쓰므로 체크포인트는 import 가 끝난 **뒤에** 넣는다.
  스크립트별 `--ckpt` 기본값은 제각각이다(주 평가 `mdd_eval_tag71.py` 는 `tag_combined`) —
  수치를 재현할 땐 README 명령처럼 명시한다.
- **기존 로컬 서빙 모델** — `deploy/mdd-api` 는 `.env` 의 `MDD_SERVE_CKPT`·`MDD_SERVE_SCORER`(`tag_all_native` 짝)로
  못 박는다. `deploy/mdd-demo` 는 이 키를 안 읽고 `mdd.pipeline` 기본값(`$SPEECH_AI_DATA` 의 `tag_all_lar` +
  `$MDD_ANALYSIS` 의 L1-prior JSON)을 써서, `setup.sh` 로 만든 추론 전용 환경에선 뜨지 않는다.
- **NPU 운영 발음 모델** — `deploy/pronunciation-demo/app.py` 계열이며 원격 `run_8932.sh`가 M3의 HF·RBLN 경로와 표시 인자를 지정한다. 로컬 기본값이나 최근 연구 체크포인트를 운영 설정으로 간주하지 않는다.
- **국적** — 동결 XLS-R 특징을 `.npz` 로 캐시하고 그 위에서 헤드를 학습한다(CPU 몇 분).
  서빙(`deploy/nationality-kd-demo/app.py`)은 KD 인코더 ONNX + numpy 헤드(`head_np.npz`)라 torch·fairseq 가
  필요 없다. 모델 위치는 `KD_MODEL_DIR`.
- **`results/`** — 문서에 쓰인 수치의 근거를 추린 사본(원본은 `$MDD_OUTPUTS`·`$NATIONALITY_OUTPUTS`).
  수치를 인용하거나 고칠 땐 여기서 출처(`source` 키)부터 확인한다.

## 새 스크립트를 쓸 때

1. 실행 진입점은 `tasks/*/scripts/{data,train,eval,score,export}/` 아래에.
   재사용 로직은 `tasks/*/src/` 안으로. **데모/서버에 라이브러리 로직을 넣지 말 것**
   (예전에 `mdd_demo.py` 가 그랬고 평가 스크립트 6개가 데모를 import 하고 있었다).
2. 파일 맨 위에 부트스트랩 5줄을 넣는다(기존 스크립트 아무거나 복사). 루트 가드가 있는
   버전이어야 한다 — 없으면 저장소 밖에서 실행할 때 무한루프다.
3. **argparse 를 모듈 최상단에서 `parse_args()` 하지 말 것.** import 부작용이 생겨
   이 모듈을 쓰는 쪽이 argv 우회를 해야 한다. `main()` 안에서 파싱한다.
4. argparse 가 없는 스크립트라면 `--help` 가드를 넣는다. 매니페스트를 재생성하는
   스크립트가 있어서, `--help` 로 본체가 돌면 산출물이 덮어써진다.
5. **3.8 문법으로 쓴다.** f-string 식 안의 백슬래시·`match` 는 SyntaxError, `list[int]`·`int | None`
   주석은 정의 시점 TypeError, `str.removeprefix`·`functools.cache` 는 없다.
6. 장치를 박지 말 것(`.cuda()` 금지). GPU 없는 머신에서도 돌아야 한다 —
   `next(model.parameters()).device` 를 읽는다(MDD 디코더는 `MDD_FORCE_CPU=1` 로 강제 CPU).
7. 다 쓰면 `$PY tests/test_imports.py`.

## 평가 잣대 (틀리면 수치가 통째로 달라진다)

- **발음**: 라벨이 3종(TAG / record / gold). **TAG 만 MDD 잣대다.**
  gold 로 재면 TAG 에서 0.965 인 공식 모델도 F1 **0.42** 로 떨어진다 — 모델이 아니라 잣대가
  틀린 것. 근거는 `$MDD_OUTPUTS/eval_official2_041.json`(`checkpoint_mdd_eu.pt`, tol0 `.4196`)이며
  **번들 `04_공식팩_검증.json` 에는 아직 병합돼 있지 않다.** 번들의 `gold잣대_대조` 는 공식이
  아니라 우리 모델(`.2235`)이니 이 주장의 근거로 인용하지 말 것. gold 는 점수 라벨 출처로만 쓴다.
- **발음 점수는 화자 단위로만 유효.** 발화 1개 PCC 는 발음·이해도 **≈0.43**,
  유창성 **0.607** 이고 20~30문장 모아야 포화한다.
- **국적**: 61 / 62 / 41 클래스가 섞여 있다. 어느 split 인지 먼저 확인할 것.
  어족·권역 병합(31/13-class)은 **중단됐다** — `archive/` 에 있고 되살리지 않는다.
- 과거 XLS-R의 weighted F1(0.9547)과 오류 F1(0.4319)은 **항상 함께 보고한다.** 질문을 받을 때까지 오류 F1 설명을 미루지 않는다. 두 값은 정상/오류 이진 채점이며 실제음·유형의 정확 진단과 다르다.
  단 정상 위치가 **95.86%** 라 전부 "정상"으로 찍어도 weighted F1 **0.9384** 가 나온다.
  즉 0.9547 의 실제 개선폭은 **+0.0163** 이다 — 이 맥락 없이 0.9547 만 인용하지 말 것.
- **0.9547 은 학습에 쓴 287개 문장 범위 안의 값이다.** 학습셋 155만 발화가 287문장뿐이라
  (문장당 5,398회 반복) 모델이 문장을 외워 표준발음을 읊는다. 홀드아웃 실험 실측:
  본 문장 weighted F1 0.9509 / 되돌림 61.8% → **안 본 문장 0.8218 / 되돌림 22.1%**.
  화자 일반화는 정상이다(중복 화자 영향 +0.0000). **문장 일반화만 깨져 있다.**
  후처리·재가중·디코딩 편향은 못 고친다 — 문장 다양성만이 검증된 처방(격차 −43%).
  근거와 재현: `tasks/pronunciation-assessment/results/08_문장암기_일반화.md`

## 하지 말 것

- **`*_preds.jsonl` 삭제** — 재채점의 원천. 지우면 GPU 재디코드 수시간.
- **`tasks/pronunciation-assessment/configs/*.yaml` 삭제** — 학습 재현의 유일한 기록.
- **`quality_flag != ok` 를 평가에서 제외** — `usable_for_aggregation=false` 는 "통계 집계에서
  빼라"는 뜻이지 "평가에서 빼라"가 아니다. 제외했던 25,146발화에 진짜 단어누락의 69.6%가 있었다.
- **가중치 파일 짝을 섞지 말 것** — `encoder_kd_l17.onnx`↔`head_np.npz`,
  `encoder_l17.onnx`↔`head.pt`, `mdd_tag_all_native`↔`score_models_v2_native.joblib`.
  shape 가 같아서 **예외 없이 답만 틀린다.** 옮기기 전에 `docs/WEIGHTS.md`.
- **`shared.paths.MDD_CKPT` 기본값(`tag_all_lar`) 변경** — `results/` 에 기록된 수치가 재현되지 않는다.
  서빙 모델은 배포 쪽(`MDD_SERVE_CKPT`)에서 바꾼다(DECISIONS #11).
- **fairseq 를 `--no-deps` 없이 설치하거나 omegaconf·hydra-core 핀을 푸는 것** — resolver 가 omegaconf 를
  올리면 hydra 설정이 통째로 깨진다.
- **`tailscale serve`** — funnel 이 꺼진다. `funnel --bg --set-path` 를 쓴다.
- **이미 쓰는 포트 재사용** — 검증은 다른 포트로 띄워 응답을 대조한 뒤 교체한다.

## 실행 환경 메모

- 긴 작업(특징 추출·디코드)은 **tmux** 로. 10분~수시간이다.
- 큰 산출물은 홈(`/`)이 아니라 `$SPEECH_AI_DATA` 나 `.env` 의 산출물 경로 쪽으로. 학습 서버는 `/` 여유가 몇 GB 뿐이다.
- 학습 서버에는 이 저장소 밖에서 도는 서비스가 있다(국적 실서비스 8900 — 타인 소유, 건드리지 말 것 →
  `deploy/nationality-api/README.md`). 포트를 뺏지 말 것.
- 속도 측정은 단독 실행으로 — 벤치마크를 동시에 돌리면 값이 4~25배 부풀려진다(`docs/ENVIRONMENT.md` 함정 9).
- `deploy/` 의 서버·데모에는 **인증이 없다.** 외부에 열 땐 앞단에 접근 제어를 둔다.

## 아직 안 끝난 것

`docs/REMAINING_ISSUES.md` 참고. 요약하면 — 골든 수치 회귀 테스트가 없고, 옛 작업 트리
(학습 서버 `/home/tta/scripts` 등)가 아직 남아 있으며(운영 서비스가 참조 중이라 단계적으로 정리),
국적 KD 모델이 실서비스(8900)에 아직 안 붙어 있다.

아래 ARIS 스킬은 학습 서버 전용이다 — `.claude/skills/` 는 `/home/tta/tools/aris` 를 가리키는 심링크이고
git 에서 제외돼(`.gitignore`) 다른 머신엔 없다.
<!-- ARIS:BEGIN -->
## ARIS Skill Scope
ARIS skills installed in this project: 22 entries.
Manifest: `.aris/installed-skills.txt` (lists every skill ARIS installed and its upstream target).
For ARIS workflows, prefer the project-local skills under `.claude/skills/` over global skills.
Do not modify or delete files inside any skill that is a symlink (symlinks point into `/home/tta/tools/aris`).
Update with: `bash /home/tta/tools/aris/tools/install_aris.sh`  (re-runnable; reconciles new/removed skills).
<!-- ARIS:END -->
