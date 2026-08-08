# CLAUDE.md

이 저장소에서 작업할 때 참고할 것. **읽는 순서: 이 파일 → `README.md` → 해당 task 의 `README.md`.**

## 이게 뭔가

한국어 음성 AI 두 가지. 백본은 둘 다 XLS-R 이지만 **모델·데이터·평가 잣대가 완전히 다르다.**

| task | 하는 일 | 모델 | 성능 |
|---|---|---|---|
| `tasks/nationality-classification/` | 외국인의 한국어 발화 → 화자 모국어/국적 | XLS-R **2B** 동결 + 헤드 | top-5 0.950 / top-1 0.798 (41클래스) |
| `tasks/pronunciation-assessment/` | 외국인의 한국어 발음 → 음소 오류 + 1~5점 | XLS-R **300m** + CTC 직접 학습 | weighted F1 0.9547 · PCC 0.662/0.683/0.762 (전체 발화 기준 · 30발화 집계 0.646/0.671/0.762) |

두 task 는 **교차 import 0건**이다. 한쪽을 고쳐도 다른 쪽은 안 깨진다.
공유하는 건 `shared/`(경로 설정 + fairseq 가드)뿐이고, **오디오 로더는 일부러 안 합쳤다**
(요구사항이 실제로 다르다 — `docs/DECISIONS.md` #1).

## 환경 (이거 모르면 아무것도 안 돌아간다)

```bash
python=/home/tta/miniconda3/envs/fairseq_xlsr/bin/python   # conda base 엔 torch 가 없다
```

python 3.8.20 · torch 2.1.2+cu118 · fairseq 0.12.2(editable, 수정 사본) · omegaconf 2.0.6 · A100 80GB ×1.
버전 조합이 서로 묶여 있어 하나만 올려도 깨진다. 자세한 건 `docs/ENVIRONMENT.md`.

## 경로 규칙 — 하드코딩 금지

코퍼스·모델·산출물은 저장소 밖에 있고 위치는 환경마다 다르다. **절대경로를 코드에 쓰지 말 것.**

```python
from shared.paths import DATA_ROOT, MDD_OUTPUTS, NAT_OUTPUTS   # .env 에서 읽는다
```

- 값은 저장소 루트 `.env`(git 제외) 또는 환경변수. 셸에서 쓰려면 `source tools/env.sh`.
- 미설정 경로는 `<SPEECH_AI_DATA 미설정>/...` 로 조립되다가 **파일을 열 때** 멈춘다.
  덕분에 데이터 없는 머신에서도 import·문법 점검이 된다. 이 성질을 깨지 말 것 —
  `shared/paths.py` 최상단에서 파생 상수를 계산하면 import 자체가 죽는다(실제로 그랬다).
- fairseq 는 반드시 `ensure_fairseq()` 뒤에 import. sys.path 에 **있기만** 해선 부족하고
  **맨 앞**이어야 namespace package 그림자를 이긴다.

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
5. 다 쓰면 `python tests/test_imports.py`.

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
- weighted F1(0.9547)과 오류 F1(0.4319)은 둘 다 사실이다. 보고엔 전자, 질문 오면 후자도 말한다.
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
- **`tailscale serve`** — funnel 이 꺼진다. `funnel --bg --set-path` 를 쓴다.
- **이미 쓰는 포트 재사용** — 검증은 다른 포트로 띄워 응답을 대조한 뒤 교체한다.

## 실행 환경 메모

- 긴 작업(특징 추출·디코드)은 **tmux** 로. 10분~수시간이다.
- 디스크가 빠듯하다(`/` 95% 사용). 큰 산출물은 홈이 아니라 `$SPEECH_AI_DATA` 쪽으로.
- 이 서버 밖에서 돌리는 서비스가 있다. 포트를 뺏지 말 것.

## 아직 안 끝난 것

`docs/REMAINING_ISSUES.md` 참고. 요약하면 — 골든 수치 회귀 테스트가 없고,
fairseq 수정 사본의 출처가 기록돼 있지 않으며, 옛 작업 트리(`/home/tta/scripts` 등)가
아직 남아 있다(운영 서비스가 참조 중이라 단계적으로 정리).
<!-- ARIS:BEGIN -->
## ARIS Skill Scope
ARIS skills installed in this project: 22 entries.
Manifest: `.aris/installed-skills.txt` (lists every skill ARIS installed and its upstream target).
For ARIS workflows, prefer the project-local skills under `.claude/skills/` over global skills.
Do not modify or delete files inside any skill that is a symlink (symlinks point into `/home/tta/tools/aris`).
Update with: `bash /home/tta/tools/aris/tools/install_aris.sh`  (re-runnable; reconciles new/removed skills).
<!-- ARIS:END -->