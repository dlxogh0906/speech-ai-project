# 환경 — 이거 모르면 아무것도 안 돌아간다

```bash
PY=python   # 이것만 쓴다
```

| | |
|---|---|
| Python | **3.8.20** (fairseq 0.12.2 + omegaconf 2.0.6 조합이 고정한다) |
| torch | 2.1.2+cu118 |
| fairseq | 0.12.2 — **editable 설치된 수정 사본** (`$FAIRSEQ_ROOT`, 기본 `$FAIRSEQ_ROOT`) |
| 하드웨어 | A100 80GB × 1 |
| 재구축 | `bash tools/create_env.sh` |

conda `base`(3.13)에는 torch가 없고 fairseq도 못 돌린다.

## 함정 1 — `import fairseq` 가 조용히 잘못된 걸 집는다

fairseq 저장소 루트에는 `__init__.py` 가 없다. 그 상위 디렉터리가 `sys.path` 에
**editable 설치 경로보다 먼저** 들어가면, 파이썬이 저장소 루트를 namespace package 로 보고
진짜 `fairseq` 패키지를 가려 버린다. 증상은 조용한 `ImportError` 나 엉뚱한 모듈 로드라서
원인을 찾는 데 오래 걸린다.

```python
from shared.fairseq_guard import ensure_fairseq
ensure_fairseq()          # fairseq import 앞에서 호출. 저장소 전체에 이미 적용돼 있다.
import fairseq
```

검증할 때는 **CWD 를 `$HOME` 가 아닌 곳**에 두고 확인할 것.

## 함정 2 — `pip install fairseq` 로는 안 된다

수정 사본이라 PyPI 버전과 동작이 다르다. `tools/create_env.sh` 는
editable `--no-deps` 설치 + `omegaconf==2.0.6` / `hydra-core==1.0.7` 핀을 건다.
omegaconf 2.0.6 이 legacy 메타데이터라서 **`pip<24.1`** 이 필요하다.

> ⚠️ **재현성 한계.** 이 fairseq 사본의 정확한 upstream 커밋과 패치 내역이 기록돼 있지 않다.
> 새 머신에서 완전 재현이 안 되는 상태다 → `docs/REMAINING_ISSUES.md` #3.

## 함정 3 — KoG2P 는 pip 패키지가 아니다

`$G2P_TOOLS/KoG2P/` 에 클론돼 있고 `g2p.py` 는 **import 시점에 optparse 로 argv 를 파싱한다.**
그래서 호출부에서 argv 를 잠시 비워야 한다.

```python
from shared.paths import ensure_kog2p
ensure_kog2p()
_argv, sys.argv = sys.argv, sys.argv[:1]
from g2p import readRules, graph2prono, ver_info
sys.argv = _argv
```

## 함정 4 — Python 3.8 문법 제약

f-string 표현식 안에 백슬래시를 못 쓴다. 워크어라운드:

```python
# SyntaxError
print(f"{'모델 \\ 데이터':14}")
# OK
print("{:14}".format("모델 \\ 데이터"))
```

## 함정 5 — 긴 작업은 tmux 로

특징 추출·디코드는 10분~수시간이다. SSH 가 끊기면 날아간다.

## 함정 6 — VS Code Remote-SSH 가 OOM 으로 죽는다

산출물 경로가 `/data`(2 TB)를 가리키면 파일 감시자가 전부 훑다가 확장 호스트가 죽는다.
`.vscode/settings.json` 의 `files.watcherExclude` 를 지우지 말 것.

## 함정 7 — CPU 에서 fp16 은 없다 (있는 게 아니라 느린 게 아니다)

```
matmul float32       6.84 ms
matmul float16   실패 — "addmm_impl_cpu_" not implemented for 'Half'
matmul bfloat16   8,615 ms   ← fp32 대비 1,260배
```

이 서버 CPU 는 AMD EPYC 7H12(Zen 2). `f16c` 는 있지만 **변환** 명령어일 뿐이고
`avx512_fp16` · `avx512_bf16` · `amx` 가 없다. **CPU 경로는 fp32 가 정답이다.**

int8 동적 양자화도 이 모델엔 안 먹힌다 — fairseq `MultiheadAttention` 이 `q_proj.bias`
를 직접 읽어 어텐션은 양자화가 깨지고, FFN 만 바꾸면 **1.05배**뿐이다(conv 특징
추출기가 병목이라 Transformer 만 줄여봐야 벽시계가 안 준다).

## 함정 8 — GPU 가 없어도 돌아가야 한다

`Decoder.__init__` 과 `forward_lprobs` 가 `.cuda().half()` 를 무조건 걸고 있어서
GPU 없는 머신에서는 **적재 단계에서 죽었다**(2026-08-06 수정). 지금은 장치·정밀도를
모델 파라미터에서 읽는다 — GPU 가 있으면 fp16 GPU, 없으면 fp32 CPU.

```bash
MDD_FORCE_CPU=1 $PY deploy/mdd-api/mdd_infer_api.py --port 8916   # 강제 CPU
```

새로 쓰는 코드도 `.cuda()` 를 박지 말 것. `next(model.parameters()).device` 를 읽는다.

## 함정 9 — 벤치마크는 단독 실행으로

여러 벤치마크를 동시에 돌렸다가 값이 **4~25배** 부풀려진 적이 있다
(국적 2B CPU 21,254 ms → 단독 실행 실측 1,832 ms). CPU 를 나눠 쓰기 때문이다.
속도를 잴 때는 다른 작업이 없는지 `uptime` 으로 확인하고 시작한다.

## 디스크

```
/       96G 중 3.5G 남음   ← 여유 없음. 큰 산출물을 홈에 만들지 말 것
/data  2.0T 중  21G 남음   ← 2026-08-07 기준. 체크포인트 정리가 필요한 시점
```
