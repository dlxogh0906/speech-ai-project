# 환경 — 이거 모르면 아무것도 안 돌아간다

```bash
PY=python   # 이것만 쓴다
```

| | |
|---|---|
| Python | **3.8.20** (fairseq 0.12.2 + omegaconf 2.0.6 조합이 고정한다) |
| torch | 2.1.2+cu118 |
| fairseq | 0.12.2 — upstream **main 브랜치** 커밋 `3d262bb` 을 editable 설치 (`$FAIRSEQ_ROOT`) |
| KoG2P | `cbc69bb` — pip 가 아니라 클론 (`$G2P_TOOLS/KoG2P`) |
| 하드웨어 | 학습: A100 80GB × 1 · 추론 검증: RTX A6000 48GB |
| 재구축 | `bash tools/create_env.sh` (두 task 통합 환경) |

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

## 함정 2 — `pip install fairseq` 도, **v0.12.2 태그도** 안 된다

체크포인트에 저장된 task 설정에 `multi_corpus_keys` 가 들어 있다. 이 필드는 **v0.12.2
태그에 없고 main 브랜치에만 있다**(`fairseq/tasks/audio_pretraining.py`). 태그로 깔면
체크포인트 로드가 이렇게 죽는다:

```
omegaconf.errors.ConfigKeyError: Key 'multi_corpus_keys' not in 'AudioFinetuningConfig'
```

fairseq 은 v0.12.2 이후 버전 번호를 올린 적이 없어서 **main 브랜치도 `__version__` 이
"0.12.2" 로 보고한다.** 그동안 "수정 사본" 으로 불리던 것의 정체가 이것이다 — 손으로 고친
사본이 아니라 태그보다 앞선 main 체크아웃이었다.

```bash
git clone https://github.com/facebookresearch/fairseq.git $FAIRSEQ_ROOT
git -C $FAIRSEQ_ROOT checkout 3d262bb25690e4eb2e7d3c1309b1e9c406ca4b99
pip install -e $FAIRSEQ_ROOT --no-deps --no-build-isolation
```

`--no-deps` 가 없으면 resolver 가 omegaconf 를 최신으로 올려 hydra 설정이 통째로 깨진다.
`--no-build-isolation` 이 없으면 격리된 빌드 환경에 torch·Cython 이 없어 빌드가 실패한다.
omegaconf 2.0.6 이 legacy 메타데이터라서 **`pip<24.1`** 도 필요하다.

> **소스 수정은 필요 없다.** 순정 upstream 으로 두 task 추론을 전 구간 검증했다
> (2026-08-08, RTX A6000). 저장소에도 커스텀 fairseq 등록 코드가 0건이다 —
> `register_task` / `register_model` / `--user-dir` 어느 것도 쓰지 않고,
> 체크포인트도 표준 `audio_finetuning` + `wav2vec_ctc` + `CtcCriterion` 구성이다.

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

## 함정 10 — conda 의 `LD_LIBRARY_PATH` 가 시스템 ffmpeg 를 깨뜨린다

anaconda 설치본이 셸 프로파일에서 `LD_LIBRARY_PATH=<anaconda>/lib` 를 걸어 두는 머신이 있다.
그러면 시스템 `ffmpeg` 가 anaconda 쪽 `libtinfo`/`libncursesw` 를 잡아 이렇게 죽는다:

```
ffmpeg: symbol lookup error: /lib/x86_64-linux-gnu/libpango-1.0.so.0: undefined symbol: g_once_init_leave_pointer
```

`deploy/nationality-kd-demo/app.py` 의 `decode()` 가 ffmpeg 를 서브프로세스로 부르므로
**국적 추론이 통째로 실패한다.** 증상은 `{"error":"오디오가 너무 짧거나 해석 불가"}` 라서
오디오 탓으로 오해하기 쉽다. 확인:

```bash
echo $LD_LIBRARY_PATH                    # anaconda/lib 가 있으면 이 함정이다
env -u LD_LIBRARY_PATH ffmpeg -version   # 이걸로 되면 확정
```

해결은 그 머신의 셸 프로파일에서 `LD_LIBRARY_PATH` 를 빼는 것이다. 임시로는
`env -u LD_LIBRARY_PATH bash deploy/infer_nationality.sh ...` 로 우회한다.

## 디스크

```
/       96G 중 3.5G 남음   ← 여유 없음. 큰 산출물을 홈에 만들지 말 것
/data  2.0T 중  21G 남음   ← 2026-08-07 기준. 체크포인트 정리가 필요한 시점
```
