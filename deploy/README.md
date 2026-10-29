# deploy — 데모와 추론 API

학습·평가 코드와 라이프사이클이 달라 분리했다. FastAPI + uvicorn 기반이며, 모델 선택은 각 진입점의 설정을 따른다.

**발음평가 NPU 운영은 `pronunciation-demo/` 계열의 M3 `w2vbert_synth_tense_v2/step_3000`이다.** `pron-api` · 포트 `8932` · NPU 7, 마지막 원격 확인은 2026-10-25 KST다(문서 갱신 10월 29일). [현재 서비스 모델·원격 설정](../docs/발음평가_현재_서비스_모델.md)과 [NPU API 안내](../docs/NPU_서버_API_안내.md)를 참고한다.

| | 무엇 | 입력 → 출력 |
|---|---|---|
| [`pronunciation-demo/`](pronunciation-demo/) | **M3 NPU 운영 계열의 통합 발음평가 앱** | `/sent.eval.kr` · `/speak` · `/word.judge`, 원격 실행 인자로 모델 지정 |
| `mdd-demo/` | 기존 XLS-R `tag_all_lar` 웹 UI | 문장 제시 → 브라우저 녹음 → 오류 위치·유형 지적 |
| `mdd-api/` | 기존 XLS-R `tag_all_native` 진단 + 점수 API | 오디오 + 문장 → 진단·점수 JSON |
| `model-compare/` | 두 모델 나란히 비교 | 같은 오디오 → 모델별 결과 대조 |
| `nationality-latency-demo/` | ONNX CPU 지연·부하 측정 | GPU vs CPU, 동접 부하 |
| `nationality-batch/` | 배치 추론·양자화 유틸 | 매니페스트 → 예측 CSV |

## 기존 로컬 API·데모 실행

아래 발음 명령은 XLS-R 경로다. NPU 운영 배포 절차와 구분한다.

```bash
source tools/env.sh
PORT=8911     # 쓰지 않는 포트를 고를 것

# 발음 진단 UI
$PY deploy/mdd-demo/app.py --port $PORT

# 발음 진단 + 점수 통합 API
$PY deploy/mdd-api/mdd_infer_api.py --port $PORT
curl -s localhost:$PORT/health
```

### 셸 스크립트 (단건 추론 · 서버 기동)

```bash
# 단건 CLI — 서버를 임시로 띄워 추론하고 끈다 (--keep 으로 유지, --url 로 재사용)
bash deploy/infer_mdd.sh sample.wav "안녕하세요 만나서 반갑습니다"   # 발음 진단 + 점수
bash deploy/infer_nationality.sh sample.wav                         # 국적 top-5 (ONNX·CPU)

# 서버 기동 (포그라운드 — tmux 안에서)
bash deploy/serve.sh mdd --port 8914     # POST /diagnose
bash deploy/serve.sh nat --port 8905     # POST /predict
```

옵션은 각 스크립트의 `--help` 참조.

### mdd-api 엔드포인트

| | |
|---|---|
| `POST /diagnose` | `audio`(파일) + `sentence` + 선택 `session_id` → 진단·점수 JSON |
| `GET /session/{sid}` | 세션에 누적된 발화들의 화자 단위 점수 |
| `DELETE /session/{sid}` | 세션 폐기 |
| `GET /health` | 헬스체크 |

라우터는 `""` 와 `"/api"` 두 프리픽스에 모두 걸려 있다. 리버스 프록시가 경로 프리픽스를
벗기지 않고 그대로 넘기는 경우를 위해서다.

> **점수는 화자 단위로만 유효하다.** 발화 1개로는 PCC 0.43 이고 20~30문장 모아야 포화한다.
> `mdd-api` 는 발화 5개 미만이면 점수를 내지 않고 진단만 반환한다.

## 기존 `mdd-api` 모델 로딩

GOP 점수 특징이 MDD forward 의 부산물이라 **모델을 한 번만 올려 진단과 점수를 둘 다 낸다.**
PyTorch 경로는 기동 시 `MDD_SERVE_CKPT`를 적재하며 점수 회귀기는 `MDD_SERVE_SCORER`로 지정한다. 기본 모델은 `tag_all_native`다. ONNX 선택 설정은 [mdd-api 안내](mdd-api/README.md)를 따른다.

## 주의

- 위 기존 로컬 API에는 인증이 없다. NPU 발음평가 API는 토큰 인증을 사용한다.
- 리버스 프록시 설정 방식에 따라 기존 라우팅이 꺼질 수 있다. 경로 마운트를 쓸 것.
- 배포 전 반드시 `source tools/env.sh` 후 `$PY tests/test_imports.py`로 저장소 상태를 확인한다.
