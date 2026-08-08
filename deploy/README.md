# deploy — 데모와 추론 API

학습·평가 코드와 라이프사이클이 달라 분리했다. 전부 FastAPI + uvicorn 이고,
`.env` 의 경로 설정을 그대로 쓴다.

| | 무엇 | 입력 → 출력 |
|---|---|---|
| `mdd-demo/` | 발음 진단 웹 UI | 문장 제시 → 브라우저 녹음 → 오류 위치·유형 지적 |
| `mdd-api/` | **MDD 진단 + 점수 3종 통합 API** | 오디오 + 문장 → 진단·점수 JSON |
| `model-compare/` | 두 모델 나란히 비교 | 같은 오디오 → 모델별 결과 대조 |
| `nationality-latency-demo/` | ONNX CPU 지연·부하 측정 | GPU vs CPU, 동접 부하 |
| `nationality-batch/` | 배치 추론·양자화 유틸 | 매니페스트 → 예측 CSV |

## 실행

```bash
source tools/env.sh
PORT=8911     # 쓰지 않는 포트를 고를 것

# 발음 진단 UI
python deploy/mdd-demo/app.py --port $PORT

# 발음 진단 + 점수 통합 API
python deploy/mdd-api/mdd_infer_api.py --port $PORT
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

## 모델 로딩

GOP 점수 특징이 MDD forward 의 부산물이라 **모델을 한 번만 올려 진단과 점수를 둘 다 낸다.**
기동 시 체크포인트(`MDD_CKPT`)를 GPU 에 올리므로 첫 요청 전까지 수십 초가 걸린다.

## 주의

- **공개 노출은 신중히.** 이 API 들은 인증이 없다. 리버스 프록시나 터널로 외부에 열 때는
  접근 제어를 앞단에 두어야 한다.
- 리버스 프록시 설정 방식에 따라 기존 라우팅이 꺼질 수 있다. 경로 마운트를 쓸 것.
- 배포 전 반드시 `python tests/test_imports.py` 로 저장소 상태를 확인한다.
