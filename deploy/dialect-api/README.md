# 사투리 분류 API · 데모 (2026-10-26)

한국어 발화를 듣고 강원 · 경상 · 전라 · 제주 · 충청 · 표준어 중 어디에 가까운지 판정한다.
요청·응답 모양은 NPU 국적 분류 API(`docs/NPU_서버_API_안내.md` 2절)와 같다. multipart 로 오디오를 올리면 JSON 이 온다.

| 항목 | 내용 |
|---|---|
| 주소 | `https://<GPU-funnel-주소>/dialect` (GPU 서버 · Tailscale Funnel) |
| 데모 화면 | 같은 주소를 브라우저로 연다. 키는 처음 한 번 넣거나, 주소 끝에 `?key=<키>` 를 붙인 링크로 받는다 |
| 인증 | `POST` 요청마다 `X-API-Key: <키>` 또는 `Authorization: Bearer <키>`. 화면과 `/health` 는 키 없이 열린다 |
| 키 | 서버 담당자에게 따로 받는다. 앱·웹 백엔드의 환경변수에만 둔다. 앱이나 페이지 코드에는 넣지 말 것 |
| 처리 | GPU 한 장에서 한 번에 1건씩. 12초 구간 하나에 약 0.1~0.3초 걸린다(옆 학습과 GPU 공유 중 실측) |
| 왕복 시간 | 새 연결이면 약 1.2~1.4초다(Funnel 의 TLS 연결만 약 0.55초). 연결을 재사용하면 그보다 짧다 — 아래 "주의" |

> AI 허브 방언 발화 데이터로 학습했다. 결과를 쓰는 앱·웹은 출처를 화면에 표기해야 한다(국적 API 와 같음).

## 경로

| 경로 | 내용 |
|---|---|
| `POST /predict` | `audio` 1개(60초까지 읽는다). 12초 이하 구간으로 나눠 각각 판정하고 평균한다. `parts`(1~5)를 주면 그 수만큼 나눈다 |
| `POST /predict_long` | `/predict` 와 같고 `parts` 기본값이 3 이다. 국적 API 와 같은 이름 |
| `POST /predict_multi` | `audio` 를 여러 개(1~5) 보낸다. 문장마다 따로 녹음한 경우다. 파일마다 판정하고 평균한다 |
| `GET /health` | 상태 · 모델 · 규칙 · 누적 요청 수 |
| `GET /` | 데모 화면 |

구간 나누기: 한 구간이 3초보다 짧아지면 구간 수를 줄이고, 12초를 넘으면 늘린다(학습 구간이 3~12초였다).
말소리가 0.3초도 안 되는 구간(긴 쉼)은 뺀다.

선택 필드 `self_report`(출신 지역 자기 신고) · `session` · `sentence_id` 는 판정에 쓰지 않는다. 저장 기록에만 남는다.
`self_report` 를 받아 두면 나중에 실제 녹음으로 정확도를 잴 수 있다. 데모 화면은 선택 항목으로 묻는다.

```bash
curl -H "X-API-Key: $DIALECT_API_KEY" -F "audio=@rec.m4a" https://<GPU-funnel-주소>/dialect/predict
curl -H "X-API-Key: $DIALECT_API_KEY" -F "audio=@s1.webm" -F "audio=@s2.webm" -F "audio=@s3.webm" \
     https://<GPU-funnel-주소>/dialect/predict_multi
```

```python
import requests
r = requests.post("https://<GPU-funnel-주소>/dialect/predict",
                  headers={"X-API-Key": KEY}, files={"audio": open("rec.wav", "rb")}, timeout=30)
print(r.status_code, r.json())
```

## 응답 (200)

코퍼스 봉인 test 의 전라 화자 녹음 3개를 `/predict_multi` 로 보낸 실제 응답이다(서버 내부 호출이라 `total_ms` 에 네트워크 시간이 없다).

```json
{
  "label": "Jeolla", "label_ko": "전라", "prob": 0.827,
  "probs": [{"label": "Jeolla", "label_ko": "전라", "prob": 0.827}, {"label": "Chungcheong", "label_ko": "충청", "prob": 0.1633},
            {"label": "Gyeongsang", "label_ko": "경상", "prob": 0.0094}, "… 강원 · 제주 · 표준어"],
  "standard": false, "low_confidence": false,
  "n": 3, "parts_requested": 3, "sec": 20.07, "voiced_sec": 20.0,
  "per_part": [{"label": "Chungcheong", "label_ko": "충청", "prob": 0.4895, "file": 0, "start": 0.0, "end": 5.84},
               {"label": "Jeolla", "label_ko": "전라", "prob": 1.0, "file": 1, "start": 0.0, "end": 6.45},
               {"label": "Jeolla", "label_ko": "전라", "prob": 0.9997, "file": 2, "start": 0.0, "end": 7.78}],
  "models": {"eat_large": {"label": "Jeolla", "label_ko": "전라", "prob": 0.6735, "standard_prob": 0.0},
             "xlsr2b": {"label": "Jeolla", "label_ko": "전라", "prob": 0.9805}},
  "rule": "eat_standard_gate_regional_eat_0.5", "encode_ms": 213, "total_ms": 433,
  "rid": "20261026-205157-874988c93814"
}
```

| 필드 | 뜻 | 화면에서 |
|---|---|---|
| `label` · `label_ko` · `prob` | 1위 지역과 확률(0~1, 구간 평균) | 그대로 보여 준다 |
| `probs` | 6개 지역 확률, 높은 순 | 막대로 보여 줄 때 |
| `standard` | 1위가 표준어인가 | |
| `low_confidence` | 1위 확률이 0.6 미만이다. 검증셋에서 이런 판정은 9.4% 였고 정답률이 0.66 이었다(나머지는 0.986) | `true` 면 "더 길게 다시 말해 보세요" |
| `n` · `parts_requested` | 실제로 판정한 구간 수 · 요청한 수 | `n` 이 적으면 말소리가 짧았던 것 |
| `per_part` | 구간별 1위와 확률. 여러 파일이면 `file`(0부터) | 디버깅용 |
| `models` | 앙상블 두 모델 각각의 1위. `standard_prob` 는 EAT 가 본 표준어 확률 | 디버깅용 |
| `rid` | 요청 번호. 응답 헤더 `X-Capture-ID` 와 같다 | 문제 제보 때 함께 |

판정 규칙: EAT 가 표준어라고 하면 표준어(확률 1.0)로 정하고, 아니면 두 모델의 지역 5개 확률을 반씩 섞는다.
검증셋에서 미리 고른 규칙이다. 그래서 표준어 판정은 `prob` 가 1.0 으로 나온다. 확신 정도는 `models.eat_large.standard_prob` 로 본다.

## 오류

모양: `{"error": "사람이 읽을 메시지", "code": "…", "rid": "…"}`

| HTTP | `code` | 원인 · 대응 |
|---|---|---|
| 401 | `unauthorized` | 키가 없거나 틀렸다 |
| 400 | `too_large` | 파일 하나가 8 MB 초과 |
| 400 | `too_many_files` | `audio` 가 0개이거나 6개 이상 |
| 400 | `decode_failed` | 오디오가 아니거나 깨진 파일 |
| 400 | `too_short` | 1초 미만 |
| 400 | `no_speech` | 말소리가 1초 미만(무음 · 작은 소리). "조금 더 길게, 가까이서" 안내 |
| 500 | `server_decoder` | 서버 변환기 문제. 서버 담당자에게 알린다 |

## 권장 흐름

1. **국적 API 를 먼저 부른다.** `native` 가 `true`(한국인)일 때만 사투리 결과를 보여 준다.
   이 모델은 한국인이 말한다고 가정한다. 외국인 발화도 대부분 "표준어"로 나온다(서버에 저장된 실제 녹음으로 확인).
2. `low_confidence` 면 재녹음을 안내한다.
3. 짧은 녹음 하나보다 5~10초 녹음 2~3개(`/predict_multi`)가 낫다. 코퍼스 기준으로 사람당 1구간 0.959 → 3구간 0.976 이다.

## 성능과 한계 — 화면 문구를 정할 때 볼 것

- 코퍼스 봉인 test: 정답률 **0.960** [0.948, 0.970] · macro-F1 0.961. 같은 코퍼스(AI 허브 방언 발화) 안의 값이다.
- **폰·노트북으로 녹음한 실제 목소리에서는 정확도를 재지 못했다.** 출신 지역을 아는 실제 녹음이 없었다.
  서버에 저장된 실제 녹음 384개를 돌려 보니 같은 사람의 녹음이라도 녹음 환경에 따라 판정이 갈렸다.
  → `docs/최종_점검_국적_사투리_2026-10-26.md` §2.4
- 표준어 학습 데이터는 녹음 방식이 다른 출처에서 왔다. 표준어 판정은 잠정적이다(`comparison_report.json` limitations).
- 결론: 재미·참고용 기능으로는 쓸 수 있지만, "정확한 출신 지역 판별"로 내세울 근거는 아직 없다.

## 저장

모든 요청의 업로드 원본과 전체 응답을 품질 분석용으로 저장한다(`shared/request_capture.py`, NPU 와 같은 형식).
입력 오류 · 인증 실패도 남는다. 키는 저장하지 않는다. 사용자 안내에 녹음 보관을 알린다(데모 화면은 알린다).
저장 녹음은 개인정보다. 서버 밖으로 옮기거나 분석하려면 담당자 승인이 필요하다.

## 주의

- **Funnel 경유라 새 연결마다 약 1초가 더 든다.** NPU 서버의 서울 직통 입구(Caddy 443)처럼 빠른 길은 아직 없다.
  빠르게 하려면 NPU Caddy 가 이 서버로 넘겨 주거나(NPU 설정 변경 필요), VM 에 공인 IP 를 받아야 한다.
- 백엔드에서 연결을 재사용(keep-alive)하면 왕복이 줄어든다.

## 운영 (내부)

| 항목 | 내용 |
|---|---|
| 프로세스 | tmux `dialect-api` → `deploy/dialect-api/start.sh` → `app.py` (127.0.0.1:8987). systemd 가 아니라서 **재부팅하면 다시 띄워야 한다** |
| 설정 | `~/.config/dialect-api/api.env` (키 · 모델 폴더 · 추가 PYTHONPATH · 포트 · 로그 · 캡처 폴더). 권한 600 |
| 로그 | `$DIALECT_LOG` — 요청마다 경로 · 판정 · 시간 한 줄 |
| 캡처 | `$SPEECH_CAPTURE_DIR/<날짜>/<rid>/` (`request.json` + `audio-NN.*`) + 날짜별 `captures.jsonl` |
| GPU | 약 11 GB (EAT Large + XLS-R 2B, FP32 가중치 · FP16 autocast) |
| 공개 경로 | `tailscale funnel --bg --set-path /dialect http://127.0.0.1:8987` · 내릴 때 `tailscale funnel --bg --https=443 --set-path /dialect off` (다른 경로는 그대로 둔다. `--https=443 off` 만 쓰면 443 전체가 내려간다) |
| 다시 띄우기 | `tmux kill-session -t dialect-api; tmux new-session -d -s dialect-api bash deploy/dialect-api/start.sh` (모델 적재 약 1분) |

서빙 경로가 평가와 같은 답을 내는지는 `tasks/dialect-classification/scripts/eval/check_serving_parity.py` 로 확인했다
(2026-10-26 · 봉인 test 200구간: 1위 일치 200/200, 확률 차이 최대 0.008).
