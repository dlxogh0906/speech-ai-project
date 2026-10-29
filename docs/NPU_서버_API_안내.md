# NPU 서버 추론 API 안내 (개발자용)

> **발음평가 모델 안내 갱신(2026-10-29):** 한국어 발음평가는 **M3 `w2vbert_synth_tense_v2/step_3000`**를 `pron-api:8932`에서 사용한다(마지막 원격 확인 10월 25일). [현재 서비스 모델·배포 이력](발음평가_현재_서비스_모델.md)을 참고한다. 기존 XLS-R `deploy/mdd-api`의 `/diagnose` 예제와 구분한다.

작성 2026-09-27 · 갱신 2026-10-05(국적 `/predict_long` · 서울 직통 입구) · 2026-10-28(사투리 분류 6절) · 대상: 앱·웹 개발자 · 서버 관리 정보는 따로 둔다(`docs/NPU_서버_운영.md`, 내부용)

NPU 서버(Rebellions RBLN-CA25 ×8) 한 대에서 음성 AI 세 가지(국적 분류 · 발음 평가 · 사투리 분류)를 HTTP API 로 제공한다.
요청은 모두 `multipart/form-data` 로 오디오 파일을 올리고, 응답은 JSON 이다.

| 서비스 | 주소 | 인증 | 동시 처리 |
|---|---|---|---|
| **국적 분류 · 서울 직통 (권장)** | 서버 담당자에게 따로 받는다. 환경변수로만 두고 문서·저장소에 적지 않는다 | **토큰 필요**: `X-API-Key: <토큰>` 또는 `Authorization: Bearer <토큰>` (`/health` 는 토큰 없이) | 8개 프로세스 · 3문장 `/predict_long` 기준 동시 8건까지 대기 없이 약 0.63초 · 최대 초당 약 12건 |
| 국적 분류 · 예비 (Funnel) | `https://<NPU-funnel-주소>` | 없음 | 같음. 직통보다 왕복이 약 0.3초 길다 |
| **발음 평가** | `https://<NPU-funnel-주소>:8443` | **토큰 필요** (서버 담당자에게 따로 받는다) | 한 번에 1건씩 · 초당 약 16건 |
| **사투리 분류 · 서울 직통 (2026-10-28)** | 서버 담당자에게 따로 받는다 | **토큰 필요**: `X-API-Key: <토큰>` 또는 `Authorization: Bearer <토큰>` (화면 `/` 와 `/health` 는 토큰 없이) | 한 번에 1건씩 · 3구간 `/predict_long` 약 0.7초 (6절) |

> 세 모델 모두 AI 허브 데이터로 학습했다. 결과를 쓰는 앱·웹은 **출처를 화면에 표기해야 한다**(4절).

---

## 1. 공통

**오디오 형식**

| 항목 | 내용 |
|---|---|
| 권장 | WAV · 16 kHz · 1채널 · 16bit (서버가 변환 없이 바로 읽어 가장 빠르다) |
| 그 밖 | 브라우저 녹음(webm/opus), 아이폰(m4a/mp4), mp3 등도 받는다. 서버가 변환한다 |
| 길이 | 국적 분류 `/predict_long` 은 **60초**까지 문장 수만큼 나눠 쓴다 · `/predict` 는 **앞 12초**만 쓴다 · 발음 평가는 **20초**까지 채점한다 · 사투리 분류는 파일마다 **60초**까지 12초 이하 구간으로 나눠 쓴다 |
| 크기 | 국적 분류 · 사투리 분류는 파일 하나 8 MB 이하. 16 kHz WAV 20초가 약 640 KB 다 |
| 말소리 | 세 서비스 모두 실제 말소리가 **1초 이상** 있어야 한다. 짧으면 400 오류 |

**백엔드에서 호출할 것.**
- 발음 평가는 CORS 헤더를 보내지 않는다. 국적 분류는 서버에 등록한 도메인에만 보낸다(지금은 로컬 개발 주소뿐). 등록하지 않은 도메인의 웹 페이지에서 `fetch` 로 부르면 브라우저가 응답을 막는다.
- 국적 분류 직통 입구 **토큰도 앱이나 웹 페이지 코드에 넣지 말 것.** 웹·앱 백엔드의 환경변수에만 둔다.
- 발음 평가 **토큰을 앱이나 웹 페이지 코드에 넣지 말 것.** 배포된 앱·페이지에서는 누구나 꺼낼 수 있다. 앱·웹의 백엔드 서버가 토큰을 들고 호출하고, 결과만 프론트로 넘긴다.
- 요청 제한 시간은 30초 정도로 둔다. 발음 평가는 한 번에 1건씩 처리해서 요청이 몰리면 대기 시간이 늘어난다.

**녹음 저장(2026-10-10 전체 적용)**: 웹·앱 구분 없이 NPU API에 전달된 원본 업로드와 전체 응답을 품질 분석용으로 저장한다. 국적 `/predict` · `/predict_long` · `/predict_multi`, 문장 발음 `/speak` · `/sent.eval.kr`, 단어 발음 `/word.judge`, 사투리 `/predict` · `/predict_long` · `/predict_multi`(10-28~) 모두 포함한다. 입력 오류·인증 실패·서버 오류도 기록한다. 파일을 보내지 않은 요청에는 음성 없이 요청 정보·응답만 남는다. 입구에서 거절돼 API에 도달하지 않은 요청은 접근 로그에만 남는다.

요청 번호는 응답 헤더 **`X-Capture-ID`**로 받는다. 국적 `/predict` · `/predict_long`의 JSON `rid`는 같은 번호다. 발음 평가의 응답 형식·점수는 유지한다. 문제를 제보할 때 시각과 이 번호를 함께 남기면 원본 음성과 서버 판정을 찾을 수 있다. `tokenId` · `userId` · 세션/기기 식별 정보를 이미 전달하고 있으면 함께 기록된다. 인증 토큰은 저장하지 않는다. 사용자 안내에 녹음 보관을 알린다.

---

## 2. 국적 분류 API

한국어 발화를 듣고 말한 사람의 국적(41종)을 추정한다. 한국인(원어민)은 `Korea` 로 나온다.

### `POST /predict_long` — 글 한 번에 읽기 (권장 · 2026-10-05)

- **요청**: 짧은 문장 1 · 3 · 5개로 된 글을 한 번에 읽은 녹음 1개(`audio`)와 문장 수(`parts`)를 보낸다.
- **처리**: 서버가 녹음을 문장 수만큼 나눠 판정하고 합친다. 녹음과 판정은 품질 분석용으로 저장한다(2026-10-09~).
- **정확도**(외국인 1위 정답, 서비스 조건): 1문장 73% · 3문장 83% · 5문장 85% 다. `/predict` 하나보다 크게 높다.
- **응답**: `top5` · `native` · `low_confidence` 는 `/predict` 와 같다. 여기에 `n`(실제 구간 수) · `parts_requested` · `voiced_sec` · `per_part` 가 더해진다.
  - `n < parts_requested` 면 글을 끝까지 읽지 않은 것이다. "끝까지 읽으면 더 정확하다"고 안내한다.
- **서울 직통 입구에서 토큰이 없거나 틀리면** 401 `{"error": "토큰이 없거나 맞지 않다", "code": "unauthorized"}` 이 온다.
- **자세한 내용**(녹음 길이 기준, 예제 코드, 문장 세트)은 `deploy/nationality-kd-demo/연동_가이드.md` 에 있다.

### `POST /predict` (예전 방식 — 파일 1개, 앞 12초)

| 필드 | 필수 | 내용 |
|---|---|---|
| `audio` | ✔ | 오디오 파일 (8 MB 이하) |
| `consent` |  | 예전 저장 규칙용 값. 2026-10-09 부터는 이 값과 관계없이 품질 분석용으로 저장한다 |
| `sentence_id` · `sentence` · `self_report` · `session` |  | 낭독 페이지용 기록 필드. 저장할 때 함께 남는다. API 로만 쓸 때는 안 보내도 된다 |

```bash
curl -F "audio=@rec.webm" -F "consent=0" https://<NPU-funnel-주소>/predict
```

```python
import requests
r = requests.post("https://<NPU-funnel-주소>/predict",
                  files={"audio": open("rec.wav", "rb")}, data={"consent": "0"}, timeout=30)
print(r.status_code, r.json())
```

```js
// Node 18+ (서버 쪽)
import { readFile } from "node:fs/promises";
const form = new FormData();
form.append("audio", new Blob([await readFile("rec.webm")]), "rec.webm");
form.append("consent", "0");
const res = await fetch("https://<NPU-funnel-주소>/predict", { method: "POST", body: form });
const body = await res.json();   // 200 이 아니면 body.code 로 원인을 본다
```

### 응답 (200)

실제 응답이다(사내 한국인, 노트북 크롬 녹음 3.9초).

```json
{
  "top5": [
    {"label": "Korea", "prob": 0.9985},
    {"label": "Philippines", "prob": 0.0006},
    {"label": "Uzbekistan", "prob": 0.0003},
    {"label": "France", "prob": 0.0002},
    {"label": "China", "prob": 0.0002}
  ],
  "sec": 3.94, "rms": 0.0461, "gain": 1.0, "silence_norm": true,
  "gate_korea": 0.9985, "native": true, "low_confidence": false,
  "encode_ms": 127, "total_ms": 257
}
```

| 필드 | 뜻 | 화면에서 |
|---|---|---|
| `top5` | 국적 상위 5개와 확률(0~1). 라벨 목록은 아래 | 그대로 보여 준다 |
| `native` | 1위가 `Korea` 인가 | `true` 면 "한국인" |
| `low_confidence` | 1위가 외국 국적인데 확률이 0.45 미만 | `true` 면 **다시 녹음**을 안내한다 |
| `gate_korea` | 확인 모델이 본 한국인 확률. 0.9 이상이면 한국인으로 판정한다 | 보통 안 쓴다 |
| `sec` | 처리한 음성 길이(초, 무음 정리 후) | 참고 |
| `rms` · `gain` · `silence_norm` | 음량·전처리 정보 | 디버깅용 |
| `encode_ms` · `total_ms` | 모델 계산 시간 · 서버 전체 처리 시간(ms). 네트워크 시간은 빠져 있다 | 참고 |

권장 흐름: `native` 면 한국인 → 아니고 `low_confidence` 면 재녹음 안내 → 그 밖에는 `top5` 를 보여 준다.

### 오류

| HTTP | `code` | 원인 · 대응 |
|---|---|---|
| 400 | `too_large` | 8 MB 초과 |
| 400 | `decode_failed` | 오디오가 아니거나 깨진 파일 |
| 400 | `too_short` | 0.5초 미만 |
| 400 | `no_speech` | 말소리가 1초 미만(무음·작은 소리). "조금 더 길게, 가까이서" 안내 |
| 500 | `server_decoder` | 서버 변환기 문제. 서버 담당자에게 알린다 |

오류 응답 모양: `{"error": "사람이 읽을 메시지", "code": "no_speech"}`

### 그 밖의 경로

| 경로 | 내용 |
|---|---|
| `GET /health` | 서버 상태 · 모델 이름 · 판정 기준값 · 누적 요청 수 |
| `GET /read` | 문장 읽기 페이지(인스타그램 공유용) |

### 국적 라벨 41종

Bangladesh · Brazil · Bulgaria · Cambodia · Chile · China · Colombia · Ethiopia · France · Germany · Hong Kong · India ·
Indonesia · Iran · Italy · Japan · Kazakhstan · **Korea** · Kyrgyzstan · Malaysia · Mongolia · Myanmar · Nepal · Pakistan ·
Peru · Philippines · Romania · Russia · Rwanda · Singapore · Spain · Tajikistan · Thailand · Turkey · Turkmenistan ·
Ukraine · United Kingdom · United States · Uzbekistan · Vietnam · Zambia

---

## 3. 발음 평가 API

문장과 녹음을 받아 음소 단위로 틀린 곳을 찾고 점수를 매긴다. 요청마다 **토큰**을 보낸다(3-8).
앱 연동에는 **`/sent.eval.kr`** 을 쓴다. SpeechSuper `sent.eval.kr` 과 요청·응답 모양이 같아서, SpeechSuper 를 쓰던 코드는 주소와 토큰만 바꾸면 된다.

### 3-1. `POST /sent.eval.kr` — 문장 하나 채점

| 필드 | 필수 | 내용 |
|---|---|---|
| `audio` | ✔ | 오디오 파일 |
| `refText` | ✔ | 읽어야 할 문장 |
| `tokenId` |  | 요청 식별자. 응답의 `recordId` 로 되돌아온다 |

SpeechSuper 방식 요청(`start.param.request.refText` · `tokenId` 를 담은 JSON 필드 + 오디오)도 그대로 받는다.

```bash
curl -H "Authorization: Bearer $TOKEN" \
     -F "audio=@rec.wav" -F "refText=독립문 앞에서 사진을 찍었어요" -F "tokenId=tok-2" \
     https://<NPU-funnel-주소>:8443/sent.eval.kr
```

```python
import requests
r = requests.post("https://<NPU-funnel-주소>:8443/sent.eval.kr",
                  headers={"Authorization": f"Bearer {TOKEN}"},
                  files={"audio": open("rec.wav", "rb")},
                  data={"refText": "독립문 앞에서 사진을 찍었어요", "tokenId": "tok-2"}, timeout=30)
result = r.json()["result"]
```

### 3-2. 응답 예 (200)

외국인 학습자 녹음의 실제 응답이다. 글자 13개 중 3개만 옮겼다.

```json
{
  "refText": "독립문 앞에서 사진을 찍었어요",
  "recordId": "tok-2",
  "result": {
    "overall": 84, "pronunciation": 89, "fluency": 70,
    "rhythm": null, "comprehensibility": null,
    "words": [
      {"word": "문", "readType": 2, "score": 96, "phonemes": [
        {"alpha": "mm", "alpha_jamo": "ㅁ", "position": "초성", "pronunciation": null, "actual": null, "actual_jamo": null, "errorType": "음운탈락"},
        {"alpha": "uu", "alpha_jamo": "ㅜ", "position": "중성", "pronunciation": 96, "actual": "uu", "actual_jamo": "ㅜ"},
        {"alpha": "nn", "alpha_jamo": "ㄴ", "position": "초성", "pronunciation": 96, "actual": "nn", "actual_jamo": "ㄴ"}]},
      {"word": "앞", "readType": 2, "score": 54, "phonemes": [
        {"alpha": "aa", "alpha_jamo": "ㅏ", "position": "중성", "pronunciation": 96, "actual": "aa", "actual_jamo": "ㅏ"},
        {"alpha": "ph", "alpha_jamo": "ㅍ", "position": "초성", "pronunciation": 12, "actual": "p0", "actual_jamo": "ㅂ", "errorType": "격음화"}]},
      {"word": "에", "readType": 0, "score": 99, "phonemes": [
        {"alpha": "ee", "alpha_jamo": "ㅔ", "position": "중성", "pronunciation": 99, "actual": "ee", "actual_jamo": "ㅔ"}]}
    ]
  },
  "_meta": {"raw_human_1to5": {"pron": 1.84, "comp": 1.89, "flu": 1.96}, "…": "그 밖의 설명 항목. 무시해도 된다"}
}
```

읽는 법: "문"의 ㅁ 은 소리를 빠뜨렸고(`actual_jamo: null`), "앞"의 ㅍ 은 ㅂ 으로 냈다(`격음화`). "에"는 정상이다.

### 3-3. 응답 필드

**`result`**

| 필드 | 값 | 뜻 |
|---|---|---|
| `overall` | 0~100 · `null` | 종합 점수. 0.6×발음 + 0.2×유창 + 0.2×이해를 쓰되, 값이 없는 칸은 빼고 비율을 맞춘다. 문장 하나일 때는 이해 점수가 없어 발음·유창만으로 계산한다 |
| `pronunciation` | 0~100 · `null` | 발음 점수. 글자 점수의 평균 |
| `fluency` | 0~100 · `null` | 유창성. 말 속도와 쉼만 보고 매긴다(발음과 독립) |
| `rhythm` · `comprehensibility` | `null` | 문장 하나로는 신뢰도가 낮아 보내지 않는다. 두 칸은 같은 값(이해 가능성)이다. 사용자 단위 점수는 3-5 의 `/speak` 로 |
| `words` | 배열 | 글자마다 한 칸 |

**`words[]`** — 글자 하나

| 필드 | 뜻 |
|---|---|
| `word` | 글자 |
| `readType` | `0` 정상 · `1` 누락(글자 소리를 전부 빠뜨림) · `2` 대치(소리 일부가 다르거나 빠짐) · `3` 오독(틀린 데다 글자 점수가 20 미만. 이때 `score` 는 `null`) · `4` 삽입(없는 소리를 넣음) |
| `score` | 글자 점수 0~100 = 음소 점수의 평균. 빠뜨려서 점수가 없는 음소는 평균에서 뺀다 |
| `phonemes` | 이 글자의 음소들 |
| `inserted` | 없는 소리를 넣은 경우에만 온다. 예: `[{"actual_jamo": "ㅡ"}]` |

**`phonemes[]`** — 음소 하나

| 필드 | 뜻 |
|---|---|
| `alpha` | 내야 할 소리. KoG2P 기호(예: `k0` 초성 ㄱ · `kf` 받침 ㄱ · `aa` ㅏ). 화면에는 `alpha_jamo` 를 쓰면 된다 |
| `alpha_jamo` | 내야 할 소리의 한글 자모 |
| `position` | `초성` · `중성` · `종성`. **실제 발음 기준**이라 연음이 되면 글자의 받침이 `초성`으로 나온다(예: "앞에서"의 ㅍ) |
| `pronunciation` | 음소 점수 0~100 = (1 − 틀렸을 확률) × 100 |
| `actual` · `actual_jamo` | 실제로 들린 소리. `null` 이면 그 소리를 빠뜨린 것이다 |
| `errorType` | **틀렸다고 판정한 음소에만** 붙는다. 값: `초성` · `종성` · `모음오류` · `경음화` · `격음화` · `비음화` · `유음화` · `음운탈락` · `음운첨가` |
| `uncertain` | `true` 면 들린 소리는 달랐지만 음소 점수가 60 이상이라 확실하지 않은 자리다 |

**`_meta`** 는 무시해도 된다. 별점(1~5)으로 보여 주려면 `_meta.raw_human_1to5` 의 사람 점수 예측값을 쓴다.

### 3-4. 화면에 그리는 법

- **점수가 `null` 이면 0 으로 바꾸지 말고 "—" 로 그린다.** `null` 은 0점이거나 판정하지 못한 자리다.
- **틀린 곳 표시는 `errorType` 이 있는 음소만 한다.** `uncertain: true` 인 자리는 지적하지 않는다. 필요하면 옅게만 표시한다.
- **점수 눈금:** 80점은 "학습자 상위 20%" 로 읽는다. 통과 기준을 두려면 `overall` 과 글자 점수 80 이상을 권한다.
- **빠뜨린 소리**는 `actual_jamo: null`, **넣은 소리**는 `inserted` 로 온다.
- 한글로 보여 줄 때는 `alpha_jamo`(내야 할 소리)와 `actual_jamo`(낸 소리)를 나란히 쓰면 된다.

**SpeechSuper 와 다른 점** (SpeechSuper 를 쓰던 코드라면 이것만 바꾸면 된다)

| | SpeechSuper | 이 API |
|---|---|---|
| 글자 점수 위치 | `words[].scores.pronunciation` | `words[].score` |
| 음소 표기 `alpha` | 한글. 낸 소리가 섞여 있다 | KoG2P 기호 = 내야 할 소리. 한글은 `alpha_jamo` |

### 3-5. `POST /speak` — 여러 문장을 모아 사용자 점수 (자체 형식)

같은 `id` 로 여러 문장을 보내면 사용자 단위 점수를 모아 준다. 발화마다 점수가 오지만, 세션 요약(`summary`) 점수는 **10문장부터** 나오고 30문장 이상을 권한다.

| 필드 | 필수 | 내용 |
|---|---|---|
| `audio` | ✔ | 오디오 파일 |
| `sentence` |  | 읽어야 할 문장. 비우면 받아쓰기만 한다 |
| `id` |  | 세션(사용자) 식별자. 기본 `default` |
| `country` |  | 국적 키(`GET /countries` 의 `key`, 예: `Vietnam`). 국적별 교정 문구가 붙는다 |

```bash
H="Authorization: Bearer $TOKEN"
curl -H "$H" -F "audio=@rec.wav" -F "sentence=학교에서 친구를 만났어요" -F "id=user-123" https://<NPU-funnel-주소>:8443/speak
curl -H "$H" "https://<NPU-funnel-주소>:8443/session?id=user-123"          # 누적 요약
curl -H "$H" -X POST -F "id=user-123" https://<NPU-funnel-주소>:8443/reset   # 세션 비우기
```

응답 주요 필드: `canon_jamo`(표준 발음 자모) · `pred_jamo`(실제로 들린 자모) ·
`sites`(자리별 판정 `일치`·`다름`·`삭제`) · `scores`(발음 `pron` · 이해 `comp` · 유창 `flu`, **1~5점**) ·
`feedback`(국적별 교정) · `vad`(무음 정리 정보) · `ms`(서버 처리 시간) · `summary`(세션 누적 요약).

### 3-6. 그 밖의 경로

| 경로 | 내용 |
|---|---|
| `GET /health` | 서버 상태 |
| `GET /sentences` | 연습 문장 목록 |
| `GET /countries` | 국적 키 목록 |
| `GET /session?id=` · `POST /reset` | 세션 요약 · 세션 비우기. 토큰 필요 |
| `GET /` · `GET /speechsuper.html` | 브라우저에서 직접 녹음해 보는 확인 화면. 처음 열 때 토큰을 묻고 그 브라우저에만 저장한다. 다시 입력하려면 주소 끝에 `?reset_token` |

### 3-7. 오류

| HTTP | `code` | 원인 |
|---|---|---|
| 401 | `unauthorized` | 토큰이 없거나 틀림 |
| 400 | `no_audio` · `no_reftext` | 오디오나 문장이 비어 있음 (`/sent.eval.kr`) |
| 400 | `too_short` | 0.3초 미만 |
| 400 | `low_speech` | 말소리가 모자람. **다시 녹음 안내**. 기준(2026-10-18~): 말한 시간 ≥ 기준 문장 음절 수 × 0.08초(최소 0.2초 · 최대 1.0초) 그리고 소리 비율 ≥ 30%. 예: 네 0.2초 · 고마워요 0.32초 · 감사합니다 0.4초 · 13음절 이상 1.0초(예전과 같음). 본문에 `speech_s`(말한 시간) · `min_speech_s`(필요한 시간) · `voiced_ratio` · `min_voiced_ratio` 가 같이 온다(예전 `audio` 칸도 그대로) |
| 400 | (메시지만) | 오디오를 해석하지 못함 · 소리가 거의 없음. `/speak` 는 0.3초 미만도 메시지만 온다 |

### 3-8. 인증 (필수)

아래 넷 중 하나로 토큰을 보낸다. **헤더 방식을 권한다** — 주소(`?token=`)에 넣으면 접속 기록에 남는다. `/health` · `/sentences` · `/countries` 는 토큰 없이 된다.

```
Authorization: Bearer <토큰>      X-API-Key: <토큰>      폼 필드 token=<토큰>      쿼리 ?token=<토큰>
```

### 3-9. 알아 둘 것

- **한 번에 1건씩 처리한다.** 동시에 여러 건이 오면 줄을 서서 처리되고 실패하지는 않는다. 동시 16건이면 응답이 약 1초 걸린다.
- **세션은 서버 메모리에만 있다.** 서버를 다시 띄우면 `/speak` 누적 기록이 사라진다.
- 학습에 없던 문장에서는 정확도가 떨어진다. 새 문장을 서비스에 넣기 전에 따로 확인한다.

---

## 4. 출처 표기 (필수)

세 모델 모두 AI 허브 데이터로 학습했다. AI 허브 이용정책상 이 데이터로 만든 모델과 서비스는 쓸 수 있지만,
**한국지능정보사회진흥원의 사업 결과물임을 밝혀야 한다.** 결과를 쓰는 앱·웹의 안내 화면(앱 정보 · 서비스 소개 등)에 아래 문구를 넣는다.

> 이 서비스는 한국지능정보사회진흥원의 「지능정보산업 인프라 조성」 사업으로 구축된 AI 허브(aihub.or.kr) 데이터를 활용해 개발했습니다.

| 서비스 | 학습에 쓴 데이터 |
|---|---|
| 국적 분류 | AI 허브: 교육용 영어·유럽어·중·일어·아시아어(중·일어 제외) 모국어 사용자의 한국어 음성 데이터(4종) · 인공지능 학습을 위한 외국인 한국어 발화 음성 데이터 · 자유대화 음성(일반남여) · 한국어 음성 / 그 밖: Zeroth-Korean(OpenSLR, CC BY 4.0) |
| 발음 평가 | AI 허브: 교육용 영어·유럽어·중·일어·아시아어(중·일어 제외) 모국어 사용자의 한국어 음성 데이터(4종) · 한국어 음성 |
| 사투리 분류 | AI 허브 방언 · 표준어 화자 음성 데이터 3종(데이터셋 번호 71517 · 71558 · 537). 이름과 이용 조건은 각 데이터셋 페이지에서 확인한다 |

- 막혀 있는 것은 **원본 데이터(오디오·라벨)를 외부에 제공하는 것**이다. 세 API 는 모델 결과만 돌려준다.
- 근거: AI 허브 [데이터 이용정책](https://aihub.or.kr/intrcn/guid/usagepolicy.do) · [FAQ](https://aihub.or.kr/aihubnews/faq/list.do) (2026-09-27 확인).
  데이터셋마다 조건이 다를 수 있으니, 상업 서비스에 넣기 전에 각 데이터셋 페이지의 이용 조건을 한 번 확인한다.

---

## 5. 속도 참고

서버 안에서 잰 처리 시간이다. 사용자 쪽에서는 업로드 시간이 더해진다.

| 서비스 | 요청 1건 | 최대 처리량 |
|---|---|---|
| 국적 분류 `/predict_long` (3문장) | 약 0.63초. 서울 직통 왕복 p50 0.68초 · p90 0.72초, Funnel 은 p50 0.96초(10-05) | 초당 약 12건 |
| 국적 분류 `/predict` | 약 0.25초 | 초당 약 35건 |
| 발음 평가 | 약 0.06초 (녹음 2~4초 기준) | 초당 약 16건 (1건씩 처리) |
| 사투리 분류 `/predict_long` (3구간) | 약 0.71초 (17.6초 녹음, 10-28). GPU 서버에서 서울 직통 왕복 8.1초 녹음 0.55초 | 1건씩 처리 — 초당 약 1.4건 (한 건 시간에서 계산) |

---

## 6. 사투리 분류 API (2026-10-28)

한국어 발화를 듣고 강원 · 경상 · 전라 · 제주 · 충청 · 표준어 중 어디에 가까운지 판정한다.
요청·응답 모양은 국적 분류와 같다(multipart 로 오디오 → JSON). 토큰은 `X-API-Key: <토큰>` 또는 `Authorization: Bearer <토큰>`.

| 경로 | 내용 |
|---|---|
| `POST /predict` | `audio` 1개(60초까지 읽는다). 12초 이하 구간으로 나눠 각각 판정하고 평균한다. `parts`(1~5)를 주면 그 수만큼 나눈다 |
| `POST /predict_long` | `/predict` 와 같고 `parts` 기본값이 3 이다 (국적과 같은 이름) |
| `POST /predict_multi` | `audio` 를 여러 개(1~5) 보낸다. 문장마다 따로 녹음한 경우다 |
| `GET /health` | 상태 · 모델 · 규칙 (토큰 없이) |
| `GET /` | 시험용 화면 (토큰을 처음 한 번 입력) |

선택 필드 `self_report`(사용자가 밝힌 출신 지역) · `session` · `sentence_id` 는 판정에 쓰지 않고 저장 기록에만 남는다.
`self_report` 를 같이 보내 주면 실제 사용자 녹음으로 정확도를 잴 수 있다.

```bash
curl -H "X-API-Key: $DIALECT_API_KEY" -F "audio=@rec.m4a" -F "self_report=경상" "$DIALECT_URL/predict_long"
```

응답(200)의 주요 필드:

| 필드 | 뜻 | 화면에서 |
|---|---|---|
| `label` · `label_ko` · `prob` | 1위 지역(`Gangwon` · `Gyeongsang` · `Jeolla` · `Jeju` · `Chungcheong` · `Standard`)과 확률(0~1, 구간 평균) | 그대로 보여 준다 |
| `probs` | 6개 지역 확률, 높은 순 | 막대로 보여 줄 때 |
| `standard` | 1위가 표준어인가 | |
| `low_confidence` | 1위 확률이 0.6 미만 — 검증셋에서 이런 판정의 정답률은 0.66 이었다(나머지 0.986) | `true` 면 "더 길게 다시 말해 보세요" |
| `n` · `per_part` | 실제로 판정한 구간 수 · 구간별 1위 | 디버깅용 |
| `rid` | 요청 번호(응답 헤더 `X-Capture-ID` 와 같다) | 문제 제보 때 함께 |

오류는 국적과 같은 모양이다: 400 `too_large` · `too_many_files` · `decode_failed` · `too_short` · `no_speech`, 401 `unauthorized`, 500 `server_decoder`.

**알아 둘 것**

- 한 번에 1건씩 처리한다. 동시에 오면 줄을 서고 실패하지는 않는다.
- 표준어 판정은 `prob` 가 1.0 으로 나온다(앙상블 규칙: 한 모델이 표준어라고 하면 표준어로 정한다).
- 코퍼스 test 정확도는 높지만(봉인 test 0.960), 서버에 쌓인 실제 사용자 녹음 점검(10-26)에서는 한국인 34건 중 17건을 지역 방언으로(출신 지역 정보가 없어 정답 여부는 모름 · 같은 자리에서 녹음한 것끼리 답이 비슷했다), 외국인 4건은 모두 표준어로 판정했다. 결과를 사용자에게 단정적으로 보여 주기 전에 `self_report` 로 실제 정확도를 먼저 쌓는 것을 권한다.
