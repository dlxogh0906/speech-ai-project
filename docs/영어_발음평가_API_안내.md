# 영어 발음 평가 API 안내 (개발자용)

작성 2026-10-14 · 대상: 앱·웹 개발자

한국인 학습자의 **영어** 문장 발음을 채점한다. 요청·응답 형식은 한국어 발음 평가 API(`/sent.eval.kr`)와 같다.
이미 한국어 API를 붙였다면 주소와 `refText`(영어 문장)만 바꾸면 된다.

| 항목 | 내용 |
|---|---|
| 주소 | `POST https://<GPU-funnel-주소>/pron-en/sent.eval` |
| 인증 | **토큰 필요**: `Authorization: Bearer <토큰>` 또는 `X-API-Key: <토큰>`. 토큰은 서버 담당자에게 따로 받는다 |
| 상태 확인 | `GET https://<GPU-funnel-주소>/pron-en/health` (토큰 없이) |
| 시험 화면 | `https://<GPU-funnel-주소>/pron-en/` (브라우저에서 녹음해 보는 데모 · 처음에 토큰을 묻는다) |
| 응답 시간 | 연결을 재사용하면 **약 0.4초**, 새 연결이면 약 1.4초(4초 문장 · 16 kHz WAV 기준, 서울에서 잰 값). 서버 채점 자체는 0.06초이고 나머지는 중계 경로 시간이다 |

> 영어 점수기는 AI 허브 데이터(「교육용 한국인의 영어 음성 데이터」)로 학습했다. 결과를 쓰는 앱·웹은 **출처를 화면에 표기해야 한다**.

---

## 1. 꼭 지킬 것

- **백엔드에서 호출한다.** CORS 헤더를 보내지 않는다. 웹 페이지의 `fetch`로 직접 부르면 브라우저가 응답을 막는다.
- **토큰을 앱이나 웹 페이지 코드에 넣지 않는다.** 배포된 앱·페이지에서는 누구나 꺼낼 수 있다. 백엔드 환경변수에만 둔다.
- 요청 제한 시간은 30초 정도로 둔다.
- **연결을 재사용한다**(HTTP keep-alive · 같은 클라이언트 객체 재사용). 새 연결은 1초 넘게 걸린다.
- 오디오는 **16 kHz WAV로 보낸다.** 44.1 kHz 원본을 그대로 보내면 올리는 데 시간이 두세 배 든다.

## 2. 요청

`multipart/form-data`로 보낸다. 둘 중 아무거나 쓴다.

**① 간편 방식**

```bash
curl -H "Authorization: Bearer $PRON_EN_TOKEN" \
     -F audio=@rec.wav -F 'refText=I think the weather is nice today.' -F tokenId=tok-1 \
     https://<GPU-funnel-주소>/pron-en/sent.eval
```

**② SpeechSuper 방식** — 한국어 API와 같다. JSON 덩어리 필드 이름은 무엇이든 된다.

```bash
curl -H "Authorization: Bearer $PRON_EN_TOKEN" \
     -F 'text={"start":{"param":{"app":{"userId":"u1"},"request":{"coreType":"sent.eval.promax","refText":"I think the weather is nice today.","tokenId":"tok-1"}}}}' \
     -F audio=@rec.wav  https://<GPU-funnel-주소>/pron-en/sent.eval
```

| 항목 | 내용 |
|---|---|
| `refText` | 읽은 영어 문장. 단어 60개까지. 숫자("3", "50%", "$5"), 축약형("don't"), 악센트("café")도 받는다. 문장부호와 영어가 아닌 글자는 무시한다 |
| `tokenId` | 아무 문자열. 응답의 `recordId`로 돌려준다 |
| `userId` | 선택 |
| 오디오 | **WAV 16 kHz · 1채널 · 16bit 권장**(변환 없이 바로 읽는다). webm/opus, m4a, mp3도 받는다(변환에 0.1초 안팎 더 든다) |
| 길이 | **30초까지 채점**한다. 넘는 부분은 버린다 |

## 3. 응답

```jsonc
{
  "refText": "We call it bear",
  "recordId": "tok-1",
  "result": {
    "overall": 86,            // 종합 = 0.6·발음 + 0.2·유창성 + 0.2·운율
    "pronunciation": 85,      // 발음
    "fluency": 82,            // 유창성(속도·쉼)
    "rhythm": 86,             // 운율
    "comprehensibility": null,// 영어에는 없음(항상 null)
    "integrity": 100,         // 읽은 단어 비율(%)
    "speed": 188,             // 분당 단어 수
    "pause_count": 0,         // 단어 사이 0.3초 이상 쉼 횟수
    "words": [
      { "word": "bear", "readType": 2, "score": 71,
        "span": {"start": 156, "end": 184},          // 보낸 녹음 기준 시각, 10 ms 단위
        "phonemes": [
          { "alpha": "b",  "alpha_ipa": "b", "position": "자음", "pronunciation": 99, "actual": "b", "actual_ipa": "b" },
          { "alpha": "eh", "alpha_ipa": "ɛ", "position": "모음", "pronunciation": 94,
            "actual": "ae", "actual_ipa": "æ",
            "errorType": "모음오류", "errorRule": "eh_ae", "errorHint": "ɛ → æ", "actual_src": "pair" },
          { "alpha": "r",  "alpha_ipa": "ɹ", "position": "자음", "pronunciation": 30,
            "actual": null, "actual_ipa": null, "errorType": "음운탈락" }
        ]
      }
    ]
  },
  "_meta": { }                // 기준 설명. 무시해도 된다
}
```

**점수 읽는 법**
- 모든 점수는 0~100이다. **0점은 `null`로 온다**(한국어 API와 같다).
- 문장 하나의 점수는 흔들린다. 실력은 **여러 문장(5문장 이상)의 평균**으로 보여 주는 것을 권한다.
- 단어·음소 점수는 소리별 점수라 문장 발음 점수와 눈금이 다르다(대체로 더 높다).

**틀린 곳 표시** — `errorType`이 있는 음소만 틀린 곳이다.
- `errorType`: `모음오류` · `자음오류` · `음운탈락`(소리를 빠뜨림)
- 한국인이 자주 틀리는 소리(f→p, v→b, r↔l, θ→s, z→s, ɪ↔iː, ɛ↔æ 등)는 `errorRule`·`errorHint`(예: `"f → p"`)가 함께 온다.
  화면에서는 "f를 p로 발음했어요"처럼 보여 주면 된다.
- `actual_src: "unclear"`는 점수는 낮지만 어떤 소리로 냈는지 근거가 없는 자리다. "부정확"으로만 표시하고 들린 소리는 보여 주지 않는다.
- `uncertain: true`는 지적하지 않은 자리다. 필요하면 "~처럼 들렸을 수 있어요" 정도로 옅게 표시한다.
- th(ð, the·this의 첫소리)는 판정하지 않는다.

**`readType`**: 0 정상 · 1 누락(단어를 통째로 빠뜨림) · 2 대치 · 3 오독 · 4 삽입(지금은 내보내지 않음)

**음소 기호**: `alpha`·`actual`은 ARPAbet 소문자(`th`, `iy`), 사람이 읽는 표기는 `alpha_ipa`·`actual_ipa`(IPA: `θ`, `iː`)다.

## 4. 오류 응답

본문은 `{"error": "설명", "code": "코드"}`다.

| HTTP | code | 뜻 · 할 일 |
|---|---|---|
| 401 | `unauthorized` | 토큰이 없거나 틀림 |
| 400 | `no_audio` | 오디오 파일이 없음 |
| 400 | `no_reftext` | `refText`가 비었거나 읽을 영어 단어가 없음 |
| 400 | `too_short` | 녹음이 너무 짧음(문장에 비해 짧은 경우 포함). 끝까지 읽도록 안내 |
| 400 | `low_speech` | 말소리가 거의 없음. 마이크에 가까이 대고 다시 녹음하도록 안내 |
| 400 | `decode_failed` | 오디오를 해석하지 못함 |
| 400 | `reftext_too_long` | 단어 60개 초과. 문장을 나눠 보낸다 |
| 503 | `server_busy` | 서버 자원 부족. 잠시 뒤 다시 |

## 5. 알아 둘 것

- 점수는 한국인 학습자 녹음과 사람 평가 점수로 맞췄다. 원어민 발음은 대체로 95점 이상이 나온다.
- 조용한 곳에서 녹음해야 정확하다. 잡음·울림이 크면 점수가 낮게 나온다.
- 녹음은 서버에 저장하지 않는다.
