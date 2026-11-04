# 운영 기록 DB · 국적별 발음오류 카탈로그

NPU 운영 서버의 요청 기록(`/data/speech-logs/captures/{pronunciation,nationality,dialect}`)을 SQLite로 모으고,
`docs/db문서자료` 예시와 같은 모양의 카탈로그(엑셀)와 히트맵(HTML)을 만든다.

```bash
bash tools/capture_db/update_all.sh        # 전부 한 번에 → docs/db문서자료/운영기록/
```

| 파일 | 어디서 | 하는 일 |
|---|---|---|
| `ingest_captures.py` | NPU (시스템 python3) | request.json → `/data/speech-logs/db/captures.sqlite`. 새로 생기거나 바뀐 요청만 넣는다 |
| `pron_nationality.py` | NPU (국적 서비스 venv) | 발음 녹음에 운영 국적 모델(CPU ONNX)을 돌려 `pron_nat`에 넣는다. 운영 기록에 안 남는다 |
| `build_catalog.py aggregate` | NPU | DB → 집계용 json (rid·해시·녹음 없음) |
| `build_catalog.py render` | A100 | json → 엑셀 카탈로그 + 히트맵 HTML (전체 / 앱 요청만) |
| `heatmap_template.html` | — | 예시 히트맵의 그리는 코드 (데이터 자리만 비움) |

DB 테이블: `requests` · `pron_utts` · `pron_words` · `pron_phonemes` · `pron_nat` · `nat_preds` · `dialect_preds` · `pron_versions`
(조회용 `v_scored` · `v_pron_flags`).

**라벨은 모두 모델 추정이다.** 오류유형·실현음소는 그 시점 운영 발음 모델의 판정(버전 P1~P6), 국적은 운영 국적 모델의 추정.
`nat_preds.actual_nationality`만 앱이 보낸 사용자 입력이다(대부분 기본값으로 보여 정답으로 쓰기 어렵다).
