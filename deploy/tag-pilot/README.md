# 사람 TAG 파일럿 P0 — 전사 화면

외국인 발화 40개(약 5분)를 전사자 2명(A · B)이 따로 들은 그대로 한글로 적는다. 작업 시간·재생 횟수는 자동 기록.
설계: `tasks/pronunciation-assessment/docs/05_사람TAG_라벨링_파일럿_설계.md` §8.

```bash
source tools/env.sh
$PY tasks/pronunciation-assessment/scripts/data/build_tag_pilot_p0.py            # 발화 고르기 → $MDD_OUTPUTS/tag_pilot/p0_v1/items.json (이미 만듦)
$PY deploy/tag-pilot/app.py --items $MDD_OUTPUTS/tag_pilot/p0_v1/items.json --port <빈 포트> --keys $MDD_OUTPUTS/tag_pilot/p0_v1/keys.json
#   → 전사자마다 자기 열쇠 링크(…/?t=<A 열쇠> · …/?t=<B 열쇠>) — 열쇠가 전사자를 정한다
$PY tasks/pronunciation-assessment/scripts/eval/tag_pilot_agreement.py --dir $MDD_OUTPUTS/tag_pilot/p0_v1   # 끝난 뒤
```

- 결과는 `labels_A.jsonl` · `labels_B.jsonl`(제출마다 한 줄, 같은 발화는 나중 것이 이김). 지우지 말 것.
- 전사자에게는 원문과 소리만 보인다(모국어 · 모델 결과 숨김). 발화마다 한 명은 빈칸, 한 명은 표준발음이 미리 채워진 칸에서 시작한다.
- AIHub 오디오를 내보낸다 — 사내 전사자만. 열쇠 없이는 127.0.0.1 에서만 뜬다. 판정 대상은 `adjudication.csv`(엑셀로 열림).
