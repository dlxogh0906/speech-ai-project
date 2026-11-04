# -*- coding: utf-8 -*-
"""국적별 '많이 틀리는 발음' — **우리 음소인식기(운영 M3·P6) 판정** 기준. dataset_error_catalog.py extract 결과를 집계한다.

- 틀리는 발음 = 정답(표준음) → 실현(우리 모델이 들은 소리 · 빠짐). 화면에 낸 지적만.
- 유형 = s0_taxonomy.classify_op — 예시 카탈로그가 자동 MDD(speech) 쪽에 쓴 규칙 그대로(KoG2P 기호 대치쌍 → 유형).
  음소 변화로 정할 수 있는 유형만 나온다(경계억양 · 강세 · 연음규칙은 사람 판정 전용, 구개음화는 다음 소리 문맥이 없어 안 나온다).
- 틀림률 = 그 나라 화자가 그 위치의 그 소리를 낼 때 모델이 이렇게 지적한 비율. 분모(소리 나온 횟수)는 dataset_gold_tags.py 의 라벨 표준음 집계.
- 참고 열: 같은 패턴을 사람 라벨(TAG)로 센 틀림률 — 모델이 사람보다 많이/적게 잡는지.
- 국적: AIHub 화자 정보(실제 값).

    $PY .../dataset_model_patterns.py [--min-spk 30 --topn 10]
"""
# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import MDD_OUTPUTS

import argparse, collections, glob, html, importlib.util, json, os

D = os.path.join(str(MDD_OUTPUTS), "analysis", "dataset_catalog_2026-11-02")
TYPES = ["초성", "종성", "모음오류", "경음화", "비음화", "격음화", "유음화", "음운탈락", "기타"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default=os.path.join(str(_r), "docs", "db문서자료", "AIHub_데이터셋"))
    ap.add_argument("--min-spk", dest="min_spk", type=int, default=30)
    ap.add_argument("--topn", type=int, default=10)
    a = ap.parse_args()
    from mdd.l1prior.s0_taxonomy import classify_op
    spec = importlib.util.spec_from_file_location("bc", str(_r / "tools" / "capture_db" / "build_catalog.py"))
    bc = importlib.util.module_from_spec(spec); spec.loader.exec_module(bc)
    ko = lambda c: bc.KO.get(c, c)
    # 분모: 라벨 표준음에서 (나라, 위치, 소리) 나온 횟수
    occ = collections.Counter()
    for f in glob.glob(os.path.join(D, "gold_tags", "part.*.json")):
        for c, pos, j, n in json.load(open(f, encoding="utf-8"))["occ"]:
            occ[(c, pos, j)] += n
    cache = {}

    def typ(alpha, actual):
        k = (alpha, actual)
        if k not in cache:
            op = {"kind": "del", "target": alpha, "actual": None} if not actual else {"kind": "sub", "target": alpha, "actual": actual}
            cache[k] = classify_op(op)[0] if alpha else "기타"
        return cache[k]

    pm, ph, utt, spk, tc = collections.Counter(), collections.Counter(), collections.Counter(), collections.defaultdict(set), collections.Counter()
    n = 0
    for f in sorted(glob.glob(os.path.join(D, "shard*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line); c = r.get("country") or "(국적 미상)"
            utt[c] += 1; spk[c].add(r["spk"]); n += 1
            for pos, alpha, aj, actual, acj, et in r["model"]:
                t = typ(alpha, actual)
                pm[(c, pos, aj, acj or "", t)] += 1; tc[(c, t)] += 1
            for pos, alpha, aj, actual, acj, et in r["human"]:
                ph[(c, pos, aj, acj or "")] += 1
    countries = sorted(utt, key=lambda c: (-len(spk[c]), c))
    big = [c for c in countries if len(spk[c]) >= a.min_spk]
    J = lambda x: x or "빠짐"
    by_c = collections.defaultdict(list)
    for (c, pos, lab, act, t), k in pm.items():
        by_c[c].append((k, pos, lab, act, t))
    top_rows, long_rows = [], []
    for c in countries:
        L = sorted(by_c[c], reverse=True); tot = sum(k for k, *_ in L)
        for rk, (k, pos, lab, act, t) in enumerate(L, 1):
            o = occ[(c, pos, lab)]
            row = [c, ko(c), len(spk[c]), rk, pos, lab, J(act), "%s→%s" % (lab, J(act)), t, k, round(100.0 * k / max(1, tot), 2),
                   round(100.0 * k / max(1, o), 2) if o else None, round(100.0 * ph[(c, pos, lab, act)] / max(1, o), 2) if o else None, o]
            long_rows.append(row)
            if rk <= a.topn:
                top_rows.append(row)
    rows_t = []
    for c in countries:
        tot = sum(tc[(c, t)] for t in TYPES)
        rows_t.append([c, ko(c), len(spk[c]), utt[c], tot] + [tc[(c, t)] for t in TYPES] + [round(100.0 * tc[(c, t)] / max(1, tot), 1) for t in TYPES])
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment
    wb = openpyxl.Workbook(); fill = PatternFill("solid", fgColor="DDE6F0")

    def sheet(name, header, rows, widths, title=None):
        ws = wb.create_sheet(name); r0 = 1
        if title:
            ws.cell(1, 1, title).font = Font(bold=True); r0 = 3
        for j, h in enumerate(header, 1):
            x = ws.cell(r0, j, h); x.font = Font(bold=True); x.fill = fill; x.alignment = Alignment(wrap_text=True, vertical="center")
        for i, row in enumerate(rows, r0 + 1):
            for j, v in enumerate(row, 1):
                ws.cell(i, j, v)
        for j, w in enumerate(widths, 1):
            ws.column_dimensions[openpyxl.utils.get_column_letter(j)].width = w
        ws.freeze_panes = ws.cell(r0 + 1, 1)

    ws = wb.active; ws.title = "00_읽는법"
    lines = ["AIHub 학습자 녹음 — 국적별 '많이 틀리는 발음' · 우리 음소인식기(운영 M3·P6) 판정", "",
             "대상: AIHub 040~043 발음 녹음(학습·검증·시험 화자) %d개 · 화자 %d명 · %d개국 · 국적은 화자 정보의 실제 값" % (n, sum(len(v) for v in spk.values()), len(countries)),
             "틀리는 발음: 정답(표준음) → 실현(우리 모델이 들은 소리, 빠짐 포함). 서비스 화면에 내는 지적만 셌다.",
             "유형: 예시 카탈로그의 자동 판정(speech)과 같은 규칙(s0_taxonomy.classify_op) — %s." % " · ".join(TYPES),
             "   경계억양·강세·연음규칙·ㄴ첨가·모음삽입은 사람 판정·삽입 정보가 필요해 모델 쪽에는 없다(사람 14유형 파일 참고).",
             "   규칙 이름 주의: ㅇ받침→ㄱ받침(비음화를 안 지킴)은 이 규칙에서 '종성'으로 나온다. 사람 태그는 같은 경우를 '비음화'라 부른다.",
             "열: '나라 지적 중 %' = 그 나라 모델 지적 합계 대비 · '틀림률(모델) %' = 그 소리가 나온 횟수 대비 모델이 이렇게 지적한 비율",
             "    '틀림률(사람) %' = 같은 패턴을 사람 라벨로 센 비율(참고) — 모델이 사람보다 크면 헛지적이 섞였거나 사람이 놓친 것, 작으면 모델이 놓친 것",
             "주의: 학습 화자 녹음(약 88%)은 모델이 학습한 녹음이라 모델 지적이 실제보다 적게 나온다. 화자가 적은 나라는 비율이 크게 흔들린다."]
    for i, l in enumerate(lines, 1):
        ws.cell(i, 1, l)
    ws["A1"].font = Font(bold=True, size=13); ws.column_dimensions["A"].width = 140
    sheet("01_국가별_유형분포", ["국가(EN)", "국가(KO)", "화자", "녹음", "지적 합계"] + ["%s 건수" % t for t in TYPES] + ["%s %%" % t for t in TYPES], rows_t,
          [16, 12, 6, 8, 9] + [8] * 18, title="★ 우리 모델 지적 · %는 그 나라 지적 합계 대비")
    hdr = ["국가(EN)", "국가(KO)", "화자", "순위", "위치", "정답", "실현", "틀리는 발음", "유형", "건수", "나라 지적 중 %", "틀림률(모델) %", "틀림률(사람) %", "그 소리 나온 횟수"]
    sheet("02_국가별_틀리는발음_상위%d" % a.topn, hdr, [r for r in top_rows if r[0] in big], [16, 12, 6, 5, 5, 5, 5, 10, 9, 8, 11, 11, 11, 11],
          title="★ 화자 %d명 이상 %d개국 · 나라마다 상위 %d개" % (a.min_spk, len(big), a.topn))
    sheet("03_전체국가_전체패턴", hdr, long_rows, [16, 12, 6, 5, 5, 5, 5, 10, 9, 8, 11, 11, 11, 11], title="★ 모든 나라 · 모든 패턴")
    os.makedirs(a.outdir, exist_ok=True)
    out_x = os.path.join(a.outdir, "AIHub_국적별_틀리는발음_우리모델.xlsx"); wb.save(out_x)
    cards = []
    for c in big:
        tot = sum(tc[(c, t)] for t in TYPES)
        t4 = " · ".join("%s %d%%" % (t, round(100.0 * v / max(1, tot))) for v, t in sorted(((tc[(c, t)], t) for t in TYPES), reverse=True)[:4] if v)
        trs = "".join("<tr><td>%d</td><td><b>%s</b> <small>%s</small></td><td>%s</td><td class=n>%s%%</td><td class=n>%s%%</td><td class=n>%s%%</td></tr>"
                      % (r[3], html.escape(r[7]), r[4], r[8], r[10], r[11], r[12]) for r in top_rows if r[0] == c)
        cards.append("<section><h3>%s <span>%s · 화자 %d명 · 녹음 %s</span></h3><p class=t>많은 유형: %s</p><table><tr><th>#</th>"
                     "<th>틀리는 발음 (정답→실현)</th><th>유형</th><th>나라 지적 중</th><th>틀림률 모델</th><th>틀림률 사람</th></tr>%s</table></section>"
                     % (ko(c), c, len(spk[c]), format(utt[c], ","), t4, trs))
    page = open(os.path.join(a.outdir, "AIHub_국적별_틀리는발음_14유형.html"), encoding="utf-8").read()
    head = page[:page.index("<main>")]
    head = head.replace("<title>국적별 많이 틀리는 발음</title>", "<title>국적별 많이 틀리는 발음 — 우리 모델</title>")
    body = ("<main><h1>국적별 많이 틀리는 발음 — 우리 음소인식기(운영 모델) 판정</h1><p class=sub>AIHub 040~043 학습자 녹음 %s개 · 화자 %s명 · 화자 %d명 이상 %d개국 · "
            "국적은 실제 값. 정답→실현 = 표준음 → 우리 모델이 들은 소리. 유형 = 예시 카탈로그 자동 판정 규칙. '틀림률 모델' = 그 소리를 낼 때 모델이 이렇게 지적한 비율 · "
            "'틀림률 사람' = 같은 패턴을 사람 라벨로 센 비율(참고). 학습 화자 녹음이 88%%라 모델 지적은 실제보다 적게 나온다.</p><div class=grid>%s</div></main></body></html>"
            % (format(n, ","), format(sum(len(v) for v in spk.values()), ","), a.min_spk, len(big), "".join(cards)))
    out_h = os.path.join(a.outdir, "AIHub_국적별_틀리는발음_우리모델.html")
    open(out_h, "w", encoding="utf-8").write(head + body)
    print("[엑셀] %s\n[그림] %s · 녹음 %d · 국가 %d(화자 %d명 이상 %d)" % (out_x, out_h, n, len(countries), a.min_spk, len(big)), flush=True)


if __name__ == "__main__":
    main()
