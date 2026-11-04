# -*- coding: utf-8 -*-
"""운영 기록 DB(captures.sqlite) → 국적별 발음오류 카탈로그(엑셀) + 음소 히트맵(HTML).

모양은 docs/db문서자료 의 예시(AIHub 기준 `국적별_한국어_발음오류_유형카탈로그.xlsx` · `음소오류_국적별_히트맵.html`)를
그대로 따른다. 히트맵은 예시의 그리는 코드(heatmap_template.html)를 쓰고 데이터만 바꾼다.

두 단계로 나눈다(녹음 · 개인 식별 값은 NPU 밖으로 나가지 않는다):
  aggregate  NPU 에서 · 표준 라이브러리만 · DB → catalog_data.json (지적 자리 · 녹음 단위 집계용 행, rid · 해시 없음)
  render     A100 에서 · openpyxl · json → 엑셀 + HTML

    python3 build_catalog.py aggregate --db /data/speech-logs/db/captures.sqlite --out catalog_data.json
    $PY tools/capture_db/build_catalog.py render --data catalog_data.json --outdir docs/db문서자료/운영기록

라벨 성격: 지적(오류유형)과 국적은 모두 **모델 추정**이다. 국적은 pron_nationality.py 가 운영 국적 모델로 붙인 값.
"""
import argparse
import collections
import datetime
import json
import os
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SOUND = {
    'k0': 'ㄱ', 'kk': 'ㄲ', 'nn': 'ㄴ', 't0': 'ㄷ', 'tt': 'ㄸ', 'rr': 'ㄹ', 'mm': 'ㅁ', 'p0': 'ㅂ', 'pp': 'ㅃ', 's0': 'ㅅ',
    'ss': 'ㅆ', 'oh': 'ㅇ(무음)', 'c0': 'ㅈ', 'cc': 'ㅉ', 'ch': 'ㅊ', 'kh': 'ㅋ', 'th': 'ㅌ', 'ph': 'ㅍ', 'h0': 'ㅎ',
    'aa': 'ㅏ', 'qq': 'ㅐ', 'ya': 'ㅑ', 'yq': 'ㅒ', 'vv': 'ㅓ', 'ee': 'ㅔ', 'yv': 'ㅕ', 'ye': 'ㅖ', 'oo': 'ㅗ', 'wa': 'ㅘ',
    'wq': 'ㅙ', 'wo': 'ㅚ', 'yo': 'ㅛ', 'uu': 'ㅜ', 'wv': 'ㅝ', 'we': 'ㅞ', 'wi': 'ㅟ', 'yu': 'ㅠ', 'xx': 'ㅡ', 'xi': 'ㅢ', 'ii': 'ㅣ',
    'kf': 'ㄱ받침', 'nf': 'ㄴ받침', 'tf': 'ㄷ받침', 'll': 'ㄹ받침', 'mf': 'ㅁ받침', 'pf': 'ㅂ받침', 'ng': 'ㅇ받침',
}
KO = {
    'Korea': '한국', 'Indonesia': '인도네시아', 'Vietnam': '베트남', 'China': '중국', 'Philippines': '필리핀', 'Japan': '일본',
    'United States': '미국', 'Thailand': '태국', 'Mongolia': '몽골', 'Russia': '러시아', 'Russian Federation': '러시아',
    'Myanmar': '미얀마', 'Taiwan': '대만', 'Spain': '스페인', 'France': '프랑스', 'Germany': '독일', 'Italy': '이탈리아',
    'Romania': '루마니아', 'Bulgaria': '불가리아', 'Singapore': '싱가포르', 'Brazil': '브라질', 'United Kingdom': '영국',
    'Uzbekistan': '우즈베키스탄', 'Kazakhstan': '카자흐스탄', 'India': '인도', 'Nepal': '네팔', 'Malaysia': '말레이시아',
    'Cambodia': '캄보디아', 'Turkey': '튀르키예', 'Egypt': '이집트', 'Mexico': '멕시코', 'Peru': '페루', 'Canada': '캐나다',
    'Australia': '호주', 'Poland': '폴란드', 'Ukraine': '우크라이나', 'Sri Lanka': '스리랑카', 'Pakistan': '파키스탄',
    'Bangladesh': '방글라데시', 'Laos': '라오스', 'Kyrgyzstan': '키르기스스탄', '(판정 안 됨)': '판정 안 됨',
}
TYPES = ["모음오류", "경음화", "격음화", "유음화", "비음화", "초성", "종성", "음운탈락"]
POSITIONS = ["초성", "중성", "종성"]


def source_of(ua):
    ua = (ua or "").lower()
    if ua.startswith("python-httpx"):
        return "앱 서버(python-httpx)"
    if ua.startswith("npu-ops-test"):
        return "운영 점검(npu-ops-test)"
    if ua.startswith("python-urllib") or ua.startswith("python-requests") or ua.startswith("curl"):
        return "스크립트 시험(urllib·requests·curl)"
    return "기타(브라우저 등)"


# ── 1단계: NPU 에서 집계용 행 만들기 ──────────────────────────────────────────
def aggregate(db, out):
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    has_nat = con.execute("SELECT count(*) FROM sqlite_master WHERE name='pron_nat'").fetchone()[0]
    nat_sql = "n.top1_label AS country, n.status AS nat_status" if has_nat else "NULL AS country, NULL AS nat_status"
    join = "LEFT JOIN pron_nat n USING (rid)" if has_nat else ""
    utts, key = [], {}
    for i, r in enumerate(con.execute("""SELECT r.rid, r.date, r.version, r.user_agent, u.ref_text, u.n_phonemes, u.n_flags, %s
                                          FROM requests r JOIN pron_utts u USING (rid) %s WHERE r.status = 200""" % (nat_sql, join))):
        key[r["rid"]] = i
        utts.append({"date": r["date"], "version": r["version"], "source": source_of(r["user_agent"]), "ref_text": r["ref_text"],
                     "n_phonemes": r["n_phonemes"], "n_flags": r["n_flags"], "country": r["country"] or "(판정 안 됨)",
                     "nat_status": r["nat_status"]})
    flags = []
    for p in con.execute("""SELECT rid, position, alpha, alpha_jamo, actual, actual_jamo, error_type
                            FROM pron_phonemes WHERE error_type IS NOT NULL"""):
        if p["rid"] in key:
            flags.append([key[p["rid"]], p["position"], p["alpha"], p["alpha_jamo"], p["actual"], p["actual_jamo"], p["error_type"]])
    versions = [list(v) for v in con.execute("SELECT version, start_time, description FROM pron_versions ORDER BY start_time")]
    meta = {"generated_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "db": db,
            "n_requests": con.execute("SELECT count(*) FROM requests WHERE service='pronunciation'").fetchone()[0],
            "date_min": min((u["date"] for u in utts), default=None), "date_max": max((u["date"] for u in utts), default=None),
            "nationality": "pron_nationality.py — 운영 국적 모델(nat-silnorm-off-20261031)을 CPU ONNX 로 · 모델 추정" if has_nat else "없음"}
    json.dump({"meta": meta, "versions": versions, "utts": utts, "flags": flags,
               "flag_cols": ["utt", "position", "alpha", "alpha_jamo", "actual", "actual_jamo", "error_type"]},
              open(out, "w", encoding="utf-8"), ensure_ascii=False)
    print("[집계] 녹음 %d · 지적 %d → %s" % (len(utts), len(flags), out), flush=True)


# ── 2단계: A100 에서 엑셀 + HTML ─────────────────────────────────────────────
def ko_c(c):
    return KO.get(c, c)


def sym_ko(s):
    return SOUND.get(s, s or "∅")


def render(data_path, outdir, template, example_xlsx, only_source=None):
    D = json.load(open(data_path, encoding="utf-8"))
    utts = D["utts"]
    flags = [dict(zip(D["flag_cols"], f)) for f in D["flags"]]
    if only_source:
        keep = {i for i, u in enumerate(utts) if u["source"] == only_source}
        flags = [f for f in flags if f["utt"] in keep]
        utts_used = [u for i, u in enumerate(utts) if i in keep]
    else:
        utts_used = utts
    for f in flags:
        f["country"] = utts[f["utt"]]["country"]; f["version"] = utts[f["utt"]]["version"]; f["source"] = utts[f["utt"]]["source"]
        f["sub"] = f["actual"] is not None and f["error_type"] != "음운탈락"
    os.makedirs(outdir, exist_ok=True)
    n_utt = collections.Counter(u["country"] for u in utts_used)
    n_ph = collections.Counter()
    for u in utts_used:
        n_ph[u["country"]] += u["n_phonemes"] or 0
    by_c = collections.defaultdict(list)
    for f in flags:
        by_c[f["country"]].append(f)
    countries = sorted(n_utt, key=lambda c: (c == "(판정 안 됨)", -len(by_c[c]), -n_utt[c], c))
    meta = D["meta"]
    kind = meta.get("kind", "ops")
    span = "%s ~ %s" % (meta["date_min"], meta["date_max"])
    tag = "" if not only_source else " · " + only_source
    n_spk = collections.defaultdict(set)
    for u in utts_used:
        if u.get("spk"):
            n_spk[u["country"]].add(u["spk"])

    # ── HTML 히트맵 (예시의 그리는 코드 그대로) ──
    groups = [("전체", "전체", "All", flags, sum(n_utt.values()))] + [(c, ko_c(c), c, by_c[c], n_utt[c]) for c in countries]
    axes = {}
    for pos in POSITIONS:
        subs = [f for f in flags if f["sub"] and f["position"] == pos]
        ct, ca = collections.Counter(f["alpha"] for f in subs), collections.Counter(f["actual"] for f in subs)
        tg = sorted(ct, key=lambda s: (-ct[s], s)); ac = sorted(ca, key=lambda s: (-ca[s], s))
        axes[pos] = {"targets": [{"sym": s, "ko": sym_ko(s)} for s in tg], "actuals": [{"sym": s, "ko": sym_ko(s)} for s in ac]}
    matrices, clist, overview = {}, [], []
    for key, name, en, fl, nu in groups:
        subs = [f for f in fl if f["sub"]]
        tot = len(subs)
        mats = {}
        for pos in POSITIONS:
            cnt = collections.Counter((f["alpha"], f["actual"], f["error_type"]) for f in subs if f["position"] == pos)
            ti = {d["sym"]: i for i, d in enumerate(axes[pos]["targets"])}
            ai = {d["sym"]: i for i, d in enumerate(axes[pos]["actuals"])}
            mats[pos] = [[ti[a], ai[b], n, round(100.0 * n / max(1, tot), 2), TYPES.index(t) if t in TYPES else len(TYPES) - 1]
                         for (a, b, t), n in cnt.most_common()]
        matrices[key] = mats
        clist.append({"key": key, "name": "%s · 녹음 %d · 지적 %d" % (name, nu, len(fl)), "en": en, "total": tot})
        tc = collections.Counter(f["error_type"] for f in fl)
        overview.append([round(100.0 * tc[t] / max(1, len(fl)), 1) for t in TYPES])
    data = {"positions": POSITIONS, "countries": clist, "axes": axes, "matrices": matrices, "types": TYPES,
            "overview": overview, "ntotal": sum(1 for f in flags if f["sub"])}
    # ── 전체 국가 한 표: 나라마다 많이 틀리는 유형 · 패턴 ──
    def pat(f):
        return "%s→%s" % (sym_ko(f["alpha"]), sym_ko(f["actual"]) if f["actual"] else "빠짐")
    allc = []
    for c in countries:
        fl = by_c[c]
        if not fl:
            continue
        tc = collections.Counter(f["error_type"] for f in fl)
        pc = collections.Counter((pat(f), f["error_type"]) for f in fl)
        allc.append({"c": c, "ko": ko_c(c), "utt": n_utt[c], "spk": len(n_spk[c]) or None, "n": len(fl),
                     "per100": round(100.0 * len(fl) / max(1, n_ph[c]), 2),
                     "types": [(t, round(100.0 * n / len(fl))) for t, n in tc.most_common(3)],
                     "pats": [(pp, t, n, round(100.0 * n / len(fl))) for (pp, t), n in pc.most_common(5)]})
    allc.sort(key=lambda r: -r["utt"])
    trs = []
    for r in allc:
        who = "%s<br><small>%s</small>" % (r["ko"], r["c"])
        size = ("화자 %d · " % r["spk"] if r["spk"] else "") + "녹음 %d" % r["utt"]
        types = " ".join('<span class="pt">%s <small>%d%%</small></span>' % tp for tp in r["types"])
        pats = " ".join('<span class="pt">%s <small>%s · %d건 %d%%</small></span>' % pp for pp in r["pats"])
        trs.append('<tr><td class="c">%s</td><td class="n">%s</td><td class="n">%d</td><td class="n">%.1f</td><td>%s</td><td>%s</td></tr>'
                   % (who, size, r["n"], r["per100"], types, pats))
    allc_html = ('<h2>국가별 자주 틀리는 유형 — 전체 국가 (%d개국)</h2><p class="sub">나라마다 지적이 많은 유형 3개와 가장 많은 틀림 패턴 5개(정답→실현). '
                 '%%는 그 나라 지적 합계 대비. 녹음·화자가 적은 나라는 비율이 크게 흔들린다.</p>'
                 '<div class="card"><div class="allwrap"><table class="allc"><thead><tr><th>국가</th><th>규모</th><th>지적</th>'
                 '<th>100자모당</th><th>많은 유형</th><th>많은 틀림 패턴 (정답→실현)</th></tr></thead><tbody>%s</tbody></table></div></div>'
                 % (len(allc), "".join(trs)))
    t = open(template, encoding="utf-8").read()
    if kind == "dataset":
        short = meta.get("short", meta.get("label", ""))
        rep = {"__TITLE__": "AIHub 학습자 녹음 — 국적별 발음오류 히트맵 (%s)" % short,
               "__H1__": "국적별 한국어 발음오류 — 음소 confusion 히트맵 (%s)" % short,
               "__SUB__": ("%s · 화자 %d명 · 녹음 %d개 · 지적 %d개(대치 %d · 빠짐 %d) · %d개국 — 국적은 화자 정보의 <b>실제 값</b>."
                           " <b>정답 → 실현</b>: %s." % (meta.get("dataset", ""), sum(len(v) for v in n_spk.values()), len(utts_used), len(flags),
                                                       sum(1 for f in flags if f["sub"]), sum(1 for f in flags if not f["sub"]), len(countries), meta["label"])),
               "__OVSUB__": "각 나라 지적 중 유형별 비중(%) — 음운탈락(빠짐) 포함. 행(국가명)을 클릭하면 위쪽 히트맵이 그 나라로 바뀝니다.",
               "__FOOT__": "생성: tools/capture_db/build_catalog.py · %s · 음소기호=KoG2P · 히트맵은 대치만(빠짐은 개요·전체 국가 표·엑셀에)." % meta["generated_at"]}
    else:
        rep = {"__TITLE__": "운영 기록 — 국적별 한국어 발음오류 음소 히트맵" + tag,
               "__H1__": "운영 기록 — 국적(추정)별 한국어 발음오류 음소 confusion 히트맵" + tag,
               "__SUB__": ("NPU 발음 API(8932) 운영 기록 · %s · 채점 녹음 %d개 · 화면에 낸 지적 %d개(대치 %d · 빠짐 %d) · 국적 %d개(추정)."
                           " <b>정답음소 → 실현음소</b>는 그 시점 운영 모델의 판정이고, 국적은 운영 국적 모델의 추정이다(사람 정답 아님)."
                           % (span, len(utts_used), len(flags), data["ntotal"], len(flags) - data["ntotal"], len(countries))),
               "__OVSUB__": "각 나라 지적 중 유형별 비중(%) — 음운탈락(빠짐) 포함. 행(국가명)을 클릭하면 위쪽 히트맵이 그 나라로 바뀝니다. 녹음이 적은 나라는 몇 건으로 비율이 크게 흔들린다.",
               "__FOOT__": ("생성: tools/capture_db/build_catalog.py · %s · 음소기호=KoG2P · 히트맵은 대치만(빠짐은 개요·엑셀에)."
                            " 운영 버전이 기간 중 바뀌었다(엑셀 04_버전별_유형)." % meta["generated_at"])}
    for k, v in rep.items():
        t = t.replace(k, v)
    t = t.replace("__ALLC__", allc_html)
    t = t.replace("__DATA__", json.dumps(data, ensure_ascii=False))
    sfx = ("" if not only_source else "_" + only_source.split("(")[0].replace(" ", ""))
    base = "운영기록" if kind == "ops" else "AIHub_" + meta.get("file_tag", "")
    if kind == "ops" and only_source:
        sfx = "_앱요청만"
    html_path = os.path.join(outdir, "%s_음소오류_국적별_히트맵%s.html" % (base, sfx))
    open(html_path, "w", encoding="utf-8").write(t)

    # ── 엑셀 카탈로그 (예시 시트 구성을 따른다) ──
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment
    ex = openpyxl.load_workbook(example_xlsx, read_only=True, data_only=True) if example_xlsx and os.path.exists(example_xlsx) else None
    wb = openpyxl.Workbook()
    head_fill = PatternFill("solid", fgColor="DDE6F0")

    def sheet(name, header, rows, widths=None, title=None):
        ws = wb.create_sheet(name)
        r0 = 1
        if title:
            ws.cell(1, 1, title).font = Font(bold=True)
            r0 = 3
        for j, h in enumerate(header, 1):
            c = ws.cell(r0, j, h); c.font = Font(bold=True); c.fill = head_fill; c.alignment = Alignment(wrap_text=True, vertical="center")
        for i, row in enumerate(rows, r0 + 1):
            for j, v in enumerate(row, 1):
                ws.cell(i, j, v)
        for j, w in enumerate(widths or [], 1):
            ws.column_dimensions[openpyxl.utils.get_column_letter(j)].width = w
        ws.freeze_panes = ws.cell(r0 + 1, 1)
        return ws

    total = len(flags)
    tcount = collections.Counter(f["error_type"] for f in flags)
    # 00 읽는법
    ws = wb.active; ws.title = "00_읽는법"
    if kind == "dataset":
        lines = ["AIHub 학습자 녹음 — 국적별 한국어 발음오류 유형 카탈로그 (%s)" % meta.get("short", ""), "",
                 "출처: " + meta.get("dataset", ""), "판정: " + meta["label"],
                 "규모: 화자 %d명 · 녹음 %d개 · 지적 %d개 · %d개국 · 생성 %s" % (sum(len(v) for v in n_spk.values()), len(utts_used), total,
                                                                   len(countries), meta["generated_at"]), "",
                 "■ 국적: AIHub 화자 정보(SpeakerMetadata.nationality)의 실제 값이다.",
                 ("■ 학습에 쓰지 않은 화자만 넣었다. 문장(RS)·낱말(RW)은 학습 문항과 같다(모델이 문장을 외운 상태 — 아래 주의)."
                  if meta.get("scope", "시험화자") == "시험화자" else
                  "■ 학습·검증·시험 화자를 모두 넣었다(05_출처별에서 나눠 볼 수 있다). 학습 화자의 녹음은 모델이 학습한 녹음이라, 우리 음소인식기 판정은 그 부분에서 오류를 크게 덜 잡는다."),
                 "■ 같은 녹음을 '우리 음소인식기' 판정과 '사람 TAG' 라벨로 따로 정리한 파일이 짝으로 있다. 둘을 비교하면 모델이 놓치는 오류가 보인다.",
                 "■ 03b_국가별_상위오류: 전체 국가를 한 표로 — 나라마다 많이 틀리는 유형 3개 · 패턴 5개.", "",
                 "■ 시트",
                 "   01_오류유형_사전 · 02_유형별_빈도 · 03_국적별_유형 · 03b_국가별_상위오류 · 04_버전별_유형 · 05_출처별(문장/낱말)",
                 "   06_대치쌍상세(정답음소→실현음소) · 07_음운탈락상세 · 08_음소별_세부 · 09_문장별 · 10_심볼범례 · 11_주의사항"]
    else:
      lines = ["운영 기록 — 국적(추정)별 한국어 발음오류 유형 카탈로그" + tag, "",
             "출처: NPU 발음 API(8932) 운영 기록(/data/speech-logs/captures/pronunciation) → captures.sqlite → 이 파일",
             "기간: %s · 채점된 녹음 %d개 · 화면에 낸 지적 %d개 · 생성 %s" % (span, len(utts_used), total, meta["generated_at"]), "",
             "■ 예시(AIHub 기준 카탈로그)와 같은 모양으로, 실제 서비스에 들어온 녹음을 정리했다.",
             "■ 라벨 성격 — 모두 모델 추정이다(사람 정답 아님)",
             "   · 오류유형·실현음소: 그 시점 운영 발음 모델과 화면 규칙의 판정(04_버전별_유형 — 기간 중 규칙이 바뀌었다)",
             "   · 국적: " + meta["nationality"],
             "   · 처음 보는 문장에서 운영 모델 지적은 10개 중 약 9개가 헛지적이다(내부 측정, 2026-10-22). 수치를 '학습자 오류 분포'로 읽지 말 것.",
             "■ 섞여 있는 것 — 05_출처별: 앱 서버 요청 외에 내부 점검·스크립트 시험이 섞여 있다. 앱 요청만 본 파일은 이름 끝 '_앱요청만'.",
             "■ 양이 적다 — 국가별 녹음이 수 개~수십 개라 비율이 크게 흔들린다. 경향을 보는 용도로만 쓸 것.", "",
             "■ 시트",
             "   01_오류유형_사전 · 02_유형별_빈도 · 03_국적별_유형 · 04_버전별_유형 · 05_출처별",
             "   06_대치쌍상세(정답음소→실현음소) · 07_음운탈락상세 · 08_음소별_세부 · 09_문장별 · 10_심볼범례 · 11_주의사항"]
    for i, l in enumerate(lines, 1):
        ws.cell(i, 1, l)
    ws["A1"].font = Font(bold=True, size=13); ws.column_dimensions["A"].width = 120
    # 01 오류유형 사전 — 정의는 예시 카탈로그의 문구를 쓴다
    defs = {}
    if ex is not None and "01_오류유형_사전" in ex.sheetnames:
        for row in list(ex["01_오류유형_사전"].iter_rows(values_only=True))[1:]:
            if row and row[0]:
                defs[row[0]] = row[1:5]
    top_pair = {}
    for t in TYPES:
        c = collections.Counter("%s→%s" % (f["alpha_jamo"], f["actual_jamo"] or "∅") for f in flags if f["error_type"] == t)
        top_pair[t] = " · ".join("%s(%d)" % kv for kv in c.most_common(3))
    rows = []
    for t in TYPES:
        d = defs.get(t, ("", "", "", ""))
        if t == "음운탈락" and not d[1]:
            d = ("분절(삭제)", "있어야 할 소리가 실현되지 않음(빠짐).", "운영: 그 자리에 대응하는 소리가 없고 강제정렬 점수 50 미만(10-21~) · 글자 통째 빠짐은 글자 단위 판정(10-22~)", "")
        rows.append([t, d[0], d[1], d[2], tcount[t], round(100.0 * tcount[t] / max(1, total), 2), top_pair[t]])
    sheet("01_오류유형_사전", ["유형", "층위", "정의(예시 카탈로그)", "판정규칙(예시 카탈로그)", "운영 건수", "운영 점유율 %", "운영 최다 (정답→실현)"],
          rows, [10, 14, 50, 50, 10, 12, 40])
    # 02 유형별 빈도
    acc, rows = 0.0, []
    for t, n in tcount.most_common():
        sh = 100.0 * n / max(1, total); acc += sh
        rows.append([t, n, round(sh, 2), round(acc, 2)])
    sheet("02_유형별_빈도", ["유형", "건수", "점유율 %", "누적 점유율 %"], rows, [12, 10, 10, 12], title="■ 운영 기록 — 화면에 낸 지적 %d개" % total)
    # 03 국적별 유형
    rows = []
    for c in countries:
        fl = by_c[c]; tc = collections.Counter(f["error_type"] for f in fl)
        rows.append([c, ko_c(c), n_utt[c], n_ph[c], len(fl), round(100.0 * len(fl) / max(1, n_ph[c]), 2)]
                    + [tc[t] for t in TYPES] + [round(100.0 * tc[t] / max(1, len(fl)), 1) for t in TYPES])
    sheet("03_국적별_유형", ["국가(EN)", "국가(KO)", "녹음 수", "자모 수", "지적 합계", "100자모당 지적"] + ["%s 건수" % t for t in TYPES] + ["%s %%" % t for t in TYPES],
          rows, [16, 12, 8, 8, 9, 10] + [9] * 16, title="★ 국적 = 운영 국적 모델의 추정 · %는 그 나라 지적 합계 대비")
    # 03b 전체 국가 — 나라마다 많이 틀리는 유형 · 패턴
    rows = []
    for r in allc:
        rows.append([r["c"], r["ko"], r["spk"], r["utt"], r["n"], r["per100"]]
                    + ["%s %d%%" % tp for tp in r["types"]] + [""] * (3 - len(r["types"]))
                    + ["%s (%s) %d건 %d%%" % pp for pp in r["pats"]] + [""] * (5 - len(r["pats"])))
    sheet("03b_국가별_상위오류", ["국가(EN)", "국가(KO)", "화자", "녹음", "지적", "100자모당 지적", "유형 1위", "유형 2위", "유형 3위",
                                  "패턴 1위", "패턴 2위", "패턴 3위", "패턴 4위", "패턴 5위"], rows,
          [16, 12, 6, 7, 7, 10, 12, 12, 12, 22, 22, 22, 22, 22], title="★ 전체 국가 한 표 — 패턴 = 정답→실현(유형) · %는 그 나라 지적 합계 대비")
    # 04 버전별
    vdesc = {v[0]: v[2] for v in D["versions"]}
    vu = collections.defaultdict(list)
    for u in utts_used:
        vu[u["version"]].append(u)
    rows = []
    for v in sorted(vu):
        fl = [f for f in flags if f["version"] == v]; tc = collections.Counter(f["error_type"] for f in fl)
        nph = sum(u["n_phonemes"] or 0 for u in vu[v])
        rows.append([v, vdesc.get(v, ""), len(vu[v]), len(fl), round(100.0 * len(fl) / max(1, nph), 2)] + [tc[t] for t in TYPES])
    sheet("04_버전별_유형", ["운영 버전", "설명", "녹음 수", "지적", "100자모당 지적"] + TYPES, rows, [8, 60, 8, 8, 12] + [9] * 8,
          title="★ 같은 녹음이라도 버전마다 지적이 다르다 — 버전이 섞인 합계를 비교할 때 주의")
    # 05 출처별
    su = collections.defaultdict(list)
    for u in utts_used:
        su[u["source"]].append(u)
    rows = []
    for s, us in sorted(su.items(), key=lambda kv: -len(kv[1])):
        rows.append([s, len(us), sum(1 for u in us if u["country"] == "Korea"),
                     sum(1 for u in us if u["country"] not in ("Korea", "(판정 안 됨)")), sum(u["n_flags"] or 0 for u in us)])
    sheet("05_출처별", ["요청 출처(user-agent)", "녹음 수", "한국 추정", "외국 추정", "지적"], rows, [36, 8, 10, 10, 8])
    # 06 대치쌍 상세
    rows = []
    for key, name, en, fl, nu in groups:
        subs = [f for f in fl if f["sub"]]
        cnt = collections.Counter((f["position"], f["alpha"], f["actual"], f["error_type"]) for f in subs)
        for rk, ((pos, a, b, t), n) in enumerate(cnt.most_common(), 1):
            rows.append([en, name, rk, pos, a, sym_ko(a), b, sym_ko(b), t, n, round(100.0 * n / max(1, len(subs)), 2)])
    sheet("06_대치쌍상세", ["국가(EN)", "국가(KO)", "국가내 순위", "위치", "정답음소(기호)", "정답음소(한글)", "실현음소(기호)", "실현음소(한글)", "오류유형", "건수", "국가내 점유율 %"],
          rows, [14, 12, 8, 6, 8, 10, 8, 10, 10, 6, 10])
    # 07 음운탈락 상세
    rows = []
    for key, name, en, fl, nu in groups:
        dl = [f for f in fl if f["error_type"] == "음운탈락"]
        for (pos, a), n in collections.Counter((f["position"], f["alpha"]) for f in dl).most_common():
            rows.append([en, name, pos, a, sym_ko(a), n, round(100.0 * n / max(1, len(dl)), 2)])
    sheet("07_음운탈락상세", ["국가(EN)", "국가(KO)", "위치", "음소(기호)", "음소(한글)", "건수", "국가 빠짐 내 %"], rows, [14, 12, 6, 8, 10, 6, 12])
    # 08 음소별 세부
    rows = []
    for pos in POSITIONS:
        fp = [f for f in flags if f["position"] == pos]
        by_a = collections.defaultdict(list)
        for f in fp:
            by_a[f["alpha"]].append(f)
        for a, fa in sorted(by_a.items(), key=lambda kv: -len(kv[1])):
            n = len(fa)
            topc = collections.Counter(ko_c(f["country"]) for f in fa).most_common(3)
            topr = collections.Counter(sym_ko(f["actual"]) if f["actual"] else "빠짐" for f in fa).most_common(3)
            rows.append([pos, a, sym_ko(a), n, round(100.0 * n / max(1, len(fp)), 1), round(100.0 * n / max(1, total), 2),
                         " · ".join("%s %d" % kv for kv in topc), " / ".join("%s %d" % kv for kv in topr)])
    sheet("08_음소별_세부", ["위치", "기호", "한글", "지적 건수", "위치내 %", "전체 %", "상위 국적(추정)", "상위 실현형(무엇으로 냈나)"],
          rows, [6, 6, 8, 9, 9, 8, 36, 40])
    # 09 문장별
    sent = collections.defaultdict(list)
    for i in (sorted(keep) if only_source else range(len(utts))):
        sent[utts[i]["ref_text"]].append(i)
    fl_by_utt = collections.defaultdict(list)
    for f in flags:
        fl_by_utt[f["utt"]].append(f)
    rows = []
    for s, ids in sorted(sent.items(), key=lambda kv: -len(kv[1])):
        fl = [f for i in ids for f in fl_by_utt.get(i, ())]
        nph = sum(utts[i]["n_phonemes"] or 0 for i in ids)
        top = collections.Counter("%s→%s" % (f["alpha_jamo"], f["actual_jamo"] or "∅") for f in fl).most_common(3)
        rows.append([s, len(ids), len(fl), round(100.0 * len(fl) / max(1, nph), 2), " · ".join("%s(%d)" % kv for kv in top)])
    sheet("09_문장별", ["읽은 문장(refText)", "녹음 수", "지적", "100자모당 지적", "최다 지적(정답→실현)"], rows, [60, 8, 8, 12, 40])
    # 10 심볼범례 — 예시 카탈로그 것을 그대로
    rows = []
    if ex is not None and "07_심볼범례" in ex.sheetnames:
        rows = [list(r) for r in list(ex["07_심볼범례"].iter_rows(values_only=True))[1:] if r and r[0]]
    else:
        rows = [[s, k, ""] for s, k in SOUND.items()]
    sheet("10_심볼범례", ["KoG2P 기호", "한글", "위치"], rows, [12, 10, 8])
    # 11 주의사항
    ws = wb.create_sheet("11_주의사항")
    if kind == "dataset":
        notes = ["이 파일을 그대로 믿으면 안 되는 지점", "",
                 "① 우리 음소인식기 판정은 학습 문장(225문항)에서 잰 값이다. 모델이 이 문장들을 외워서, 틀린 발음도 표준발음으로 받아 적는 일이 많다",
                 "   (학습 문장에서 진짜 오류를 잡는 비율 약 32%). 사람 TAG 파일과 같이 볼 것 — 둘의 차이가 모델이 놓치는 오류다.",
                 "② 오류유형은 운영 판정(mdd.pipeline.ctype) 이름이다. 예시 카탈로그(s0_taxonomy.classify_op)와 이름은 같지만 규칙이 조금 다르다.",
                 "③ 낱말(RW) 녹음은 짧아서 운영 API 는 상당수를 '말소리 부족'으로 거절한다. 여기서는 거절하지 않고 모두 판정했다.",
                 "④ 나라별 화자 수가 크게 다르다(03b_국가별_상위오류의 '화자'). 화자가 적은 나라는 한두 사람의 습관이 나라 전체처럼 보인다.",
                 "⑤ 국적과 모국어는 다르다(예: 미국 국적의 비영어 모국어 화자). 이 파일은 국적 기준이다."]
    else:
      notes = ["이 파일을 그대로 믿으면 안 되는 지점", "",
             "① 오류유형은 운영 판정(mdd.pipeline.ctype)이다. 예시 카탈로그(s0_taxonomy.classify_op)와 이름은 같지만 규칙이 조금 다르다.",
             "   예: ㄹ받침↔ㄴ받침 혼동의 이름이 두 규칙에서 다를 수 있다. 예시와 숫자를 직접 비교할 때 주의.",
             "② 화면에 낸 지적만 셌다. 운영이 '애매'로 숨긴 자리(uncertain)는 넣지 않았다.",
             "③ 국적은 녹음 1개로 낸 모델 추정이다. 실제 앱 요청에서 낯선 입력이 미국 등으로 쏠리는 경향이 있다.",
             "④ 한국(추정) 녹음 상당수는 내부 시험·원어민 점검이다(05_출처별). 학습자 분포를 보려면 '_앱요청만' 파일을 볼 것.",
             "⑤ 버전이 섞여 있다(04_버전별_유형). P6(10-22~)이 지금 운영이다.",
             "⑥ 같은 문장을 여러 번 읽은 녹음이 많다(09_문장별) — 문장 하나의 특성이 국가 비율을 끌고 갈 수 있다."]
    for i, l in enumerate(notes, 1):
        ws.cell(i, 1, l)
    ws["A1"].font = Font(bold=True); ws.column_dimensions["A"].width = 120
    xlsx_path = os.path.join(outdir, "%s_국적별_한국어_발음오류_유형카탈로그%s.xlsx" % (base, sfx))
    wb.save(xlsx_path)
    print("[그림] %s\n[엑셀] %s · 녹음 %d · 지적 %d · 국가 %d" % (html_path, xlsx_path, len(utts_used), total, len(countries)), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    a1 = sp.add_parser("aggregate"); a1.add_argument("--db", required=True, help="captures.sqlite (운영 서버 /data/speech-logs/db/)"); a1.add_argument("--out", default="catalog_data.json")
    a2 = sp.add_parser("render")
    a2.add_argument("--data", default="catalog_data.json"); a2.add_argument("--outdir", default="report")
    a2.add_argument("--template", default=os.path.join(HERE, "heatmap_template.html"))
    a2.add_argument("--example-xlsx", default="", help="정의 문구 · 심볼범례를 가져올 예시 카탈로그(없으면 기본값)")
    a2.add_argument("--only-source", default=None, help="예: '앱 서버(python-httpx)' — 그 출처의 요청만")
    a = ap.parse_args()
    if a.cmd == "aggregate":
        aggregate(a.db, a.out)
    else:
        render(a.data, a.outdir, a.template, a.example_xlsx, a.only_source)


if __name__ == "__main__":
    main()
