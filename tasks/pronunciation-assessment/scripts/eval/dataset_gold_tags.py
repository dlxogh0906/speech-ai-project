# -*- coding: utf-8 -*-
"""AIHub 학습자 녹음의 **사람 오류 태그(14유형)**를 음소에 붙여 국적별로 '어떤 발음을 많이 틀리나'를 센다.

- 사람 태그: pronunciation_lab json 의 RecordingMetadata.phonemic.error_tags[] = {error_tag, start, end} (시간창만 있다)
- 음소 시간: 같은 json 의 phonemic.phones[] — 매니페스트 표준음(.canon)과 순서가 같으면(표본 98%) 자리 q 의 시간창으로 쓴다
- 실제 발음: 매니페스트 TAG(.ltr)를 표준음 축에 투영 — 자리마다 정답(표준음) → 실현(사람이 들은 소리 · 빠짐)
- 유형: 음소 변화가 있는 자리에 겹치는 태그(가장 많이 겹치는 것). 겹치는 태그가 없으면 '(태그 없음)'
- 국적: SpeakerMetadata.nationality(실제 값)
대상은 dataset_error_catalog.py 와 같은 녹음(시험·검증: manifest_full_tag_replay10, 학습: manifest_tag_rebuilt, 문항 코드 있는 것).

    $PY .../dataset_gold_tags.py extract --shard 0/8      # 조각마다 집계 json
    $PY .../dataset_gold_tags.py report                   # 합쳐서 엑셀 + HTML
"""
# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import DATA_ROOT, MDD_OUTPUTS

import argparse, collections, glob, importlib.util, json, os, re, time, zlib

OUT_DIR = os.path.join(str(MDD_OUTPUTS), "analysis", "dataset_catalog_2026-11-02", "gold_tags")
AIHUB = str(DATA_ROOT / "aihub")
CORE14 = ["초성", "종성", "모음오류", "경음화", "비음화", "격음화", "연음규칙", "유음화", "경계억양 오류", "ㄴ-첨가", "구개음화", "강세 오류",
          "음운탈락", "모음삽입"]
EXTRA = ["기타", "반복", "간투사"]
VOWELS = set("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
ITEM_RE = re.compile(r"-(R[WSP]\d{3})-")


def position(seq, i):
    if seq[i] in VOWELS:
        return "중성"
    return "초성" if (i + 1 < len(seq) and seq[i + 1] in VOWELS) else "종성"


def sources():
    M1 = os.path.join(str(MDD_OUTPUTS), "manifest_full_tag_replay10")
    M2 = os.path.join(str(MDD_OUTPUTS), "manifest_tag_rebuilt")
    return [("test", M1, "test"), ("valid", M1, "valid"), ("train", M2, "train")]


def cmd_extract(a):
    spec = importlib.util.spec_from_file_location("ecs", str(_pl.Path(__file__).with_name("eval_canonical_source.py")))
    ecs = importlib.util.module_from_spec(spec); spec.loader.exec_module(ecs)
    shard, n_shard = (int(x) for x in a.shard.split("/"))
    lab_idx = {}
    for d in sorted(glob.glob(AIHUB + "/04*/01-1*/*/02.라벨링데이터/*_pronunciation_lab")):
        for e in os.scandir(d):
            if e.name.endswith(".json"):
                lab_idx[e.name[:-5]] = e.path
    meta = json.load(open(os.path.join(os.path.dirname(OUT_DIR), "speaker_meta.json"), encoding="utf-8"))
    C = {"tags": collections.Counter(), "pat": collections.Counter(), "occ": collections.Counter(), "utt": collections.Counter(),
         "spk": collections.defaultdict(set), "stat": collections.Counter()}
    t0 = time.time(); n = 0
    for split, M, f in sources():
        with open(os.path.join(M, f + ".tsv"), encoding="utf-8") as fh:
            fh.readline(); rows = [l.rstrip("\n").split("\t")[0] for l in fh]
        can = open(os.path.join(M, f + ".canon"), encoding="utf-8").read().splitlines()
        ltr = open(os.path.join(M, f + ".ltr"), encoding="utf-8").read().splitlines()
        seen = set()
        for i, p in enumerate(rows):
            if p in seen or not ITEM_RE.search(p):
                continue
            seen.add(p)
            if zlib.crc32(p.encode()) % n_shard != shard:
                continue
            stem = os.path.basename(p)[:-4]
            labc, tg = can[i].split(), ltr[i].split()
            if not labc or any(len(x) != 1 for x in labc + tg):
                continue
            country = (meta.get(stem[:5]) or {}).get("nationality") or "(국적 미상)"
            lp = lab_idx.get(stem)
            if not lp:
                C["stat"]["라벨 json 없음"] += 1
                continue
            try:
                ph = json.load(open(lp, encoding="utf-8"))["RecordingMetadata"]["phonemic"]
            except Exception:
                C["stat"]["라벨 json 깨짐"] += 1
                continue
            n += 1
            C["utt"][country] += 1; C["spk"][country].add(stem[:5])
            etags = [(e.get("error_tag"), float(e.get("start") or 0), float(e.get("end") or 0)) for e in (ph.get("error_tags") or [])]
            for t, _, _ in etags:
                C["tags"][(country, t)] += 1
            phones = ph.get("phones") or []
            aligned = [x.get("phone") for x in phones] == labc
            C["stat"]["음소열 같음" if aligned else "음소열 다름(유형 못 붙임)"] += 1
            tag = ecs.proj(labc, tg)
            for q, c in enumerate(labc):
                pos = position(labc, q)
                C["occ"][(country, pos, c)] += 1
                if tag[q] == c:
                    continue
                typ = "(태그 없음)"
                if aligned and etags:
                    s0, s1 = float(phones[q].get("start") or 0), float(phones[q].get("end") or 0)
                    best = max(((min(s1, e1) - max(s0, e0), t) for t, e0, e1 in etags), default=(0, None))
                    if best[0] > 0:
                        typ = best[1]
                elif not aligned:
                    typ = "(시간 정렬 불가)"
                C["pat"][(country, pos, c, tag[q] or "", typ)] += 1
            if n % 20000 == 0:
                print("  %d · %.0f녹음/s" % (n, n / (time.time() - t0)), flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump({"tags": [list(k) + [v] for k, v in C["tags"].items()], "pat": [list(k) + [v] for k, v in C["pat"].items()],
               "occ": [list(k) + [v] for k, v in C["occ"].items()], "utt": dict(C["utt"]),
               "spk": {k: sorted(v) for k, v in C["spk"].items()}, "stat": dict(C["stat"])},
              open(os.path.join(OUT_DIR, "part.%d-of-%d.json" % (shard, n_shard)), "w", encoding="utf-8"), ensure_ascii=False)
    print("[끝] 녹음 %d · %s · %.0f초" % (n, dict(C["stat"]), time.time() - t0), flush=True)


KO = None


def ko(c):
    global KO
    if KO is None:
        spec = importlib.util.spec_from_file_location("bc", str(_r / "tools" / "capture_db" / "build_catalog.py"))
        bc = importlib.util.module_from_spec(spec); spec.loader.exec_module(bc); KO = bc.KO
    return KO.get(c, c)


def cmd_report(a):
    tags, pat, occ, utt, spk, stat = collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter(), collections.defaultdict(set), collections.Counter()
    for f in sorted(glob.glob(os.path.join(OUT_DIR, "part.*.json"))):
        d = json.load(open(f, encoding="utf-8"))
        for *k, v in d["tags"]: tags[tuple(k)] += v
        for *k, v in d["pat"]: pat[tuple(k)] += v
        for *k, v in d["occ"]: occ[tuple(k)] += v
        utt.update(d["utt"]); stat.update(d["stat"])
        for k, v in d["spk"].items(): spk[k].update(v)
    countries = sorted(utt, key=lambda c: (-len(spk[c]), c))
    big = [c for c in countries if len(spk[c]) >= a.min_spk]
    J = lambda x: x or "빠짐"
    # 국가별 14유형 분포
    rows14 = []
    for c in countries:
        tc = {t: tags[(c, t)] for t in CORE14}; tot = sum(tc.values())
        ex = {t: tags[(c, t)] for t in EXTRA}
        rows14.append([c, ko(c), len(spk[c]), utt[c], tot] + [tc[t] for t in CORE14] + [round(100.0 * tc[t] / max(1, tot), 1) for t in CORE14]
                      + [ex[t] for t in EXTRA])
    # 국가별 틀리는 발음
    by_c = collections.defaultdict(list)
    for (c, pos, lab, act, typ), n in pat.items():
        by_c[c].append((n, pos, lab, act, typ))
    top_rows, long_rows = [], []
    for c in countries:
        L = sorted(by_c[c], reverse=True); tot = sum(n for n, *_ in L)
        for rk, (n, pos, lab, act, typ) in enumerate(L, 1):
            rate = 100.0 * n / max(1, occ[(c, pos, lab)])
            r = [c, ko(c), len(spk[c]), rk, pos, lab, J(act), "%s→%s" % (lab, J(act)), typ, n, round(100.0 * n / max(1, tot), 2), round(rate, 2), occ[(c, pos, lab)]]
            long_rows.append(r)
            if rk <= a.topn:
                top_rows.append(r)
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
    tot_tags = sum(v for (c, t), v in tags.items() if t in CORE14)
    lines = ["AIHub 학습자 녹음 — 국적별 '많이 틀리는 발음' · 사람 오류 태그 14유형", "",
             "대상: AIHub 040~043 발음 녹음(학습·검증·시험 화자) %d개 · 화자 %d명 · %d개국" % (sum(utt.values()), sum(len(v) for v in spk.values()), len(countries)),
             "오류유형: 사람이 단 오류 태그(pronunciation_lab error_tags) 14유형 — 예시 카탈로그의 gold 와 같은 체계. 14유형 태그 %d개." % tot_tags,
             "   %s" % " · ".join(CORE14),
             "   그 밖의 태그(%s)는 발음이 아니라 유창성·미분류라 14유형에서 뺐다(01 시트 끝 열에 따로)." % " · ".join(EXTRA), "",
             "■ 틀리는 발음 = 정답(표준음) → 실현(사람이 들은 소리, 빠짐 포함). 자리마다 겹치는 사람 태그로 유형을 붙였다.",
             "   · '나라 오류 중 %' = 그 나라 음소 오류 합계 대비",
             "   · '틀림률 %' = 그 나라 화자가 그 위치의 그 소리를 낼 때 이렇게 틀린 비율(같은 소리가 나온 횟수 대비)",
             "■ 경계억양·강세 오류는 음소가 바뀌지 않는 운율 오류라 '틀리는 발음' 표에는 거의 나오지 않는다(01 시트에서 본다).",
             "■ '(태그 없음)': 사람 전사(TAG)는 다르게 적었지만 오류 태그가 겹치지 않은 자리. '(시간 정렬 불가)': 라벨 음소열이 달라 시간을 못 맞춘 녹음(약 2%).",
             "■ 화자가 적은 나라는 한두 사람의 습관이 나라 전체처럼 보인다. 02 시트는 화자 %d명 이상 나라만." % a.min_spk,
             "정리 상태: %s" % dict(stat)]
    for i, l in enumerate(lines, 1):
        ws.cell(i, 1, l)
    ws["A1"].font = Font(bold=True, size=13); ws.column_dimensions["A"].width = 130
    sheet("01_국가별_14유형분포", ["국가(EN)", "국가(KO)", "화자", "녹음", "14유형 합계"] + ["%s 건수" % t for t in CORE14] + ["%s %%" % t for t in CORE14] + EXTRA,
          rows14, [16, 12, 6, 8, 10] + [8] * 28 + [7] * 3, title="★ 사람 오류 태그 기준 · %는 그 나라 14유형 합계 대비")
    hdr = ["국가(EN)", "국가(KO)", "화자", "순위", "위치", "정답", "실현", "틀리는 발음", "오류유형(사람 태그)", "건수", "나라 오류 중 %", "틀림률 %", "그 소리 나온 횟수"]
    sheet("02_국가별_틀리는발음_상위%d" % a.topn, hdr, [r for r in top_rows if r[0] in big], [16, 12, 6, 5, 5, 5, 5, 10, 14, 8, 10, 8, 10],
          title="★ 화자 %d명 이상 %d개국 · 나라마다 상위 %d개" % (a.min_spk, len(big), a.topn))
    sheet("03_전체국가_전체패턴", hdr, long_rows, [16, 12, 6, 5, 5, 5, 5, 10, 14, 8, 10, 8, 10], title="★ 모든 나라 · 모든 패턴")
    out_x = os.path.join(a.outdir, "AIHub_국적별_틀리는발음_14유형.xlsx")
    os.makedirs(a.outdir, exist_ok=True); wb.save(out_x)
    # HTML — 전체 국가 한 화면
    import html as _h
    cards = []
    for c in big:
        tc = [(t, tags[(c, t)]) for t in CORE14]; tot = sum(v for _, v in tc)
        t3 = " · ".join("%s %d%%" % (t, round(100.0 * v / max(1, tot))) for t, v in sorted(tc, key=lambda x: -x[1])[:4] if v)
        trs = "".join("<tr><td>%d</td><td><b>%s</b> <small>%s</small></td><td>%s</td><td class=n>%s%%</td><td class=n>%s%%</td><td class=n>%s</td></tr>"
                      % (r[3], _h.escape(r[7]), r[4], _h.escape(r[8]), r[10], r[11], format(r[9], ",")) for r in top_rows if r[0] == c)
        cards.append("<section><h3>%s <span>%s · 화자 %d명 · 녹음 %s</span></h3><p class=t>많은 유형: %s</p>"
                     "<table><tr><th>#</th><th>틀리는 발음 (정답→실현)</th><th>유형</th><th>나라 오류 중</th><th>틀림률</th><th>건수</th></tr>%s</table></section>"
                     % (ko(c), c, len(spk[c]), format(utt[c], ","), t3, trs))
    page = """<!doctype html><html lang=ko><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>국적별 많이 틀리는 발음</title><style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#1d2330;--muted:#6b7385;--line:#e3e6ec;--acc:#3b5bdb}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#14171d;--card:#1c2028;--ink:#e6e9ef;--muted:#9aa3b2;--line:#2c323d;--acc:#7c9cff}}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,-apple-system,"Noto Sans KR",sans-serif}
main{max-width:1200px;margin:0 auto;padding:20px 16px}h1{font-size:22px;margin:0 0 6px}p.sub{color:var(--muted);margin:0 0 16px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:14px}
section{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 14px}
h3{margin:0 0 4px;font-size:16px}h3 span{font-weight:400;color:var(--muted);font-size:12px}p.t{margin:0 0 8px;color:var(--muted);font-size:12.5px}
table{border-collapse:collapse;width:100%;font-size:12.5px}th,td{border-bottom:1px solid var(--line);padding:4px 5px;text-align:left}
th{color:var(--muted);font-weight:600}td.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}small{color:var(--muted)}
</style></head><body><main><h1>국적별 많이 틀리는 발음 — 사람 오류 태그 14유형</h1>
<p class=sub>AIHub 040~043 학습자 녹음 __NUTT__개 · 화자 __NSPK__명 · 화자 __MIN__명 이상 __NBIG__개국. 정답→실현 = 표준음 → 사람이 들은 소리. 유형 = 그 자리에 사람이 단 오류 태그.
'나라 오류 중' = 그 나라 음소 오류 합계 대비 · '틀림률' = 그 나라 화자가 그 소리를 낼 때 이렇게 틀린 비율.</p>
<div class=grid>__CARDS__</div></main></body></html>"""
    page = (page.replace("__NUTT__", format(sum(utt.values()), ",")).replace("__NSPK__", format(sum(len(v) for v in spk.values()), ","))
                .replace("__MIN__", str(a.min_spk)).replace("__NBIG__", str(len(big))).replace("__CARDS__", "".join(cards)))
    out_h = os.path.join(a.outdir, "AIHub_국적별_틀리는발음_14유형.html")
    open(out_h, "w", encoding="utf-8").write(page)
    print("[엑셀] %s\n[그림] %s · 국가 %d(화자 %d명 이상 %d) · 정리 %s" % (out_x, out_h, len(countries), a.min_spk, len(big), dict(stat)), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("extract"); e.add_argument("--shard", default="0/1")
    r = sp.add_parser("report"); r.add_argument("--outdir", default=os.path.join(str(_r), "docs", "db문서자료", "AIHub_데이터셋"))
    r.add_argument("--min-spk", dest="min_spk", type=int, default=30); r.add_argument("--topn", type=int, default=10)
    a = ap.parse_args()
    {"extract": cmd_extract, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
