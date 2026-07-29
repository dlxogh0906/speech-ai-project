# -*- coding: utf-8 -*-
"""
s9_gold_catalog.py
~~~~~~~~~~~~~~~~~~
gold(pronunciation_lab) 측 **국적별 발음오류 위치 카탈로그**.
Speech_lab 카탈로그(s8, 실현형 대치 '무엇으로 바뀌나')의 상보 —
gold 는 '어느 음소(위치)에 어떤 유형의 오류가 사람 판정으로 찍히나'를 준다.

입력: _mdd_analysis/<pron_tag>/records_pronlab.jsonl  (s2 산출, 스트리밍)
      각 record.errors[] = {error_type, target_phone, pos, ...}

출력(_mdd_analysis/<pron_tag>/error_catalog_gold/):
  gold_error_catalog.csv          L1 × error_type × target음소(위치) 카운트/비중
  L1_발음오류_gold카탈로그.md      국적별: 유형별 어느 음소가 주로 걸리나
  (콘솔) 주요국 미리보기

실행:
  python s9_gold_catalog.py --pron-tag 040_041_042_043_both_pronlab --min-files 500 --topn 8
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists(): _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import os
import csv
import json
import argparse
import collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # scripts/ 하위 -> 데이터 루트는 부모
from mdd.l1prior.s0_kog2p_wrapper import ONSET, NUCLEUS, CODA
from mdd.l1prior.s0_symbol_map import SYMBOL_TO_JAMO
from mdd.l1prior.s0_taxonomy import CORE_TAGS
from mdd.l1prior.s8_error_catalog import SOUND, KO, role, ko, label  # 재사용

ONSET_S, NUC_S, CODA_S = set(ONSET), set(NUCLEUS), set(CODA)


def main(pron_tag, min_files, topn):
    d = os.path.join(ROOT, "_mdd_analysis", pron_tag)
    jsonl = os.path.join(d, "records_pronlab.jsonl")
    outdir = os.path.join(d, "error_catalog_gold")
    os.makedirs(outdir, exist_ok=True)

    # L1 -> error_type -> (target_sym) -> count ;  L1 -> files ; L1 -> type -> count
    tri = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    l1_files = collections.Counter()
    l1_type = collections.defaultdict(collections.Counter)
    l1_errs = collections.Counter()

    n = 0
    with open(jsonl, encoding="utf-8") as f:
        for line in f:
            n += 1
            try:
                r = json.loads(line)
            except Exception:
                continue
            L1 = r.get("L1") or "UNKNOWN"
            l1_files[L1] += 1
            for e in r.get("errors", []):
                et = e.get("error_type")
                if not et:
                    continue
                l1_type[L1][et] += 1
                l1_errs[L1] += 1
                tp = e.get("target_phone")
                if tp:
                    tri[L1][et][tp] += 1
            if n % 200000 == 0:
                print(f"    ... {n:,} lines")
    print(f"[*] 스트리밍 완료: {n:,} records, {len(l1_files)} L1")

    order = [L1 for L1, c in l1_files.most_common() if c >= min_files]

    # ── enriched CSV ──
    with open(os.path.join(outdir, "gold_error_catalog.csv"), "w",
              newline="", encoding="utf-8-sig") as fo:
        w = csv.writer(fo)
        w.writerow(["L1", "error_type", "target_position", "target_sym",
                    "target_ko", "count", "share_in_type_pct"])
        for L1 in order:
            for et in CORE_TAGS:
                cnt = tri[L1].get(et)
                if not cnt:
                    continue
                tot = sum(cnt.values())
                for sym, c in cnt.most_common():
                    w.writerow([L1, et, role(sym), sym, ko(sym), c,
                                round(c / tot * 100, 2)])

    # ── 마크다운 ──
    L = ["# 국적(L1)별 gold 발음오류 위치 카탈로그 — pronunciation_lab\n",
         f"_생성: s9_gold_catalog.py · `{pron_tag}` · 파일≥{min_files} → {len(order)}개국_\n",
         "## 읽는 법",
         "- gold = 사람이 태깅한 error_tags(유형+시간창). 시간창을 phones 타임라인에 겹쳐 **대상 음소(위치)** 를 특정.",
         "- Speech_lab 카탈로그(s8)가 '무엇으로 바뀌나(대치)'라면, 여기는 '어느 음소에 어떤 유형이 걸리나(위치·유형)'.",
         "- 표기: `nf=ㄴ받침(종성)`. **비중%** = 그 유형 내에서 해당 음소가 차지하는 비율.\n",
         "## 국적별 (유형 → 대상음소)\n"]
    for L1 in order:
        nm = f"{L1} ({KO.get(L1, '')})".strip()
        tot_e = l1_errs[L1]
        L.append(f"### {nm}")
        L.append(f"- gold 파일 **{l1_files[L1]:,}** · 오류태그 **{tot_e:,}**건")
        # 유형 상위(카운트순)
        top_types = [t for t, _ in l1_type[L1].most_common() if t in CORE_TAGS][:6]
        for et in top_types:
            cnt = tri[L1].get(et)
            if not cnt:
                continue
            tote = sum(cnt.values())
            items = ", ".join(f"{label(s)} {c:,}건({c/tote*100:.0f}%)"
                              for s, c in cnt.most_common(topn))
            share = l1_type[L1][et] / tot_e * 100 if tot_e else 0
            L.append(f"- **{et}** (유형내 {l1_type[L1][et]:,}건, 전체의 {share:.0f}%): {items}")
        L.append("")
    open(os.path.join(outdir, "L1_발음오류_gold카탈로그.md"), "w",
         encoding="utf-8").write("\n".join(L))

    print(f"[완료] {outdir}")
    print(f"  대상 {len(order)}개국")
    for L1 in ["China", "Vietnam", "Japan", "Philippines"]:
        if L1 not in order:
            continue
        # 종성 유형의 대상 음소 top3
        cnt = tri[L1].get("종성")
        if cnt:
            s = ", ".join(f"{label(k)}({v})" for k, v in cnt.most_common(3))
            print(f"    [{L1} 종성→대상음소] {s}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pron-tag", default="040_041_042_043_both_pronlab")
    ap.add_argument("--min-files", type=int, default=500)
    ap.add_argument("--topn", type=int, default=8)
    a = ap.parse_args()
    main(a.pron_tag, a.min_files, a.topn)
