# -*- coding: utf-8 -*-
"""
s3_merge_schema.py
~~~~~~~~~~~~~~~~~~
Speech_lab(실현형 MDD)와 pronunciation_lab(gold 위치·유형)을 **같은 통합 스키마**로
합쳐 하나의 분석 뷰를 만든다. (작업지시서 5절 "통합 적재")

주의: 두 소스는 '측정 대상'이 다르다.
  - speech_lab : 실현형 대치/삭제/삽입(약라벨, 대량)  → err_type 자동태깅
  - pronun_lab : 사람이 붙인 위치·유형(강라벨, gold)   → err_type gold
따라서 **카운트를 단순 합산하지 않는다**(이중집계 위험).
대신 L1 × error_type 를 source 별로 나란히 두는 '비교 매트릭스'로 병합한다.

입력(각 배치 산출물의 집계 CSV):
  _mdd_analysis/<speech_tag>/l1_errortype_counts.csv       (speech_lab, usable)
  _mdd_analysis/<speech_tag>/l1_summary.csv
  _mdd_analysis/<pron_tag>/l1_errortype_counts.csv         (pronun_lab, gold)
  _mdd_analysis/<pron_tag>/l1_summary.csv

출력(_mdd_analysis/<out_tag>/):
  l1_source_matrix_long.csv   L1 × error_type × source(카운트·율) 롱포맷
  l1_source_rate_wide.csv     L1 × error_type, speech_rate/gold_rate 병렬
  merge_coverage.csv          L1별 두 소스 존재/파일수/오류수
  merged_records.jsonl        (옵션 --concat) 두 jsonl 물리 병합(대용량)

실행:
  python s3_merge_schema.py --speech-tag 040_041_042_043_both \
                            --pron-tag  040_041_042_043_both_pronlab
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import os
import csv
import json
import argparse
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # scripts/ 하위 -> 데이터 루트는 부모
from mdd.l1prior.s0_taxonomy import CORE_TAGS  # noqa: E402


def _load_counts(d):
    p = os.path.join(d, "l1_errortype_counts.csv")
    return pd.read_csv(p, encoding="utf-8-sig").set_index("L1")


def _load_summary(d):
    p = os.path.join(d, "l1_summary.csv")
    return pd.read_csv(p, encoding="utf-8-sig").set_index("L1")


def main(speech_tag, pron_tag, out_tag, concat):
    mdd = os.path.join(ROOT, "_mdd_analysis")
    sd = os.path.join(mdd, speech_tag)
    pd_ = os.path.join(mdd, pron_tag)
    outdir = os.path.join(mdd, out_tag or (speech_tag + "_MERGED"))
    os.makedirs(outdir, exist_ok=True)

    sc = _load_counts(sd)    # speech counts
    pc = _load_counts(pd_)   # pron gold counts
    ss = _load_summary(sd)
    ps = _load_summary(pd_)

    # 분모: speech=usable records, pron=files
    s_den = ss["usable"] if "usable" in ss else ss.iloc[:, 0]
    p_den = ps["files"] if "files" in ps else ps.iloc[:, 0]

    # 공통 error_type(core 우선), 공통·합집합 L1
    et_cols = [c for c in CORE_TAGS if c in sc.columns or c in pc.columns]
    all_L1 = sorted(set(sc.index) | set(pc.index))

    # ── 롱포맷: L1 × error_type × source ──
    long_rows = []
    for L1 in all_L1:
        for et in et_cols:
            s_cnt = int(sc.loc[L1, et]) if (L1 in sc.index and et in sc.columns) else 0
            p_cnt = int(pc.loc[L1, et]) if (L1 in pc.index and et in pc.columns) else 0
            sden = float(s_den.get(L1, 0) or 0)
            pden = float(p_den.get(L1, 0) or 0)
            if L1 in sc.index:
                long_rows.append([L1, et, "speech_lab", s_cnt,
                                  round(s_cnt / sden, 5) if sden else 0])
            if L1 in pc.index:
                long_rows.append([L1, et, "pronun_lab", p_cnt,
                                  round(p_cnt / pden, 5) if pden else 0])
    with open(os.path.join(outdir, "l1_source_matrix_long.csv"), "w",
              newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["L1", "error_type", "source", "count", "rate_per_record"])
        w.writerows(long_rows)

    # ── wide: speech_rate vs gold_rate 병렬 ──
    wide_rows = []
    for L1 in all_L1:
        for et in et_cols:
            s_cnt = int(sc.loc[L1, et]) if (L1 in sc.index and et in sc.columns) else 0
            p_cnt = int(pc.loc[L1, et]) if (L1 in pc.index and et in pc.columns) else 0
            sden = float(s_den.get(L1, 0) or 0)
            pden = float(p_den.get(L1, 0) or 0)
            wide_rows.append([
                L1, et,
                round(s_cnt / sden, 5) if sden else 0,
                round(p_cnt / pden, 5) if pden else 0,
                s_cnt, p_cnt,
            ])
    with open(os.path.join(outdir, "l1_source_rate_wide.csv"), "w",
              newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["L1", "error_type", "speech_rate", "gold_rate",
                    "speech_count", "gold_count"])
        w.writerows(wide_rows)

    # ── coverage ──
    cov_rows = []
    for L1 in all_L1:
        cov_rows.append([
            L1,
            "Y" if L1 in sc.index else "",
            "Y" if L1 in pc.index else "",
            int(s_den.get(L1, 0) or 0),
            int(p_den.get(L1, 0) or 0),
        ])
    with open(os.path.join(outdir, "merge_coverage.csv"), "w",
              newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["L1", "in_speech", "in_gold", "speech_records", "gold_files"])
        w.writerows(cov_rows)

    both = [r for r in cov_rows if r[1] == "Y" and r[2] == "Y"]
    print(f"[완료] {outdir}")
    print(f"  L1 합집합 {len(all_L1)} | 두 소스 공통 {len(both)}")
    print(f"  error_type(core) {len(et_cols)}종")

    # ── 옵션: 물리 병합 jsonl(대용량) ──
    if concat:
        outp = os.path.join(outdir, "merged_records.jsonl")
        n = 0
        with open(outp, "w", encoding="utf-8") as fo:
            for src, tag in ((sd, "speech"), (pd_, "gold")):
                for fn in ("records.jsonl", "records_pronlab.jsonl"):
                    fp = os.path.join(src, fn)
                    if not os.path.isfile(fp):
                        continue
                    for line in open(fp, encoding="utf-8"):
                        fo.write(line)
                        n += 1
        print(f"  [concat] merged_records.jsonl 총 {n:,}줄 -> {outp}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--speech-tag", default="040_041_042_043_both")
    ap.add_argument("--pron-tag", default="040_041_042_043_both_pronlab")
    ap.add_argument("--out-tag", default=None)
    ap.add_argument("--concat", action="store_true",
                    help="두 jsonl 물리 병합(대용량, 기본 off)")
    a = ap.parse_args()
    main(a.speech_tag, a.pron_tag, a.out_tag, a.concat)
