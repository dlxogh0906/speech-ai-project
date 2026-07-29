# -*- coding: utf-8 -*-
"""
s1_run_batch.py
~~~~~~~~~~~~~~~
Speech_lab (LAR 파일만) 대량 처리 실행판. 멀티프로세싱.

  각 레코드 -> s1_speechlab_mdd.build_record (정제->G2P->MDD->품질필터)
  -> 레코드 JSONL 저장(전량) + L1별 집계(usable=ok 만) CSV 저장

출력 폴더: _mdd_analysis/<split>/
  records.jsonl                 전량(품질플래그 포함)
  l1_summary.csv                L1별 레코드수/usable/오류수/평균err_ratio/간투사·반복
  l1_errortype_counts.csv       L1 × error_type 카운트(usable)
  l1_errortype_rate.csv         L1 × error_type 비율(=count/usable_records)
  substitution_pairs.csv        L1 × (target->actual) 대치쌍 카운트(usable, op=sub) = L1 prior 원천
  quality_report.csv            품질플래그 분포

실행(PowerShell, 프로젝트 루트에서):
  $env:PYTHONUTF8=1
  python s1_run_batch.py --groups 040 --split both --workers 8
옵션:
  --groups 040 041 042 043   (기본 040 영어)
  --split  Training|Validation|both   (기본 both)
  --workers N   (기본 = CPU-1)
  --limit  N    (디버그: 파일 N개만)
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists(): _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import os
import sys
import json
import glob
import argparse
import collections
import multiprocessing as mp

# 워커에서 import 되도록 모듈 경로 보장
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mdd.l1prior.s1_speechlab_mdd import build_record

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # scripts/ 하위 -> 데이터 루트는 부모
GROUP_DIRS = {
    "040": "040.교육용 영어 모국어 사용자의 한국어 음성 데이터",
    "041": "041.교육용 유럽어 모국어 사용자의 한국어 음성 데이터",
    "042": "042.교육용 중·일어 모국어 사용자의 한국어 음성 데이터",
    "043": "043.교육용 아시아어(중·일어 제외) 사용자의 한국어 음성 데이터",
}


def find_speechlab_dirs(groups, split):
    """그룹×스플릿의 *_Speech_lab 폴더 경로 목록."""
    dirs = []
    splits = ["Training", "Validation"] if split == "both" else [split]
    for g in groups:
        base = os.path.join(ROOT, GROUP_DIRS[g], "01-1.정식개방데이터")
        for sp in splits:
            prefix = "TL" if sp == "Training" else "VL"
            p = os.path.join(base, sp, "02.라벨링데이터", f"{prefix}_Speech_lab")
            if os.path.isdir(p):
                dirs.append((g, sp, p))
    return dirs


def _worker(fpath):
    """파일 1개 -> 통합 스키마 레코드(dict) 또는 None."""
    try:
        r = json.load(open(fpath, encoding="utf-8"))
    except Exception:
        return None
    if not r.get("RecordingMetadata", {}).get("orthographic"):
        return None
    try:
        return build_record(r, os.path.basename(fpath))
    except Exception as e:
        return {"_error": str(e), "file": os.path.basename(fpath)}


def run(groups, split, workers, limit, outroot):
    dirs = find_speechlab_dirs(groups, split)
    if not dirs:
        print("[!] Speech_lab 폴더를 못 찾음. --groups/--split 확인.")
        return
    files = []
    for g, sp, p in dirs:
        fs = sorted(glob.glob(os.path.join(p, "*LAR*.json")))
        files.extend(fs)
        print(f"  {g}/{sp}: LAR {len(fs):,}건  ({p})")
    if limit:
        files = files[:limit]
    print(f"[*] 총 처리 대상: {len(files):,}건 | workers={workers}")

    tag = "_".join(groups) + "_" + split
    outdir = os.path.join(outroot, tag)
    os.makedirs(outdir, exist_ok=True)

    # 집계 컨테이너
    l1_et = collections.defaultdict(collections.Counter)     # L1 -> error_type -> n (usable)
    l1_sub = collections.defaultdict(collections.Counter)    # L1 -> "t>a" -> n (usable, sub)
    l1_stat = collections.defaultdict(lambda: collections.Counter())
    flag_dist = collections.Counter()
    n_ok = n_err = n_none = 0

    jsonl_path = os.path.join(outdir, "records.jsonl")
    fout = open(jsonl_path, "w", encoding="utf-8")

    with mp.Pool(workers) as pool:
        for i, rec in enumerate(pool.imap_unordered(_worker, files, chunksize=64), 1):
            if rec is None:
                n_none += 1
            elif "_error" in rec:
                n_err += 1
            else:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_ok += 1
                L1 = rec.get("L1") or "UNKNOWN"
                flag = rec["quality_flag"]
                flag_dist[flag] += 1
                st = l1_stat[L1]
                st["records"] += 1
                st["disfluency"] += len(rec["disfluencies"])
                if rec["usable_for_aggregation"]:
                    st["usable"] += 1
                    st["errors"] += rec["n_errors"]
                    st["err_ratio_sum"] += rec["err_ratio"]
                    for e in rec["errors"]:
                        l1_et[L1][e["error_type"]] += 1
                        if e["op"] == "sub":
                            l1_sub[L1][f'{e["target_phone"]}>{e["actual_phone"]}'] += 1
            if i % 5000 == 0:
                print(f"    ... {i:,}/{len(files):,}  (ok={n_ok:,} skip={n_none:,} err={n_err:,})")
    fout.close()

    # ── CSV 저장 ──
    import csv

    def wcsv(name, header, rows):
        with open(os.path.join(outdir, name), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f); w.writerow(header); w.writerows(rows)

    # l1_summary
    rows = []
    for L1, st in sorted(l1_stat.items()):
        usable = st["usable"]
        rows.append([L1, st["records"], usable, st["errors"], st["disfluency"],
                     round(st["err_ratio_sum"] / usable, 4) if usable else 0,
                     round(st["errors"] / usable, 3) if usable else 0])
    wcsv("l1_summary.csv",
         ["L1", "records", "usable", "total_errors", "disfluencies",
          "mean_err_ratio", "errors_per_usable"], rows)

    # error_type 목록(전체에서 등장한 것)
    all_et = sorted({et for c in l1_et.values() for et in c})
    # counts
    rows = [[L1] + [l1_et[L1].get(et, 0) for et in all_et] for L1 in sorted(l1_et)]
    wcsv("l1_errortype_counts.csv", ["L1"] + all_et, rows)
    # rate (per usable record)
    rows = []
    for L1 in sorted(l1_et):
        u = l1_stat[L1]["usable"] or 1
        rows.append([L1] + [round(l1_et[L1].get(et, 0) / u, 4) for et in all_et])
    wcsv("l1_errortype_rate.csv", ["L1"] + all_et, rows)

    # substitution pairs (long)
    rows = []
    for L1 in sorted(l1_sub):
        for pair, n in l1_sub[L1].most_common():
            t, a = pair.split(">")
            rows.append([L1, t, a, n])
    wcsv("substitution_pairs.csv", ["L1", "target_phone", "actual_phone", "count"], rows)

    # quality report
    wcsv("quality_report.csv", ["quality_flag", "count"],
         [[k, v] for k, v in flag_dist.most_common()])

    # ── 콘솔 요약 ──
    print("\n[완료] 출력:", outdir)
    print(f"  레코드 저장: {n_ok:,} | 스킵(전사없음): {n_none:,} | 에러: {n_err:,}")
    print(f"  품질분포: {dict(flag_dist)}")
    tot = sum(flag_dist.values()) or 1
    print(f"  usable(ok) 비율: {100*flag_dist['ok']/tot:.1f}%")
    print(f"  L1(국적) 종류: {len(l1_stat)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", default=["040"], choices=list(GROUP_DIRS))
    ap.add_argument("--split", default="both", choices=["Training", "Validation", "both"])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--outroot", default=os.path.join(ROOT, "_mdd_analysis"))
    a = ap.parse_args()
    run(a.groups, a.split, a.workers, a.limit, a.outroot)
