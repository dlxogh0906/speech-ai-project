# -*- coding: utf-8 -*-
"""
국적별 발음 오류(error_tag) 추출·집계 스크립트
-------------------------------------------------
대상 : 040~043 교육용 한국어 음성 데이터셋의 *pronunciation_lab* JSON
       (error_tags 는 pronunciation_lab 에만 존재 / Speech_lab 에는 없음)

동작 :
  1) ROOT 아래의 모든 pronunciation_lab 폴더를 자동 탐색
  2) 각 JSON 에서  nationality / language / proficiency / topik_level
     + 음소수(phones) / 단어수(words) / error_tags 를 추출
  3) (dataset, split, nationality, language, proficiency, topik_level) 단위로 집계
  4) 4종 CSV 로 저장 (UTF-8-SIG → 엑셀에서 한글 정상)

산출물 (기본: ROOT/_error_tag_analysis/) :
  - agg_full.csv                : 모든 집계키 x error_tag 카운트 (마스터, 여기서 원하는 대로 피벗)
  - by_nationality_tag_long.csv : 국적 x 태그 롱포맷 + 정규화율
  - by_nationality_pivot.csv    : 국적 x 태그 카운트 피벗
  - nationality_summary.csv     : 국적별 요약(파일수/오류율/상위태그)

사용 예:
  python analyze_error_tags.py                      # 전체 스캔
  python analyze_error_tags.py --datasets 043       # 043 만
  python analyze_error_tags.py --split Validation   # 검증셋만 (프로토타입 권장)
  python analyze_error_tags.py --workers 8
  python analyze_error_tags.py --include-dup        # ' (2)' 중복폴더 포함
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]


import os
import re
import sys
import csv
import json
import time
import argparse
import collections
from concurrent.futures import ProcessPoolExecutor, as_completed

# 콘솔 한글 출력 깨짐 방지 (Python 3.7+)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DATASET_RE = re.compile(r"(\d{3})\.교육용")     # 경로에서 040~043 추출
CHUNK = 2000                                    # 워커에 넘길 파일 묶음 크기


# ---------------------------------------------------------------------------
# 경로 메타 파싱
# ---------------------------------------------------------------------------
def parse_path_meta(path):
    m = DATASET_RE.search(path)
    dataset = m.group(1) if m else "UNK"
    split = "Validation" if (os.sep + "Validation" + os.sep) in path or "/Validation/" in path else "Training"
    return dataset, split


# ---------------------------------------------------------------------------
# 워커: 파일 묶음 하나를 읽어 부분 집계 반환
#   반환 구조:
#   { key_tuple : {"n_files":int,"n_err_files":int,"phones":int,
#                  "words":int,"tags":{tag:count}} }
#   key_tuple = (dataset, split, nationality, language, proficiency, topik_level)
# ---------------------------------------------------------------------------
def process_chunk(paths):
    agg = {}
    bad = 0
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                d = json.load(f)
        except Exception:
            bad += 1
            continue

        sm = d.get("SpeakerMetadata", {}) or {}
        nationality = sm.get("nationality") or "UNKNOWN"
        language = sm.get("language") or "UNKNOWN"
        proficiency = sm.get("proficiency") or "UNKNOWN"
        topik_level = sm.get("topik_level")
        topik_level = "NA" if topik_level is None else str(topik_level)

        phon = (d.get("RecordingMetadata", {}) or {}).get("phonemic", {}) or {}
        phones = phon.get("phones") or []
        words = phon.get("words") or []
        etags = phon.get("error_tags") or []

        dataset, split = parse_path_meta(path)
        key = (dataset, split, nationality, language, proficiency, topik_level)

        rec = agg.get(key)
        if rec is None:
            rec = {"n_files": 0, "n_err_files": 0, "phones": 0,
                   "words": 0, "tags": collections.Counter()}
            agg[key] = rec

        rec["n_files"] += 1
        rec["phones"] += len(phones)
        rec["words"] += len(words)
        if etags:
            rec["n_err_files"] += 1
            for e in etags:
                t = e.get("error_tag")
                if t:
                    rec["tags"][t] += 1

    # Counter -> dict (피클링 안전)
    for rec in agg.values():
        rec["tags"] = dict(rec["tags"])
    return agg, len(paths), bad


# ---------------------------------------------------------------------------
# 부분 집계 병합
# ---------------------------------------------------------------------------
def merge_into(total, part):
    for key, rec in part.items():
        t = total.get(key)
        if t is None:
            t = {"n_files": 0, "n_err_files": 0, "phones": 0,
                 "words": 0, "tags": collections.Counter()}
            total[key] = t
        t["n_files"] += rec["n_files"]
        t["n_err_files"] += rec["n_err_files"]
        t["phones"] += rec["phones"]
        t["words"] += rec["words"]
        for tag, c in rec["tags"].items():
            t["tags"][tag] += c


# ---------------------------------------------------------------------------
# pronunciation_lab 폴더 탐색
# ---------------------------------------------------------------------------
def find_pron_folders(root, datasets, split_filter, include_dup):
    folders = []
    for dirpath, dirnames, _ in os.walk(root):
        base = os.path.basename(dirpath)
        if "pronunciation_lab" not in base.lower():
            continue
        if (not include_dup) and (" (2)" in dirpath):
            continue
        ds, sp = parse_path_meta(dirpath)
        if datasets and ds not in datasets:
            continue
        if split_filter and sp != split_filter:
            continue
        folders.append(dirpath)
    return sorted(folders)


def iter_chunks(folders):
    """폴더별로 파일 경로를 모아 CHUNK 단위로 yield (메모리 절약)"""
    buf = []
    for folder in folders:
        try:
            with os.scandir(folder) as it:
                for entry in it:
                    if entry.is_file() and entry.name.lower().endswith(".json"):
                        buf.append(entry.path)
                        if len(buf) >= CHUNK:
                            yield buf
                            buf = []
        except FileNotFoundError:
            continue
    if buf:
        yield buf


# ---------------------------------------------------------------------------
# CSV 출력
# ---------------------------------------------------------------------------
def write_outputs(total, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # 전체 태그 집합(동적)
    all_tags = set()
    for rec in total.values():
        all_tags.update(rec["tags"].keys())
    all_tags = sorted(all_tags)

    # 1) agg_full.csv --------------------------------------------------------
    p1 = os.path.join(out_dir, "agg_full.csv")
    with open(p1, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        header = ["dataset", "split", "nationality", "language",
                  "proficiency", "topik_level", "n_files", "n_err_files",
                  "total_phones", "total_words", "total_errors"] + all_tags
        w.writerow(header)
        for key, rec in sorted(total.items()):
            total_err = sum(rec["tags"].values())
            row = list(key) + [rec["n_files"], rec["n_err_files"],
                               rec["phones"], rec["words"], total_err]
            row += [rec["tags"].get(t, 0) for t in all_tags]
            w.writerow(row)

    # 국적 단위로 접기 --------------------------------------------------------
    by_nat = {}
    for (ds, sp, nat, lang, prof, tk), rec in total.items():
        n = by_nat.get(nat)
        if n is None:
            n = {"n_files": 0, "n_err_files": 0, "phones": 0,
                 "words": 0, "tags": collections.Counter()}
            by_nat[nat] = n
        n["n_files"] += rec["n_files"]
        n["n_err_files"] += rec["n_err_files"]
        n["phones"] += rec["phones"]
        n["words"] += rec["words"]
        for t, c in rec["tags"].items():
            n["tags"][t] += c

    # 2) by_nationality_tag_long.csv ----------------------------------------
    p2 = os.path.join(out_dir, "by_nationality_tag_long.csv")
    with open(p2, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["nationality", "error_tag", "count",
                    "rate_per_100phones", "rate_per_utterance"])
        for nat in sorted(by_nat):
            rec = by_nat[nat]
            ph = rec["phones"] or 1
            nf = rec["n_files"] or 1
            for t in all_tags:
                c = rec["tags"].get(t, 0)
                w.writerow([nat, t, c,
                            round(c / ph * 100, 4),
                            round(c / nf, 4)])

    # 3) by_nationality_pivot.csv -------------------------------------------
    p3 = os.path.join(out_dir, "by_nationality_pivot.csv")
    with open(p3, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["nationality", "n_files", "total_phones", "total_errors"] + all_tags)
        for nat in sorted(by_nat):
            rec = by_nat[nat]
            w.writerow([nat, rec["n_files"], rec["phones"], sum(rec["tags"].values())]
                       + [rec["tags"].get(t, 0) for t in all_tags])

    # 4) nationality_summary.csv --------------------------------------------
    p4 = os.path.join(out_dir, "nationality_summary.csv")
    with open(p4, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["nationality", "n_files", "n_err_files", "err_file_pct",
                    "total_phones", "total_errors", "errors_per_100phones",
                    "top1", "top2", "top3"])
        for nat in sorted(by_nat, key=lambda x: -by_nat[x]["n_files"]):
            rec = by_nat[nat]
            nf = rec["n_files"] or 1
            ph = rec["phones"] or 1
            top = collections.Counter(rec["tags"]).most_common(3)
            top = [f"{t}:{c}" for t, c in top] + ["", "", ""]
            w.writerow([nat, rec["n_files"], rec["n_err_files"],
                        round(rec["n_err_files"] / nf * 100, 2),
                        rec["phones"], sum(rec["tags"].values()),
                        round(sum(rec["tags"].values()) / ph * 100, 4),
                        top[0], top[1], top[2]])

    return [p1, p2, p3, p4], all_tags, by_nat


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    default_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # scripts/ 하위 -> 데이터 루트는 부모
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=default_root, help="데이터셋 최상위 폴더")
    ap.add_argument("--out", default=None, help="출력 폴더 (기본: ROOT/_error_tag_analysis)")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="처리할 데이터셋 코드 (예: 040 043). 미지정 시 전체")
    ap.add_argument("--split", choices=["Training", "Validation"], default=None,
                    help="특정 split 만 처리")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--include-dup", action="store_true",
                    help="' (2)' 중복 폴더도 포함")
    args = ap.parse_args()

    root = args.root
    out_dir = args.out or os.path.join(root, "_error_tag_analysis")
    datasets = set(args.datasets) if args.datasets else None

    folders = find_pron_folders(root, datasets, args.split, args.include_dup)
    print(f"[탐색] pronunciation_lab 폴더 {len(folders)}개")
    for fo in folders:
        print("   -", os.path.relpath(fo, root))
    if not folders:
        print("처리할 폴더가 없습니다. --root / --datasets 옵션을 확인하세요.")
        return

    total = {}
    n_files = n_bad = 0
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = []
        # 백프레셔: 동시에 너무 많은 future 를 쌓지 않도록 제한
        pending = set()
        max_inflight = args.workers * 4
        chunk_gen = iter_chunks(folders)

        def submit_next():
            try:
                paths = next(chunk_gen)
            except StopIteration:
                return False
            pending.add(ex.submit(process_chunk, paths))
            return True

        for _ in range(max_inflight):
            if not submit_next():
                break

        while pending:
            done = next(as_completed(pending))
            pending.discard(done)
            part, nread, bad = done.result()
            merge_into(total, part)
            n_files += nread
            n_bad += bad
            submit_next()
            if n_files // 50000 != (n_files - nread) // 50000:
                rate = n_files / max(time.time() - t0, 1e-6)
                print(f"   ...처리 {n_files:,}개  (bad {n_bad})  {rate:,.0f} files/s")

    print(f"[완료] 총 {n_files:,}개 처리 (파싱실패 {n_bad}), {time.time()-t0:.1f}s")

    paths, all_tags, by_nat = write_outputs(total, out_dir)
    print(f"[출력] {out_dir}")
    for p in paths:
        print("   -", os.path.basename(p))

    # 콘솔 요약: 국적별 파일수 상위 + 전체 태그 분포
    print("\n[국적별 파일 수 상위 15]")
    for nat in sorted(by_nat, key=lambda x: -by_nat[x]["n_files"])[:15]:
        rec = by_nat[nat]
        te = sum(rec["tags"].values())
        r = te / (rec["phones"] or 1) * 100
        print(f"   {nat:28s} files={rec['n_files']:>8,}  errors={te:>8,}  "
              f"per100ph={r:5.2f}")

    print("\n[error_tag 종류]", ", ".join(all_tags))


if __name__ == "__main__":
    main()
