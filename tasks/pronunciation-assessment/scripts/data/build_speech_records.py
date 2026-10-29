# -*- coding: utf-8 -*-
"""040~043 Speech 라벨(LAR) → records.jsonl — l1prior `s1_run_batch.py` 와 **같은 워커**를 지금 AIHub 배치에서 돌린다.

`s1_run_batch.py` 는 옛 폴더 이름("040.교육용 영어 …", 공백)과 저장소 기준 경로를 가정해 이 서버에서 폴더를 못 찾는다.
여기서는 aihubshell 이 푼 이름("040.교육용_영어_…")을 glob 으로 찾고, 레코드는 그 스크립트의 `_worker`
(→ `s1_speechlab_mdd.build_record`: 간투사 정제 → KoG2P → 정렬 → quality_flag)를 그대로 부른다.

출력은 원래와 같은 자리·이름이다: `$MDD_ANALYSIS/<groups>_<split>/records.jsonl` (+ quality_report.json).
`build_lar_mix.py`(무오류 LAR 선별)·`eval_word_skip_record.py` 가 이 파일을 읽는다.
문서 기준값: 4개 언어군 Train+Val LAR **268,533건** (docs/L1prior_인수인계.md).

    $PY tasks/pronunciation-assessment/scripts/data/build_speech_records.py --groups 040 041 042 043 --workers 12
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import DATA_ROOT, MDD_ANALYSIS

import argparse, collections, glob, json, os, time
import multiprocessing as mp


def find_dirs(root, groups, splits):
    out = []
    for g in groups:
        bases = glob.glob(os.path.join(root, f"{g}.*", "01-1.정식개방데이터"))
        if len(bases) != 1:
            raise SystemExit(f"{g} 데이터셋 폴더를 하나로 못 찾았다: {bases}")
        for sp in splits:
            pre = "TL" if sp == "Training" else "VL"
            d = os.path.join(bases[0], sp, "02.라벨링데이터", f"{pre}_Speech_lab")
            if not os.path.isdir(d):
                raise SystemExit(f"라벨 폴더가 없다: {d}")
            out.append((g, sp, d))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--groups", nargs="+", default=["040", "041", "042", "043"])
    ap.add_argument("--split", default="both", choices=["Training", "Validation", "both"])
    ap.add_argument("--aihub_root", default=str(DATA_ROOT / "aihub"))
    ap.add_argument("--outroot", default=str(MDD_ANALYSIS))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    from mdd.l1prior.s1_run_batch import _worker          # 원래 배치의 워커 그대로

    splits = ["Training", "Validation"] if a.split == "both" else [a.split]
    files, per = [], collections.Counter()
    for g, sp, d in find_dirs(a.aihub_root, a.groups, splits):
        fs = sorted(glob.glob(os.path.join(d, "*LAR*.json")))
        files += fs; per[f"{g}/{sp}"] = len(fs)
        print(f"  {g}/{sp}: LAR {len(fs):,}", flush=True)
    tag = "_".join(a.groups) + "_" + a.split
    outdir = os.path.join(a.outroot, tag)
    out = os.path.join(outdir, "records.jsonl")
    if os.path.exists(out) and not a.overwrite:
        raise SystemExit(f"[stop] {out} 가 이미 있다 — --overwrite 로만 다시 만든다")
    os.makedirs(outdir, exist_ok=True)
    print(f"[*] LAR 라벨 {len(files):,}건 · workers {a.workers} → {out}", flush=True)

    t0 = time.time(); flags = collections.Counter(); n_ok = n_none = n_err = 0; errs = []
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fo, mp.Pool(a.workers) as pool:
        for i, rec in enumerate(pool.imap_unordered(_worker, files, chunksize=64), 1):
            if rec is None:
                n_none += 1
            elif "_error" in rec:
                n_err += 1
                if len(errs) < 20:
                    errs.append(rec)
            else:
                fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_ok += 1; flags[rec["quality_flag"]] += 1
            if i % 20000 == 0:
                print(f"    {i:,}/{len(files):,} ok={n_ok:,} skip={n_none:,} err={n_err:,} ({time.time()-t0:.0f}s)", flush=True)
    os.replace(tmp, out)
    rep = {"groups": a.groups, "split": a.split, "label_files": len(files), "per_group_split": dict(per),
           "records": n_ok, "skipped_no_orthographic": n_none, "errors": n_err, "error_examples": errs,
           "quality_flags": dict(flags), "doc_reference_LAR_total": 268533, "sec": round(time.time() - t0, 1)}
    json.dump(rep, open(os.path.join(outdir, "quality_report.json"), "w"), ensure_ascii=False, indent=1)
    print(f"[완료] records {n_ok:,} · 전사없음 {n_none:,} · 오류 {n_err:,} · 품질 {dict(flags)} · "
          f"(문서 LAR 268,533) · {rep['sec']}s → {out}", flush=True)
    return 1 if n_err else 0


if __name__ == "__main__":
    _sys.exit(main())
