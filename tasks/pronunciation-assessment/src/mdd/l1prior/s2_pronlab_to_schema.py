# -*- coding: utf-8 -*-
"""
s2_pronlab_to_schema.py
~~~~~~~~~~~~~~~~~~~~~~~~
pronunciation_lab(gold) 을 Speech_lab 과 **같은 통합 오류 스키마**로 적재한다.
(작업지시서 5절 · 로드맵 갈래 A "위치·유형은 error_tags")

pronunciation_lab 1건 구조(실측 확인):
  RecordingMetadata.prompt                 낭독 지문(정답 텍스트)
  RecordingMetadata.phonemic.phones[]      {phone: 자모, start, end}  ← 표준음 + 타임스탬프
  RecordingMetadata.phonemic.words[]       {word, start, end}
  RecordingMetadata.phonemic.error_tags[]  {error_tag, start, end}    ← gold(위치=시간창만)

핵심 처리:
  - error_tags 는 phone 인덱스를 직접 주지 않는다 → **시간창 겹침**으로 position 매핑.
  - phones 티어는 '표준음'이므로 actual_phone(실현형)은 gold 에 없다 → null.
    (실현형은 Speech_lab LAR MDD 담당. 두 소스는 source 로 구분·상보.)
  - target_phone = 겹친 phone 자모 → KoG2P 기호(s0_symbol_map).

출력(_mdd_analysis/<tag>_pronlab/):
  records_pronlab.jsonl          통합 스키마(전량, source=pronun_lab)
  l1_errortype_counts.csv        L1 × error_type 카운트(gold)
  l1_utterance_labels.csv        발화별 (L1, 있는 error_type 집합) — B1 벤치마크 입력
  l1_summary.csv                 L1별 파일/오류/phones
  pron_quality_report.csv        태그 유무 분포

실행(PowerShell, 프로젝트 루트):
  $env:PYTHONUTF8=1
  python s2_pronlab_to_schema.py --groups 040 041 042 043 --split both --workers 10
데모(파일 몇 개만 출력):
  python s2_pronlab_to_schema.py --demo --n 3
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mdd.l1prior.s0_kog2p_wrapper import text_to_phones
from mdd.l1prior.s0_symbol_map import jamo_seq_to_kog2p

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # scripts/ 하위 -> 데이터 루트는 부모
GROUP_DIRS = {
    "040": "040.교육용 영어 모국어 사용자의 한국어 음성 데이터",
    "041": "041.교육용 유럽어 모국어 사용자의 한국어 음성 데이터",
    "042": "042.교육용 중·일어 모국어 사용자의 한국어 음성 데이터",
    "043": "043.교육용 아시아어(중·일어 제외) 사용자의 한국어 음성 데이터",
}
# core 14(기타·간투사·반복 제외) — 기존 gold 통계 기준과 동일
from mdd.l1prior.s0_taxonomy import CORE_TAGS, SUPRASEG_FLUENCY  # noqa: E402
DROP_TAGS = {"기타", "간투사", "반복"}


def _overlap(a0, a1, b0, b1):
    """두 시간창 [a0,a1],[b0,b1] 겹치면 겹친 길이(>0), 아니면 0."""
    lo, hi = max(a0, b0), min(a1, b1)
    return max(0.0, hi - lo)


def build_pron_record(rec, fname=None):
    """pronunciation_lab dict -> 통합 스키마 레코드(dict)."""
    sm = rec.get("SpeakerMetadata", {}) or {}
    rm = rec.get("RecordingMetadata", {}) or {}
    ev = rec.get("EvaluationMetadata", {}) or {}
    phon = rm.get("phonemic", {}) or {}
    prompt = rm.get("prompt", "") or ""

    phones = phon.get("phones") or []      # [{phone,start,end}]
    etags = phon.get("error_tags") or []   # [{error_tag,start,end}]

    # phones 자모열 -> KoG2P 기호(순서 보존, 타임스탬프 유지)
    jamo = [p.get("phone") for p in phones]
    kog = jamo_seq_to_kog2p([j if j else "?" for j in jamo])  # [(sym,src,role)]
    phone_tl = []   # (idx, sym, jamo, start, end)
    for idx, (p, ks) in enumerate(zip(phones, kog)):
        phone_tl.append((idx, ks[0], p.get("phone"),
                         float(p.get("start", 0) or 0), float(p.get("end", 0) or 0)))

    target = list(text_to_phones(prompt))   # KoG2P 정답열(공통 축)

    errors = []
    tag_types = []          # 이 발화에 존재하는 error_type(중복 포함)
    for e in etags:
        tag = e.get("error_tag")
        if not tag:
            continue
        tag_types.append(tag)
        t0 = float(e.get("start", 0) or 0)
        t1 = float(e.get("end", 0) or 0)
        # 시간창 겹치는 phone 인덱스들
        hits = [(idx, sym, jm, ov) for (idx, sym, jm, s, en) in phone_tl
                if (ov := _overlap(t0, t1, s, en)) > 0]
        hits.sort(key=lambda x: -x[3])   # 겹침 큰 순
        if hits:
            pos = hits[0][0]
            tphone = hits[0][1]
            tjamo = hits[0][2]
            span_idx = sorted(h[0] for h in hits)
        else:
            pos = None; tphone = None; tjamo = None; span_idx = []
        errors.append({
            "pos": pos, "target_phone": tphone, "target_jamo": tjamo,
            "actual_phone": None,          # gold 미기록(실현형은 Speech_lab)
            "op": None,
            "error_type": tag,
            "t_start": round(t0, 3), "t_end": round(t1, 3),
            "span_phone_idx": span_idx,
            "is_core": tag in CORE_TAGS,
            "is_suprseg": tag in SUPRASEG_FLUENCY,
            "source": "pronun_lab", "confidence": "gold",
            "rationale": "gold error_tag; pos=시간창 겹침 매핑",
        })

    core_types = sorted({t for t in tag_types if t in CORE_TAGS})

    return {
        "utterance_id": rec.get("sound_id"),
        "file": fname,
        "task": (fname.split("-")[5][:3] if fname else None),
        "user_id": rec.get("UserID"),
        "L1": sm.get("nationality"), "language": sm.get("language"),
        "proficiency": sm.get("proficiency"),
        "recording_level": sm.get("recording_level"),
        "topik_level": sm.get("topik_level"),
        "prompt": prompt,
        "orthographic_raw": None, "orthographic_clean": None,
        "disfluencies": [],
        "target_phones": target,
        "actual_phones": None,             # gold 실현형 없음
        "n_target": len(target),
        "n_phones_tl": len(phone_tl),
        "n_errors": len(errors),
        "n_core_errors": sum(1 for e in errors if e["is_core"]),
        "err_types_core": core_types,
        "has_error": bool(tag_types),
        "quality_flag": "gold",
        "usable_for_aggregation": True,
        "errors": errors,
        "eval": {"Delivery": None, "LanguageUse": None, "Content": None,
                 "PronunProf": ev.get("PronunProfEval"),
                 "Fluency": ev.get("FluencyEval"),
                 "Comprehend": ev.get("ComprehendEval")},
        "source": "pronun_lab",
    }


# ── 배치 ────────────────────────────────────────────────────────
def find_pron_dirs(groups, split):
    dirs = []
    splits = ["Training", "Validation"] if split == "both" else [split]
    for g in groups:
        base = os.path.join(ROOT, GROUP_DIRS[g], "01-1.정식개방데이터")
        for sp in splits:
            prefix = "TL" if sp == "Training" else "VL"
            p = os.path.join(base, sp, "02.라벨링데이터", f"{prefix}_pronunciation_lab")
            if os.path.isdir(p):
                dirs.append((g, sp, p))
    return dirs


def _worker(fpath):
    try:
        r = json.load(open(fpath, encoding="utf-8-sig"))
    except Exception:
        return None
    if not (r.get("RecordingMetadata", {}) or {}).get("phonemic"):
        return None
    try:
        return build_pron_record(r, os.path.basename(fpath))
    except Exception as e:
        return {"_error": str(e), "file": os.path.basename(fpath)}


def run(groups, split, workers, limit, outroot):
    dirs = find_pron_dirs(groups, split)
    if not dirs:
        print("[!] pronunciation_lab 폴더를 못 찾음. --groups/--split 확인.")
        return
    files = []
    for g, sp, p in dirs:
        fs = sorted(glob.glob(os.path.join(p, "*.json")))
        files.extend(fs)
        print(f"  {g}/{sp}: {len(fs):,}건  ({p})")
    if limit:
        files = files[:limit]
    print(f"[*] 총 처리 대상: {len(files):,}건 | workers={workers}")

    tag = "_".join(groups) + "_" + split + "_pronlab"
    outdir = os.path.join(outroot, tag)
    os.makedirs(outdir, exist_ok=True)

    l1_et = collections.defaultdict(collections.Counter)   # L1 -> error_type -> n
    l1_stat = collections.defaultdict(collections.Counter)
    flag = collections.Counter()
    n_ok = n_err = n_none = 0

    jsonl_path = os.path.join(outdir, "records_pronlab.jsonl")
    lab_path = os.path.join(outdir, "l1_utterance_labels.csv")
    fout = open(jsonl_path, "w", encoding="utf-8")
    import csv
    flab = open(lab_path, "w", newline="", encoding="utf-8-sig")
    wlab = csv.writer(flab)
    wlab.writerow(["utterance_id", "L1", "proficiency", "task",
                   "n_target", "has_error", "err_types_core"])

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
                flag["has_error" if rec["has_error"] else "no_error"] += 1
                st = l1_stat[L1]
                st["files"] += 1
                st["phones"] += rec["n_phones_tl"]
                st["errors"] += rec["n_errors"]
                st["core_errors"] += rec["n_core_errors"]
                if rec["has_error"]:
                    st["err_files"] += 1
                for e in rec["errors"]:
                    l1_et[L1][e["error_type"]] += 1
                wlab.writerow([rec["utterance_id"], L1, rec.get("proficiency"),
                               rec.get("task"), rec["n_target"],
                               int(rec["has_error"]),
                               "|".join(rec["err_types_core"])])
            if i % 20000 == 0:
                print(f"    ... {i:,}/{len(files):,}  (ok={n_ok:,} skip={n_none:,} err={n_err:,})")
    fout.close(); flab.close()

    def wcsv(name, header, rows):
        with open(os.path.join(outdir, name), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f); w.writerow(header); w.writerows(rows)

    all_et = sorted({et for c in l1_et.values() for et in c})
    rows = [[L1] + [l1_et[L1].get(et, 0) for et in all_et] for L1 in sorted(l1_et)]
    wcsv("l1_errortype_counts.csv", ["L1"] + all_et, rows)

    rows = []
    for L1, st in sorted(l1_stat.items()):
        rows.append([L1, st["files"], st["err_files"], st["phones"],
                     st["errors"], st["core_errors"],
                     round(st["errors"] / (st["phones"] or 1) * 100, 4)])
    wcsv("l1_summary.csv",
         ["L1", "files", "err_files", "total_phones", "total_errors",
          "core_errors", "errors_per_100phones"], rows)

    wcsv("pron_quality_report.csv", ["flag", "count"],
         [[k, v] for k, v in flag.most_common()])

    print("\n[완료] 출력:", outdir)
    print(f"  레코드: {n_ok:,} | 스킵: {n_none:,} | 에러: {n_err:,}")
    print(f"  태그유무: {dict(flag)}")
    print(f"  L1(국적): {len(l1_stat)} | error_type: {len(all_et)}")


def _demo(n):
    import glob as _g
    d = os.path.join(ROOT, GROUP_DIRS["040"], "01-1.정식개방데이터",
                     "Training", "02.라벨링데이터", "TL_pronunciation_lab")
    shown = 0
    for f in sorted(_g.glob(os.path.join(d, "*.json"))):
        try:
            r = json.load(open(f, encoding="utf-8-sig"))
        except Exception:
            continue
        rec = build_pron_record(r, os.path.basename(f))
        if not rec["errors"]:
            continue
        print(json.dumps(rec, ensure_ascii=False, indent=2))
        print("=" * 70)
        shown += 1
        if shown >= n:
            break


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--groups", nargs="+", default=["040"], choices=list(GROUP_DIRS))
    ap.add_argument("--split", default="both", choices=["Training", "Validation", "both"])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--outroot", default=os.path.join(ROOT, "_mdd_analysis"))
    a = ap.parse_args()
    if a.demo:
        _demo(a.n)
    else:
        run(a.groups, a.split, a.workers, a.limit, a.outroot)
