# -*- coding: utf-8 -*-
"""AIHub 131(505 도메인) 라벨 인덱스 — 라벨 JSON × 음성 파일 → 41라벨.

라벨 규칙은 배포 모델을 만들 때와 같다(`kd_encoder_prep.py` 의 c40):
    motherTongue → alias(`configs/fine_grained_label_alias_merged6.json`)
    → 영어·스페인어는 화자의 **최다 거주국**으로 나눔(SPLIT: US/GB · PE/CO/ES/CL, 그 밖의 나라는 버림)
    → 그 밖의 모국어는 LANG2C 의 단일 국가(목록에 없는 모국어는 버림) → 영문 헤드 라벨(country41_ko2en.json)
131 화자는 대부분 한국 거주(residence=KR)라 영어·스페인어 화자 상당수가 버려진다 — 옛 규칙 그대로다.

음성은 Training 원천데이터(TS_*) · Validation 원천데이터(VS_*) 아래 wav 를 파일명(stem)으로 잇는다.
라벨은 있는데 음성이 아직 없는 행(다운로드 전)은 `audio_path` 가 빈칸이다 — 매니페스트 단계에서 거른다.
화자 키는 `131:<SpeakerID>` (웹마이크 재학습의 index_131.csv 와 같다).

출력 (--out_dir)
    index_131_labeled.csv   stem,aihub_split,speaker,mother_tongue,country,label_en,audio_path,duration_sec
    index_131_labeled.json  모국어·라벨·버린 행 통계

    $PY tasks/nationality-classification/scripts/data/build_131_index.py --out_dir $NATIONALITY_OUTPUTS/webmic_kd
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import DATA_ROOT, NAT_OUTPUTS, NAT_TASK

import argparse, collections, csv, glob, json, os, time
from multiprocessing import Pool

BYTES_PER_SEC = 16000 * 2


def walk(root, ext):
    out, stack = [], [root]
    while stack:
        d = stack.pop()
        try:
            it = os.scandir(d)
        except OSError:
            continue
        with it:
            for e in it:
                if e.is_dir(follow_symlinks=False):
                    stack.append(e.path)
                elif e.name.endswith(ext):
                    out.append((e.name[: -len(ext)], e.path, e.stat().st_size if ext == ".wav" else 0))
    return out


def _walk_job(a):
    return a, walk(a[0], a[1])


def read_label(path):
    for enc in ("utf-8-sig", "cp949"):
        try:
            j = json.load(open(path, encoding=enc))
            break
        except (UnicodeDecodeError, ValueError):
            j = None
        except OSError:
            return None
    if not isinstance(j, dict):
        return None
    fi = j.get("file_info") or {}
    stem = os.path.splitext(j.get("fileName") or os.path.basename(path))[0]
    return (stem, (j.get("SpeakerID") or fi.get("speakerID") or "").strip(),
            ((j.get("skill_info") or {}).get("motherTongue") or "").strip(),
            ((j.get("residence_info") or {}).get("country") or "").strip())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="")
    ap.add_argument("--out_dir", default=str(NAT_OUTPUTS / "webmic_kd"))
    ap.add_argument("--workers", type=int, default=12)
    A = ap.parse_args()
    from country_classifier.make_country40_probe import SPLIT, LANG2C

    root = A.root or glob.glob(os.path.join(str(DATA_ROOT), "aihub", "131.*", "01.데이터*"))[0]
    alias = json.load(open(NAT_TASK / "configs" / "fine_grained_label_alias_merged6.json", encoding="utf-8"))["aliases"]
    ko2en = json.load(open(NAT_TASK / "configs" / "country41_ko2en.json", encoding="utf-8"))
    t0 = time.time()

    jobs = []
    for sp in ["1.Training", "2.Validation"]:
        for d in sorted(glob.glob(os.path.join(root, sp, "라벨링데이터*", "*"))):
            if os.path.isdir(d):
                jobs.append((d, ".json", sp, "label"))
        for d in sorted(glob.glob(os.path.join(root, sp, "원천데이터*", "*"))):
            if os.path.isdir(d):
                jobs.append((d, ".wav", sp, "audio"))
    labels_files, wav = [], {}
    with Pool(A.workers) as pool:
        for (d, ext, sp, kind), items in pool.imap_unordered(_walk_job, [(j[0], j[1], j[2], j[3]) for j in jobs]):
            if kind == "label":
                labels_files += [(sp, p) for _, p, _ in items]
            else:
                for st, p, sz in items:
                    wav[st] = (p, round(max(sz - 44, 0) / BYTES_PER_SEC, 2))
            print(f"  [scan] {sp} {os.path.basename(d):20} {kind} {len(items):>8,} ({time.time()-t0:.0f}s)", flush=True)

    rows, bad = [], 0
    with Pool(A.workers) as pool:
        for (sp, _), rec in zip(labels_files, pool.imap(read_label, [p for _, p in labels_files], chunksize=512)):
            if rec is None or not rec[1] or not rec[2]:
                bad += 1
                continue
            rows.append((sp,) + rec)
    print(f"  [label] {len(rows):,}행 · 읽기 실패/빈 값 {bad} ({time.time()-t0:.0f}s)", flush=True)

    # 화자별 모국어(최다) · 최다 거주국
    spk_mt, spk_cc, spk_split = collections.defaultdict(collections.Counter), collections.defaultdict(collections.Counter), {}
    for sp, st, spk, mt, cc in rows:
        spk_mt[spk][alias.get(mt, mt)] += 1
        spk_cc[spk][cc] += 1
        spk_split.setdefault(spk, set()).add(sp)
    mt_incons = sum(len(c) > 1 for c in spk_mt.values())

    def label_of(spk):
        mt = spk_mt[spk].most_common(1)[0][0]
        cc = spk_cc[spk].most_common(1)[0][0]
        ko = SPLIT[mt].get(cc) if mt in SPLIT else LANG2C.get(mt)
        return mt, cc, (ko2en[ko] if ko else "")

    stats = collections.Counter()
    drop_mt = collections.Counter()
    spk_by_label = collections.defaultdict(set)
    out_csv = os.path.join(A.out_dir, "index_131_labeled.csv")
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stem", "aihub_split", "speaker", "mother_tongue", "country", "label_en", "audio_path", "duration_sec"])
        for sp, st, spk, _, _ in sorted(rows, key=lambda r: (r[0], r[1])):
            mt, cc, lb = label_of(spk)
            ap_, dur = wav.get(st, ("", ""))
            if not lb:
                drop_mt[f"{mt}/{cc}"] += 1
            else:
                spk_by_label[lb].add(spk)
            stats[(sp, bool(lb), bool(ap_))] += 1
            w.writerow([st, sp, f"131:{spk}", mt, cc, lb, ap_, dur])
    meta = {"rows": len(rows), "speakers": len(spk_mt), "speakers_inconsistent_mother_tongue": mt_incons,
            "speakers_in_both_splits": sum(len(v) > 1 for v in spk_split.values()),
            "rows_by_split_labeled_audio": {f"{a}|라벨{'O' if b else 'X'}|음성{'O' if c else 'X'}": n
                                            for (a, b, c), n in sorted(stats.items())},
            "speakers_per_label": {k: len(v) for k, v in sorted(spk_by_label.items(), key=lambda x: -len(x[1]))},
            "dropped_mother_tongue_country": dict(drop_mt.most_common(40)),
            "wav_indexed": len(wav), "elapsed_sec": round(time.time() - t0)}
    json.dump(meta, open(os.path.join(A.out_dir, "index_131_labeled.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(json.dumps(meta, ensure_ascii=False, indent=1))
    print(f"[saved] {out_csv}")


if __name__ == "__main__":
    main()
