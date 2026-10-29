# -*- coding: utf-8 -*-
"""화자 단위 train : valid : test = 8 : 1 : 1 매니페스트 폴더를 만든다 (2026-10-19).

원본(lab131full_zeroth_r16k)은 이미 화자 단위로 나뉘어 있다 — 화자 비율 train 85% · valid 5% · test 10%, 겹치는 화자 0.
여기서는 test 를 그대로 두고 valid 만 화자의 10% 로 늘린다.
  - test 화자 · test 행은 그대로 둔다(test.csv · test_koraug_zeroth.csv 복사). run_zeroth 와 그 앞 학습들도 이 화자를
    학습에 쓰지 않았으므로, 같은 시험지로 run_zeroth 와 비교할 수 있다.
  - (클래스, 출처)마다 valid 목표 = round(그 칸의 전체 화자 × --valid_frac) (화자 5명 이상이면 최소 1명).
    모자라는 만큼 train 화자를 무작위로 valid 로 옮긴다. 옮긴 화자의 train 행은 전부 train 에서 뺀다.
    valid 에는 옛 valid 와 같은 몫만 뽑아 넣는다 — 웹: 단어(RW) 외 6 · 단어 2 / 131: 5 / Kspon · Zeroth: 5.
    나머지 행은 어디에도 쓰지 않는다.
  - 출처는 web040~043 · p505 · Korea 의 KSPON · ZEROTH · K001 이다. K001 은 증강본(kor_aug)만 있어 옮기지 않는다.
  - KsponSpeech 는 화자 ID 가 없어 1,000발화 폴더(묶음)를 화자로 쓴다. 대화가 이웃 묶음에 걸칠 수 있으므로,
    valid · test 묶음과 번호가 ±1 인 train 묶음을 train 에서 뺀다(kspon · kor_aug 둘 다).
  - 같은 화자 · 같은 파일이 두 split 에 있거나, 위 Kspon 규칙이 깨지면 아무것도 쓰지 않고 실패한다.

    $PY tasks/nationality-classification/scripts/data/make_speaker_split.py            # → <src>_spk811
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import NAT_OUTPUTS

import argparse, collections, csv, json, os, random, re, shutil, time
from pathlib import Path

TESTS = ("test.csv", "test_koraug_zeroth.csv")
COPY = ("prep_meta.json", "restore_meta.json")
KSPON = re.compile(r"^KSPON:KsponSpeech_(\d+)$")


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        return r.fieldnames, list(r)


def group_of(r):
    """(클래스, 출처) 칸의 출처. Korea 는 화자 ID 앞머리(KSPON · ZEROTH · K001)로 나눈다 — kor_aug 와 kspon 이 같은 묶음을 쓴다."""
    if r["label_en"] == "Korea":
        return r["speaker"].split(":", 1)[0]
    return r["domain"]


def valid_take(rows, idx, g, rng):
    """옮긴 화자 한 명의 train 행 번호 → valid 에 넣을 행 번호 (옛 valid 와 같은 몫)."""
    if g.startswith("web"):
        other = [i for i in idx if rows[i]["task"] != "RW"]
        word = [i for i in idx if rows[i]["task"] == "RW"]
        return rng.sample(other, min(6, len(other))) + rng.sample(word, min(2, len(word)))
    if g == "KSPON":
        idx = [i for i in idx if rows[i]["domain"] == "kspon"]            # 증강본(kor_aug)은 valid 에 안 넣는다
    return rng.sample(idx, min(5, len(idx)))


def hours(rs):
    return round(sum(float(r["duration_sec"] or 0) for r in rs) / 3600, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=f"{NAT_OUTPUTS}/webmic_kd/lab131full_zeroth_r16k", help="원본 매니페스트 폴더")
    ap.add_argument("--dst", default="", help="기본 <src>_spk811")
    ap.add_argument("--valid_frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    src = Path(a.src)
    dst = Path(a.dst or f"{str(src).rstrip('/')}_spk811")
    fields, tr = read_csv(src / "train.csv")
    _, va = read_csv(src / "valid.csv")
    tests = {fn: read_csv(src / fn)[1] for fn in TESTS}
    te_all = [r for fn in TESTS for r in tests[fn]]

    # 화자 → 칸 · 원래 split
    cell, home = {}, {}
    for split, rs in (("train", tr), ("valid", va), ("test", te_all)):
        for r in rs:
            c = (r["label_en"], group_of(r))
            if cell.setdefault(r["speaker"], c) != c:
                raise SystemExit(f"[split] 한 화자가 두 칸에 있다 — {c} / {cell[r['speaker']]}")
            if home.setdefault(r["speaker"], split) != split:
                raise SystemExit(f"[split] 원본에서 이미 한 화자가 {home[r['speaker']]} · {split} 에 있다")

    # valid 로 옮길 train 화자
    by_cell = collections.defaultdict(lambda: collections.defaultdict(list))
    for s, c in cell.items():
        by_cell[c][home[s]].append(s)
    rng = random.Random(a.seed)
    moved = set()
    for c in sorted(by_cell):
        if c[1] == "K001":
            continue
        n = sum(len(v) for v in by_cell[c].values())
        want = round(n * a.valid_frac)
        if n >= 5:
            want = max(1, want)
        pool = sorted(by_cell[c]["train"])
        k = min(max(0, want - len(by_cell[c]["valid"])), len(pool) - 1)
        if k > 0:
            moved.update(rng.sample(pool, k))

    # 행 나누기
    tr_idx = collections.defaultdict(list)
    for i, r in enumerate(tr):
        tr_idx[r["speaker"]].append(i)
    new_va = list(va)
    for s in sorted(moved):
        new_va += [tr[i] for i in sorted(valid_take(tr, tr_idx[s], cell[s][1], rng))]
    keep = [r for r in tr if r["speaker"] not in moved]

    # Kspon 묶음 이웃 빼기
    bundle = lambda s: int(KSPON.match(s).group(1)) if KSPON.match(s) else None
    held = {bundle(r["speaker"]) for r in new_va + te_all} - {None}
    near = {b + d for b in held for d in (-1, 1)} - held
    before = len(keep)
    drop_b = {bundle(r["speaker"]) for r in keep if bundle(r["speaker"]) in near}
    keep = [r for r in keep if bundle(r["speaker"]) not in near]
    kspon_dropped = before - len(keep)

    # 누수 검사 — 하나라도 걸리면 쓰지 않는다
    spk = {"train": {r["speaker"] for r in keep}, "valid": {r["speaker"] for r in new_va}, "test": {r["speaker"] for r in te_all}}
    stem = lambda p: os.path.basename(p).rsplit(".", 1)[0]
    files = {"train": [stem(r["audio_path"]) for r in keep], "valid": [stem(r["audio_path"]) for r in new_va],
             "test": [stem(r["audio_path"]) for r in te_all]}
    bad = []
    for x, y in (("train", "valid"), ("train", "test"), ("valid", "test")):
        if spk[x] & spk[y]:
            bad.append(f"화자 {x}∩{y} {len(spk[x] & spk[y])}명")
        if set(files[x]) & set(files[y]):
            bad.append(f"파일 {x}∩{y} {len(set(files[x]) & set(files[y]))}개")
    for x in ("train", "valid"):
        if len(files[x]) != len(set(files[x])):
            bad.append(f"{x} 안 중복 파일 {len(files[x]) - len(set(files[x]))}개")
    for fn in TESTS:
        fs = [stem(r["audio_path"]) for r in tests[fn]]
        if len(fs) != len(set(fs)):
            bad.append(f"{fn} 안 중복 파일")
    tr_b = {bundle(s) for s in spk["train"]} - {None}
    if {b + d for b in held for d in (-1, 0, 1)} & tr_b:
        bad.append("Kspon train 묶음이 valid·test 묶음과 이웃한다")
    if {r["label_en"] for r in new_va + te_all} - {r["label_en"] for r in keep}:
        bad.append("valid·test 에만 있는 클래스")
    if bad:
        raise SystemExit("[split] 누수 검사 실패 — 쓰지 않았다: " + " · ".join(bad))

    # 클래스별 화자 비율
    per_class = {}
    for c in sorted({r["label_en"] for r in keep}):
        n = {k: sum(1 for s in v if cell[s][0] == c) for k, v in spk.items()}
        tot = sum(n.values())
        per_class[c] = {**n, "frac": {k: round(v / tot, 3) for k, v in n.items()}}
    nspk = {k: len(v) for k, v in spk.items()}
    allspk = sum(nspk.values())

    dst.mkdir(parents=True, exist_ok=True)
    for fn, rs in (("train.csv", keep), ("valid.csv", new_va)):
        with open(dst / fn, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader(); w.writerows(rs)
    for fn in TESTS + COPY:
        if (src / fn).is_file():
            shutil.copy2(src / fn, dst / fn)
    (dst / "speaker_split.json").write_text(json.dumps(
        {"_설명": "화자 → split (make_speaker_split.py). 서버 밖으로 내보내지 말 것 — AIHub 화자 정보",
         "assign": {s: k for k, v in spk.items() for s in sorted(v)}}, ensure_ascii=False), encoding="utf-8")
    meta = {"src": str(src), "seed": a.seed, "valid_frac": a.valid_frac, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "speakers": nspk, "speaker_frac": {k: round(v / allspk, 3) for k, v in nspk.items()},
            "rows": {"train": len(keep), "valid": len(new_va), **{fn: len(tests[fn]) for fn in TESTS}},
            "hours": {"train": hours(keep), "valid": hours(new_va), **{fn: hours(tests[fn]) for fn in TESTS}},
            "classes": {"train": len(per_class), "valid": len({r["label_en"] for r in new_va}),
                        "test": len({r["label_en"] for r in te_all})},
            "moved_train_to_valid": len(moved), "train_rows_removed_with_moved": len(tr) - before,
            "kspon_bundles_dropped_near_heldout": len(drop_b), "kspon_rows_dropped": kspon_dropped,
            "leak_checks": "화자 · 파일 split 간 0 · split 안 중복 0 · Kspon 이웃 묶음 0",
            "per_class": per_class}
    (dst / "split_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[split] 화자 train {nspk['train']:,} · valid {nspk['valid']:,} · test {nspk['test']:,} "
          f"({meta['speaker_frac']['train']:.1%} · {meta['speaker_frac']['valid']:.1%} · {meta['speaker_frac']['test']:.1%}) · "
          f"행 train {len(tr):,} → {len(keep):,} · valid {len(va):,} → {len(new_va):,} · test 그대로 · "
          f"옮긴 화자 {len(moved):,} · Kspon 이웃 묶음 {len(drop_b)}개({kspon_dropped:,}행) 뺌 · 누수 검사 통과 → {dst}", flush=True)


if __name__ == "__main__":
    main()
