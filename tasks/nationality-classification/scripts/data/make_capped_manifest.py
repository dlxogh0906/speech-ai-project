# -*- coding: utf-8 -*-
"""다운샘플링 실험용 — 학습 목록(train.csv)에서 클래스마다 행 수를 상한 이하로 줄인 매니페스트 폴더를 만든다.

대조군(원본 폴더)과 **클래스 비율만** 다르게 하려고 이렇게 줄인다.
  - 클래스 c 의 목표 행 수 = min(원래 행 수, --cap). --cap min 이면 가장 적은 클래스의 행 수 — 모든 클래스가 같아진다
  - 클래스 안 출처(domain: web040~043 · p505 · kspon · zeroth …) 비율은 그대로 둔다 — 출처마다 같은 비율로 줄인다
  - 출처 안에서는 화자를 번갈아 한 발화씩 뽑는다(단어 과제 RW 는 화자 안에서 뒤로) — 화자를 최대한 남긴다
  - valid · test 는 그대로 복사한다(같은 시험지)
  - 대조군과 같은 최대 스텝을 주려면 몇 epoch 이 필요한지 cap_meta.json 의 matched_epochs 에 적는다.
    finetune_full_encoder.py 와 같은 길이 버킷 배치(max_sec · max_batch_sec · max_batch · grad_accum)로 센다.

    $PY scripts/data/make_capped_manifest.py --cap 2000            # → <src>_cap2k
    $PY scripts/data/make_capped_manifest.py --src <폴더> --cap 10000 --dst <폴더>
    $PY scripts/data/make_capped_manifest.py --src <폴더> --cap min                # → <src>_capmin
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import NAT_OUTPUTS

import argparse, collections, csv, json, math, random, shutil, time
from pathlib import Path

COPY = ("valid.csv", "test.csv", "test_koraug_zeroth.csv", "speaker_split.json", "prep_meta.json", "restore_meta.json")


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        return r.fieldnames, list(r)


def quotas(sizes, target):
    """출처별 행 수 → 합이 target 인 출처별 몫 (같은 비율, 최대 나머지 반올림)."""
    total = sum(sizes.values())
    if target >= total:
        return dict(sizes)
    raw = {k: n * target / total for k, n in sizes.items()}
    q = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: q[k] - raw[k])[:target - sum(q.values())]:
        q[k] += 1
    return q


def pick(rows, idx, quota, rng):
    """한 (클래스, 출처) 안에서 화자를 번갈아 quota 개 → 행 번호."""
    by_spk = collections.defaultdict(list)
    for i in idx:
        by_spk[rows[i]["speaker"]].append(i)
    order = sorted(by_spk)
    rng.shuffle(order)
    queues = {}
    for s in order:
        first = [i for i in by_spk[s] if rows[i]["task"] != "RW"]
        last = [i for i in by_spk[s] if rows[i]["task"] == "RW"]
        rng.shuffle(first); rng.shuffle(last)
        queues[s] = collections.deque(first + last)
    out = []
    while len(out) < quota:
        for s in order:
            if queues[s]:
                out.append(queues[s].popleft())
                if len(out) >= quota:
                    break
        order = [s for s in order if queues[s]]
        if not order:
            break
    return out


def steps_per_epoch(rows, a):
    from country_classifier.kd_student import LengthBucketBatchSampler
    bs = LengthBucketBatchSampler([float(r["duration_sec"]) for r in rows], a.max_sec, a.max_batch_sec,
                                  a.max_batch, shuffle=True, seed=0)
    return math.ceil(len(bs) / a.grad_accum)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=f"{NAT_OUTPUTS}/webmic_kd/lab131full_zeroth_r16k", help="대조군 매니페스트 폴더")
    ap.add_argument("--cap", required=True, help="클래스당 최대 학습 행 (min = 가장 적은 클래스의 행 수)")
    ap.add_argument("--dst", default="", help="기본 <src>_cap<N>k")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ref_epochs", type=int, default=5, help="대조군 epoch — 이만큼의 스텝에 맞춘 epoch 을 계산한다")
    ap.add_argument("--max_sec", type=float, default=8.0, help="아래 넷은 finetune_full_encoder.py 기본값과 같아야 한다")
    ap.add_argument("--max_batch_sec", type=float, default=64.0)
    ap.add_argument("--max_batch", type=int, default=24)
    ap.add_argument("--grad_accum", type=int, default=4)
    a = ap.parse_args()

    src = Path(a.src)
    fields, rows = read_csv(src / "train.csv")
    cap = min(collections.Counter(r["label_en"] for r in rows).values()) if a.cap == "min" else int(a.cap)
    name = "capmin" if a.cap == "min" else f"cap{cap // 1000}k" if cap % 1000 == 0 else f"cap{cap}"
    dst = Path(a.dst or f"{str(src).rstrip('/')}_{name}")

    groups = collections.defaultdict(lambda: collections.defaultdict(list))     # 클래스 → 출처 → 행 번호
    for i, r in enumerate(rows):
        groups[r["label_en"]][r["domain"]].append(i)
    rng = random.Random(a.seed)
    keep, per_class = [], {}
    for c in sorted(groups):
        sizes = {d: len(v) for d, v in groups[c].items()}
        q = quotas(sizes, min(sum(sizes.values()), cap))
        took = []
        for d in sorted(groups[c]):
            took += pick(rows, groups[c][d], q[d], rng)
        keep += took
        per_class[c] = {"rows_src": sum(sizes.values()), "rows_dst": len(took),
                        "spk_src": len({rows[i]["speaker"] for v in groups[c].values() for i in v}),
                        "spk_dst": len({rows[i]["speaker"] for i in took})}
    keep.sort()                                                                  # 원래 행 순서
    out_rows = [rows[i] for i in keep]

    dst.mkdir(parents=True, exist_ok=True)
    with open(dst / "train.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(out_rows)
    for fn in COPY:
        if (src / fn).is_file():
            shutil.copy2(src / fn, dst / fn)

    s_src, s_dst = steps_per_epoch(rows, a), steps_per_epoch(out_rows, a)
    hours = lambda rs: round(sum(min(float(r["duration_sec"] or 0), a.max_sec) for r in rs) / 3600, 1)
    sizes_dst = sorted(v["rows_dst"] for v in per_class.values())
    meta = {"src": str(src), "cap": cap, "cap_arg": a.cap, "seed": a.seed, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "rows": {"src": len(rows), "dst": len(out_rows)},
            "hours_le_max_sec": {"src": hours(rows), "dst": hours(out_rows)},
            "speakers": {"src": sum(v["spk_src"] for v in per_class.values()),
                         "dst": sum(v["spk_dst"] for v in per_class.values())},
            "rows_per_class_dst": {"min": sizes_dst[0], "max": sizes_dst[-1]},
            "steps_per_epoch": {"src": s_src, "dst": s_dst},
            "ref_epochs": a.ref_epochs, "matched_epochs": max(1, round(a.ref_epochs * s_src / s_dst)),
            "per_class": per_class,
            "note": "클래스 안 출처 비율은 원본과 같다. valid·test 는 원본 복사. matched_epochs 는 대조군 ref_epochs 와 "
                    "최대 스텝을 맞춘 값(조기 종료 규칙은 같게 둔다)."}
    (dst / "cap_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    lost = meta["speakers"]["src"] - meta["speakers"]["dst"]
    print(f"[cap {cap:,}] 학습 {len(rows):,} → {len(out_rows):,}행 · {meta['hours_le_max_sec']['src']} → "
          f"{meta['hours_le_max_sec']['dst']} 시간 · 화자 {meta['speakers']['dst']:,}/{meta['speakers']['src']:,} "
          f"(빠짐 {lost:,}) · 클래스당 {sizes_dst[0]:,}~{sizes_dst[-1]:,}행 · 스텝/epoch {s_src:,} → {s_dst:,} · "
          f"대조군 {a.ref_epochs} epoch 와 같은 스텝 = {meta['matched_epochs']} epoch → {dst}", flush=True)


if __name__ == "__main__":
    main()
