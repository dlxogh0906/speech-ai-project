# -*- coding: utf-8 -*-
"""화자 수 실험용 — 몇 나라의 학습 화자를 N명으로 줄인 매니페스트 폴더를 만든다 (2026-10-22).

꼬리 국가(학습 화자 ≤15)가 안 오르는 까닭이 정말 화자 수인지, 화자를 몇 명 모으면 되는지 재려고,
화자가 많은 나라의 학습 화자를 일부러 줄여서 학습한다. 줄인 나라 말고는 대조군(원본 폴더)과 같다.
  - --spec "Germany=15,Italy=30x10,..." — 나라=화자 수. xK 를 붙이면 화자마다 K행만 쓴다(부스 수집처럼 한 사람 K문장)
  - 화자는 학습 행이 --min_rows 이상인 사람에서 고른다. 고르는 순서는 나라마다 seed 로 한 번 정하고,
    출처(web041 · p505 …) 비율이 앞에서부터 유지되게 섞는다 → 같은 seed 면 15명은 60명의 부분집합이다
    (두 번의 학습에서 같은 나라를 다른 조건으로 줄여도 화자가 겹쳐서 비교가 깨끗하다)
  - xK 는 읽은 문장(RS) → 다른 문장·자유발화 → 단어(RW) 순으로 화자 안에서 K행을 고른다
  - valid · test 는 그대로 복사한다(같은 시험지). 대조군 5 epoch 와 같은 최대 스텝 = abl_meta.json 의 matched_epochs
  - abl_meta.json 에는 나라별 행 · 화자 수만 적는다(화자 ID 는 적지 않는다)

    $PY scripts/data/make_speaker_ablation_manifest.py --src <대조군 폴더> --spec "Germany=15,Japan=30x10" --dst <폴더>
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import argparse, collections, csv, json, math, random, re, shutil, time
from pathlib import Path

COPY = ("valid.csv", "test.csv", "test_koraug_zeroth.csv", "speaker_split.json", "prep_meta.json", "restore_meta.json")


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        return r.fieldnames, list(r)


def parse_spec(spec):
    """"Germany=15,Japan=30x10" → {나라: (화자 수, 화자당 행 수 또는 None)}"""
    out = {}
    for part in filter(None, (p.strip() for p in spec.split(","))):
        m = re.fullmatch(r"(.+?)=(\d+)(?:x(\d+))?", part)
        if not m:
            raise SystemExit(f"--spec 항목을 못 읽었다: {part!r} (예: Germany=15 · Japan=30x10)")
        out[m.group(1).strip()] = (int(m.group(2)), int(m.group(3)) if m.group(3) else None)
    return out


def speaker_order(spk_rows, rows, seed, country):
    """화자 → 고르는 순서. 출처마다 섞은 뒤 (k + u_d) / n_d 위치로 끼워 넣어 앞부분도 출처 비율을 따른다."""
    rng = random.Random(f"{seed}|order|{country}")
    by_dom = collections.defaultdict(list)
    for s in sorted(spk_rows):
        dom = collections.Counter(rows[i]["domain"] for i in spk_rows[s]).most_common(1)[0][0]
        by_dom[dom].append(s)
    pos = []
    for d in sorted(by_dom):
        v = by_dom[d]; rng.shuffle(v); u = rng.random()
        pos += [((k + u) / len(v), d, s) for k, s in enumerate(v)]
    return [s for _, _, s in sorted(pos)]


def pick_rows(idx, rows, k, seed, spk):
    """화자 한 명의 행에서 k개 — 읽은 문장(RS) → 다른 문장·자유발화 → 단어(RW)."""
    rng = random.Random(f"{seed}|rows|{spk}")
    tiers = ([i for i in idx if rows[i]["task"] == "RS"],
             [i for i in idx if rows[i]["task"] not in ("RS", "RW")],
             [i for i in idx if rows[i]["task"] == "RW"])
    out = []
    for t in tiers:
        t = sorted(t); rng.shuffle(t)
        out += t[:k - len(out)]
    return out


def steps_per_epoch(rows, a):
    from country_classifier.kd_student import LengthBucketBatchSampler
    bs = LengthBucketBatchSampler([float(r["duration_sec"]) for r in rows], a.max_sec, a.max_batch_sec,
                                  a.max_batch, shuffle=True, seed=0)
    return math.ceil(len(bs) / a.grad_accum)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="대조군 매니페스트 폴더")
    ap.add_argument("--spec", required=True, help='나라=화자 수[xK], 쉼표로 — 예 "Germany=15,Japan=30x10"')
    ap.add_argument("--dst", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min_rows", type=int, default=10, help="이만큼 학습 행이 있는 화자에서만 고른다")
    ap.add_argument("--ref_epochs", type=int, default=5, help="대조군 epoch — 이만큼의 스텝에 맞춘 epoch 을 계산한다")
    ap.add_argument("--max_sec", type=float, default=8.0, help="아래 넷은 finetune_full_encoder.py 기본값과 같아야 한다")
    ap.add_argument("--max_batch_sec", type=float, default=64.0)
    ap.add_argument("--max_batch", type=int, default=24)
    ap.add_argument("--grad_accum", type=int, default=4)
    a = ap.parse_args()

    src, dst = Path(a.src), Path(a.dst)
    spec = parse_spec(a.spec)
    fields, rows = read_csv(src / "train.csv")
    by_c = collections.defaultdict(lambda: collections.defaultdict(list))      # 나라 → 화자 → 행 번호
    for i, r in enumerate(rows):
        by_c[r["label_en"]][r["speaker"]].append(i)
    missing = [c for c in spec if c not in by_c]
    if missing:
        raise SystemExit(f"학습 목록에 없는 나라: {missing}")

    drop, per_class = set(), {}
    for c, (n, k) in spec.items():
        spk_rows = by_c[c]
        elig = {s: v for s, v in spk_rows.items() if len(v) >= a.min_rows}
        order = speaker_order(elig, rows, a.seed, c)
        if n > len(order):
            print(f"[경고] {c}: 고를 수 있는 화자 {len(order)}명 < {n}명 — 모두 쓴다", flush=True)
        chosen = order[:n]
        keep = []
        for s in chosen:
            keep += pick_rows(spk_rows[s], rows, k, a.seed, s) if k else spk_rows[s]
        keep = set(keep)
        drop |= {i for v in spk_rows.values() for i in v} - keep
        doms = collections.Counter(rows[i]["domain"] for i in keep)
        tasks = collections.Counter(rows[i]["task"] for i in keep)
        per_class[c] = {"cond": f"{n}x{k}" if k else str(n), "spk_src": len(spk_rows), "spk_eligible": len(elig),
                        "spk_dst": len(chosen), "rows_src": sum(len(v) for v in spk_rows.values()), "rows_dst": len(keep),
                        "rows_per_spk_dst": round(len(keep) / max(1, len(chosen)), 1),
                        "domains_dst": dict(sorted(doms.items())), "tasks_dst": dict(tasks.most_common())}
    out_rows = [r for i, r in enumerate(rows) if i not in drop]

    dst.mkdir(parents=True, exist_ok=True)
    with open(dst / "train.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(out_rows)
    for fn in COPY:
        if (src / fn).is_file():
            shutil.copy2(src / fn, dst / fn)

    s_src, s_dst = steps_per_epoch(rows, a), steps_per_epoch(out_rows, a)
    meta = {"src": str(src), "spec": a.spec, "seed": a.seed, "min_rows": a.min_rows,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "rows": {"src": len(rows), "dst": len(out_rows)},
            "steps_per_epoch": {"src": s_src, "dst": s_dst},
            "ref_epochs": a.ref_epochs, "matched_epochs": max(1, round(a.ref_epochs * s_src / s_dst)),
            "per_class": per_class,
            "note": "spec 에 없는 나라는 원본 그대로. valid·test 는 원본 복사. 같은 seed 면 나라마다 화자 순서가 같아 "
                    "작은 조건의 화자가 큰 조건에 들어 있다. matched_epochs 는 대조군 ref_epochs 와 최대 스텝을 맞춘 값."}
    (dst / "abl_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    for c, v in per_class.items():
        print(f"  {c}: {v['cond']} → 화자 {v['spk_dst']}/{v['spk_src']} · 행 {v['rows_src']:,} → {v['rows_dst']:,} "
              f"(화자당 {v['rows_per_spk_dst']}) · 출처 {v['domains_dst']}", flush=True)
    print(f"[화자 수 실험] 학습 {len(rows):,} → {len(out_rows):,}행 · 스텝/epoch {s_src:,} → {s_dst:,} · "
          f"대조군 {a.ref_epochs} epoch 와 같은 스텝 = {meta['matched_epochs']} epoch → {dst}", flush=True)


if __name__ == "__main__":
    main()
