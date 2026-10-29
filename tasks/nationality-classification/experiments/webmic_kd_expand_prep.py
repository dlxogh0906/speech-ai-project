# -*- coding: utf-8 -*-
"""국가 클래스 확장 파일럿 데이터 (2026-10-01) — 기존 run_zeroth 데이터(lab131full_zeroth)에 AIHub 040~043 의
41개국 밖 국가를 덧붙인다. 기존 행은 순서를 바꾸지 않고 **뒤에 붙여** 교사 신호(teacher_*.npz)의 행 번호가 그대로 맞게 한다.

새 국가는 기존 빌드(webmic_kd_prep.py)와 같은 규칙을 따른다.
  · 1초 미만 제외 · 화자 키 md5 순서로 test(n≥3 이면 max(1, 10%)) → valid(n≥8 이면 max(1, 5%)) → train
  · 학습: 화자를 돌아가며 한 발화씩 --cap 까지(단어 과제는 뒤로) · valid 화자당 문장 6 + 단어 2 · test 화자당 문장 12 + 단어 4
  · 040~043 의 화자 번호는 데이터셋 사이에 겹치지 않아(13,478명 중 0명) 화자 키 = 사람이다 → split 은 화자 독립

    $PY tasks/nationality-classification/experiments/webmic_kd_expand_prep.py --countries Azerbaijan --out_dir <폴더>
출력: train.csv · valid.csv · test.csv(기존 test_koraug_zeroth + 새 국가) · teacher_*.npz 링크 · prep_meta.json
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import argparse, collections, csv, importlib.util, json, os, random

from shared.paths import NAT_OUTPUTS


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base_dir", default=str(NAT_OUTPUTS / "webmic_kd" / "lab131full_zeroth"))
    ap.add_argument("--base_test", default="test_koraug_zeroth.csv", help="base_dir 안의 기준 시험 파일")
    ap.add_argument("--index_dir", default=str(NAT_OUTPUTS / "webmic_kd"))
    ap.add_argument("--countries", required=True, help="쉼표 구분 — index_040_043.csv 의 nationality 값")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cap", type=int, default=20000)
    ap.add_argument("--min_sec", type=float, default=1.0)
    ap.add_argument("--eval_sent", type=int, default=12)
    ap.add_argument("--eval_word", type=int, default=4)
    ap.add_argument("--valid_sent", type=int, default=6)
    ap.add_argument("--valid_word", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    A = ap.parse_args()
    spec = importlib.util.spec_from_file_location("webmic_kd_prep", str(_pl.Path(__file__).parent / "webmic_kd_prep.py"))
    P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)       # h · round_robin · per_speaker 재사용
    os.makedirs(A.out_dir, exist_ok=True)
    rd = lambda f: list(csv.DictReader(open(os.path.join(A.base_dir, f), encoding="utf-8")))
    base = {"train": rd("train.csv"), "valid": rd("valid.csv"), "test": rd(A.base_test)}
    base_spk = {s for rows in base.values() for s in (r["speaker"] for r in rows)}
    countries = [c.strip() for c in A.countries.split(",") if c.strip()]
    web = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in csv.DictReader(open(os.path.join(A.index_dir, "index_040_043.csv"), encoding="utf-8")):
        c = r["nationality"]
        if c not in countries or r["label_en"] or float(r["duration_sec"] or 0) < A.min_sec:
            continue
        web[c][r["speaker"]].append({"audio_path": r["audio_path"], "label_en": c, "speaker": r["speaker"],
                                     "domain": "web" + r["dataset"], "task": r["task"], "duration_sec": r["duration_sec"]})
    new = {"train": [], "valid": [], "test": []}; stats = {}; assign = {}
    for c in countries:
        spk = web[c]; order = sorted(spk, key=P.h); n = len(order)
        assert n >= 3, f"{c}: 화자 {n}명 — 시험 화자를 만들 수 없다"
        n_te = max(1, round(0.10 * n)); n_va = 0 if n < 8 else max(1, round(0.05 * n))
        for i, s in enumerate(order):
            assign[s] = "test" if i < n_te else ("valid" if i < n_te + n_va else "train")
        part = lambda split: {s: rows for s, rows in spk.items() if assign[s] == split}
        rng = random.Random(f"{A.seed}:{c}")
        tr = P.round_robin(part("train"), A.cap, rng)
        va = P.per_speaker(part("valid"), A.valid_sent, A.valid_word, rng)
        te = P.per_speaker(part("test"), A.eval_sent, A.eval_word, rng)
        new["train"] += tr; new["valid"] += va; new["test"] += te
        stats[c] = {"spk": {k: sum(assign[s] == k for s in spk) for k in ("train", "valid", "test")},
                    "utt": {"train": len(tr), "valid": len(va), "test": len(te)},
                    "test_sent": sum(r["task"] != "RW" for r in te), "test_word": sum(r["task"] == "RW" for r in te)}
    # 화자 독립 확인: 새 화자끼리 split 이 겹치지 않고, 기존 데이터 화자와도 겹치지 않는다
    seen = collections.defaultdict(set)
    for split, rows in new.items():
        for r in rows:
            seen[r["speaker"]].add(split)
    leak_split = [s for s, v in seen.items() if len(v) > 1]
    leak_base = sorted(set(seen) & base_spk)
    assert not leak_split and not leak_base, (leak_split[:5], leak_base[:5])
    hdr = list(base["train"][0].keys())
    for split, fn in (("train", "train.csv"), ("valid", "valid.csv"), ("test", "test.csv")):
        with open(os.path.join(A.out_dir, fn), "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=hdr); w.writeheader()
            for r in base[split] + new[split]:                     # 기존 행이 앞 — 교사 신호 행 번호 유지
                w.writerow({k: r.get(k, "") for k in hdr})
    for fn in ("teacher_train.npz", "teacher_valid.npz"):
        src, dst = os.path.join(A.base_dir, fn), os.path.join(A.out_dir, fn)
        if os.path.exists(src) and not os.path.exists(dst):
            os.symlink(src, dst)
    meta = {"base_dir": A.base_dir, "countries": countries, "args": vars(A), "per_country": stats,
            "rows": {k: {"base": len(base[k]), "new": len(new[k])} for k in base},
            "checks": {"new_speaker_split_overlap": 0, "new_vs_base_speaker_overlap": 0},
            "new_speaker_assign": assign}
    json.dump(meta, open(os.path.join(A.out_dir, "prep_meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for c, s in stats.items():
        print(f"[{c}] 화자 train {s['spk']['train']} · valid {s['spk']['valid']} · test {s['spk']['test']} | 발화 train {s['utt']['train']:,} · valid {s['utt']['valid']} · test {s['utt']['test']} (문장 {s['test_sent']} · 단어 {s['test_word']})")
    print(f"[saved] {A.out_dir} · train {len(base['train'])+len(new['train']):,} · valid {len(base['valid'])+len(new['valid']):,} · test {len(base['test'])+len(new['test']):,}")


if __name__ == "__main__":
    main()
