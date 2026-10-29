# -*- coding: utf-8 -*-
"""P5 — hold-out 학습 매니페스트에 **의사라벨 발화(새 문장)**를 섞는다. 스크리닝(4k step) 전용.

섞는 비율은 **오디오 시간 기준**이다. 트레이너가 배치를 오디오 길이로 채우므로, 행 수로 맞추면 긴 의사라벨
발화(중위 12초 vs 기존 2.8초)가 연산을 독차지한다. 의사라벨 발화는 필요한 만큼 반복해 비율을 맞춘다.
valid/valid_unseen 은 기존 hold-out 매니페스트를 링크한다(test 는 링크하지 않는다 — 스크리닝은 test 를 안 본다).

필터는 `build_pseudo_labels.py` 가 남긴 확신도·거리로 여기서 고른다 — 같은 전사에서 기준만 달리한 변형을 만들기 위해.

    $PY .../build_p5_manifest.py --base $MDD_OUTPUTS/manifest_holdout_v1 --pseudo $MDD_OUTPUTS/pseudo_p5_v1_all \\
        --keep-frac 0.7 --max-dist 0.10 --audio-share 0.25 --out $MDD_OUTPUTS/manifest_p5a_clean
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import MDD_OUTPUTS

import argparse, collections, json, os, random, re
import numpy as np

HANGUL = re.compile(u"[^가-힣]")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--pseudo", required=True)
    ap.add_argument("--holdout", default=os.path.join(str(MDD_OUTPUTS), "holdout_prompts_v1.json"))
    ap.add_argument("--keep-frac", dest="keep_frac", type=float, default=0.7)
    ap.add_argument("--max-dist", dest="max_dist", type=float, default=0.10)
    ap.add_argument("--audio-share", dest="share", type=float, default=0.25, help="학습 오디오 시간 중 의사라벨 비율")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.path.exists(a.out) and os.listdir(a.out):
        raise SystemExit("이미 있다(덮어쓰기 금지): %s" % a.out)
    rng = random.Random(a.seed)

    # 의사라벨 고르기
    pl = open(os.path.join(a.pseudo, "pseudo.tsv"), encoding="utf-8").read().splitlines()
    proot, prow = pl[0].strip(), pl[1:]
    pltr = open(os.path.join(a.pseudo, "pseudo.ltr"), encoding="utf-8").read().splitlines()
    pmeta = [json.loads(l) for l in open(os.path.join(a.pseudo, "pseudo.meta.jsonl"), encoding="utf-8")]
    ok = [i for i, m in enumerate(pmeta) if m["pseudo_dist"] <= a.max_dist]
    thr = float(np.percentile([pmeta[i]["pseudo_conf"] for i in ok], 100 * (1 - a.keep_frac))) if ok else 1.0
    keep = [i for i in ok if pmeta[i]["pseudo_conf"] >= thr]
    # 누출 재확인 — hold-out 문항 텍스트가 들어 있으면 안 된다
    H = json.load(open(a.holdout, encoding="utf-8"))
    hold = sorted(set(HANGUL.sub(u"", v.get("norm") or v.get("prompt") or u"") for v in H["items"].values()) - {u""})
    leak = [i for i in keep if any(h in HANGUL.sub(u"", pmeta[i].get("prompt") or u"") for h in hold)]
    assert not leak, "hold-out 문항이 든 의사라벨 %d개" % len(leak)
    pfr = [int(prow[i].split("\t")[1]) for i in keep]
    p_hours = sum(pfr) / 16000.0 / 3600

    # 기존 학습 행
    bl = open(os.path.join(a.base, "train.tsv"), encoding="utf-8").read().splitlines()
    broot, brow = bl[0].strip(), bl[1:]
    bltr = open(os.path.join(a.base, "train.ltr"), encoding="utf-8").read().splitlines()
    bsrc = open(os.path.join(a.base, "train.source"), encoding="utf-8").read().splitlines()
    b_hours = sum(int(r.split("\t")[1]) for r in brow) / 16000.0 / 3600
    need = b_hours * a.share / (1 - a.share)          # 의사라벨이 차지해야 할 시간
    reps, frac = divmod(need / max(1e-9, p_hours), 1.0)
    reps = int(reps)
    extra = rng.sample(keep, int(round(frac * len(keep))))
    prow_idx = keep * reps + extra

    def absrow(root, r):
        path, fr = r.split("\t")
        return os.path.join(root, path).lstrip("/"), fr
    rows = [(absrow(broot, r), l, s) for r, l, s in zip(brow, bltr, bsrc)]
    rows += [(absrow(proot, prow[i]), pltr[i], "pseudo") for i in prow_idx]
    rng.shuffle(rows)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "train.tsv"), "w", encoding="utf-8") as ft, \
         open(os.path.join(a.out, "train.ltr"), "w", encoding="utf-8") as fl, \
         open(os.path.join(a.out, "train.source"), "w", encoding="utf-8") as fs:
        ft.write("/\n")
        for (p, fr), l, s in rows:
            ft.write("%s\t%s\n" % (p, fr)); fl.write(l + "\n"); fs.write(s + "\n")
    np.savetxt(os.path.join(a.out, "weights_ones.txt"), np.ones(len(rows), np.float32), fmt="%.3f")
    for f in os.listdir(a.base):          # valid 계열과 사전만 링크 — test 는 링크하지 않는다
        if f.startswith("valid") or f == "dict.ltr.txt":
            os.symlink(os.path.join(a.base, f), os.path.join(a.out, f))
    src = collections.Counter(s for _, _, s in rows)
    sec = collections.Counter()
    for (p, fr), _, s in rows:
        sec[s] += int(fr) / 16000.0
    stats = {"base": os.path.abspath(a.base), "pseudo": os.path.abspath(a.pseudo),
             "filter": {"keep_frac": a.keep_frac, "max_dist": a.max_dist, "conf_threshold": round(thr, 4)},
             "pseudo_unique": len(keep),
             "pseudo_unique_sentences": len(set(HANGUL.sub(u"", pmeta[i].get("prompt") or u"") for i in keep)),
             "pseudo_speakers": len(set(pmeta[i].get("spk") for i in keep)),
             "pseudo_hours_unique": round(p_hours, 1), "pseudo_repeat": reps, "pseudo_extra": len(extra),
             "pseudo_mean_dist": round(float(np.mean([pmeta[i]["pseudo_dist"] for i in keep])), 4) if keep else None,
             "rows": dict(src), "hours": dict((k, round(v / 3600, 1)) for k, v in sec.items()),
             "audio_share": dict((k, round(v / sum(sec.values()), 4)) for k, v in sec.items()),
             "holdout_leak_in_pseudo": 0}
    json.dump(stats, open(os.path.join(a.out, "stats.json"), "w"), ensure_ascii=False, indent=1)
    print("[p5] 의사라벨 %d발화(문장 %d · 화자 %d · %.1fh) × %d(+%d) · 행 %s · 시간비율 %s"
          % (len(keep), stats["pseudo_unique_sentences"], stats["pseudo_speakers"], p_hours, reps, len(extra),
             stats["rows"], stats["audio_share"]))
    print("[saved] %s" % a.out)


if __name__ == "__main__":
    main()
