# -*- coding: utf-8 -*-
"""학습이 실제로 **무엇을 몇 번 봤는지** 센다 — 트레이너의 배치 순서를 그대로 재현한다.

`train_w2vbert_ctc.batches_by_frames`(길이순 묶음 → epoch 시드로 순서 섞기)는 결정적이라, 같은 매니페스트·시드·
배치 크기면 N 업데이트 동안 본 행을 정확히 알 수 있다(업데이트 1회 = --accum 개 배치).
의사라벨 비율을 바꾸면 gold(TAG) 노출도 같이 줄기 때문에, 비교할 때 이 표를 함께 남긴다.

    $PY .../count_exposure.py --manifest $MDD_OUTPUTS/manifest_p5b_noisy --steps 4000 8000 \\
        --pseudo-meta $MDD_OUTPUTS/pseudo_p5_v1_all/pseudo.meta.jsonl --out <json>
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
_sys.path.insert(0, str(_r / "tasks/pronunciation-assessment/scripts/train"))

import argparse, collections, json, os, re
from train_w2vbert_ctc import batches_by_frames, read_split

HANGUL = re.compile(u"[^가-힣]")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--steps", nargs="+", type=int, default=[4000])
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--batch-frames", dest="batch_frames", type=int, default=1_600_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pseudo-meta", dest="pseudo_meta", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    items, _ = read_split(a.manifest, "train")
    src = open(os.path.join(a.manifest, "train.source"), encoding="utf-8").read().splitlines()
    assert len(src) == len(items)
    sent_of = {}
    if a.pseudo_meta:
        for l in open(a.pseudo_meta, encoding="utf-8"):
            m = json.loads(l); sent_of[m["stem"]] = HANGUL.sub(u"", m.get("prompt") or u"")
    stem = lambda i: os.path.basename(items[i][0])[:-4]
    need = max(a.steps) * a.accum
    seq, epoch = [], 0
    while len(seq) < need:
        seq += batches_by_frames(items, a.batch_frames, shuffle_seed=a.seed + epoch); epoch += 1
    res = {"manifest": os.path.abspath(a.manifest), "accum": a.accum, "batch_frames": a.batch_frames, "seed": a.seed,
           "train_rows": len(items), "by_step": {}}
    for st in a.steps:
        rows = [i for b in seq[:st * a.accum] for i in b]
        cnt = collections.Counter(src[i] for i in rows)
        hrs = collections.Counter()
        for i in rows:
            hrs[src[i]] += items[i][1] / 16000.0 / 3600
        uniq = collections.defaultdict(set)
        for i in rows:
            uniq[src[i]].add(stem(i))
        r = {"updates": st, "batches": st * a.accum, "row_exposures": dict(cnt),
             "hours_seen": dict((k, round(v, 1)) for k, v in hrs.items()),
             "unique_utterances_seen": dict((k, len(v)) for k, v in uniq.items())}
        if "pseudo" in uniq:
            r["unique_pseudo_sentences_seen"] = len(set(sent_of.get(s, s) for s in uniq["pseudo"]))
            r["pseudo_exposures_per_unique_utt"] = round(cnt["pseudo"] / float(max(1, len(uniq["pseudo"]))), 2)
        gold = sum(v for k, v in cnt.items() if k in ("foreign_clean", "foreign_mis"))
        r["gold_foreign_exposures"] = gold
        res["by_step"][str(st)] = r
        print("[%s @%d] 노출 %s · gold 외국인 %d · 시간 %s · 고유 %s%s" % (
            os.path.basename(a.manifest), st, dict(cnt), gold, r["hours_seen"], r["unique_utterances_seen"],
            (" · 본 의사라벨 문장 %d" % r["unique_pseudo_sentences_seen"]) if "pseudo" in uniq else ""))
    if a.out:
        json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
