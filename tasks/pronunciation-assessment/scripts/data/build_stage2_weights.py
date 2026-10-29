# -*- coding: utf-8 -*-
"""Stage 2 **행별 loss 가중치** 생성 — H0~H4 실험용. 학습 전에 분포·편중을 먼저 출력한다.

가중은 `train_w2vbert_ctc.py --weights` 가 받는다(Σ w·loss / Σ w). 행 복제가 아니라서 같은 문항·화자가
늘어나지 않는다. 행마다 TAG 와 canonical 을 정렬해 오류 범주를 붙인다:

  insertion : 표준발음에 없는 소리가 끼어듦(삽입)
  hard_pair : 같은 계열 안에서 세기만 다른 자음 대치 — 요청 6쌍 + valid 에서 되돌림 ≥0.6 인 쌍
  vowel     : 모음 자리의 대치(다른 모음으로)

가중식(범주가 겹쳐도 폭발하지 않게 합산 후 상한):
    w = min(w_max, 1 + b·[insertion] + b·[hard_pair] + b·[vowel])
    H0 = 전부 1 · H1 = insertion 만 · H2 = hard_pair 만 · H3 = vowel 만 · H4 = 셋 다
원어민 replay 행은 항상 1.

    $PY .../build_stage2_weights.py --manifest $MDD_OUTPUTS/manifest_holdout_v1 --bonus 0.5 --w-max 2.0
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import MDD_OUTPUTS

import argparse, collections, gzip, json, os
import numpy as np
import edlib

VOWELS = set(u"ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
REQUIRED_PAIRS = [u"ㅅ>ㅆ", u"ㅆ>ㅅ", u"ㅈ>ㅊ", u"ㄱ>ㅋ", u"ㄸ>ㄷ", u"ㅉ>ㅈ"]
PRIORITY_VOWELS = u"ㅛㅜㅡㅗ"


def categories(canon, tag, pairs):
    c, t = u"".join(canon), u"".join(tag)
    if not c or not t:
        return False, set(), False, set()
    a = edlib.getNiceAlignment(edlib.align(c, t, task="path"), c, t)
    ins, hit, vow, vset = False, set(), False, set()
    for qc, ts in zip(a["query_aligned"], a["target_aligned"]):
        if qc == "-":
            ins = True
            continue
        if ts == "-" or ts == qc:
            continue
        if (qc, ts) in pairs:
            hit.add(u"%s>%s" % (qc, ts))
        if qc in VOWELS and ts in VOWELS:
            vow = True; vset.add(qc)
    return ins, hit, vow, vset


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--cache", default=os.path.join(str(MDD_OUTPUTS), "cache", "foreign_tag_lab_parsed.jsonl.gz"))
    ap.add_argument("--valid-pairs", dest="valid_pairs", default=os.path.join(str(MDD_OUTPUTS), "analysis", "hard_pairs_from_valid.json"),
                    help="valid 에서 뽑은 대치쌍 되돌림 표(test 는 쓰지 않는다)")
    ap.add_argument("--min-reversion", dest="min_rev", type=float, default=0.6)
    ap.add_argument("--min-n", dest="min_n", type=int, default=30)
    ap.add_argument("--bonus", type=float, default=0.5)
    ap.add_argument("--w-max", dest="w_max", type=float, default=2.0)
    ap.add_argument("--out-dir", dest="out_dir", default=None, help="기본: <manifest>/weights_b<bonus>")
    a = ap.parse_args()
    out = a.out_dir or os.path.join(a.manifest, "weights_b%s" % a.bonus)
    if os.path.exists(out) and os.listdir(out):
        raise SystemExit("이미 있다(덮어쓰기 금지): %s" % out)
    os.makedirs(out, exist_ok=True)

    # 쌍 목록 — 요청 6쌍 + valid 에서 되돌림이 높은 **자음** 쌍(모음쌍은 vowel 범주가 맡는다)
    pairs = set(tuple(p.split(">")) for p in REQUIRED_PAIRS)
    extra = []
    for r in json.load(open(a.valid_pairs, encoding="utf-8")):
        c, t = r["pair"].split(">")
        if r["n"] >= a.min_n and r["reversion"] >= a.min_rev and c not in VOWELS and t not in VOWELS:
            if (c, t) not in pairs:
                extra.append(r["pair"])
            pairs.add((c, t))
    print("[pairs] 요청 %s + valid 추가 %s" % (REQUIRED_PAIRS, extra), flush=True)

    rows = open(os.path.join(a.manifest, "train.tsv"), encoding="utf-8").read().splitlines()[1:]
    ltr = open(os.path.join(a.manifest, "train.ltr"), encoding="utf-8").read().splitlines()
    src = open(os.path.join(a.manifest, "train.source"), encoding="utf-8").read().splitlines()
    assert len(rows) == len(ltr) == len(src)
    stems = [os.path.basename(r.split("\t")[0])[:-4] for r in rows]
    need = set(s for s, so in zip(stems, src) if so != "native")
    canon, item, spk = {}, {}, {}
    with gzip.open(a.cache, "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["stem"] in need:
                canon[d["stem"]] = d.get("canon") or []; item[d["stem"]] = d.get("item"); spk[d["stem"]] = d.get("spk")
    miss = len(need) - len(canon)
    print("[rows] train %d · 외국인 %d(고유 발화 %d) · canonical 못 찾음 %d" % (len(rows), sum(1 for x in src if x != "native"),
                                                                    len(need), miss), flush=True)
    assert miss == 0, "캐시에 없는 발화가 있다"

    N = len(rows)
    f_ins = np.zeros(N, bool); f_pair = np.zeros(N, bool); f_vow = np.zeros(N, bool)
    pair_rows = collections.Counter(); vow_rows = collections.Counter()
    memo = {}
    for i, (st, so) in enumerate(zip(stems, src)):
        if so == "native":
            continue
        if st not in memo:
            memo[st] = categories(canon[st], ltr[i].split(), pairs)
        ins, hit, vow, vset = memo[st]
        f_ins[i], f_pair[i], f_vow[i] = ins, bool(hit), vow
        for h in hit: pair_rows[h] += 1
        for v in vset: vow_rows[v] += 1
    native = np.array([so == "native" for so in src])
    b = a.bonus
    W = {"H0": np.ones(N, np.float32),
         "H1": np.minimum(a.w_max, 1 + b * f_ins).astype(np.float32),
         "H2": np.minimum(a.w_max, 1 + b * f_pair).astype(np.float32),
         "H3": np.minimum(a.w_max, 1 + b * f_vow).astype(np.float32),
         "H4": np.minimum(a.w_max, 1 + b * f_ins + b * f_pair + b * f_vow).astype(np.float32)}
    for k in W:
        W[k][native] = 1.0

    # 편중 검사 — 문항·화자별 (가중 몫 / 무가중 몫)
    items = np.array([item.get(s, "native") if so != "native" else "native" for s, so in zip(stems, src)])
    spks = np.array([spk.get(s, "native") if so != "native" else "native" for s, so in zip(stems, src)])

    def concentration(w, keys):
        tot_w, tot_n = float(w.sum()), float(len(w))
        uk, inv = np.unique(keys, return_inverse=True)
        sw = np.bincount(inv, weights=w); sn = np.bincount(inv).astype(float)
        ratio = (sw / tot_w) / (sn / tot_n)
        order = np.argsort(-ratio)
        return {"max_ratio": round(float(ratio.max()), 3), "min_ratio": round(float(ratio.min()), 3),
                "top": [(str(uk[j]), round(float(ratio[j]), 3), int(sn[j])) for j in order[:5]],
                "max_share_weighted": round(float((sw / tot_w).max()), 4)}

    rep = {"manifest": os.path.abspath(a.manifest), "bonus": b, "w_max": a.w_max,
           "pairs_required": REQUIRED_PAIRS, "pairs_added_from_valid": extra,
           "pair_rule": "valid(replay10 valid · step_38000 예측)에서 n≥%d · 되돌림≥%.2f 인 자음쌍" % (a.min_n, a.min_rev),
           "rows": N, "native_rows": int(native.sum()),
           "category_rows": {"insertion": int(f_ins.sum()), "hard_pair": int(f_pair.sum()), "vowel": int(f_vow.sum()),
                             "ins&pair": int((f_ins & f_pair).sum()), "ins&vowel": int((f_ins & f_vow).sum()),
                             "pair&vowel": int((f_pair & f_vow).sum()), "all3": int((f_ins & f_pair & f_vow).sum())},
           "category_share_of_rows": {"insertion": round(float(f_ins.mean()), 4), "hard_pair": round(float(f_pair.mean()), 4),
                                      "vowel": round(float(f_vow.mean()), 4)},
           "hard_pair_rows": dict(pair_rows.most_common()),
           "vowel_rows_by_canonical_vowel": dict(vow_rows.most_common()),
           "priority_vowel_rows": dict((v, vow_rows.get(v, 0)) for v in PRIORITY_VOWELS),
           "experiments": {}}
    for k, w in W.items():
        np.savetxt(os.path.join(out, "weights_%s.txt" % k), w, fmt="%.3f")
        tot = float(w.sum())
        rep["experiments"][k] = {
            "mean_weight": round(float(w.mean()), 4), "max_weight": round(float(w.max()), 3),
            "rows_weight_gt1": round(float((w > 1).mean()), 4),
            "effective_share": {"insertion": round(float(w[f_ins].sum()) / tot, 4),
                                "hard_pair": round(float(w[f_pair].sum()) / tot, 4),
                                "vowel": round(float(w[f_vow].sum()) / tot, 4)},
            "item_concentration": concentration(w, items), "speaker_concentration": concentration(w, spks)}
    json.dump(rep, open(os.path.join(out, "weights_report.json"), "w"), ensure_ascii=False, indent=1)
    print("[cat ] 행 비율 %s" % rep["category_share_of_rows"])
    print("[cat ] 겹침 %s" % {k: v for k, v in rep["category_rows"].items() if "&" in k or k == "all3"})
    print("[pair] %s" % rep["hard_pair_rows"])
    print("[vow ] 우선 모음 %s" % rep["priority_vowel_rows"])
    for k, e in rep["experiments"].items():
        print("[%s] 평균 %.3f 최대 %.2f · 가중>1 행 %.1f%% · 실효몫 %s · 문항 최대배율 %.2f · 화자 최대배율 %.2f"
              % (k, e["mean_weight"], e["max_weight"], 100 * e["rows_weight_gt1"], e["effective_share"],
                 e["item_concentration"]["max_ratio"], e["speaker_concentration"]["max_ratio"]))
    print("[saved] %s" % out)


if __name__ == "__main__":
    main()
