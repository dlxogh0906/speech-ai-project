# -*- coding: utf-8 -*-
"""사람 TAG 파일럿 P0 — 전사할 발화 고르기(docs/05_사람TAG_라벨링_파일럿_설계.md §3·§4).

AIHub 131 외국인 읽기 발화 중 **P5b 의사라벨에 쓴 발화**에서 고른다 — 나중에 같은 오디오로 "사람 라벨 vs 의사라벨"을
비교할 수 있게. hold-out 문항은 의사라벨 풀을 만들 때 이미 뺐다(`build_pseudo_labels.py`).

- 길이 3~8초(전사 부담) · 문장과 화자는 겹치지 않게 · 모국어는 풀 분포를 따른다(상위 언어 최소 1개)
- 교사가 표준발음과 다르게 들은 발화(pseudo_dist ≥ 0.1)를 절반 — 오류가 있는 발화가 섞이도록
- 두 전사자 교차 배정: 짝수 번호는 A 가 빈칸 · B 가 표준발음 미리 채움, 홀수는 반대(미리 채움 끌림 측정)

    $PY .../build_tag_pilot_p0.py --out $MDD_OUTPUTS/tag_pilot/p0_v1
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import MDD_OUTPUTS

import argparse, collections, json, os, random


def spaced(hangul, text):
    """표준발음 한글에 원문 띄어쓰기를 입힌다 — 음절 수가 원문과 같을 때만(대부분 같다)."""
    words = [sum(1 for ch in w if u"가" <= ch <= u"힣") for w in text.split()]
    words = [n for n in words if n]
    if sum(words) != len(hangul):
        return hangul
    out, k = [], 0
    for n in words:
        out.append(hangul[k:k + n]); k += n
    return u" ".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", default=os.path.join(str(MDD_OUTPUTS), "pseudo_p5_v1_all"))
    ap.add_argument("--out", default=os.path.join(str(MDD_OUTPUTS), "tag_pilot", "p0_v1"))
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--min-sec", dest="min_sec", type=float, default=3.0)
    ap.add_argument("--max-sec", dest="max_sec", type=float, default=8.0)
    ap.add_argument("--conf", type=float, default=0.9197, help="P5b 필터와 같은 값(manifest_p5b_noisy/stats.json)")
    ap.add_argument("--max-dist", dest="max_dist", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from mdd.hangul_jamo import compose
    from mdd.word_judge import KoCanonical

    lines = open(os.path.join(a.pool, "pseudo.tsv"), encoding="utf-8").read().splitlines()
    root, rows = lines[0].strip(), lines[1:]
    meta = [json.loads(l) for l in open(os.path.join(a.pool, "pseudo.meta.jsonl"), encoding="utf-8")]
    ltr = open(os.path.join(a.pool, "pseudo.ltr"), encoding="utf-8").read().splitlines()
    assert len(rows) == len(meta) == len(ltr)
    cand = [i for i, m in enumerate(meta) if m["pseudo_conf"] >= a.conf and m["pseudo_dist"] <= a.max_dist
            and a.min_sec <= m["sec"] <= a.max_sec]
    rng = random.Random(a.seed)
    rng.shuffle(cand)

    # 모국어 배분: 풀 비율대로(최대 나머지 방식), 상위 언어는 최소 1개
    by = collections.defaultdict(list)
    for i in cand:
        by[meta[i]["mother_tongue"]].append(i)
    langs = sorted(by, key=lambda k: -len(by[k]))
    share = dict((k, a.n * len(by[k]) / len(cand)) for k in langs)
    alloc = dict((k, int(share[k])) for k in langs)
    for k in sorted(langs, key=lambda k: -(share[k] - alloc[k])):
        if sum(alloc.values()) >= a.n:
            break
        alloc[k] += 1
    for k in langs[:10]:                              # 상위 10개 언어는 최소 1개 — 많은 쪽에서 1개씩 덜어 온다
        while alloc[k] == 0:
            big = max(langs, key=lambda x: alloc[x]); alloc[big] -= 1; alloc[k] = 1

    picked, used_item, used_spk = [], set(), set()
    for k in langs:
        want, hi = alloc[k], 0
        for i in by[k]:                               # 오류 있는 발화(dist ≥ 0.1)와 없는 발화를 번갈아
            if want == 0:
                break
            m = meta[i]
            if m["item"] in used_item or m["spk"] in used_spk:
                continue
            need_hi = hi < (alloc[k] + 1) // 2
            if (m["pseudo_dist"] >= 0.1) != need_hi and len([j for j in by[k] if j not in picked]) > want * 3:
                continue
            picked.append(i); used_item.add(m["item"]); used_spk.add(m["spk"])
            hi += m["pseudo_dist"] >= 0.1; want -= 1
    rng.shuffle(picked)

    K = KoCanonical()
    items = []
    for n, i in enumerate(picked):
        m = meta[i]
        canon = K(m["prompt"])
        items.append({"id": n, "stem": m["stem"], "wav": os.path.join(root, rows[i].split("\t")[0]), "sec": m["sec"],
                      "text": m["prompt"], "mother_tongue": m["mother_tongue"], "spk": m["spk"], "item": m["item"],
                      "canon": canon, "canon_hangul": spaced(compose(canon), m["prompt"]),
                      "pseudo": [t for t in ltr[i].split() if t != "|"], "pseudo_dist": m["pseudo_dist"],
                      "prefill": {"A": "blank" if n % 2 == 0 else "canon", "B": "canon" if n % 2 == 0 else "blank"}})
    os.makedirs(a.out, exist_ok=True)
    json.dump(items, open(os.path.join(a.out, "items.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    c = collections.Counter(it["mother_tongue"] for it in items)
    print("[p0] 후보 %d → %d개 · 총 %.1f분 · 오류 의심(dist≥0.1) %d개 · 모국어 %s"
          % (len(cand), len(items), sum(it["sec"] for it in items) / 60,
             sum(it["pseudo_dist"] >= 0.1 for it in items), dict(c.most_common())), flush=True)
    print("[saved] %s" % os.path.join(a.out, "items.json"))


if __name__ == "__main__":
    main()
