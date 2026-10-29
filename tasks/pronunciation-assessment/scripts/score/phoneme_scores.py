# -*- coding: utf-8 -*-
"""음소별 발음 점수 (0~100) — "초성 ㅊ 54점 · 받침 ㅇ 66점" 화면용.

입력은 `extract_gop_feats.py` 캐시(`canon`·`gops_m`·`stem`). GPU 없이 CPU 로만 돈다.

  1. 자리마다 **margin GOP**(목표 − 목표 제외 1등)를 쓴다. 기존 maxnorm 은 96.7% 가 0 으로 뭉쳐 순위가 안 선다.
  2. 자리를 **초성 / 종성(받침) / 모음 × 자모**로 묶는다. 같은 ㄱ 이라도 초성과 받침은 다른 소리다.
  3. 묶음마다 전체 화자 분포를 만들어 **백분위 → 0~100** 으로 환산한다.
     음소마다 난이도가 달라(받침 ㅇ 은 원래 어렵다) 절대값을 그대로 쓰면 비교가 안 된다.
  4. 검증: TAG(사람이 들은 실제 발음)와 표준발음을 자리별로 맞대어 **실제 오류 여부**를 만들고,
     점수가 그것을 가려내는지(AUC) · 화자별 점수가 그 화자의 실제 오류율과 반대로 가는지(상관) 본다.

    $PY .../phoneme_scores.py --feats $MDD_OUTPUTS/score_phoneme/gop_margin_test.jsonl \\
        --manifest $MDD_OUTPUTS/manifest_tag_rebuilt --split test --out $MDD_OUTPUTS/score_phoneme

산출
  norms.json        묶음별 분포(백분위 기준점) — 서비스에서 점수 환산에 쓴다
  speakers.jsonl    화자별 묶음 점수(0~100) + 발화 수
  report.md         검증 수치와 상위/하위 음소
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import argparse, collections, json, os

import numpy as np

from mdd.jamo_to_kog2p import VOWEL

VOWELS = set(VOWEL)
MIN_POS = 200        # 묶음이 이보다 적으면 분포를 못 믿는다
MIN_SPK = 5          # 화자 점수를 낼 최소 자리 수


def context(canon, i):
    """자리 i 의 묶음 이름. 모음 / 초성 / 받침(종성)."""
    c = canon[i]
    if c in VOWELS:
        return "모음 " + c
    coda = c == "ㅇ" or not (i + 1 < len(canon) and canon[i + 1] in VOWELS)
    return ("받침 " if coda else "초성 ") + c


def align_flags(canon, tag):
    """표준발음 자리별 '실제로 틀렸나'(TAG 와 다른가). 삽입은 버린다."""
    import edlib
    c, t = "".join(canon), "".join(tag)
    if not c:
        return []
    if not t:
        return [1] * len(c)
    a = edlib.getNiceAlignment(edlib.align(c, t, task="path"), c, t)
    out = []
    for x, y in zip(a["query_aligned"], a["target_aligned"]):
        if x == "-":
            continue
        out.append(0 if x == y else 1)
    return out


def auc(scores, labels):
    """오류 판별력. 1 = 오류이고 점수가 낮을수록 오류여야 하므로,
    '정상 자리 점수 > 오류 자리 점수'일 확률을 돌려준다(0.5 = 무작위)."""
    s = np.asarray(scores, dtype=np.float64); y = np.asarray(labels)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return None
    r = s.argsort().argsort().astype(np.float64) + 1          # 오름차순 순위
    return 1.0 - float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feats", required=True, help="extract_gop_feats.py 산출 jsonl (gops_m 포함)")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-pos", dest="min_pos", type=int, default=MIN_POS)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    tsv = open(os.path.join(a.manifest, a.split + ".tsv"), encoding="utf-8").read().splitlines()[1:]
    ltr = [l.split() for l in open(os.path.join(a.manifest, a.split + ".ltr"), encoding="utf-8").read().splitlines()]
    tag = {os.path.basename(r.split("\t")[0])[:-4]: ltr[i] for i, r in enumerate(tsv)}

    pool = collections.defaultdict(list)          # 묶음 → margin 값 (분포용)
    per_spk = collections.defaultdict(lambda: collections.defaultdict(list))
    ev_s, ev_y, ev_g = [], [], []                 # 검증용: margin · 실제오류 · maxnorm
    spk_err = collections.defaultdict(lambda: [0, 0])
    n_utt = 0
    for line in open(a.feats, encoding="utf-8"):
        r = json.loads(line)
        canon, gm = r.get("canon"), r.get("gops_m")
        if not canon or not gm or len(canon) != len(gm):
            continue
        n_utt += 1
        spk = r["stem"].split("-")[0]
        flags = align_flags(list(canon), tag.get(r["stem"], []))
        g_old = r.get("gops") or [None] * len(gm)
        for i, ch in enumerate(canon):
            k = context(canon, i)
            pool[k].append(gm[i]); per_spk[spk][k].append(gm[i])
            if i < len(flags):
                ev_s.append(gm[i]); ev_y.append(flags[i]); ev_g.append(g_old[i])
                spk_err[spk][0] += flags[i]; spk_err[spk][1] += 1

    # ── 묶음별 분포(백분위 기준점) ──
    norms = {}
    for k, v in pool.items():
        if len(v) < a.min_pos:
            continue
        q = np.percentile(np.asarray(v), np.arange(0, 101, 2))
        norms[k] = {"n": len(v), "q": [round(float(x), 4) for x in q]}
    print(f"[분포] 발화 {n_utt:,} · 묶음 {len(norms)} (자리 {len(ev_s):,})", flush=True)

    def to100(k, x):
        q = norms[k]["q"]
        return float(np.clip(np.interp(x, q, np.arange(0, 101, 2)), 0, 100))

    # ── 화자별 점수 ──
    out_spk = open(os.path.join(a.out, "speakers.jsonl"), "w", encoding="utf-8")
    spk_scores = {}
    for spk, d in per_spk.items():
        sc = {k: round(to100(k, float(np.mean(v))), 1) for k, v in d.items() if k in norms and len(v) >= MIN_SPK}
        if not sc:
            continue
        spk_scores[spk] = sc
        weak = sorted(sc.items(), key=lambda kv: kv[1])[:5]
        out_spk.write(json.dumps({"speaker": spk, "n_pos": sum(len(v) for v in d.values()),
                                  "scores": sc, "weak": weak}, ensure_ascii=False) + "\n")
    out_spk.close()

    # ── 검증 ──
    y = np.asarray(ev_y)
    a_margin = auc(ev_s, y)
    valid_old = [i for i, g in enumerate(ev_g) if g is not None]
    a_old = auc([ev_g[i] for i in valid_old], y[valid_old]) if valid_old else None
    zero_old = (np.asarray([ev_g[i] for i in valid_old]) == 0).mean() if valid_old else None
    # 화자 단위: 묶음 평균점수 vs 실제 오류율
    xs, ys = [], []
    for spk, sc in spk_scores.items():
        e, n = spk_err[spk]
        if n >= 50:
            xs.append(float(np.mean(list(sc.values())))); ys.append(e / n)
    corr = float(np.corrcoef(xs, ys)[0, 1]) if len(xs) > 10 else None

    rank = sorted(((k, float(np.mean(v)), len(v)) for k, v in pool.items() if k in norms), key=lambda t: t[1])
    L = ["# 음소별 발음 점수 — margin GOP + 백분위 환산", "",
         f"- 자료: `{os.path.basename(a.feats)}` · 발화 {n_utt:,} · 자리 {len(ev_s):,} · 묶음 {len(norms)} · 화자 {len(spk_scores):,}",
         f"- **오류 판별력(AUC)** margin **{a_margin:.4f}**" + (f" · 기존 maxnorm {a_old:.4f}" if a_old else ""),
         f"- 기존 maxnorm 이 0 으로 뭉치는 비율: {zero_old*100:.1f}%" if zero_old is not None else "",
         f"- 화자 평균점수 vs 실제 오류율 상관: {corr:.4f} (음수여야 정상)" if corr is not None else "", "",
         "## 어려운 묶음 10 (평균 margin 낮은 순)", "", "| 묶음 | 평균 margin | 자리 수 |", "|---|--:|--:|"]
    for k, m, n in rank[:10]:
        L.append(f"| {k} | {m:.2f} | {n:,} |")
    L += ["", "## 쉬운 묶음 10", "", "| 묶음 | 평균 margin | 자리 수 |", "|---|--:|--:|"]
    for k, m, n in rank[-10:][::-1]:
        L.append(f"| {k} | {m:.2f} | {n:,} |")
    json.dump(norms, open(os.path.join(a.out, "norms.json"), "w", encoding="utf-8"), ensure_ascii=False)
    open(os.path.join(a.out, "report.md"), "w", encoding="utf-8").write("\n".join(x for x in L if x != "") + "\n")
    print("\n".join(x for x in L if x != ""), flush=True)


if __name__ == "__main__":
    main()
