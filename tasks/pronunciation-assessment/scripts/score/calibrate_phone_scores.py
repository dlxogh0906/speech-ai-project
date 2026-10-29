# -*- coding: utf-8 -*-
"""음소 점수 보정 — margin GOP 를 **"이 소리가 틀렸을 확률"** 로 바꾼다.

왜 바꾸나
    지금 점수는 백분위(`norms.json`)라 **50 이 평균**이다. 맞게 발음해도 50 근처가 나오고,
    같은 발화 안에서 6점과 94점이 섞여 화면이 이상해 보인다(실측 2026-09-23).
    백분위는 "남들 대비 순위"일 뿐 "맞았나"를 뜻하지 않는다.

무엇으로 바꾸나
    자리마다 정답이 있다 — 사람이 들은 실제 발음(TAG)이 표준발음과 같은가.
    margin 으로 이 정답을 가려내는 능력은 이미 AUC 0.937 이다(05/음소별 점수 검증).
    그 margin 을 **오류 확률**로 보정하면 점수 = (1 − 오류확률) × 100 이 되고,
    "80 점 이상이면 정확" 같은 절대 기준이 성립한다.

방법
    묶음(초성/받침/모음 × 자모)마다 로지스틱 회귀 1개. 표본이 적은 묶음은 전체 모형으로 물러난다.
    학습 70% · 검증 30% 로 나눠, 검증셋에서 보정이 맞는지(예측 확률 ↔ 실제 오류율) 확인한다.

    $PY .../calibrate_phone_scores.py --feats $MDD_OUTPUTS/score_phoneme/gop_margin_test.jsonl \\
        --manifest $MDD_OUTPUTS/manifest_tag_rebuilt --split test --out $MDD_OUTPUTS/score_phoneme
산출: calibration.json (묶음별 계수) · calibration_report.md
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
from mdd.session import phone_context

VOWELS = set(VOWEL)
MIN_FIT = 500          # 이보다 적은 묶음은 전체 모형을 쓴다
MIN_ERR = 20           # 오류 사례가 이보다 적어도 전체 모형


def align_flags(canon, tag):
    """표준발음 자리별 '실제로 틀렸나'(1=틀림). 삽입은 버린다."""
    import edlib
    c, t = "".join(canon), "".join(tag)
    if not c:
        return []
    if not t:
        return [1] * len(c)
    a = edlib.getNiceAlignment(edlib.align(c, t, task="path"), c, t)
    out = []
    for x, y in zip(a["query_aligned"], a["target_aligned"]):
        if x != "-":
            out.append(0 if x == y else 1)
    return out


def fit(x, y):
    """로지스틱 회귀 1변수. (a, b) 로 p(틀림) = sigmoid(a·x + b)."""
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(max_iter=1000)
    m.fit(np.asarray(x).reshape(-1, 1), np.asarray(y))
    return float(m.coef_[0][0]), float(m.intercept_[0])


def prob(a, b, x):
    return 1.0 / (1.0 + np.exp(-(a * np.asarray(x) + b)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feats", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    tsv = open(os.path.join(a.manifest, a.split + ".tsv"), encoding="utf-8").read().splitlines()[1:]
    ltr = [l.split() for l in open(os.path.join(a.manifest, a.split + ".ltr"), encoding="utf-8").read().splitlines()]
    tag = {os.path.basename(r.split("\t")[0])[:-4]: ltr[i] for i, r in enumerate(tsv)}

    X = collections.defaultdict(list)      # 묶음 → margin
    Y = collections.defaultdict(list)      # 묶음 → 틀렸나
    for line in open(a.feats, encoding="utf-8"):
        r = json.loads(line)
        canon, gm = r.get("canon"), r.get("gops_m")
        if not canon or not gm or len(canon) != len(gm):
            continue
        flags = align_flags(list(canon), tag.get(r["stem"], []))
        for i in range(min(len(flags), len(gm))):
            k = phone_context(canon, i, VOWELS)
            X[k].append(gm[i]); Y[k].append(flags[i])

    rng = np.random.RandomState(a.seed)
    allx = np.concatenate([np.asarray(v) for v in X.values()])
    ally = np.concatenate([np.asarray(v) for v in Y.values()])
    tr = rng.rand(len(allx)) < 0.7
    ga, gb = fit(allx[tr], ally[tr])                      # 전체(물러날 곳)
    cal = {"_global": {"a": ga, "b": gb, "n": int(len(allx)), "err_rate": float(ally.mean())}}
    ev_p, ev_y = [], []
    for k in X:
        x, y = np.asarray(X[k]), np.asarray(Y[k])
        m = rng.rand(len(x)) < 0.7
        if len(x) >= MIN_FIT and y[m].sum() >= MIN_ERR and y[m].sum() < m.sum():
            ka, kb = fit(x[m], y[m])
        else:
            ka, kb = ga, gb
        cal[k] = {"a": ka, "b": kb, "n": int(len(x)), "err_rate": float(y.mean()),
                  "own_fit": bool(len(x) >= MIN_FIT and y[m].sum() >= MIN_ERR)}
        ev_p += list(prob(ka, kb, x[~m])); ev_y += list(y[~m])
    ev_p, ev_y = np.asarray(ev_p), np.asarray(ev_y)
    score = (1 - ev_p) * 100

    # 검증 ① 보정이 맞는가(예측 오류율 ↔ 실제 오류율) ② 맞은/틀린 자리의 점수 분포
    L = ["# 음소 점수 보정 — margin → 오류 확률", "",
         f"- 자리 {len(allx):,} · 묶음 {len(X)} · 실제 오류율 {ally.mean()*100:.2f}%",
         f"- 묶음별로 따로 맞춘 것 {sum(1 for k,v in cal.items() if k!='_global' and v['own_fit'])}개, 나머지는 전체 모형",
         "", "## 보정 확인 (검증셋)", "", "| 예측 점수 구간 | 자리 수 | 실제 오류율 | 기대 오류율 |", "|---|--:|--:|--:|"]
    for lo, hi in ((0, 40), (40, 60), (60, 80), (80, 90), (90, 101)):
        m = (score >= lo) & (score < hi)
        if m.sum():
            L.append(f"| {lo}~{hi} | {int(m.sum()):,} | {ev_y[m].mean()*100:.1f}% | {(1-score[m]/100).mean()*100:.1f}% |")
    ok, bad = score[ev_y == 0], score[ev_y == 1]
    L += ["", "## 점수 분포", "", "| 자리 | 중앙값 | 하위 10% | 상위 10% |", "|---|--:|--:|--:|",
          f"| 맞게 발음한 자리 | {np.median(ok):.0f} | {np.percentile(ok,10):.0f} | {np.percentile(ok,90):.0f} |",
          f"| 틀리게 발음한 자리 | {np.median(bad):.0f} | {np.percentile(bad,10):.0f} | {np.percentile(bad,90):.0f} |",
          "", f"- 맞은 자리 중 80점 이상 비율 **{(ok>=80).mean()*100:.1f}%**",
          f"- 틀린 자리 중 80점 미만 비율 **{(bad<80).mean()*100:.1f}%**"]
    os.makedirs(a.out, exist_ok=True)
    json.dump(cal, open(os.path.join(a.out, "calibration.json"), "w", encoding="utf-8"), ensure_ascii=False)
    open(os.path.join(a.out, "calibration_report.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    main()
