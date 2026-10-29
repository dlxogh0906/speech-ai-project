# -*- coding: utf-8 -*-
"""쌍별 임계값 오프라인 실험 — 재학습 없이 판정 기준만 쌍마다 옮겨서 H4 가중의 이득을 따라잡을 수 있나(results/19).

SupCon 탐침이 저장한 자리별 로그확률(`extract_phone_embeddings.py` → `<prefix>_<split>.npz`)만 쓴다. GPU 불필요.
표준 자모 c 인 자리 중 실제(TAG)가 c 또는 a 인 곳에서 점수 s = log p(a) − log p(c) 로 "a 로 말했다"를 지적한다.

  τ=0        : 두 자모 중 더 높은 쪽 — 모델 기본 판정의 대용(그리디 전사와 같지는 않다)
  전역 τ      : 모든 쌍에 같은 τ 하나(valid 에서 쌍을 모은 F1 최대)
  쌍별 τ      : 쌍마다 따로(valid 에서 그 쌍 F1 최대)

임계값은 valid(본 문항)에서만 고르고 valid_unseen(처음 보는 문항)은 마지막에 한 번 본다.
한계: 자리 구간은 TAG(실제 발음)에 강제정렬한 것이다 — 서비스는 표준발음에 정렬하므로 이 값은 **분리 가능성의 상한**에 가깝다.

    $PY .../pair_threshold_offline.py --out $MDD_OUTPUTS/analysis/supcon_probe/pair_threshold.json
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import MDD_OUTPUTS

import argparse, json, os

import numpy as np

PAIRS = [("ㅅ", "ㅆ"), ("ㄱ", "ㅋ"), ("ㄱ", "ㄲ"), ("ㅈ", "ㅊ"), ("ㅈ", "ㅉ"),
         ("ㄷ", "ㅌ"), ("ㄷ", "ㄸ"), ("ㅂ", "ㅍ"), ("ㅂ", "ㅃ")]


def load(prefix, split):
    z = np.load("%s_%s.npz" % (prefix, split), allow_pickle=True)
    return {"T": list(z["targets"]), "lp": z["lp"].astype(np.float32), "actual": z["actual"], "canon": z["canon"]}


def pair_sites(D, c, a):
    ic, ia = D["T"].index(c), D["T"].index(a)
    m = (D["canon"] == ic) & ((D["actual"] == ic) | (D["actual"] == ia))
    return D["lp"][m, ia] - D["lp"][m, ic], (D["actual"][m] == ia)


def stats(s, y, tau):
    f = s > tau
    tp, fp = int((f & y).sum()), int((f & ~y).sum())
    npos, nneg = int(y.sum()), int((~y).sum())
    rec = tp / max(1, npos); prec = tp / max(1, tp + fp)
    return {"n_err": npos, "n_ok": nneg, "flag": tp + fp, "recall": rec, "precision": prec,
            "f1": 2 * prec * rec / max(1e-9, prec + rec), "fp_rate": fp / max(1, nneg), "fp": fp}


def best_tau(s, y, grid):
    return max(grid, key=lambda t: (stats(s, y, t)["f1"], -abs(t)))


def auc(s, y):
    p, n = s[y], s[~y]
    if not len(p) or not len(n):
        return float("nan")
    r = np.argsort(np.argsort(np.concatenate([p, n]))) + 1
    return float((r[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    base = os.path.join(str(MDD_OUTPUTS), "analysis", "supcon_probe")
    ap.add_argument("--models", nargs="+", default=["holdout_v1_H0", "holdout_v1_H4_b0.5"])
    ap.add_argument("--dir", default=base)
    ap.add_argument("--out", default=os.path.join(base, "pair_threshold.json"))
    a = ap.parse_args()
    grid = np.round(np.arange(-8, 8.01, 0.1), 2)
    res = {}
    for mname in a.models:
        pre = os.path.join(a.dir, mname)
        V, U = load(pre, "valid"), load(pre, "valid_unseen")
        # 전역 τ: valid 에서 쌍을 모아 F1 최대
        sv = [pair_sites(V, c, x) for c, x in PAIRS]
        S = np.concatenate([s for s, _ in sv]); Y = np.concatenate([y for _, y in sv])
        tg = best_tau(S, Y, grid)
        R = {"global_tau": float(tg), "pairs": {}}
        for (c, x), (s_v, y_v) in zip(PAIRS, sv):
            tp_ = best_tau(s_v, y_v, grid)
            s_u, y_u = pair_sites(U, c, x)
            R["pairs"][c + "→" + x] = {
                "pair_tau": float(tp_),
                "valid": {"auc": auc(s_v, y_v), "tau0": stats(s_v, y_v, 0.0), "pair": stats(s_v, y_v, tp_)},
                "unseen": {"auc": auc(s_u, y_u), "tau0": stats(s_u, y_u, 0.0), "global": stats(s_u, y_u, tg),
                           "pair": stats(s_u, y_u, tp_)}}
        res[mname] = R
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    H0, H4 = a.models[0], a.models[-1]
    print("처음 보는 문항(valid_unseen) — 보존(재현율) / 헛지적률(실제 c 인데 a 로 지적) · 임계값은 valid 에서 고름")
    print("%-6s %5s %5s | %-15s %-15s %-15s | %-15s %-15s | AUC H0→H4" % (
        "쌍", "오류", "정상", "H0 τ=0", "H0 전역τ", "H0 쌍별τ", "H4 τ=0", "H4 쌍별τ"))
    f = lambda d: "%5.1f%% / %4.1f%%" % (100 * d["recall"], 100 * d["fp_rate"])
    for k in res[H0]["pairs"]:
        p0, p4 = res[H0]["pairs"][k], res[H4]["pairs"][k]
        u0, u4 = p0["unseen"], p4["unseen"]
        print("%-6s %5d %5d | %-15s %-15s %-15s | %-15s %-15s | %.3f→%.3f  (τ H0 %.1f · H4 %.1f)" % (
            k, u0["tau0"]["n_err"], u0["tau0"]["n_ok"], f(u0["tau0"]), f(u0["global"]), f(u0["pair"]),
            f(u4["tau0"]), f(u4["pair"]), u0["auc"], u4["auc"], p0["pair_tau"], p4["pair_tau"]))
    print("전역 τ: H0 %.1f · H4 %.1f" % (res[H0]["global_tau"], res[H4]["global_tau"]))
    print("[saved] %s" % a.out)


if __name__ == "__main__":
    main()
