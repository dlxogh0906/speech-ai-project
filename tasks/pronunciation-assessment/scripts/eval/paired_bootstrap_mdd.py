# -*- coding: utf-8 -*-
"""경량화 변형 vs 기준 모델 — MDD 정확도 차이의 **쌍 부트스트랩 신뢰구간**.

부분집합(1만 발화)에서 오류 F1 의 표본 오차는 ±0.01 수준이라 "차이 없음"을 점추정치 비교로는 말할 수 없다.
같은 발화에 두 모델을 태운 **쌍 비교**라 차이의 분산은 훨씬 작다 — 그 구간을 잰다.

채점은 `mdd_eval_fixedref.py` 와 같다(고정 정답지 · canonical 축 · 자모). 그 스크립트의 `diff_vs_canon` 을
그대로 import 한다. 발화마다 오류 클래스 TP/FP/FN/TN 을 세고, 화자 단위로 묶어 재표집한다
(같은 화자의 발화는 서로 상관이 있어 발화 단위 재표집은 구간을 과소평가한다).
점추정치는 fixedref 의 jamo 결과와 **정확히 같아야 한다** — 스크립트가 스스로 대조한다(--fixedref_json).

    $PY tasks/pronunciation-assessment/scripts/eval/paired_bootstrap_mdd.py --manifest <dir> \\
        --ref preds_torch_gpu.jsonl --cand preds_int8.jsonl [--cand ...] --out boot.json
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import argparse, collections, importlib.util, json, os
import numpy as np


def load_fixedref():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mdd_eval_fixedref.py")
    spec = importlib.util.spec_from_file_location("mdd_eval_fixedref", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_preds(path, n):
    P = [None] * n
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        P[d["i"]] = d["pred"]
    if any(p is None for p in P):
        raise SystemExit(f"{path}: 빠진 행이 있다")
    return P


def metrics(c):
    """c = [TP, FP, FN, TN] 합계 → 오류 F1 · weighted F1 (sklearn average='weighted' 와 같은 식)."""
    tp, fp, fn, tn = c
    def f1(t, p_, n_):
        pr = t / max(1, t + p_); rc = t / max(1, t + n_)
        return 2 * pr * rc / max(1e-12, pr + rc)
    f_err = f1(tp, fp, fn)
    f_ok = f1(tn, fn, fp)
    s_err, s_ok = tp + fn, tn + fp
    return f_err, (f_err * s_err + f_ok * s_ok) / max(1, s_err + s_ok)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--ref", required=True, help="기준 예측 캐시")
    ap.add_argument("--cand", action="append", required=True, help="비교 대상 예측 캐시 (여러 번 줄 수 있다)")
    ap.add_argument("--fixedref_json", action="append", default=[],
                    help="같은 순서로 ref, cand... 의 mdd_eval_fixedref 결과 — 점추정치 대조용(선택)")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    fx = load_fixedref()
    M = a.manifest
    rel = [l.split("\t")[0] for l in open(f"{M}/test.tsv").read().splitlines()[1:]]
    canon = [l.split() for l in open(f"{M}/test.canon", encoding="utf-8").read().splitlines()]
    manual = [l.split() for l in open(f"{M}/test.ltr", encoding="utf-8").read().splitlines()]
    N = len(rel)
    spk = [os.path.basename(r).split("-")[0] for r in rel]
    X = [fx.diff_vs_canon("".join(canon[i]), "".join(manual[i])) for i in range(N)]

    def counts(P):
        out = np.zeros((N, 4), np.int64)
        for i in range(N):
            p = "".join(t for t in (P[i] or []) if t in fx.JAMO)
            y = fx.diff_vs_canon("".join(canon[i]), p)
            x = X[i]
            tp = sum(u & v for u, v in zip(x, y))
            out[i] = (tp, sum(y) - tp, sum(x) - tp, len(x) - sum(x) - sum(y) + tp)
        return out

    paths = [a.ref] + a.cand
    C = [counts(load_preds(p, N)) for p in paths]

    spk_ids = sorted(set(spk))
    sidx = {s: k for k, s in enumerate(spk_ids)}
    S = [np.zeros((len(spk_ids), 4), np.int64) for _ in paths]
    for i, s in enumerate(spk):
        for m in range(len(paths)):
            S[m][sidx[s]] += C[m][i]

    point = [metrics(S[m].sum(0)) for m in range(len(paths))]
    checks = []
    for m, fj in enumerate(a.fixedref_json):
        ref_j = json.load(open(fj))["jamo"]
        ok = bool(abs(round(float(point[m][0]), 4) - ref_j["err_f1"]) < 1e-9
                  and abs(round(float(point[m][1]), 4) - ref_j["f1_weighted"]) < 1e-9)
        checks.append({"preds": paths[m], "fixedref": fj, "match": ok,
                       "ours": [round(point[m][0], 4), round(point[m][1], 4)],
                       "fixedref_vals": [ref_j["err_f1"], ref_j["f1_weighted"]]})
        if not ok:
            raise SystemExit(f"[stop] 점추정치가 fixedref 와 다르다 — 구현 오류: {checks[-1]}")

    rng = np.random.default_rng(a.seed)
    boots = [[] for _ in a.cand]
    for _ in range(a.n_boot):
        pick = rng.integers(0, len(spk_ids), len(spk_ids))
        r_err, r_w = metrics(S[0][pick].sum(0))
        for k in range(len(a.cand)):
            c_err, c_w = metrics(S[k + 1][pick].sum(0))
            boots[k].append((c_err - r_err, c_w - r_w))

    rep = {"manifest": M, "n_utts": N, "n_speakers": len(spk_ids), "n_boot": a.n_boot, "unit": "speaker",
           "scorer": "fixedref jamo (canonical-anchored)", "ref": {"preds": a.ref, "err_f1": point[0][0],
                                                                    "f1_weighted": point[0][1]},
           "point_checks": checks, "cands": []}
    print(f"기준 {os.path.basename(a.ref)}: 오류 F1 {point[0][0]:.4f} · weighted F1 {point[0][1]:.4f} "
          f"({N:,}발화 · 화자 {len(spk_ids):,} · 부트스트랩 {a.n_boot})")
    for k, p in enumerate(a.cand):
        b = np.array(boots[k])
        d_err, d_w = point[k + 1][0] - point[0][0], point[k + 1][1] - point[0][1]
        lo_e, hi_e = np.percentile(b[:, 0], [2.5, 97.5])
        lo_w, hi_w = np.percentile(b[:, 1], [2.5, 97.5])
        n_diff_utts = int((C[k + 1] != C[0]).any(1).sum())
        rep["cands"].append({"preds": p, "err_f1": point[k + 1][0], "f1_weighted": point[k + 1][1],
                             "d_err_f1": d_err, "d_err_f1_ci95": [float(lo_e), float(hi_e)],
                             "d_f1_weighted": d_w, "d_f1_weighted_ci95": [float(lo_w), float(hi_w)],
                             "utts_with_different_scoring": n_diff_utts})
        print(f"  {os.path.basename(p)}: 오류 F1 {point[k+1][0]:.4f} (Δ {d_err:+.4f}, 95% [{lo_e:+.4f}, {hi_e:+.4f}]) · "
              f"weighted F1 {point[k+1][1]:.4f} (Δ {d_w:+.4f}, 95% [{lo_w:+.4f}, {hi_w:+.4f}]) · "
              f"채점이 달라진 발화 {n_diff_utts:,}")
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    print(f"→ {a.out}")
    return 0


if __name__ == "__main__":
    _sys.exit(main())
