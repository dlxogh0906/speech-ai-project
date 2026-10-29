# -*- coding: utf-8 -*-
"""SupCon **오프라인 탐침** — 인코더를 고정한 채, 세기 계열 자음을 가를 정보가 표현 안에 있는지 잰다.

입력은 `scripts/eval/extract_phone_embeddings.py` 의 npz(train · valid · valid_unseen). 비교하는 네 가지:
  head      CTC 헤드 그대로 — 두 자모의 로그확률 차이
  head_lr   헤드 출력(14자모 로그확률)을 로지스틱으로 재보정 — "헤드가 이미 내놓는 정보"의 상한
  emb_lr    고정 임베딩 위 로지스틱 — "표현 안에 있는 정보"
  supcon    임베딩 → 작은 투영망(SupCon 손실, 같은 실제 자모끼리 모으고 다른 자모는 밀어냄) → 로지스틱

판단: emb_lr·supcon 이 head_lr 보다 **처음 보는 문항(valid_unseen)에서도** 뚜렷이 높으면, 표현에는 정보가 있는데
헤드가 못 쓰는 것 → CTC+SupCon 결합 학습에 기대할 게 있다. 비슷하면 헤드가 이미 다 쓰고 있는 것 → 기대가 작다.

지표(임계값·클래스 비율에 흔들리지 않게):
  pair AUC  표준 c 인 자리에서 "실제 a 로 말함" vs "실제 c 로 말함" 을 점수 log p(a) − log p(c) 로 가르는 AUC
  family balanced accuracy  계열(ㄱㄲㅋ 등) 안에서 실제 자모 맞히기, 클래스 균형 정확도
학습은 train(본 문항·학습 화자)만, 평가는 valid(본 문항)·valid_unseen(처음 보는 문항). test 는 보지 않는다.

    $PY .../probe_supcon_offline.py --prefix $MDD_OUTPUTS/analysis/supcon_probe/holdout_v1_H0 --layer 24
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import argparse, json, os, time
import numpy as np

FAMILIES = {u"ㄱ계열": u"ㄱㄲㅋ", u"ㄷ계열": u"ㄷㄸㅌ", u"ㅂ계열": u"ㅂㅃㅍ", u"ㅈ계열": u"ㅈㅉㅊ", u"ㅅ계열": u"ㅅㅆ"}
PAIRS = [u"ㅅ>ㅆ", u"ㅆ>ㅅ", u"ㅈ>ㅊ", u"ㄱ>ㅋ", u"ㄸ>ㄷ", u"ㅉ>ㅈ", u"ㄱ>ㄲ", u"ㅋ>ㄱ", u"ㄷ>ㄸ", u"ㅂ>ㅍ", u"ㅍ>ㅂ"]


def load(path, layer):
    d = np.load(path)
    return {"X": d["emb_%d" % layer].astype(np.float32), "lp": d["lp"].astype(np.float32),
            "y": d["actual"], "c": d["canon"], "targets": [str(t) for t in d["targets"]]}


def supcon_train(X, y, dim=128, steps=2000, per_class=32, tau=0.1, seed=0):
    """클래스 균형 배치의 Supervised Contrastive(Khosla 2020). 같은 계열 자모가 한 배치에 늘 함께 있어 hard negative 가 된다."""
    import torch
    torch.manual_seed(seed); rng = np.random.RandomState(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mu, sd = X.mean(0), X.std(0) + 1e-5
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 512), torch.nn.ReLU(), torch.nn.Linear(512, dim)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    cls = [np.where(y == k)[0] for k in range(int(y.max()) + 1)]
    cls = [c for c in cls if len(c) >= 2]
    Xt = torch.from_numpy((X - mu) / sd).to(dev)
    for it in range(steps):
        idx = np.concatenate([rng.choice(c, per_class, replace=len(c) < per_class) for c in cls])
        z = torch.nn.functional.normalize(net(Xt[idx]), dim=1)
        lab = torch.from_numpy(y[idx]).to(dev)
        sim = z @ z.t() / tau
        eye = torch.eye(len(idx), device=dev, dtype=torch.bool)
        sim = sim.masked_fill(eye, -1e9)
        logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        pos = (lab[:, None] == lab[None, :]) & ~eye
        loss = -(logp * pos).sum(1) / pos.sum(1).clamp(min=1)
        loss = loss.mean()
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()

    def project(Z):
        with torch.no_grad():
            out = []
            for s in range(0, len(Z), 8192):
                out.append(torch.nn.functional.normalize(net(torch.from_numpy((Z[s:s + 8192] - mu) / sd).to(dev)), dim=1).cpu().numpy())
            return np.concatenate(out) if out else np.zeros((0, dim), np.float32)
    return project, float(loss.item())


def fit_lr(F, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(F)
    m = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced").fit(sc.transform(F), y)
    return lambda Z: m.predict_log_proba(sc.transform(Z)), list(m.classes_)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", required=True, help="<prefix>_{train,valid,valid_unseen}.npz")
    ap.add_argument("--layer", type=int, default=24)
    ap.add_argument("--supcon-steps", dest="steps", type=int, default=2000)
    ap.add_argument("--min-err", dest="min_err", type=int, default=20, help="AUC 를 보고할 최소 오류 자리 수")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score
    t0 = time.time()
    D = dict((s, load("%s_%s.npz" % (a.prefix, s), a.layer)) for s in ("train", "valid", "valid_unseen"))
    T = D["train"]["targets"]; ti = dict((t, i) for i, t in enumerate(T))
    print("[data] %s · layer %d · 구간 train %d · valid %d · valid_unseen %d" % (
        os.path.basename(a.prefix), a.layer, len(D["train"]["y"]), len(D["valid"]["y"]), len(D["valid_unseen"]["y"])), flush=True)

    project, sc_loss = supcon_train(D["train"]["X"], D["train"]["y"], steps=a.steps)
    P = dict((s, project(D[s]["X"])) for s in D)
    print("[supcon] 마지막 손실 %.3f (%.0fs)" % (sc_loss, time.time() - t0), flush=True)

    res = {"prefix": a.prefix, "layer": a.layer, "families": {}, "pairs": {}}
    feats = {"head_lr": lambda s: D[s]["lp"], "emb_lr": lambda s: D[s]["X"], "supcon": lambda s: P[s]}
    for fam, members in FAMILIES.items():
        ids = [ti[m] for m in members]
        sel = dict((s, np.isin(D[s]["y"], ids) & np.isin(D[s]["c"], ids)) for s in D)
        models = {}
        for name, fget in feats.items():
            models[name] = fit_lr(fget("train")[sel["train"]], D["train"]["y"][sel["train"]])
        res["families"][fam] = {}
        for s in ("valid", "valid_unseen"):
            m = sel[s]; y = D[s]["y"][m]
            if m.sum() < 20:
                continue
            row = {"n": int(m.sum()), "n_error": int((D[s]["c"][m] != y).sum())}
            head_pred = np.array(ids)[D[s]["lp"][m][:, ids].argmax(1)]
            row["head"] = round(float(balanced_accuracy_score(y, head_pred)), 4)
            for name, (pred, classes) in models.items():
                lp = pred(feats[name](s)[m])
                row[name] = round(float(balanced_accuracy_score(y, np.array(classes)[lp.argmax(1)])), 4)
            res["families"][fam][s] = row
        # 쌍별 AUC — 같은 계열 모델로
        for pair in PAIRS:
            c_, a_ = pair.split(">")
            if c_ not in members or a_ not in members:
                continue
            res["pairs"].setdefault(pair, {})
            for s in ("valid", "valid_unseen"):
                m = (D[s]["c"] == ti[c_]) & np.isin(D[s]["y"], [ti[c_], ti[a_]])
                yy = (D[s]["y"][m] == ti[a_]).astype(int)
                if yy.sum() < a.min_err or (1 - yy).sum() < a.min_err:
                    res["pairs"][pair][s] = {"n_error": int(yy.sum()), "note": "표본 부족"}
                    continue
                row = {"n_error": int(yy.sum()), "n_correct": int((1 - yy).sum())}
                row["head"] = round(float(roc_auc_score(yy, D[s]["lp"][m][:, ti[a_]] - D[s]["lp"][m][:, ti[c_]])), 4)
                for name, (pred, classes) in models.items():
                    lp = pred(feats[name](s)[m])
                    row[name] = round(float(roc_auc_score(yy, lp[:, classes.index(ti[a_])] - lp[:, classes.index(ti[c_])])), 4)
                res["pairs"][pair][s] = row
    out = a.out or "%s_probe_L%d.json" % (a.prefix, a.layer)
    json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)
    print("\n계열 균형정확도 (valid → valid_unseen)  head / head_lr / emb_lr / supcon")
    for fam, r in res["families"].items():
        for s in ("valid", "valid_unseen"):
            if s in r:
                x = r[s]
                print("  %-5s %-12s n=%6d 오류 %5d | %.3f / %.3f / %.3f / %.3f" % (fam, s, x["n"], x["n_error"], x["head"], x["head_lr"], x["emb_lr"], x["supcon"]))
    print("\n쌍별 AUC (실제 오류 vs 정상, 표준 자모 고정)  head / head_lr / emb_lr / supcon")
    for pair, r in res["pairs"].items():
        for s in ("valid", "valid_unseen"):
            x = r.get(s, {})
            if "head" in x:
                print("  %-5s %-12s 오류 %4d | %.3f / %.3f / %.3f / %.3f" % (pair, s, x["n_error"], x["head"], x["head_lr"], x["emb_lr"], x["supcon"]))
            else:
                print("  %-5s %-12s 오류 %4s | 표본 부족" % (pair, s, x.get("n_error", "-")))
    print("[saved] %s (%.0fs)" % (out, time.time() - t0))


if __name__ == "__main__":
    main()
