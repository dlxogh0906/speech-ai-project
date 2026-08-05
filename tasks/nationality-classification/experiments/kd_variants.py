# -*- coding: utf-8 -*-
"""KD 기법 변형 비교 — 꼬리 붕괴(clean macro5 −1.7%p)를 겨냥한 5갈래.

바닐라 KD(kd_2b_to_300m.py)의 결론: 교사가 아는 도메인 +1.8%p, 대신 꼬리 하락
(① 043 무지 누출 ② 교사 자신의 꼬리 약세 전이). 이를 갈래별로 공략한다:

  ls_e*        Label Smoothing 대조군 — 이득이 '교사 지식'인지 '그냥 smoothing'인지 판별
  dkd_*        Decoupled KD (Zhao'22) — KL 을 target(TCKD)/non-target(NCKD) 분리.
               dkd_a0_b1 은 순수 dark knowledge(NCKD)만 — 꼬리 뭉갬이 어느 항 탓인지 분리
  gate_*       교사 확신도(마진) 게이팅 — 실측: 043 마진 중앙 0.355 vs 505 0.988.
               도메인 마스크보다 원리적(교사가 모르는 '행'을 스스로 강등)
  ens_self     앙상블 교사 = 2B 교사 + CE 학생(Born-Again 유사) 소프트 평균
  featkd_*     특징 KD — 닫힌형 ridge 로 1024→1920 투영 후 교사 헤드 합성.
               featkd_ridge 는 학습 0회(직접 평가), featkd_kd 는 그 합성 로짓을 교사로 KD

전 변형 공통: sweep300 스케줄 재현, 행별 손실 배합, valid macro5 선택, clean 분해 병기.
주의: 변형이 많을수록 valid 다중선택 편향이 커진다 — 표 전체를 보고할 것(승자만 X).

    $PY tasks/nationality-classification/experiments/kd_variants.py --quick   # 스모크
    $PY tasks/nationality-classification/experiments/kd_variants.py           # 전체 (~2h CPU)
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), str(_r / "tasks/nationality-classification/experiments"),
                 *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import NAT_TASK

import argparse, json, os, time
from collections import Counter
import numpy as np

import kd_2b_to_300m as K                       # 인덱스·gather·교사로짓·평가 재사용
from country_classifier.make_speaker_split import assign_speakers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--alpha", type=float, default=0.3, help="게이트·앙상블 변형의 기본 α (바닐라 최적)")
    ap.add_argument("--T", type=float, default=2.0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(NAT_TASK / "outputs" / "kd_variants" / "kd_variants.json"))
    A = ap.parse_args()
    import torch, torch.nn as nn, torch.nn.functional as F
    dev = A.device
    t0 = time.time()
    T0 = A.T

    print("[1/6] 인덱스·split·오염 마스크", flush=True)
    s_path, s_row, t_path, t_row, sps, lab, dom = K.build_index()
    ko2en = json.load(open(NAT_TASK / "configs" / "country41_ko2en.json", encoding="utf-8"))
    assert len(set(ko2en.values())) == len(ko2en)
    lab_en = np.array([ko2en[l] for l in lab], dtype=object)
    classes = sorted(set(lab_en.tolist())); c2i = {c: i for i, c in enumerate(classes)}
    y = np.array([c2i[c] for c in lab_en], np.int64)
    items = [{"speaker_id": s, "label_name": l, "num_frames": 0} for s, l in zip(sps, lab_en)]
    split = np.array([assign_speakers(items, [8, 1, 1], seed=42)[s] for s in sps])
    t_items = [{"speaker_id": s, "label_name": l, "num_frames": 0}
               for s, l, d in zip(sps, lab, dom) if d != "asia043"]
    t_assign = assign_speakers(t_items, [8, 1, 1], seed=42)
    t_train_spk = {s for s, a in t_assign.items() if a == "train"}
    N = len(sps)

    if A.quick:
        rng = np.random.RandomState(0)
        keep = np.zeros(N, bool); keep[rng.choice(N, N // 20, replace=False)] = True
        s_path, s_row, t_path, t_row = s_path[keep], s_row[keep], t_path[keep], t_row[keep]
        sps, y, dom, split = sps[keep], y[keep], dom[keep], split[keep]
        N = keep.sum(); A.epochs = 3
        print(f"    [quick] {N:,}행 · {A.epochs}ep", flush=True)

    tr, va, te = (np.where(split == s)[0] for s in ["train", "valid", "test"])
    ytr, yva, yte = y[tr], y[va], y[te]; dte = dom[te]
    contam_te = np.array([s in t_train_spk for s in sps[te]])

    print("[2/6] 교사 로짓 + 확신도 마진", flush=True)
    TL, t_ko, _ = K.teacher_logits(t_path, t_row, sps)
    t_en = [ko2en[l] for l in t_ko]
    assert len(t_en) == len(set(t_en)) == len(classes)
    TL = TL[:, [t_en.index(c) for c in classes]]
    P1 = np.exp(TL - TL.max(1, keepdims=True)); P1 /= P1.sum(1, keepdims=True)
    srt = np.sort(P1, 1); margin = (srt[:, -1] - srt[:, -2]).astype(np.float32)
    del P1, srt

    print("[3/6] 학생 특징", flush=True)
    X = K.gather(s_path, s_row, K.S_LAYER, 1024, "학생")
    mu = X[tr].mean(0, keepdims=True); sd = X[tr].std(0, keepdims=True) + 1e-6
    Xtr, Xva, Xte = (X[tr] - mu) / sd, (X[va] - mu) / sd, (X[te] - mu) / sd

    print("[4/6] 특징 KD: ridge 투영 1024→1920 (닫힌형, 학습 0회)", flush=True)
    XT = K.gather(t_path, t_row, K.T_LAYER, 1920, "교사특징")
    import torch as th_
    Xs_t = th_.from_numpy(X[tr]); Xt_t = th_.from_numpy(XT[tr])
    G = (Xs_t.T @ Xs_t).numpy(); Cxy = (Xs_t.T @ Xt_t).numpy()   # (1024,1024),(1024,1920)
    del Xs_t, Xt_t
    import torch as _t
    thd = _t.load(K.TEACHER_HEAD, map_location="cpu")
    Wt = thd["head"]["weight"].numpy().astype(np.float32); bt = thd["head"]["bias"].numpy().astype(np.float32)
    mut = np.asarray(thd["mu"], np.float32); sdt = np.asarray(thd["sd"], np.float32)
    Wt = Wt[[t_en.index(c) for c in classes]]; bt = bt[[t_en.index(c) for c in classes]]
    ridge = {}
    for lam in [1e1, 1e2, 1e3]:
        Pmat = np.linalg.solve(G + lam * np.eye(1024, dtype=np.float64), Cxy).astype(np.float32)
        # 합성 선형모델: logits = ((X@P − mu_t)/sd_t)@Wt.T + bt
        def flog(Z, Pm=Pmat):
            return ((Z @ Pm - mut) / sdt) @ Wt.T + bt
        vm = K.macro5(flog(X[va]), yva)
        ridge[lam] = (vm, Pmat)
        print(f"    λ={lam:g}: valid_macro5={vm:.4f}", flush=True)
    lam_best = max(ridge, key=lambda k: ridge[k][0])
    Pbest = ridge[lam_best][1]
    FL = (X @ Pbest - mut) / sdt @ Wt.T + bt if False else ((X @ Pbest - mut) / sdt) @ Wt.T + bt
    del XT, G, Cxy

    At = torch.from_numpy(Xtr).to(dev); Yt = torch.from_numpy(ytr).to(dev)
    Tt = torch.from_numpy(TL[tr]).to(dev)
    Mg = torch.from_numpy(margin[tr]).to(dev)
    nn_ = len(Yt)

    def sched_train(loss_fn):
        """sweep300 스케줄 그대로. loss_fn(z, j) → 스칼라."""
        torch.manual_seed(42)
        head = nn.Linear(Xtr.shape[1], len(classes)).to(dev)
        opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
        def logits(Z):
            head.eval()
            with torch.no_grad(): return head(torch.from_numpy(Z).to(dev)).cpu().numpy()
        best = -1; bw = None
        for ep in range(A.epochs):
            head.train(); pm = torch.randperm(nn_, device=dev)
            for i in range(0, nn_, 4096):
                j = pm[i:i + 4096]
                loss_fn(head(At[j]), j).backward(); opt.step(); opt.zero_grad()
            m = K.macro5(logits(Xva), yva)
            if m > best: best = m; bw = {k: v.detach().clone() for k, v in head.state_dict().items()}
        head.load_state_dict(bw)
        return head, best, logits

    def kl_to(z, soft_p, T):
        return F.kl_div(F.log_softmax(z / T, 1), soft_p, reduction="none").sum(1)

    VAR = {}
    # ── 대조군: CE / LS ──
    VAR["baseline_ce"] = lambda: sched_train(lambda z, j: F.cross_entropy(z, Yt[j]))
    for e in (0.05, 0.1):
        VAR[f"ls_e{e}"] = lambda e=e: sched_train(
            lambda z, j: F.cross_entropy(z, Yt[j], label_smoothing=e))
    # ── 바닐라 KD (참조 재현) ──
    a0 = A.alpha
    def mk_vanilla():
        Pt = F.softmax(Tt / T0, 1)
        def lf(z, j):
            ce = F.cross_entropy(z, Yt[j], reduction="none")
            return ((1 - a0) * ce + a0 * (T0 * T0) * kl_to(z, Pt[j], T0)).mean()
        return sched_train(lf)
    VAR["kd_vanilla"] = mk_vanilla
    # ── DKD ──
    def mk_dkd(a, b):
        def f():
            Pt = F.softmax(Tt / T0, 1)
            pt_t = Pt.gather(1, Yt.view(-1, 1)).squeeze(1).clamp(1e-6, 1 - 1e-6)
            NEG = torch.full_like(Tt, -1e9)
            Tn = Tt.scatter(1, Yt.view(-1, 1), NEG)              # target 마스킹
            Ptn = F.softmax(Tn / T0, 1)
            def lf(z, j):
                ce = F.cross_entropy(z, Yt[j], reduction="none")
                ps = F.softmax(z / T0, 1)
                ps_t = ps.gather(1, Yt[j].view(-1, 1)).squeeze(1).clamp(1e-6, 1 - 1e-6)
                ptj = pt_t[j]
                tckd = ptj * (ptj / ps_t).log() + (1 - ptj) * ((1 - ptj) / (1 - ps_t)).log()
                zn = z.scatter(1, Yt[j].view(-1, 1), torch.full_like(z[:, :1], -1e9))
                nckd = kl_to(zn, Ptn[j], T0)
                return (ce + (a * tckd + b * nckd) * (T0 * T0)).mean()
            return sched_train(lf)
        return f
    VAR["dkd_a1_b0.5"] = mk_dkd(1.0, 0.5)
    VAR["dkd_a1_b2"] = mk_dkd(1.0, 2.0)
    VAR["dkd_a0_b1"] = mk_dkd(0.0, 1.0)         # 순수 dark knowledge
    # ── 확신도 게이팅 ──
    def mk_gate(kind):
        def f():
            Pt = F.softmax(Tt / T0, 1)
            g = Mg if kind == "cont" else (Mg > 0.5).float()
            def lf(z, j):
                ce = F.cross_entropy(z, Yt[j], reduction="none")
                w = a0 * g[j]
                return ((1 - w) * ce + w * (T0 * T0) * kl_to(z, Pt[j], T0)).mean()
            return sched_train(lf)
        return f
    VAR["gate_margin"] = mk_gate("cont")
    VAR["gate_th0.5"] = mk_gate("th")
    # ── 특징 KD 를 교사로 (합성 로짓) ──
    def mk_featkd():
        Ft = torch.from_numpy(FL[tr]).to(dev)
        Pf = F.softmax(Ft / T0, 1)
        def lf(z, j):
            ce = F.cross_entropy(z, Yt[j], reduction="none")
            return ((1 - a0) * ce + a0 * (T0 * T0) * kl_to(z, Pf[j], T0)).mean()
        return sched_train(lf)
    VAR["featkd_kd"] = mk_featkd

    print(f"[5/6] 변형 학습 ({len(VAR)}개 + ens_self, 각 {A.epochs}ep)", flush=True)
    runs = {"meta": {"n": int(N), "epochs": A.epochs, "alpha": a0, "T": T0, "quick": A.quick,
                     "ridge_lambda": lam_best,
                     "selection_note": "변형 다수의 valid 선택 — 다중비교 편향 있음. 표 전체로 판단할 것."},
            "variants": {}}
    # 학습 0회 변형: ridge 직접 평가
    r0 = K.evaluate(FL[te], yte, dte, contam=contam_te)
    r0["valid_macro5"] = round(float(ridge[lam_best][0]), 4); r0["trainings"] = 0
    runs["variants"]["featkd_ridge_direct"] = r0
    print(f"    featkd_ridge_direct(λ={lam_best:g}): test m5={r0['micro5']} mac5={r0['macro5']} m1={r0['micro1']} | 043 top1={r0['by_domain']['043']['top1']}", flush=True)

    base_logits_tr = None
    for name, fn in VAR.items():
        hd, vm, lg = fn()
        r = K.evaluate(lg(Xte), yte, dte, contam=contam_te)
        r["valid_macro5"] = round(float(vm), 4)
        runs["variants"][name] = r
        print(f"    {name}: valid={vm:.4f} | test m5={r['micro5']} mac5={r['macro5']} m1={r['micro1']} | 043 top1={r['by_domain']['043']['top1']} | clean mac5={r['clean']['macro5']}", flush=True)
        if name == "baseline_ce":
            base_logits_tr = lg(Xtr)                      # 앙상블 교사용
    # ── 앙상블 교사 (2B + CE 학생) ──
    Bt = torch.from_numpy(base_logits_tr).to(dev)
    Pe = (F.softmax(Tt / T0, 1) + F.softmax(Bt / T0, 1)) / 2
    def lf_ens(z, j):
        ce = F.cross_entropy(z, Yt[j], reduction="none")
        return ((1 - a0) * ce + a0 * (T0 * T0) * kl_to(z, Pe[j], T0)).mean()
    hd, vm, lg = sched_train(lf_ens)
    r = K.evaluate(lg(Xte), yte, dte, contam=contam_te); r["valid_macro5"] = round(float(vm), 4)
    runs["variants"]["ens_self"] = r
    print(f"    ens_self: valid={vm:.4f} | test m5={r['micro5']} mac5={r['macro5']} m1={r['micro1']} | 043 top1={r['by_domain']['043']['top1']} | clean mac5={r['clean']['macro5']}", flush=True)

    print("[6/6] 저장", flush=True)
    base = runs["variants"]["baseline_ce"]
    for name, r in runs["variants"].items():
        if name == "baseline_ce": continue
        r["d_micro1"] = round(r["micro1"] - base["micro1"], 4)
        r["d_macro5"] = round(r["macro5"] - base["macro5"], 4)
        r["d_clean_macro5"] = round(r["clean"]["macro5"] - base["clean"]["macro5"], 4)
        r["d_043_top1"] = round(r["by_domain"]["043"]["top1"] - base["by_domain"]["043"]["top1"], 4)
    runs["meta"]["elapsed_sec"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(A.out), exist_ok=True)
    json.dump(runs, open(A.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[saved] {A.out} · {runs['meta']['elapsed_sec']}s")


if __name__ == "__main__":
    main()
