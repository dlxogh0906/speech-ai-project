# -*- coding: utf-8 -*-
"""음소 단위 오발음 판별기(phone detector) — 문장에 구애받지 않는 발음 진단.

입력은 `extract_phone_feats.py` 가 만든 **표준발음 자리별 음향 증거**다(원어민 Stage 1 모델 · 강제정렬).
판별기는 발화의 음소열 위에서 도는 작은 트랜스포머이고, 자리마다 "오발음인가"(이진)와
"실제로 낸 소리가 무엇인가"(46자모 + 삭제, 보조 과제)를 낸다. 받아쓰기를 하지 않으므로 문장을 외울 길이 없다.

채점은 `eval_actual_ctc.py` 의 MDD 정의와 같다(표준발음 축):
    TR=오발음&지적 · FA=오발음&미지적 · FR=정상&지적 · TA=정상&미지적
    Precision=TR/(TR+FR) · Recall=TR/(TR+FA) · FRR=FR/(TA+FR)
정렬에 실패한 발화의 자리는 전부 "미지적"으로 센다(빼지 않는다).
임계값은 valid_unseen 에서 F1 최대로 고르고 test_unseen 에 그대로 적용한다.

    $PY tasks/pronunciation-assessment/scripts/train/train_phone_detector.py --feats <dir> --out <dir>/run1
    # 절제: --no_emb(은닉 벡터 제외) · --layers 0(문맥 제외) · --no_l1 · --no_phone · --train_frac 0.25
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import argparse, collections, json, math, os, time

from mdd.phone_detector import build_model, cond_logits      # 모델 정의는 서빙과 공유한다(가중치 키가 같아야 한다)

import numpy as np


# ─────────────────────────── 데이터 ───────────────────────────
class Split:
    def __init__(self, d, use_emb=True, l1_vocab=None):
        ld = lambda n: np.load(os.path.join(d, n), mmap_mode="r")
        self.info = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
        self.off = np.load(os.path.join(d, "offsets.npy")); self.ok = np.load(os.path.join(d, "ok.npy"))
        self.scal = np.asarray(ld("scal.npy"), np.float32)
        self.lps = np.asarray(ld("lpp_spike.npy")); self.lpn = np.asarray(ld("lpp_span.npy"))
        self.cid = np.asarray(ld("canon_id.npy"), np.int64); self.rid = np.asarray(ld("real_id.npy"), np.int64)
        self.y = np.asarray(ld("y.npy"), np.int64)
        self.el = np.asarray(ld("emb_last.npy")) if use_emb else None
        self.em = np.asarray(ld("emb_mid.npy")) if use_emb and os.path.exists(os.path.join(d, "emb_mid.npy")) else None
        self.utt = [json.loads(l) for l in open(os.path.join(d, "utt.jsonl"), encoding="utf-8")]
        self.n = len(self.utt)
        self.len = np.diff(self.off)
        self.l1_vocab = l1_vocab
        self.l1 = None

    def set_l1(self, vocab):
        self.l1 = np.array([vocab.get(u.get("language") or "?", 0) for u in self.utt], np.int64)

    def get(self, k):
        a, e = int(self.off[k]), int(self.off[k + 1])
        return (self.scal[a:e], self.lps[a:e], self.lpn[a:e], self.cid[a:e], self.y[a:e], self.rid[a:e],
                None if self.el is None else self.el[a:e], None if self.em is None else self.em[a:e])


class Multi:
    """여러 학습 split 을 하나처럼 — 은닉 벡터(수십 GB)를 이어붙이지 않고 발화 단위로 해당 split 의 memmap 에서 읽는다."""
    def __init__(self, parts):
        self.parts = parts
        self.info = parts[0].info
        self.cum = np.cumsum([0] + [p.n for p in parts])
        self.n = int(self.cum[-1])
        self.len = np.concatenate([p.len for p in parts]); self.ok = np.concatenate([p.ok for p in parts])
        self.y = np.concatenate([p.y for p in parts]); self.utt = [u for p in parts for u in p.utt]
        self.scal, self.lps, self.el, self.em = parts[0].scal, parts[0].lps, parts[0].el, parts[0].em      # 통계·차원 확인용(첫 split)
        self.l1 = None

    def set_l1(self, vocab):
        for p in self.parts:
            p.set_l1(vocab)
        self.l1 = np.concatenate([p.l1 for p in self.parts])

    def get(self, k):
        s = int(np.searchsorted(self.cum, k, side="right") - 1)
        return self.parts[s].get(k - int(self.cum[s]))


def make_batch(S, ks, dev, torch, stats, use_emb):
    L = int(S.len[ks].max()); B = len(ks)
    V = S.lps.shape[1]
    small = np.zeros((B, L, S.scal.shape[1] + 3 * V), np.float32)
    cid = np.zeros((B, L), np.int64); y = np.zeros((B, L), np.float32); rid = np.full((B, L), -100, np.int64)
    mask = np.zeros((B, L), bool)
    el = np.zeros((B, L, S.el.shape[1]), np.float16) if use_emb else None
    em = np.zeros((B, L, S.em.shape[1]), np.float16) if use_emb and S.em is not None else None
    for b, k in enumerate(ks):
        sc_, sp_, sn_, cid_, y_, rid_, el_, em_ = S.get(int(k)); n = len(cid_)
        sp, sn = sp_.astype(np.float32), sn_.astype(np.float32)
        canon_lp = sp[np.arange(n), cid_][:, None]
        small[b, :n] = np.concatenate([(sc_ - stats["m"]) / stats["s"], sp / 10.0, sn / 10.0, (sp - canon_lp) / 10.0], -1)
        cid[b, :n] = cid_; y[b, :n] = y_; mask[b, :n] = True
        r = np.array(rid_, np.int64); r[r < 0] = V                # 삭제 = 클래스 V
        rid[b, :n] = r
        if use_emb:
            el[b, :n] = el_
            if em is not None:
                em[b, :n] = em_
    t = lambda x, dt=None: None if x is None else torch.from_numpy(x).to(dev, non_blocking=True).to(dt) if dt else torch.from_numpy(x).to(dev)
    return (t(small), t(cid), t(el, torch.float32), t(em, torch.float32), torch.from_numpy(S.l1[ks]).to(dev),
            t(mask), t(y), t(rid))


# ─────────────────────────── 채점 ───────────────────────────
def mdd_at(y, flag):
    tr = int((flag & (y == 1)).sum()); fa = int((~flag & (y == 1)).sum())
    fr = int((flag & (y == 0)).sum()); ta = int((~flag & (y == 0)).sum())
    p = tr / (tr + fr) if tr + fr else 0.0; r = tr / (tr + fa) if tr + fa else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0,
            "frr": round(fr / max(1, ta + fr), 4), "far": round(fa / max(1, fa + tr), 4), "TR": tr, "FA": fa, "FR": fr, "TA": ta}


def best_f1_threshold(y, s):
    o = np.argsort(-s); ys = y[o]; tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
    f1 = 2 * tp / np.maximum(1, 2 * tp + fp + (ys.sum() - tp))
    k = int(f1.argmax())
    return float(s[o][k]), float(f1[k])


def curve_points(y, s):
    from sklearn.metrics import average_precision_score, roc_auc_score
    o = np.argsort(-s); ys = y[o]; tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
    P, N = ys.sum(), (1 - ys).sum()
    rec, prec, frr = tp / P, tp / np.maximum(1, tp + fp), fp / N
    at = lambda arr, v: int(np.searchsorted(arr, v))
    out = {"auc": round(float(roc_auc_score(y, s)), 4), "ap": round(float(average_precision_score(y, s)), 4)}
    for v in (0.01, 0.02, 0.05):
        k = min(len(frr) - 1, at(frr, v)); out[f"recall@frr{v}"] = round(float(rec[k]), 4); out[f"prec@frr{v}"] = round(float(prec[k]), 4)
    for v in (0.5, 0.74):
        k = min(len(rec) - 1, at(rec, v)); out[f"prec@recall{v}"] = round(float(prec[k]), 4); out[f"frr@recall{v}"] = round(float(frr[k]), 4)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feats", required=True, help="extract_phone_feats.py 의 --out 폴더")
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", default="train", help="학습 split. 쉼표로 여러 개(train,train_extra)"); ap.add_argument("--valid", default="valid_unseen")
    ap.add_argument("--tests", default="test_unseen,valid")
    ap.add_argument("--dim", type=int, default=256); ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4); ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=8); ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--pos_weight", type=float, default=3.0)
    ap.add_argument("--aux", type=float, default=0.5, help="실제 발음(오발음 자리 조건부) 보조 손실 가중치")
    ap.add_argument("--aux_all", type=float, default=0.3, help="전 자리 실제 발음 예측 보조 손실(정규화용) 가중치 · 0 이면 끔")
    ap.add_argument("--train_frac", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=0, help="배치 조립 스레드 수(0=메인). 데이터가 클 때 memmap 읽기를 겹친다")
    ap.add_argument("--no_emb", action="store_true"); ap.add_argument("--no_phone", action="store_true")
    ap.add_argument("--no_l1", action="store_true"); ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(a.out, exist_ok=True)
    use_emb = not a.no_emb

    t0 = time.time()
    parts = [Split(d if os.path.isabs(d) else os.path.join(a.feats, d), use_emb) for d in a.train.split(",")]
    tr = parts[0] if len(parts) == 1 else Multi(parts)          # --train train,train_extra (절대경로도 된다)
    l1_vocab = {"?": 0}
    for u in tr.utt:
        l1_vocab.setdefault(u.get("language") or "?", len(l1_vocab))
    splits = {a.valid: Split(os.path.join(a.feats, a.valid), use_emb)}
    for s in a.tests.split(","):
        if s and os.path.isdir(os.path.join(a.feats, s)):
            splits[s] = Split(os.path.join(a.feats, s), use_emb)
    for S in [tr] + list(splits.values()):
        S.set_l1(l1_vocab)
    p0 = parts[0]; tok_ok = np.repeat(p0.ok, p0.len)             # 정규화 통계는 첫 split 에서(분포가 같다)
    stats = {"m": p0.scal[tok_ok].mean(0), "s": p0.scal[tok_ok].std(0) + 1e-6}
    V = tr.lps.shape[1]
    print(f"[data] train {tr.n:,}발화/{len(tr.y):,}자리(오발음 {tr.y.mean():.2%}) · L1 {len(l1_vocab)} · 적재 {time.time()-t0:.0f}s", flush=True)

    model = build_model(torch, nn, tr.scal.shape[1] + 3 * V, tr.el.shape[1] if use_emb else 0,
                        use_emb and tr.em is not None, len(l1_vocab), V, a).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    ks_all = np.where(tr.ok)[0]
    if a.train_frac < 1:
        ks_all = np.random.RandomState(a.seed).permutation(ks_all)[: int(len(ks_all) * a.train_frac)]
    steps = a.epochs * math.ceil(len(ks_all) / a.bs)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    warm = max(1, int(0.1 * steps))                       # 선형 워밍업 → 코사인 감쇠 (스텝이 아주 적어도 0 으로 나누지 않는다)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    pw = torch.tensor(a.pos_weight, device=dev)
    print(f"[model] {n_par/1e6:.2f}M params · train utts {len(ks_all):,} · steps {steps:,}", flush=True)

    def predict(S):
        model.eval(); s = np.zeros(len(S.y), np.float32); rp = np.full((len(S.y), 3), -1, np.int64)
        ks_ok = np.where(S.ok)[0]; ks_ok = ks_ok[np.argsort(S.len[ks_ok])]
        with torch.no_grad():
            for i in range(0, len(ks_ok), 1024):
                ks = ks_ok[i:i + 1024]
                small, cid, el, em, l1, mask, y, rid = make_batch(S, ks, dev, torch, stats, use_emb)
                lo, lr_ = model(small, cid, el, em, l1, mask)
                pr = torch.sigmoid(lo).float().cpu().numpy()
                rr = cond_logits(lr_.float(), cid).topk(3, -1).indices.cpu().numpy()
                for b, k in enumerate(ks):
                    n = int(S.len[k]); s[S.off[k]:S.off[k + 1]] = pr[b, :n]; rp[S.off[k]:S.off[k + 1]] = rr[b, :n]
        return s, rp                                        # 정렬 실패 발화는 점수 0 = 미지적

    best, log, step = (-1, None), [], 0
    for ep in range(a.epochs):
        model.train(); perm = np.random.permutation(ks_all); tl, nb = 0.0, 0
        for i in range(0, len(perm), a.bs):
            ks = perm[i:i + a.bs]
            small, cid, el, em, l1, mask, y, rid = make_batch(tr, ks, dev, torch, stats, use_emb)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                lo, lr_ = model(small, cid, el, em, l1, mask)
            lo, lr_ = lo.float(), lr_.float()
            l_err = F.binary_cross_entropy_with_logits(lo[mask], y[mask], pos_weight=pw)
            em_ = mask & (y > 0.5)                              # "무엇으로 냈나"는 오발음 자리에서만 배운다 —
            if em_.any():                                       # 전 자리로 배우면 96% 인 정상 자리 때문에 표준음 복사가 된다
                l_aux = F.cross_entropy(cond_logits(lr_, cid)[em_], rid[em_])
            else:
                l_aux = lo.sum() * 0
            loss = l_err + a.aux * l_aux
            if a.aux_all > 0:
                loss = loss + a.aux_all * F.cross_entropy(model.last_all.float()[mask], rid[mask])
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            tl += float(loss); nb += 1; step += 1
        Sv = splits[a.valid]; sv, _ = predict(Sv)
        th, f1 = best_f1_threshold(Sv.y, sv); cp = curve_points(Sv.y, sv)
        log.append({"epoch": ep + 1, "loss": round(tl / nb, 4), "valid_f1": round(f1, 4), "valid_auc": cp["auc"], "valid_ap": cp["ap"], "th": round(th, 4)})
        print(f"  ep {ep+1}/{a.epochs} loss {tl/nb:.4f} · {a.valid} F1 {f1:.4f} AUC {cp['auc']} AP {cp['ap']} th {th:.3f} ({time.time()-t0:.0f}s)", flush=True)
        if f1 > best[0]:
            best = (f1, th); torch.save({"model": model.state_dict(), "args": vars(a), "stats": {k: v.tolist() for k, v in stats.items()},
                                         "l1_vocab": l1_vocab, "threshold": th, "symbols": tr.info["symbols"]}, os.path.join(a.out, "detector.pt"))

    ck = torch.load(os.path.join(a.out, "detector.pt"), map_location=dev); model.load_state_dict(ck["model"]); th = ck["threshold"]
    rep = {"args": vars(a), "params_M": round(n_par / 1e6, 3), "train_utts": int(len(ks_all)), "threshold": th, "log": log,
           "feats_ckpt": tr.info["ckpt"], "results": {}}
    for name, S in splits.items():
        s, rp = predict(S)
        r = {"utts": S.n, "sites": int(len(S.y)), "err_rate": round(float(S.y.mean()), 4), "align_fail_utts": int((~S.ok).sum()),
             "detector": mdd_at(S.y, s >= th), "curve": curve_points(S.y, s)}
        gop = -S.scal[:, 1]; gop[~np.repeat(S.ok, S.len)] = -1e9
        if name == a.valid:
            rep["gop_threshold"] = best_f1_threshold(S.y, gop)[0]
        r["gop_only"] = {**mdd_at(S.y, gop >= rep.get("gop_threshold", 0.0)), **{"auc": curve_points(S.y, gop)["auc"]}}
        err = (S.y == 1) & np.repeat(S.ok, S.len)
        gold = np.where(S.rid[err] < 0, S.lps.shape[1], S.rid[err])
        r["realized_acc_on_err_sites"] = round(float((rp[err][:, 0] == gold).mean()), 4)
        r["realized_top3_on_err_sites"] = round(float((rp[err] == gold[:, None]).any(1).mean()), 4)
        hit = err & (s >= th)                                   # 실제로 지적한 오발음 자리에서의 정확도(사용자가 보는 값)
        gh = np.where(S.rid[hit] < 0, S.lps.shape[1], S.rid[hit])
        r["realized_acc_on_flagged_err_sites"] = round(float((rp[hit][:, 0] == gh).mean()), 4) if hit.any() else None
        by = collections.defaultdict(list)
        for k, u in enumerate(S.utt):
            by["task:" + str(u.get("task"))].append(k); by["L1:" + str(u.get("language"))].append(k)
        r["breakdown"] = {}
        for g, ks in sorted(by.items(), key=lambda kv: -len(kv[1]))[:14]:
            m = np.zeros(len(S.y), bool)
            for k in ks:
                m[S.off[k]:S.off[k + 1]] = True
            if S.y[m].sum() >= 30:
                r["breakdown"][g] = {"utts": len(ks), **{x: mdd_at(S.y[m], s[m] >= th)[x] for x in ("precision", "recall", "f1", "frr")}}
        rep["results"][name] = r
        d = r["detector"]
        print(f"[{name}] P {d['precision']} R {d['recall']} F1 {d['f1']} FRR {d['frr']} · AUC {r['curve']['auc']} AP {r['curve']['ap']} · "
              f"GOP단독 F1 {r['gop_only']['f1']} · 실제발음 top1 {r['realized_acc_on_err_sites']} top3 {r['realized_top3_on_err_sites']} "
              f"(지적한 자리 {r['realized_acc_on_flagged_err_sites']})", flush=True)
    json.dump(rep, open(os.path.join(a.out, "report.json"), "w"), ensure_ascii=False, indent=1)
    print(f"→ {a.out}/report.json ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
