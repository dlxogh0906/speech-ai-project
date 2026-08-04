# -*- coding: utf-8 -*-
"""지식증류: 2B(layer31, 0702_country41_sent_best) → 300m(layer17) 선형 헤드.

두 인코더 모두 **동결**이고 특징 캐시가 발화 단위로 정렬돼 있어 GPU 없이 CPU 로 돈다.
데이터 조립·분할·학습 루프는 `scripts/train/sweep300_layers.py` 를 **그대로 재현**한다
— CE baseline 이 출하본(0705, test micro5 0.9331)을 재현하는지가 파이프라인 검증이다.

교사 로짓: X31 → clip/nan → (x−mu_t)/sd_t → W_t·x+b_t  (교사 head.pt 의 mu/sd 사용)
학생 학습: 같은 행의 X17 로 CE / CE+KD 를 같은 스케줄로 학습, valid macro5 로 선택.
KD 손실:  (1−α)·CE(hard) + α·T²·KL( logσ(z_s/T) ‖ σ(z_t/T) )

라벨 주의: 교사 라벨은 한글 가나다순, 학생은 영문 알파벳순 — labels_en 으로 재매핑한다.
유럽 041 주의: 교사 캐시(전체+sent_mask)와 학생 캐시(문장전용)의 행 순서가 달라
6조각 순열(euro_sent_row_perm.npy)로 대응시킨다. 근거: 화자열 전단사 + 동일 인코더
지문 대조(cos 0.9993 vs 동일화자 오짝 0.9565). 생성·검증은 scripts/data/recover_euro_sent_perm.py.
교사 미학습 도메인(043)은 --kd_domains 로 KD 를 끌 수 있다(행별 배합 — CE 는 전액 유지).

공정성 주의(적대적 검증에서 확인된 것):
- '같은 assign_speakers seed42' 라도 인덱스 구성이 다르면 배정이 달라진다(해시 기반이 아님).
  교사(0702, 043 없음)의 train 화자 일부가 학생 test 에 들어간다 → teacher_on_test 는
  오염 수치이며, 본 스크립트는 교사-train 화자 제외 clean 부분집합을 병기한다.
- KD Δ 에는 2차 경로(교사 헤드가 학생 test 화자 발화로 적합됨)가 있어 clean/contam 분해를
  함께 기록한다. (α,T) 선택은 valid 15조합 vs baseline 1회의 비대칭임을 meta 에 남긴다.
- CE baseline 은 sweep300 의 '통계적' 재현이다(device RNG 차이로 셔플 순서는 다름).

    $PY tasks/nationality-classification/experiments/kd_2b_to_300m.py --quick   # 스모크
    $PY tasks/nationality-classification/experiments/kd_2b_to_300m.py           # 전체 (~30분 CPU)
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import DATA_ROOT, NAT_DATA, NAT_OUTPUTS, NAT_TASK

import argparse, csv, json, os, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

from country_classifier.make_country40_probe import SPLIT, LANG2C
from country_classifier.make_speaker_split import assign_speakers

# ── 캐시 위치 (sweep300_layers.py / merge_country41_euro.py 와 동일) ──
S_FULL = Path(f"{DATA_ROOT}/taeho300_layers_full")            # 학생 505
S_KOR = Path(f"{DATA_ROOT}/taeho300_layers_korean/train.npz")
S_EURO = Path(f"{DATA_ROOT}/taeho300_layers_euro_sent/train.npz")
S_ASIA = Path(f"{DATA_ROOT}/taeho300_layers_asia/train.npz")
T_FULL = Path(f"{DATA_ROOT}/taeho_layers_full")               # 교사 505
T_KOR = Path(f"{DATA_ROOT}/taeho_layers_korean/korean.npz")
T_EURO = Path(f"{DATA_ROOT}/taeho_layers_euro/train.npz")
T_ASIA = Path(f"{DATA_ROOT}/taeho_layers_asia/train.npz")
EURO_MASK = Path(f"{DATA_ROOT}/taeho_layers_euro/sent_mask_ge4.npy")
EURO_PERM = Path(f"{DATA_ROOT}/taeho_layers_euro/euro_sent_row_perm.npy")
EURO_MAN = f"{NAT_DATA}/euro_country40_manifest.csv"
ASIA_MAN = f"{NAT_DATA}/asian_country40_manifest.csv"
TEACHER_HEAD = f"{NAT_OUTPUTS}/0702_country41_sent_best/head.pt"

S_LAYER, T_LAYER = 17, 31


def _layer_index(npz, want):
    li = npz["layers"].tolist().index(want)
    return li


def _prep(x):
    """sweep300/merge 와 동일한 전처리: fp32 + nan→0 + clip."""
    return np.clip(np.nan_to_num(x.astype(np.float32)), -50, 50)


def build_index():
    """sweep300_layers.py 의 인덱스 구성을 그대로 재현 + 교사쪽 (파일,행) 대응을 추가.

    반환 행 순서·필터가 sweep300 과 완전히 같아야 split(seed42)이 동일해진다.
    """
    rows = list(csv.DictReader(open(f"{NAT_DATA}/fine_labels.csv", encoding="utf-8")))
    spk_mt = {}; spk_cc = defaultdict(Counter)
    for r in rows:
        spk_mt[r["speaker_id"]] = r["mother_tongue"]
        spk_cc[r["speaker_id"]][(r["country"] or "").strip()] += 1
    spk_c = {s: c.most_common(1)[0][0] for s, c in spk_cc.items()}

    def c40(s):
        mt = spk_mt.get(s)
        return SPLIT[mt].get(spk_c.get(s)) if mt in SPLIT else LANG2C.get(mt)

    e2l = {r["speaker_id"]: r["label_name"] for r in csv.DictReader(open(EURO_MAN, encoding="utf-8"))}
    a2l = {}
    for r in csv.DictReader(open(ASIA_MAN, encoding="utf-8")):
        a2l.setdefault(r["speaker_id"], r["label_name"])

    # 유럽: 학생행 → 교사(전체 캐시) 행 매핑 (+ 런타임 무결성 검증 — 드리프트 시 즉사)
    mask = np.load(EURO_MASK)
    perm = np.load(EURO_PERM)                       # 학생 euro i행 = 교사(마스크후) perm[i]행
    assert np.array_equal(np.sort(perm), np.arange(len(perm))), "euro perm 이 전단사가 아님"
    _t_spk = np.asarray(np.load(str(T_EURO), allow_pickle=True)["speaker"], dtype=object)[mask]
    _s_spk = np.asarray(np.load(str(S_EURO), allow_pickle=True)["speaker"], dtype=object)
    assert len(perm) == len(_s_spk) and (_t_spk[perm] == _s_spk).all(), \
        "euro perm 화자열 불일치 — perm/mask/캐시 버전 드리프트 (scripts/data/recover_euro_sent_perm.py 로 재생성)"
    del _t_spk, _s_spk
    maskidx = np.where(mask)[0]                     # 마스크후 j행 = 교사(전체) maskidx[j]행
    euro_t_row = maskidx[perm]                      # 학생 euro i행 = 교사(전체) euro_t_row[i]행

    s_path, s_row, t_path, t_row = [], [], [], []
    sps, lab, dom = [], [], []

    def add(spath, tpath, speaker_iter, labeler, domain, trow_of=lambda i: i):
        for i, s in enumerate(speaker_iter):
            l = labeler(str(s))
            if l is None: continue
            s_path.append(spath); s_row.append(i)
            t_path.append(tpath); t_row.append(int(trow_of(i)))
            sps.append(str(s)); lab.append(l); dom.append(domain)

    for sp in ["train", "valid", "test"]:                       # 505
        d = np.load(str(S_FULL / f"{sp}.npz"), allow_pickle=True)
        add(str(S_FULL / f"{sp}.npz"), str(T_FULL / f"{sp}.npz"), d["speaker"], c40, "asia")
    kd = np.load(str(S_KOR), allow_pickle=True)                 # korean (필터 없음)
    for i, s in enumerate(kd["speaker"]):
        s_path.append(str(S_KOR)); s_row.append(i)
        t_path.append(str(T_KOR)); t_row.append(i)
        sps.append("KOR:" + str(s)); lab.append("한국어"); dom.append("korean")
    de = np.load(str(S_EURO), allow_pickle=True)                # euro (문장전용)
    add(str(S_EURO), str(T_EURO), de["speaker"], lambda s: e2l.get(s), "euro",
        trow_of=lambda i: euro_t_row[i])
    da = np.load(str(S_ASIA), allow_pickle=True)                # asia043
    add(str(S_ASIA), str(T_ASIA), da["speaker"], lambda s: a2l.get(s), "asia043")

    return (np.array(s_path), np.array(s_row), np.array(t_path), np.array(t_row),
            np.array(sps), np.array(lab, dtype=object), np.array(dom))


def gather(paths, rows, layer_want, dim, tag):
    """(파일,행,층) 인덱스로 (N,dim) fp32 특징 조립. 파일당 1회 로드."""
    N = len(paths)
    out = np.empty((N, dim), np.float32)
    for p in dict.fromkeys(paths.tolist()):
        z = np.load(p, allow_pickle=True)
        li = _layer_index(z, layer_want)
        X = _prep(z["X"][:, li, :])
        m = paths == p
        out[m] = X[rows[m]]
        del X, z
        print(f"    [{tag}] {os.path.basename(os.path.dirname(p))}/{os.path.basename(p)} 완료", flush=True)
    return out


def teacher_logits(t_paths, t_rows, sps):
    """교사 로짓 (N,41) — 교사 head.pt 의 mu/sd/가중치로, 교사 라벨 순서 그대로."""
    import torch
    th = torch.load(TEACHER_HEAD, map_location="cpu")
    W = th["head"]["weight"].numpy().astype(np.float32)         # (41,1920)
    b = th["head"]["bias"].numpy().astype(np.float32)
    mu = np.asarray(th["mu"], np.float32); sd = np.asarray(th["sd"], np.float32)
    X = gather(t_paths, t_rows, T_LAYER, W.shape[1], "교사")
    Z = (X - mu) / sd
    del X
    return Z @ W.T + b, list(th["labels"]), list(th["labels_en"])


def macro5(L, yy):
    i5 = np.argpartition(-L, 4, axis=1)[:, :5]
    h = np.array([a in r for a, r in zip(yy, i5)])
    return np.mean([h[yy == c].mean() for c in sorted(set(yy.tolist()))])


def evaluate(L, yy, dte=None, contam=None):
    """contam: 교사-train 화자 행 마스크 — clean(제외) 수치를 병기해 2차 누수 영향을 분해."""
    pred = L.argmax(1); t5 = np.argpartition(-L, 4, 1)[:, :5]
    h1 = pred == yy; h5 = np.array([a in r for a, r in zip(yy, t5)])
    pres = sorted(set(yy.tolist()))
    out = {"micro1": round(float(h1.mean()), 4), "micro5": round(float(h5.mean()), 4),
           "macro5": round(float(np.mean([h5[yy == c].mean() for c in pres])), 4),
           "macro1": round(float(np.mean([h1[yy == c].mean() for c in pres])), 4)}
    if dte is not None:
        out["by_domain"] = {}
        for dd, nm in [("asia", "505"), ("euro", "041"), ("asia043", "043"), ("korean", "korean")]:
            mk = dte == dd
            if mk.sum():
                out["by_domain"][nm] = {"n": int(mk.sum()),
                                        "top1": round(float(h1[mk].mean()), 4),
                                        "top5": round(float(h5[mk].mean()), 4)}
                if contam is not None and (mk & ~contam).sum():
                    ck = mk & ~contam
                    out["by_domain"][nm]["clean_n"] = int(ck.sum())
                    out["by_domain"][nm]["clean_top1"] = round(float(h1[ck].mean()), 4)
                    out["by_domain"][nm]["clean_top5"] = round(float(h5[ck].mean()), 4)
    if contam is not None:
        cl = ~contam
        pres_c = sorted(set(yy[cl].tolist()))
        out["clean"] = {"n": int(cl.sum()),
                        "micro1": round(float(h1[cl].mean()), 4),
                        "micro5": round(float(h5[cl].mean()), 4),
                        "macro5": round(float(np.mean([h5[cl & (yy == c)].mean() for c in pres_c])), 4)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", default="0.3,0.5,0.7,0.9,1.0")
    ap.add_argument("--temps", default="1,2,4")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--kd_domains", default="all",
                    help="'all' 또는 콤마목록(asia,korean,euro) — 목록 밖 도메인은 KD α=0 (교사 미학습 043 제외용)")
    ap.add_argument("--quick", action="store_true", help="행 5%% 표본·짧은 학습으로 파이프라인만 검증")
    ap.add_argument("--device", default="cpu", help="GPU 는 omniASR 학습 중 — 기본 cpu")
    ap.add_argument("--out", default=str(NAT_TASK / "outputs" / "kd_2b_to_300m.json"))
    A = ap.parse_args()
    import torch, torch.nn as nn, torch.nn.functional as F
    dev = A.device
    t0 = time.time()

    print("[1/5] 인덱스 구성 (sweep300 재현)", flush=True)
    s_path, s_row, t_path, t_row, sps, lab, dom = build_index()
    ko2en = json.load(open(NAT_TASK / "configs" / "country41_ko2en.json", encoding="utf-8"))
    assert len(set(ko2en.values())) == len(ko2en), "ko2en 이 단사가 아님 — 클래스 병합 발생"
    lab_en = np.array([ko2en[l] for l in lab], dtype=object)
    classes = sorted(set(lab_en.tolist())); c2i = {c: i for i, c in enumerate(classes)}
    y = np.array([c2i[c] for c in lab_en], np.int64)
    items = [{"speaker_id": s, "label_name": l, "num_frames": 0} for s, l in zip(sps, lab_en)]
    assign = assign_speakers(items, [8, 1, 1], seed=42)
    split = np.array([assign[s] for s in sps])
    N = len(sps)
    print(f"    {N:,}행 · {len(classes)}클래스 · split {Counter(split.tolist())}", flush=True)

    # ── 교사(0702) 자신의 split 재구성 → 오염(교사-train 화자) 마스크 ──
    # 교사 인덱스 = 본 인덱스에서 043 블록 제외, 라벨은 교사 학습과 같은 한글.
    # '같은 seed42'여도 인덱스 구성이 다르면 배정이 달라진다(assign_speakers 는 해시가 아님).
    t_items = [{"speaker_id": s, "label_name": l, "num_frames": 0}
               for s, l, d in zip(sps, lab, dom) if d != "asia043"]
    t_assign = assign_speakers(t_items, [8, 1, 1], seed=42)
    t_train_spk = {s for s, a in t_assign.items() if a == "train"}
    t_valid_spk = {s for s, a in t_assign.items() if a == "valid"}

    if A.quick:
        rng = np.random.RandomState(0)
        keep = np.zeros(N, bool); keep[rng.choice(N, N // 20, replace=False)] = True
        s_path, s_row, t_path, t_row = s_path[keep], s_row[keep], t_path[keep], t_row[keep]
        sps, y, dom, split = sps[keep], y[keep], dom[keep], split[keep]
        N = keep.sum(); A.epochs = 3
        print(f"    [quick] {N:,}행 표본 · {A.epochs}ep", flush=True)

    tr, va, te = (np.where(split == s)[0] for s in ["train", "valid", "test"])
    ytr, yva, yte = y[tr], y[va], y[te]; dte = dom[te]

    # 오염 마스크: 학생 test/valid 행 중 화자가 교사-train 에 있던 것
    contam_te = np.array([s in t_train_spk for s in sps[te]])
    contam_va = np.array([s in t_train_spk for s in sps[va]])
    contam_meta = {
        "test_rows_contam": int(contam_te.sum()), "test_rows": int(len(te)),
        "test_contam_rate": round(float(contam_te.mean()), 4),
        "valid_rows_contam": int(contam_va.sum()),
        "test_speakers_in_teacher_train": len(set(sps[te]) & t_train_spk),
        "test_speakers_in_teacher_valid": len(set(sps[te]) & t_valid_spk),
        "by_domain_contam_rate": {nm: round(float(contam_te[dte == dd].mean()), 4)
                                  for dd, nm in [("asia", "505"), ("euro", "041"),
                                                 ("asia043", "043"), ("korean", "korean")]
                                  if (dte == dd).sum()},
    }
    print(f"    [오염] 학생 test {contam_meta['test_rows_contam']:,}/{contam_meta['test_rows']:,}행"
          f"({contam_meta['test_contam_rate']:.1%})이 교사-train 화자 · 도메인별 {contam_meta['by_domain_contam_rate']}", flush=True)

    print("[2/5] 교사 로짓 계산 (2B layer31 + 0702 head)", flush=True)
    TL, t_labels_ko, t_labels_en = teacher_logits(t_path, t_row, sps)
    # 교사 라벨(한글순) → 학생 클래스(영문순) 재배열
    t_en = [ko2en[l] for l in t_labels_ko]
    assert len(t_en) == len(set(t_en)) == len(classes), "교사 라벨 영문 변환에 중복/누락"
    assert set(t_en) == set(classes), f"라벨 집합 불일치: {set(t_en) ^ set(classes)}"
    reorder = [t_en.index(c) for c in classes]
    TL = TL[:, reorder]                                        # (N,41) 학생 클래스 순서
    t_test = evaluate(TL[te], yte, dte, contam=contam_te)
    print(f"    교사@test  micro1={t_test['micro1']} micro5={t_test['micro5']} macro5={t_test['macro5']}"
          f"  |  clean(교사-train 화자 제외) micro1={t_test['clean']['micro1']} micro5={t_test['clean']['micro5']}", flush=True)
    for nm, v in t_test.get("by_domain", {}).items():
        cl = f"  clean_top1={v['clean_top1']}" if "clean_top1" in v else ""
        print(f"      [{nm:6}] n={v['n']:>7,} top1={v['top1']} top5={v['top5']}{cl}", flush=True)

    print("[3/5] 학생 특징 로드 (300m layer17)", flush=True)
    X = gather(s_path, s_row, S_LAYER, 1024, "학생")
    mu = X[tr].mean(0, keepdims=True); sd = X[tr].std(0, keepdims=True) + 1e-6
    Xtr, Xva, Xte = (X[tr] - mu) / sd, (X[va] - mu) / sd, (X[te] - mu) / sd
    del X

    # KD 도메인 마스크 (train 행 기준)
    if A.kd_domains == "all":
        kd_w_tr = np.ones(len(tr), np.float32)
    else:
        allow = set(A.kd_domains.split(","))
        kd_w_tr = np.array([1.0 if d in allow else 0.0 for d in dom[tr]], np.float32)
        print(f"    KD 허용 도메인 {sorted(allow)} → train {int(kd_w_tr.sum()):,}/{len(tr):,}행", flush=True)

    At = torch.from_numpy(Xtr).to(dev); Yt = torch.from_numpy(ytr).to(dev)
    Tt = torch.from_numpy(TL[tr]).to(dev); Kw = torch.from_numpy(kd_w_tr).to(dev)
    nn_ = len(Yt)

    def train_probe(alpha, T):
        """sweep300 의 train_probe 와 동일 스케줄. alpha=0 이면 순수 CE(=baseline)."""
        torch.manual_seed(42)
        head = nn.Linear(Xtr.shape[1], len(classes)).to(dev)
        opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
        def logits(Z):
            head.eval()
            with torch.no_grad():
                return head(torch.from_numpy(Z).to(dev)).cpu().numpy()
        best = -1; bw = None
        for ep in range(A.epochs):
            head.train(); pm = torch.randperm(nn_, device=dev)
            for i in range(0, nn_, 4096):
                j = pm[i:i + 4096]
                z = head(At[j])
                if alpha > 0:
                    # 행별 배합: loss_i = (1−α·K_i)·CE_i + α·K_i·T²·KL_i
                    # (마스크 행도 CE 는 전액 유지 — 전 도메인 마스크 시 CE까지 감쇠되던 버그 수정)
                    ce = F.cross_entropy(z, Yt[j], reduction="none")            # (B,)
                    kl = F.kl_div(F.log_softmax(z / T, 1), F.softmax(Tt[j] / T, 1),
                                  reduction="none").sum(1)                      # (B,)
                    w = alpha * Kw[j]
                    loss = ((1 - w) * ce + w * (T * T) * kl).mean()
                else:
                    loss = F.cross_entropy(z, Yt[j])
                loss.backward(); opt.step(); opt.zero_grad()
            m = macro5(logits(Xva), yva)
            if m > best:
                best = m; bw = {k: v.detach().clone() for k, v in head.state_dict().items()}
        head.load_state_dict(bw)
        return head, best, logits

    print("[4/5] CE baseline (sweep300 재현 — 출하본과 대조용)", flush=True)
    head0, v0, lg0 = train_probe(0.0, 1.0)
    base = evaluate(lg0(Xte), yte, dte, contam=contam_te); base["valid_macro5"] = round(float(v0), 4)
    print(f"    baseline  valid_macro5={v0:.4f} | test micro5={base['micro5']} macro5={base['macro5']} micro1={base['micro1']}", flush=True)
    print(f"    (출하본 0705: micro5 0.9331 macro5 0.6170 micro1 0.7510 — 근접해야 정상)", flush=True)

    print("[5/5] KD 스윕", flush=True)
    runs = {"baseline_ce": base, "teacher_on_test": t_test,
            "shipped_0705": {"micro5": 0.9331, "macro5": 0.6170, "micro1": 0.7510},
            "contamination": contam_meta, "kd": []}
    best_kd = None
    for alpha in [float(x) for x in A.alphas.split(",")]:
        for T in [float(x) for x in A.temps.split(",")]:
            hd, vm, lg = train_probe(alpha, T)
            r = evaluate(lg(Xte), yte, dte, contam=contam_te)
            r.update({"alpha": alpha, "T": T, "valid_macro5": round(float(vm), 4)})
            runs["kd"].append(r)
            tag = ""
            if best_kd is None or vm > best_kd[0]:
                best_kd = (vm, r, {k: v.detach().cpu().clone() for k, v in hd.state_dict().items()}); tag = "  ← valid 최고"
            print(f"    α={alpha} T={T}: valid_macro5={vm:.4f} | test micro5={r['micro5']} "
                  f"macro5={r['macro5']} micro1={r['micro1']}{tag}", flush=True)

    sel = best_kd[1]
    runs["selected_by_valid"] = sel
    runs["delta_vs_baseline"] = {k: round(sel[k] - base[k], 4) for k in ("micro1", "micro5", "macro5", "macro1")}
    if "clean" in sel and "clean" in base:
        runs["delta_vs_baseline_clean"] = {k: round(sel["clean"][k] - base["clean"][k], 4)
                                           for k in ("micro1", "micro5", "macro5")}
    runs["meta"] = {"n": int(N), "classes": len(classes), "epochs": A.epochs,
                    "kd_domains": A.kd_domains, "quick": A.quick,
                    "student_layer": S_LAYER, "teacher_layer": T_LAYER,
                    "selection_note": "KD 는 (α,T) 15조합 중 valid macro5 최고 vs baseline 1회 — 비대칭 선택",
                    "elapsed_sec": round(time.time() - t0, 1)}

    # 선택된 헤드 저장 (배포·재사용 가능하게 — 재학습 불필요)
    if not A.quick:
        hd_dir = os.path.join(os.path.dirname(A.out), "kd_heads")
        os.makedirs(hd_dir, exist_ok=True)
        torch.save({"head": best_kd[2], "mu": mu, "sd": sd, "labels": classes, "c2i": c2i,
                    "layer": S_LAYER, "feat_dim": 1024, "max_sec": 12, "backbone": "xlsr_300m",
                    "kind": "country41_kd_from_2B", "alpha": sel["alpha"], "T": sel["T"],
                    "teacher": TEACHER_HEAD},
                   os.path.join(hd_dir, "kd_best_head.pt"))
        torch.save({"head": {k: v.detach().cpu().clone() for k, v in head0.state_dict().items()},
                    "mu": mu, "sd": sd, "labels": classes, "c2i": c2i,
                    "layer": S_LAYER, "feat_dim": 1024, "max_sec": 12, "backbone": "xlsr_300m",
                    "kind": "country41_ce_rebaseline"},
                   os.path.join(hd_dir, "ce_baseline_head.pt"))
        runs["meta"]["saved_heads"] = hd_dir
    os.makedirs(os.path.dirname(A.out), exist_ok=True)
    json.dump(runs, open(A.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n[선택: valid macro5] α={sel['alpha']} T={sel['T']} → test micro5={sel['micro5']} "
          f"macro5={sel['macro5']} micro1={sel['micro1']}")
    print(f"[Δ vs baseline] {runs['delta_vs_baseline']}")
    print(f"[saved] {A.out} · {runs['meta']['elapsed_sec']}s")


if __name__ == "__main__":
    main()
