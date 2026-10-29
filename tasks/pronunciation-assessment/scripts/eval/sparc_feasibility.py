# -*- coding: utf-8 -*-
"""조음 복원(SPARC) 피드백의 가능성 검증 — "복원된 조음·음원 궤적이 정상 발음과 오발음을 가르는가".

입력
  SPARC 인코딩   `sparc_encode_split.py` 의 <split>.shard*.npz (EMA 12 + loudness · pitch · periodicity, 50 Hz)
  사람 경계      매니페스트 meta 의 `lab_phones` — 표준발음 자리와 1:1 (start, end, 자모)
  라벨 · GOP     `extract_phone_feats.py` 출력(y · real_id · scal) — 같은 발화 순서 · 같은 자리 순서

방법: 자리마다 구간 통계(평균·최소·최대·처음·끝·변화량)와 앞뒤 3프레임 문맥을 뽑고, 음소 계열별로
      **화자 단위 5겹 교차검증** 그래디언트 부스팅의 out-of-fold AUC 를 잰다. 특징 묶음을 바꿔 가며 비교한다.
        gop        강제정렬 GOP 스칼라(기준선)          ema     혀·턱·입술 좌표
        source     loudness · pitch · periodicity      all     gop + ema + source
      AUC(all) − AUC(gop) 가 그 계열에서 조음 정보가 **추가로** 주는 몫이다.
그림: 계열별 평균 궤적(정상 vs 오발음) — 발표·보고서에 넣을 "AI 가 복원한 조음 차이".

    $PY tasks/pronunciation-assessment/scripts/eval/sparc_feasibility.py --enc <enc dir> --feats <phone feats dir> \
        --manifest $MDD_OUTPUTS/manifest_foreign_clean_v2 --splits valid_unseen,valid --out <dir>
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import argparse, collections, glob, json, os

import numpy as np

VOW = set("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
STOPS = set("ㄱㄲㅋㄷㄸㅌㅂㅃㅍㅈㅉㅊㅅㅆ")
CH = ["TDx", "TDy", "TBx", "TBy", "TTx", "TTy", "LIx", "LIy", "ULx", "ULy", "LLx", "LLy", "loud", "pitch", "period"]
FPS = 50.0


def position(canon, j):
    x = canon[j]
    if x in VOW:
        return "V"
    return "coda" if (x == "ㅇ" or j + 1 == len(canon) or canon[j + 1] not in VOW) else "onset"


def family(x, pos):
    if pos == "V":
        return "vowel"
    if pos == "onset" and x in STOPS:
        return "onset_obstruent"            # 평음·경음·격음 혼동이 몰린 곳 — 음원(source) 단서
    if pos == "coda":
        return "coda"                       # ㄹ·ㄴ·ㅇ·ㅁ 받침 — 위치(EMA) 단서
    return "onset_sonorant"


def tok_feats(F, f0, f1):
    seg = F[f0:max(f1, f0 + 1)]
    st = np.concatenate([seg.mean(0), seg.min(0), seg.max(0), seg[0], seg[-1], seg[-1] - seg[0]])
    nxt = F[f1:f1 + 3]; prv = F[max(0, f0 - 3):f0]
    ctx = np.concatenate([nxt.mean(0) if len(nxt) else seg[-1], prv.mean(0) if len(prv) else seg[0]])
    return np.concatenate([st, ctx, [np.log(max(1, f1 - f0))]])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--enc", required=True); ap.add_argument("--feats", required=True)
    ap.add_argument("--manifest", required=True); ap.add_argument("--splits", default="valid_unseen,valid")
    ap.add_argument("--out", required=True); ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import GroupKFold
    os.makedirs(a.out, exist_ok=True)

    X, G, Y, SPK, FAM, PH, REAL, TRAJ = [], [], [], [], [], [], [], []
    n_utt = n_skip = 0
    for sp in a.splits.split(","):
        canon = [l.split() for l in open(os.path.join(a.manifest, f"{sp}.canon"), encoding="utf-8")]
        meta = [json.loads(l) for l in open(os.path.join(a.manifest, f"{sp}.meta.jsonl"), encoding="utf-8")]
        fd = os.path.join(a.feats, sp)
        y = np.load(os.path.join(fd, "y.npy")); rid = np.load(os.path.join(fd, "real_id.npy")); sc = np.load(os.path.join(fd, "scal.npy"))
        off = np.load(os.path.join(fd, "offsets.npy")); ok = np.load(os.path.join(fd, "ok.npy"))
        sym = json.load(open(os.path.join(fd, "meta.json"), encoding="utf-8"))["symbols"]
        enc = {}
        for f in sorted(glob.glob(os.path.join(a.enc, f"{sp}.shard*.npz"))):
            z = np.load(f)
            for k, i in enumerate(z["idx"]):
                enc[int(i)] = z["frames"][z["offsets"][k]:z["offsets"][k + 1]].astype(np.float32)
        for i, (c, m) in enumerate(zip(canon, meta)):
            lab = m.get("lab_phones") or []
            F = enc.get(i)
            if F is None or len(F) < 3 or not ok[i] or [p[2] for p in lab] != c:
                n_skip += 1; continue
            n_utt += 1
            F = F.copy()
            v = F[:, 13] > 1                                        # pitch → 발화 중앙값 대비 반음(무성 구간은 0)
            med = np.median(F[v, 13]) if v.any() else 1.0
            F[:, 13] = np.where(v, 12 * np.log2(np.maximum(F[:, 13], 1e-3) / med), 0.0)
            for j, (s, e, _) in enumerate(lab):
                f0 = int(np.clip(np.floor(s * FPS), 0, len(F) - 1)); f1 = int(np.clip(np.ceil(e * FPS), f0 + 1, len(F)))
                pos = position(c, j); t = int(off[i]) + j
                X.append(tok_feats(F, f0, f1)); G.append(sc[t]); Y.append(int(y[t])); SPK.append(m["spk"])
                FAM.append(family(c[j], pos)); PH.append(c[j] + ":" + pos); REAL.append(sym[rid[t]] if rid[t] >= 0 else "∅")
                g = np.linspace(f0 - 0.5 * (f1 - f0), f1 + 0.5 * (f1 - f0), 12).round().astype(int).clip(0, len(F) - 1)
                TRAJ.append(F[g])                                   # 구간 ±50% 를 12점으로 시간 정규화
    X, G, Y = np.array(X, np.float32), np.array(G, np.float32), np.array(Y)
    SPK, FAM, PH, REAL, TRAJ = np.array(SPK), np.array(FAM), np.array(PH), np.array(REAL), np.array(TRAJ, np.float32)
    nC = len(CH)
    cols = {"ema": [k for k in range(X.shape[1] - 1) if k % nC < 12], "source": [k for k in range(X.shape[1] - 1) if k % nC >= 12]}
    dur = X[:, -1:]
    ph_ids = {p: k for k, p in enumerate(sorted(set(PH)))}
    phc = np.array([ph_ids[p] for p in PH], np.float32)[:, None]
    sets = {"gop": np.hstack([G, phc]), "ema": np.hstack([X[:, cols["ema"]], dur, phc]),
            "source": np.hstack([X[:, cols["source"]], dur, phc]),
            "ema+source": np.hstack([X[:, :-1], dur, phc]), "all": np.hstack([G, X, phc])}
    print(f"[data] 발화 {n_utt:,}(제외 {n_skip:,}) · 자리 {len(Y):,} · 오발음 {Y.mean():.2%} · 화자 {len(set(SPK)):,}", flush=True)

    rep = {"n_utts": n_utt, "n_sites": int(len(Y)), "families": {}}
    for fam in ["all"] + sorted(set(FAM)):
        m = np.ones(len(Y), bool) if fam == "all" else FAM == fam
        if Y[m].sum() < 100:
            continue
        r = {"sites": int(m.sum()), "errors": int(Y[m].sum()),
             "top_errors": collections.Counter(f"{p}→{q}" for p, q, yy in zip(PH[m], REAL[m], Y[m]) if yy).most_common(8)}
        for name, Z in sets.items():
            oof = np.zeros(m.sum())
            for trn, tst in GroupKFold(a.folds).split(Z[m], Y[m], SPK[m]):
                clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0,
                                                     class_weight="balanced", random_state=0)
                clf.fit(Z[m][trn], Y[m][trn]); oof[tst] = clf.predict_proba(Z[m][tst])[:, 1]
            r[name] = {"auc": round(float(roc_auc_score(Y[m], oof)), 4), "ap": round(float(average_precision_score(Y[m], oof)), 4)}
        rep["families"][fam] = r
        print(f"[{fam:16s}] 자리 {r['sites']:>7,} 오발음 {r['errors']:>5,} | " +
              " · ".join(f"{k} {r[k]['auc']:.3f}" for k in sets) + f" | 조음 추가 몫 {r['all']['auc'] - r['gop']['auc']:+.3f}", flush=True)
    json.dump(rep, open(os.path.join(a.out, "sparc_feasibility.json"), "w"), ensure_ascii=False, indent=1)

    try:                                                            # 그림 — 한글 폰트가 없을 수 있어 라벨은 로마자
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cases = [("coda L (ㄹ받침): tongue-tip height TTy", PH == "ㄹ:coda", 5, {"correct": REAL == "ㄹ", "→ n (ㄴ)": REAL == "ㄴ", "deleted": REAL == "∅"}),
                 ("tense onset (ㄲㄸㅃㅉ): periodicity", np.isin(PH, ["ㄲ:onset", "ㄸ:onset", "ㅃ:onset", "ㅉ:onset"]), 14,
                  {"correct": Y == 0, "→ lenis (ㄱㄷㅂㅈ)": np.isin(REAL, list("ㄱㄷㅂㅈ")), "→ aspirated (ㅋㅌㅍㅊ)": np.isin(REAL, list("ㅋㅌㅍㅊ"))}),
                 ("tense onset (ㄲㄸㅃㅉ): pitch (semitones)", np.isin(PH, ["ㄲ:onset", "ㄸ:onset", "ㅃ:onset", "ㅉ:onset"]), 13,
                  {"correct": Y == 0, "→ lenis (ㄱㄷㅂㅈ)": np.isin(REAL, list("ㄱㄷㅂㅈ")), "→ aspirated (ㅋㅌㅍㅊ)": np.isin(REAL, list("ㅋㅌㅍㅊ"))}),
                 ("vowel o (ㅗ) vs eo (ㅓ), correct tokens: lower-lip x (protrusion)", (Y == 0) & np.isin(PH, ["ㅗ:V", "ㅓ:V"]), 10,
                  {"o (ㅗ, rounded)": PH == "ㅗ:V", "eo (ㅓ, unrounded)": PH == "ㅓ:V"})]
        fig, axs = plt.subplots(1, len(cases), figsize=(5.2 * len(cases), 3.8))
        t = np.linspace(-0.5, 1.5, 12)
        for ax, (title, base, ch, groups) in zip(axs, cases):
            for lab, g in groups.items():
                mm = base & g
                if mm.sum() < 15:
                    continue
                mu, se = TRAJ[mm][:, :, ch].mean(0), TRAJ[mm][:, :, ch].std(0) / np.sqrt(mm.sum())
                ax.plot(t, mu, label=f"{lab} (n={mm.sum():,})"); ax.fill_between(t, mu - 2 * se, mu + 2 * se, alpha=0.2)
            ax.axvspan(0, 1, color="0.92"); ax.set_title(title, fontsize=9); ax.set_xlabel("time (0–1 = phone span)"); ax.legend(fontsize=7)
        plt.tight_layout(); plt.savefig(os.path.join(a.out, "sparc_trajectories.png"), dpi=130)
        print(f"→ {a.out}/sparc_trajectories.png", flush=True)
    except Exception as e:
        print(f"[그림 생략] {type(e).__name__}: {e}", flush=True)
    print(f"→ {a.out}/sparc_feasibility.json", flush=True)


if __name__ == "__main__":
    main()
