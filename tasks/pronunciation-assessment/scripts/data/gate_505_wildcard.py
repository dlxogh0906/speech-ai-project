# -*- coding: utf-8 -*-
"""판별기 신뢰도로 505 라벨을 거른다 → 와일드카드 CTC 학습용 fairseq 데이터 폴더.

505 는 읽은 문장은 알지만 **어느 자리를 틀리게 읽었는지 모른다.** 표준발음을 전부 정답으로 주면 오발음을 표준발음이라고
가르치게 된다(되돌림의 원인). 그래서 자리마다 판별기의 오발음 확률 p 를 보고
    p <  tau   → 표준발음 자모를 정답으로 (판별기가 "정상"이라고 확신하는 자리)
    p >= tau   → `*`  (wildcard_ctc 가 "blank 가 아닌 아무 자모"로 다룬다. 연속한 `*` 는 하나로 합친다 — 탈락·삽입도 흡수)
로 라벨을 쓴다. `*` 비율이 --max_wild_frac 를 넘는 발화(다른 문장을 읽었거나 잡음)는 버린다.
원어민 음향 기준을 잃지 않도록 KsponSpeech 를 --native_ratio(시간 비율)만큼 섞는다. 검증은 Kspon dev(와일드카드 없음).

tau 선택 근거는 TAG(test_unseen)에서 잰다: 남기는 정상 자리 비율 vs 정답으로 새어 들어가는 오발음 비율 — `--report_tag <split 폴더>`.

    $PY .../gate_505_wildcard.py --pool $MDD_OUTPUTS/manifest_505_weak_v1 --feats <feats>/pool --tau 0.10 --out $MDD_OUTPUTS/manifest_b1_wildcard_505
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import MDD_OUTPUTS

import argparse, collections, json, os, random, shutil

import numpy as np


def tag_report(d, taus):
    """TAG split(정답 있음)에서 tau 별 품질: 정상 자리를 얼마나 남기고, 남긴 라벨에 오발음이 얼마나 섞이나."""
    p, y = np.load(os.path.join(d, "p_err.npy")), np.load(os.path.join(d, "y.npy")).astype(bool)
    m = p >= 0
    out = []
    for t in taus:
        keep = m & (p < t)
        out.append({"tau": t, "kept_frac": round(float(keep.sum() / m.sum()), 4),
                    "kept_normal_frac": round(float((keep & ~y).sum() / (m & ~y).sum()), 4),
                    "contamination_in_kept": round(float((keep & y).sum() / max(1, keep.sum())), 5),
                    "errors_masked_frac": round(float((~keep & y & m).sum() / (m & y).sum()), 4)})
    return {"naive_contamination": round(float(y[m].mean()), 5), "by_tau": out}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", default=str(MDD_OUTPUTS / "manifest_505_weak_v1"))
    ap.add_argument("--feats", required=True, help="pool 의 특징 폴더(p_err.npy 포함)")
    ap.add_argument("--tau", type=float, default=0.10)
    ap.add_argument("--max_wild_frac", type=float, default=0.35)
    ap.add_argument("--native", default=str(MDD_OUTPUTS / "manifest_kspon_stage1"))
    ap.add_argument("--native_ratio", type=float, default=1.0, help="505 시간 대비 원어민 시간")
    ap.add_argument("--report_tag", default=None, help="정답이 있는 TAG 특징 split 폴더 — tau 표를 찍는다")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    taus = [0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]
    rep = {"tau": a.tau, "max_wild_frac": a.max_wild_frac}
    if a.report_tag:
        rep["tag_check"] = tag_report(a.report_tag, taus)
        print("[TAG 점검] 표준발음을 전부 정답으로 줄 때 오염", rep["tag_check"]["naive_contamination"])
        for r in rep["tag_check"]["by_tau"]:
            print(f"  tau {r['tau']:.2f}: 정상 자리 {r['kept_normal_frac']:.1%} 유지 · 남긴 라벨의 오염 {r['contamination_in_kept']:.3%} · 오발음의 {r['errors_masked_frac']:.1%} 를 * 로 가림")

    with open(os.path.join(a.pool, "pool.tsv"), encoding="utf-8") as f:
        root = f.readline().rstrip("\n"); rows = [l.rstrip("\n").split("\t") for l in f]
    canon = [l.split() for l in open(os.path.join(a.pool, "pool.canon"), encoding="utf-8")]
    meta = [json.loads(l) for l in open(os.path.join(a.pool, "pool.meta.jsonl"), encoding="utf-8")]
    p = np.load(os.path.join(a.feats, "p_err.npy")); off = np.load(os.path.join(a.feats, "offsets.npy")); ok = np.load(os.path.join(a.feats, "ok.npy"))
    assert len(rows) == len(canon) == len(ok), (len(rows), len(canon), len(ok))

    os.makedirs(a.out, exist_ok=True)
    n_keep = n_drop = 0; frames_505 = 0; tok_tot = tok_wild = 0
    by_topik = collections.defaultdict(lambda: [0, 0]); by_lang = collections.defaultdict(lambda: [0, 0]); wf_all = []
    lines_tsv, lines_ltr = [], []
    for i, (row, c) in enumerate(zip(rows, canon)):
        if not ok[i]:
            n_drop += 1; continue
        pi = p[off[i]:off[i + 1]]
        wild = pi >= a.tau
        wf = float(wild.mean()); wf_all.append(wf)
        k = str(meta[i].get("topik")); by_topik[k][0] += int(wild.sum()); by_topik[k][1] += len(wild)
        g = meta[i].get("language") or "?"; by_lang[g][0] += int(wild.sum()); by_lang[g][1] += len(wild)
        if wf > a.max_wild_frac:
            n_drop += 1; continue
        lab, prev = [], None
        for s, w in zip(c, wild):
            t = "*" if w else s
            if t == "*" and prev == "*":
                continue
            lab.append(t); prev = t
        lines_tsv.append(f"{os.path.join(root, row[0]).lstrip('/')}\t{row[1]}"); lines_ltr.append(" ".join(lab))
        n_keep += 1; frames_505 += int(row[1]); tok_tot += len(wild); tok_wild += int(wild.sum())

    rng = random.Random(a.seed)
    with open(os.path.join(a.native, "train.tsv"), encoding="utf-8") as f:
        nroot = f.readline().rstrip("\n"); nrows = [l.rstrip("\n").split("\t") for l in f]
    nltr = open(os.path.join(a.native, "train.ltr"), encoding="utf-8").read().splitlines()
    order = list(range(len(nrows))); rng.shuffle(order)
    need, got, n_nat = frames_505 * a.native_ratio, 0, 0
    for j in order:
        if got >= need:
            break
        lines_tsv.append(f"{os.path.join(nroot, nrows[j][0]).lstrip('/')}\t{nrows[j][1]}"); lines_ltr.append(nltr[j])
        got += int(nrows[j][1]); n_nat += 1
    idx = list(range(len(lines_tsv))); rng.shuffle(idx)
    with open(os.path.join(a.out, "train.tsv"), "w", encoding="utf-8") as ft, open(os.path.join(a.out, "train.ltr"), "w", encoding="utf-8") as fl:
        ft.write("/\n")
        for j in idx:
            ft.write(lines_tsv[j] + "\n"); fl.write(lines_ltr[j] + "\n")
    with open(os.path.join(a.native, "valid.tsv"), encoding="utf-8") as f, open(os.path.join(a.out, "valid.tsv"), "w", encoding="utf-8") as ft:
        vroot = f.readline().rstrip("\n"); ft.write("/\n")
        for l in f:
            pth, n = l.rstrip("\n").split("\t"); ft.write(f"{os.path.join(vroot, pth).lstrip('/')}\t{n}\n")
    shutil.copy(os.path.join(a.native, "valid.ltr"), os.path.join(a.out, "valid.ltr"))
    shutil.copy(os.path.join(a.pool, "dict.ltr.txt"), os.path.join(a.out, "dict.ltr.txt"))

    rate = lambda d: {k: round(v[0] / max(1, v[1]), 4) for k, v in sorted(d.items(), key=lambda kv: -kv[1][1])[:12]}
    rep.update({"utts_505_kept": n_keep, "utts_505_dropped": n_drop, "hours_505": round(frames_505 / 16000 / 3600, 1),
                "utts_native": n_nat, "hours_native": round(got / 16000 / 3600, 1), "wildcard_token_rate_kept": round(tok_wild / max(1, tok_tot), 4),
                "wild_frac_percentiles": {q: round(float(np.percentile(wf_all, q)), 4) for q in (50, 90, 99)},
                "wildcard_rate_by_topik": rate(by_topik), "wildcard_rate_by_language": rate(by_lang)})
    json.dump(rep, open(os.path.join(a.out, "stats.json"), "w"), ensure_ascii=False, indent=1)
    print(f"[완료] {a.out} · 505 {n_keep:,}발화/{rep['hours_505']}h (제외 {n_drop:,}) + 원어민 {n_nat:,}발화/{rep['hours_native']}h · "
          f"* 비율 {rep['wildcard_token_rate_kept']:.2%}")
    print("  TOPIK 별 * 비율:", rep["wildcard_rate_by_topik"])
    print("  모국어별 * 비율:", rep["wildcard_rate_by_language"])


if __name__ == "__main__":
    main()
