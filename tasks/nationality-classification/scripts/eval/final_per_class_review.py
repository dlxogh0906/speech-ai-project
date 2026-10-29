# -*- coding: utf-8 -*-
"""최종 점검 — 국적 모델 클래스별 성능 표 (2026-10-26).

test 로짓 한 벌과 같은 체크포인트의 valid 로짓으로 클래스마다 잰다.
  - 웹마이크 문장·자유발화(단어 RW 제외): 발화 1개 top-1 · top-5(화자 복원추출 95% 구간), top-1 정밀도, 많이 간 오답
  - 화자 단위: 같은 화자의 문장 k개(3 · 5 · 전부)를 log-softmax 평균으로 합친 top-1 · top-5
  - 웹 단어(RW) · 131 읽기 top-1 · top-5
  - 한국: 시험 묶음별 정답률, 외국인을 한국으로 부른 비율
  - 전체: 확신도 문턱별 응답 비율 · 정답률(온도와 문턱은 valid 로 정함), 길이 · 과제별, 녹음 묶음(web040~043) 단서

    $PY scripts/eval/final_per_class_review.py --train <train.csv> --test <test.csv> --logits <test logits.npz> \
        --valid <valid.csv> --valid_logits <valid logits.npz> --name "300m 전체학습" --out <폴더>
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import argparse, collections, csv, json, os
import numpy as np

KO = {"Bangladesh": "방글라데시", "Brazil": "브라질", "Bulgaria": "불가리아", "Cambodia": "캄보디아", "Chile": "칠레",
      "China": "중국", "Colombia": "콜롬비아", "Ethiopia": "에티오피아", "France": "프랑스", "Germany": "독일",
      "Hong Kong": "홍콩", "India": "인도", "Indonesia": "인도네시아", "Iran": "이란", "Italy": "이탈리아", "Japan": "일본",
      "Kazakhstan": "카자흐스탄", "Kyrgyzstan": "키르기스스탄", "Malaysia": "말레이시아", "Mongolia": "몽골",
      "Myanmar": "미얀마", "Nepal": "네팔", "Pakistan": "파키스탄", "Peru": "페루", "Philippines": "필리핀",
      "Romania": "루마니아", "Russia": "러시아", "Rwanda": "르완다", "Singapore": "싱가포르", "Spain": "스페인",
      "Tajikistan": "타지키스탄", "Thailand": "태국", "Turkey": "튀르키예", "Turkmenistan": "투르크메니스탄",
      "Ukraine": "우크라이나", "United Kingdom": "영국", "United States": "미국", "Uzbekistan": "우즈베키스탄",
      "Vietnam": "베트남", "Zambia": "잠비아", "Korea": "한국"}
FREE = {"LAR", "ATQ", "SPT", "RP", "PDT", "EIG", "SSO"}      # 웹 자유발화 · 질문 답하기 계열 (RS = 읽은 문장, RW = 단어)
DUR_BINS = [(0, 2), (2, 4), (4, 8), (8, 1e9)]


def read_csv(p):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_logits(p, n):
    d = np.load(p, allow_pickle=True)
    if len(d["ok"]) != n:
        raise SystemExit(f"{p}: 행 {len(d['ok']):,} ≠ 시험지 {n:,}")
    return [str(x) for x in d["labels"]], d["L"].astype(np.float64), d["ok"].astype(bool)


def log_softmax(L, T=1.0):
    z = L / T
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def fit_temperature(L, y):
    grid = np.arange(0.5, 5.001, 0.05)
    nll = [-log_softmax(L, t)[np.arange(len(y)), y].mean() for t in grid]
    j = int(np.argmin(nll))
    return float(grid[j]), float(nll[j]), float(-log_softmax(L, 1.0)[np.arange(len(y)), y].mean())


def ece(conf, correct, bins=15):
    e = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for a, b in zip(edges[:-1], edges[1:]):
        m = (conf > a) & (conf <= b)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(e)


def rank_of(L, y):
    """정답 로짓보다 큰 로짓 개수 = 정답 순위(0 이면 1위)."""
    return (L > L[np.arange(len(y)), y][:, None]).sum(1)


def speaker_groups(spk, idx):
    g = collections.OrderedDict()
    for i in idx:
        g.setdefault(spk[i], []).append(i)
    return [np.array(v) for v in g.values()]


def boot_ci(groups, x, n, rng):
    if not groups:
        return [None, None]
    s = np.array([x[g].sum() for g in groups], np.float64); c = np.array([len(g) for g in groups], np.float64)
    bs = []
    for _ in range(n):
        k = rng.integers(0, len(groups), len(groups))
        bs.append(s[k].sum() / c[k].sum())
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def agg_speaker(groups, LS, y, k, draws, rng):
    """화자마다 발화 k개(None 이면 전부)를 log-softmax 평균으로 합쳐 top-1 · top-5. 화자별 평균을 돌려준다."""
    out = []
    for g in groups:
        if k is not None and len(g) < k:
            continue
        sets = [g] if (k is None or len(g) == k) else [rng.choice(g, k, replace=False) for _ in range(draws)]
        t1 = t5 = 0.0
        for s in sets:
            m = LS[s].mean(0); c = y[s[0]]
            r = int((m > m[c]).sum())
            t1 += r == 0; t5 += r < 5
        out.append((t1 / len(sets), t5 / len(sets)))
    return out


def rate(x):
    return float(np.mean(x)) if len(x) else None


def grade(r, min_spk):
    """앱에서 쓸 수준 등급. 시험 화자가 min_spk 명 이상일 때만 매긴다.
    A 바로 사용 = 5문장 합산 top-1 ≥ 0.90 이고 top-1 정밀도 ≥ 0.70 · B 후보 제시 = 1문장 top-5 ≥ 0.90 · C 어려움 = 그 밖.
    웹 문장 시험 화자가 모자라고 131 읽기 화자가 충분하면 131 발화 top-1 ≥ 0.90 → A, top-5 ≥ 0.90 → B 로 대신 매긴다."""
    if r["class"] == "Korea":
        return "조건부", "스튜디오 · 낭독 한국인만 시험 (웹마이크 한국인 없음)"
    if r["test_spk"] >= min_spk:
        if r["spk5_top1"] is not None and r["spk5_top1"] >= 0.9 and (r["precision"] or 0) >= 0.7:
            return "A", "웹 문장"
        return ("B" if r["top5"] >= 0.9 else "C"), "웹 문장"
    if r["r131_spk"] >= min_spk:
        if r["r131_top1"] >= 0.9:
            return "A", "131 읽기 (웹 시험 화자 %d명)" % r["test_spk"]
        return ("B" if r["r131_top5"] >= 0.9 else "C"), "131 읽기 (웹 시험 화자 %d명)" % r["test_spk"]
    return "판단 불가", "시험 화자 %d명" % r["test_spk"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--logits", required=True)
    ap.add_argument("--valid", required=True)
    ap.add_argument("--valid_logits", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--draws", type=int, default=20, help="화자당 k문장 묶음을 뽑는 횟수")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min_spk", type=int, default=5, help="등급을 매길 최소 시험 화자 수")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    train = read_csv(a.train)
    tr_spk = collections.defaultdict(set); tr_rows = collections.Counter(); camp = collections.defaultdict(set)
    for r in train:
        tr_spk[r["label_en"]].add(r["speaker"]); tr_rows[r["label_en"]] += 1
        if r["domain"].startswith("web"):
            camp[r["domain"]].add(r["label_en"])

    rows = read_csv(a.test)
    labels, L, ok = load_logits(a.logits, len(rows))
    ci = {c: i for i, c in enumerate(labels)}
    y = np.array([ci[r["label_en"]] for r in rows])
    spk = [r["speaker"] for r in rows]
    dom = np.array([r["domain"] for r in rows]); task = np.array([r["task"] for r in rows])
    dur = np.array([float(r["duration_sec"] or 0) for r in rows])
    web = np.array([d.startswith("web") for d in dom]) & ok
    sent = web & (task != "RW"); word = web & (task == "RW"); r131 = (dom == "p505") & ok

    vrows = read_csv(a.valid)
    vlab, Lv, okv = load_logits(a.valid_logits, len(vrows))
    if vlab != labels:
        raise SystemExit("valid 와 test 의 라벨 순서가 다르다")
    yv = np.array([ci[r["label_en"]] for r in vrows])
    T, nll_T, nll_1 = fit_temperature(Lv[okv], yv[okv])

    LS = log_softmax(L, T)
    pred = L.argmax(1); rk = rank_of(L, y)
    top1 = rk == 0; top5 = rk < 5

    def tier(c):
        n = len(tr_spk[c])
        return "꼬리 (≤15)" if n <= 15 else ("중간 (16~100)" if n <= 100 else "머리 (>100)")

    per = []
    idx_sent = np.where(sent)[0]
    for c in labels:
        k = ci[c]
        own = np.where(sent & (y == k))[0]
        groups = speaker_groups(spk, own)
        p_mask = sent & (pred == k)
        conf = collections.Counter(int(pred[i]) for i in own if pred[i] != k)
        row = {"class": c, "ko": KO.get(c, c), "tier": tier(c) if c != "Korea" else "한국",
               "train_spk": len(tr_spk[c]), "train_rows": tr_rows[c],
               "test_spk": len(groups), "test_utt": int(len(own)),
               "top1": rate(top1[own]), "top5": rate(top5[own]),
               "top1_ci": boot_ci(groups, top1, a.n_boot, rng), "top5_ci": boot_ci(groups, top5, a.n_boot, rng),
               "pred_n": int(p_mask.sum()), "precision": rate((y[p_mask] == k)) if p_mask.any() else None,
               "confusions": [[labels[j], KO.get(labels[j], labels[j]), n / len(own)] for j, n in conf.most_common(3)]}
        if row["top1"] is not None and row["precision"] is not None and (row["top1"] + row["precision"]) > 0:
            row["f1"] = 2 * row["top1"] * row["precision"] / (row["top1"] + row["precision"])
        else:
            row["f1"] = None
        for kk, tag in ((3, "spk3"), (5, "spk5"), (None, "spk_all")):
            res = agg_speaker(groups, LS, y, kk, a.draws, rng)
            row[tag + "_n"] = len(res)
            row[tag + "_top1"] = rate([r[0] for r in res]); row[tag + "_top5"] = rate([r[1] for r in res])
        for m, tag in ((word, "word"), (r131, "r131")):
            o = np.where(m & (y == k))[0]
            row[tag + "_utt"] = int(len(o)); row[tag + "_spk"] = len(set(spk[i] for i in o))
            row[tag + "_top1"] = rate(top1[o]); row[tag + "_top5"] = rate(top5[o])
        row["grade"], row["grade_basis"] = grade(row, a.min_spk)
        per.append(row)

    # 한국
    kor = {}
    kk = ci["Korea"]
    for d in sorted(set(dom[y == kk])):
        o = np.where((dom == d) & (y == kk) & ok)[0]
        kor[d] = {"utt": int(len(o)), "spk": len(set(spk[i] for i in o)), "top1": rate(top1[o])}
    foreign = (y != kk) & ok
    kor["foreign_as_korea"] = {"all": rate(pred[foreign] == kk), "web_sent": rate(pred[sent & (y != kk)] == kk),
                               "web_word": rate(pred[word & (y != kk)] == kk), "r131": rate(pred[r131 & (y != kk)] == kk)}

    # 전체 (웹 문장)
    countries = [r for r in per if r["test_utt"] > 0 and r["class"] != "Korea"]
    allgroups = speaker_groups(spk, idx_sent)
    overall = {"utt": int(sent.sum()), "spk": len(allgroups), "countries": len(countries),
               "top1": rate(top1[sent]), "top5": rate(top5[sent]),
               "top1_ci": boot_ci(allgroups, top1, a.n_boot, rng), "top5_ci": boot_ci(allgroups, top5, a.n_boot, rng),
               "country_mean_top1": float(np.mean([r["top1"] for r in countries])),
               "country_mean_top5": float(np.mean([r["top5"] for r in countries]))}
    for kk2, tag in ((3, "spk3"), (5, "spk5"), (None, "spk_all")):
        res = agg_speaker(allgroups, LS, y, kk2, a.draws, rng)
        overall[tag + "_n"] = len(res)
        overall[tag + "_top1"] = rate([r[0] for r in res]); overall[tag + "_top5"] = rate([r[1] for r in res])
    overall["word_top1"] = rate(top1[word]); overall["word_top5"] = rate(top5[word]); overall["word_utt"] = int(word.sum())
    overall["r131_top1"] = rate(top1[r131]); overall["r131_top5"] = rate(top5[r131]); overall["r131_utt"] = int(r131.sum())
    tiers = {}
    for t in ("머리 (>100)", "중간 (16~100)", "꼬리 (≤15)"):
        g = [r for r in countries if r["tier"] == t]
        tiers[t] = {"countries": len(g), "test_spk": sum(r["test_spk"] for r in g),
                    "top1": float(np.mean([r["top1"] for r in g])), "top5": float(np.mean([r["top5"] for r in g])),
                    "spk5_top1": float(np.mean([r["spk5_top1"] for r in g if r["spk5_top1"] is not None]))
                    if any(r["spk5_top1"] is not None for r in g) else None}
    overall["tiers"] = tiers

    # 길이 · 과제
    by_len = []
    for lo, hi in DUR_BINS:
        m = sent & (dur >= lo) & (dur < hi)
        by_len.append({"bin": f"{lo}~{hi if hi < 1e8 else ''}초", "utt": int(m.sum()), "top1": rate(top1[m]), "top5": rate(top5[m])})
    for lo, hi in DUR_BINS:
        m = word & (dur >= lo) & (dur < hi)
        by_len.append({"bin": f"단어 {lo}~{hi if hi < 1e8 else ''}초", "utt": int(m.sum()), "top1": rate(top1[m]), "top5": rate(top5[m])})
    by_task = {}
    for name, m in (("읽은 문장 (RS)", web & (task == "RS")), ("자유발화 · 질문 답하기", web & np.isin(task, list(FREE))),
                    ("단어 (RW)", word)):
        by_task[name] = {"utt": int(m.sum()), "top1": rate(top1[m]), "top5": rate(top5[m]),
                         "dur_median": float(np.median(dur[m])) if m.any() else None}

    # 확신도 문턱 — 온도 T 는 valid NLL 로, 문턱은 valid 웹 문장에서 정한다
    vdom = np.array([r["domain"] for r in vrows]); vtask = np.array([r["task"] for r in vrows])
    vsent = np.array([d.startswith("web") for d in vdom]) & (vtask != "RW") & okv
    Pv = np.exp(log_softmax(Lv, T)); confv = Pv.max(1); corv = Pv.argmax(1) == yv
    P = np.exp(LS); conf = P.max(1); cor = top1
    sel = []
    targets = [("valid 정답률 0.90", 0.90), ("valid 정답률 0.95", 0.95)]
    for name, tgt in targets:
        thr = None
        for t in np.arange(0.0, 1.0, 0.01):
            m = vsent & (confv >= t)
            if m.sum() >= 50 and corv[m].mean() >= tgt:
                thr = float(t); break
        if thr is None:
            continue
        m = sent & (conf >= thr)
        sel.append({"rule": name, "thr": thr, "valid_cov": rate(confv[vsent] >= thr),
                    "test_cov": float(m.sum() / sent.sum()), "test_acc": rate(cor[m]),
                    "test_acc_rest": rate(cor[sent & (conf < thr)]), "test_top5_rest": rate(top5[sent & (conf < thr)])})
    for thr in (0.5, 0.7, 0.9):
        m = sent & (conf >= thr)
        sel.append({"rule": f"확신도 ≥ {thr}", "thr": thr, "valid_cov": rate(confv[vsent] >= thr),
                    "test_cov": float(m.sum() / sent.sum()), "test_acc": rate(cor[m]),
                    "test_acc_rest": rate(cor[sent & (conf < thr)]), "test_top5_rest": rate(top5[sent & (conf < thr)])})
    calib = {"T": T, "valid_nll_T1": nll_1, "valid_nll_T": nll_T,
             "test_ece_T1": ece(np.exp(log_softmax(L, 1.0)).max(1)[sent], cor[sent]),
             "test_ece_T": ece(conf[sent], cor[sent]),
             "test_mean_conf_T1": float(np.exp(log_softmax(L, 1.0)).max(1)[sent].mean()),
             "test_mean_conf_T": float(conf[sent].mean()), "test_acc": rate(cor[sent])}

    # 녹음 묶음 단서: top-1 이 그 발화의 묶음(web040~043)에 있는 나라인가
    in_camp = np.array([labels[pred[i]] in camp.get(dom[i], set()) for i in idx_sent])
    wrong = ~top1[idx_sent]
    exp_wrong = np.mean([(len(camp.get(dom[i], set())) - 1) / (len(labels) - 1) for i in idx_sent[wrong]])
    campaign = {"sizes": {d: len(s) for d, s in sorted(camp.items())},
                "top1_in_campaign": float(in_camp.mean()), "wrong_in_campaign": float(in_camp[wrong].mean()),
                "wrong_in_campaign_if_uniform": float(exp_wrong), "wrong_n": int(wrong.sum())}

    summ = {"name": a.name, "logits": a.logits, "test": a.test, "valid_logits": a.valid_logits,
            "overall": overall, "korea": kor, "by_len": by_len, "by_task": by_task, "selective": sel,
            "calibration": calib, "campaign": campaign}
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summ, f, ensure_ascii=False, indent=1)
    with open(os.path.join(a.out, "per_class.json"), "w", encoding="utf-8") as f:
        json.dump(per, f, ensure_ascii=False, indent=1)
    keys = ["class", "ko", "tier", "grade", "grade_basis", "train_spk", "train_rows", "test_spk", "test_utt", "top1", "top1_ci", "top5", "top5_ci",
            "precision", "pred_n", "f1", "spk3_n", "spk3_top1", "spk3_top5", "spk5_n", "spk5_top1", "spk5_top5",
            "spk_all_n", "spk_all_top1", "spk_all_top5", "word_utt", "word_top1", "word_top5", "r131_spk", "r131_utt",
            "r131_top1", "r131_top5", "confusions"]
    with open(os.path.join(a.out, "per_class.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f); w.writerow(keys)
        for r in per:
            out = []
            for k in keys:
                v = r.get(k)
                if isinstance(v, float):
                    v = round(v, 4)
                elif k.endswith("_ci"):
                    v = "" if v[0] is None else f"{v[0]:.3f}~{v[1]:.3f}"
                elif k == "confusions":
                    v = " · ".join(f"{x[1]} {x[2]:.0%}" for x in v)
                elif v is None:
                    v = ""
                out.append(v)
            w.writerow(out)
    print(json.dumps({"overall": {k: v for k, v in overall.items() if k != "tiers"}, "calibration": calib,
                      "campaign": campaign}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
