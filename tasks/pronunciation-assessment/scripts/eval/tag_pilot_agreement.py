# -*- coding: utf-8 -*-
"""사람 TAG 파일럿 P0 분석 — 작업 시간 · 전사자 간 일치 · 미리 채움 효과 · 판정 대상 모으기 · 통과 판정(docs/05 §8).

`deploy/tag-pilot` 이 남긴 labels_A.jsonl · labels_B.jsonl(발화마다 마지막 제출)과 items.json 을 읽는다.
한글 전사는 `mdd.hangul_jamo.decompose` 로 TAG 와 같은 자모열로 바꾸고, **표준발음 자리 기준**으로 비교한다
(자리마다 일치 "=" · 낸 자모 · 빠짐 "-", 자리 사이에 끼어든 소리는 삽입).

보고하는 다섯 가지(+ 참고):
1. 발화당 전사 시간 중앙값(오디오 1초당 포함)
2. 두 전사자 실제 발음 일치율 — 자모 일치도 · 자리 일치율 · 자모열 완전 일치
3. 오류 위치 일치 — Cohen's κ · 양성 일치(Dice) · A 를 기준으로 본 B 의 정밀도/재현율
4. 미리 채움의 시간 절감 — 같은 발화의 빈칸 전사 시간 대비(교차 배정이라 전사자 효과가 상쇄된다)
5. 미리 채움의 표준발음 끌림 — 같은 발화에서 "표준발음과 다름" 자리 비율이 얼마나 줄었나
+ 유형별 불일치율(ㅅ/ㅆ · ㄱ계열 · 삭제 · 삽입 등) · 사람 ↔ 의사라벨 · 통과 조건 판정
판정 대상(두 사람이 다르게 쓴 자리)은 정답을 정하지 않고 `adjudication.jsonl` · `.csv` 로 따로 모은다.

    $PY .../tag_pilot_agreement.py --dir $MDD_OUTPUTS/tag_pilot/p0_v1
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

# 통과 조건 — 결과를 보기 전에 정했다(2026-10-04, docs/05 §8). 바꾸면 문서에 이유와 날짜를 남길 것.
KAPPA_PASS, KAPPA_COND = 0.60, 0.40            # 오류 위치 κ: 이상 통과 · 사이 조건부(규칙서 보완) · 미만 불통과
MAX_SEC_PER_UTT, MAX_SEC_PER_AUDIO = 120.0, 25.0
PREFILL_MIN_SAVING, PREFILL_MAX_PULL = 0.20, 0.25   # 시간 20% 이상 줄고 끌림(상대) 25% 이하면 P1 에 미리 채움 사용
HUMAN_OVER_PSEUDO = 0.20                       # κ(사람 A, B) − κ(사람 합의, 의사라벨) 이 이 이상

FAMILIES = [("ㅅ/ㅆ", set(u"ㅅㅆ")), ("ㄱ/ㄲ/ㅋ", set(u"ㄱㄲㅋ")), ("ㄷ/ㄸ/ㅌ", set(u"ㄷㄸㅌ")),
            ("ㅂ/ㅃ/ㅍ", set(u"ㅂㅃㅍ")), ("ㅈ/ㅉ/ㅊ", set(u"ㅈㅉㅊ"))]
VOWELS = set(u"ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")


def sites(canon, hyp):
    """표준발음 자리마다 ("=" | 낸 자모 | "-" 빠짐) + 자리 앞 삽입 목록(len(canon)+1 칸)."""
    import edlib
    st, ins = ["-"] * len(canon), [[] for _ in range(len(canon) + 1)]
    if not canon:
        ins[0] = list(hyp)
        return st, ins
    if not hyp:
        return st, ins
    c, h = "".join(canon), "".join(hyp)
    a = edlib.getNiceAlignment(edlib.align(c, h, task="path"), c, h)
    i = 0
    for x, y in zip(a["query_aligned"], a["target_aligned"]):
        if x == "-":
            ins[i].append(y)
            continue
        st[i] = "=" if x == y else ("-" if y == "-" else y)
        i += 1
    return st, ins


def err_type(c, va, vb):
    if "-" in (va, vb):
        return "삭제"
    vals = set(v for v in (va, vb) if v != "=")
    for name, fam in FAMILIES:
        if c in fam and vals <= fam:
            return name
    return "모음" if c in VOWELS else "기타 자음"


def kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    if not len(a):
        return float("nan")
    po = float((a == b).mean())
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def load_labels(d, who):
    p = os.path.join(d, "labels_%s.jsonl" % who)
    out = {}
    if os.path.exists(p):
        for l in open(p, encoding="utf-8"):
            r = json.loads(l)
            out[r["id"]] = r
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="items.json · labels_A.jsonl · labels_B.jsonl 이 있는 폴더")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from mdd.hangul_jamo import decompose
    import edlib
    items = dict((it["id"], it) for it in json.load(open(os.path.join(a.dir, "items.json"), encoding="utf-8")))
    L = {w: load_labels(a.dir, w) for w in ("A", "B")}
    res = {"n_items": len(items), "n_labeled": {w: len(L[w]) for w in L},
           "n_unclear": {w: sum(bool(r.get("flag")) for r in L[w].values()) for w in L}}

    # 1. 작업 시간
    allt = [r["time_sec"] for w in L for r in L[w].values() if not r.get("flag")]
    alls = [r["time_sec"] / items[r["id"]]["sec"] for w in L for r in L[w].values() if not r.get("flag")]
    res["time"] = {"median_sec_per_utt": float(np.median(allt)) if allt else None,
                   "median_sec_per_audio_sec": float(np.median(alls)) if alls else None, "by": {}}
    for w in L:
        for mode in ("blank", "canon"):
            R = [r for r in L[w].values() if r["mode"] == mode and not r.get("flag")]
            if R:
                res["time"]["by"]["%s_%s" % (w, mode)] = {
                    "n": len(R), "median_sec": float(np.median([r["time_sec"] for r in R])),
                    "plays_median": float(np.median([r["plays"] for r in R]))}

    both = sorted(i for i in items if i in L["A"] and i in L["B"] and not L["A"][i].get("flag") and not L["B"][i].get("flag"))
    if not both:
        json.dump(res, open(a.out or os.path.join(a.dir, "agreement.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("두 사람이 모두 전사한 발화가 아직 없다 — A %d · B %d / %d" % (len(L["A"]), len(L["B"]), len(items)))
        return

    seq_eq, char_agree, site_agree = 0, [], []
    errA, errB = [], []
    by_type = collections.defaultdict(lambda: [0, 0])          # 유형 → [불일치, 한 명 이상 오류]
    adj = []
    pair_time, pair_pull = [], []
    rate = {"blank": [0, 0], "canon": [0, 0]}
    hp_h, hp_p = [], []
    for i in both:
        it = items[i]; canon = it["canon"]
        ra, rb = L["A"][i], L["B"][i]
        ja, _ = decompose(ra["text"]); jb, _ = decompose(rb["text"])
        seq_eq += ja == jb
        char_agree.append(1 - edlib.align("".join(ja), "".join(jb))["editDistance"] / max(1, len(ja), len(jb)))
        sa, ia = sites(canon, ja); sb, ib = sites(canon, jb)
        site_agree.append(np.mean([x == y for x, y in zip(sa, sb)] + [x == y for x, y in zip(ia, ib)]))
        for k in range(len(canon)):
            ea, eb = sa[k] != "=", sb[k] != "="
            errA.append(ea); errB.append(eb)
            if ea or eb:
                t = err_type(canon[k], sa[k], sb[k])
                by_type[t][1] += 1
                if sa[k] != sb[k]:
                    by_type[t][0] += 1
                    adj.append({"id": i, "stem": it["stem"], "kind": "자리", "pos": k, "canon": canon[k],
                                "A": sa[k], "B": sb[k], "type": t})
        for k in range(len(canon) + 1):
            if ia[k] or ib[k]:
                by_type["삽입"][1] += 1
                if ia[k] != ib[k]:
                    by_type["삽입"][0] += 1
                    adj.append({"id": i, "stem": it["stem"], "kind": "삽입", "pos": k, "canon": "",
                                "A": "".join(ia[k]), "B": "".join(ib[k]), "type": "삽입"})
        # 미리 채움 효과 — 같은 발화의 빈칸 전사 vs 미리 채움 전사(교차 배정이라 전사자 효과가 상쇄된다)
        rec = {ra["mode"]: (ra, sa), rb["mode"]: (rb, sb)}
        if set(rec) == {"blank", "canon"}:
            (rbk, sbk), (rcn, scn) = rec["blank"], rec["canon"]
            pair_time.append(1 - rcn["time_sec"] / max(1e-6, rbk["time_sec"]))
            eb_, ec_ = np.mean([x != "=" for x in sbk]), np.mean([x != "=" for x in scn])
            pair_pull.append((eb_, ec_))
            for m, s_ in (("blank", sbk), ("canon", scn)):
                rate[m][0] += sum(x != "=" for x in s_); rate[m][1] += len(s_)
        # 사람 합의(두 사람이 같은 판단을 한 자리) ↔ 의사라벨
        sp, _ = sites(canon, it["pseudo"])
        for k in range(len(canon)):
            if (sa[k] != "=") == (sb[k] != "="):
                hp_h.append(sa[k] != "="); hp_p.append(sp[k] != "=")

    errA, errB = np.asarray(errA), np.asarray(errB)
    tp = int((errA & errB).sum())
    agree = {"n_both": len(both), "sequence_exact": seq_eq / len(both), "char_agreement": float(np.mean(char_agree)),
             "site_agreement": float(np.mean(site_agree)), "kappa": kappa(errA, errB),
             "dice": 2 * tp / max(1, int(errA.sum() + errB.sum())),
             "B_precision_vs_A": tp / max(1, int(errB.sum())), "B_recall_vs_A": tp / max(1, int(errA.sum())),
             "error_rate_A": float(errA.mean()), "error_rate_B": float(errB.mean()), "sites": int(len(errA))}
    res["agreement"] = agree
    res["by_type"] = dict((t, {"disagree": v[0], "flagged": v[1], "disagree_rate": v[0] / max(1, v[1])}) for t, v in by_type.items())
    pb = sum(x for x, _ in pair_pull) / max(1, len(pair_pull)); pc = sum(y for _, y in pair_pull) / max(1, len(pair_pull))
    res["prefill"] = {"n_pairs": len(pair_time), "time_saving_median": float(np.median(pair_time)) if pair_time else None,
                      "error_site_rate_blank": pb, "error_site_rate_canon": pc,
                      "pull_relative": 1 - pc / pb if pb > 0 else None,
                      "pooled": dict((m, v[0] / max(1, v[1])) for m, v in rate.items())}
    kh = kappa(hp_h, hp_p)
    res["human_vs_pseudo"] = {"sites": len(hp_h), "kappa": kh, "human_error_rate": float(np.mean(hp_h)),
                              "pseudo_error_rate": float(np.mean(hp_p))}

    # 통과 판정(미리 정한 조건)
    k = agree["kappa"]
    v = {"1_일치도": "통과" if k >= KAPPA_PASS else ("조건부(규칙서 보완)" if k >= KAPPA_COND else "불통과(규칙서 고쳐 P0 재실시)"),
         "2_시간": "통과" if (res["time"]["median_sec_per_utt"] <= MAX_SEC_PER_UTT and
                           res["time"]["median_sec_per_audio_sec"] <= MAX_SEC_PER_AUDIO) else "불통과(비용 재산정)",
         "3_미리채움": ("P1 에 미리 채움 사용" if (res["prefill"]["time_saving_median"] or 0) >= PREFILL_MIN_SAVING
                    and (res["prefill"]["pull_relative"] or 0) <= PREFILL_MAX_PULL else "P1 은 빈칸으로"),
         "4_사람>의사라벨": "통과" if k - kh >= HUMAN_OVER_PSEUDO else "불통과(사람 라벨의 추가 가치가 불분명)"}
    res["verdict"] = v

    # 판정 대상 — 정답을 정하지 않고 모은다
    for r in adj:
        it = items[r["id"]]
        r.update({"text": it["text"], "canon_hangul": it["canon_hangul"], "A_text": L["A"][r["id"]]["text"],
                  "B_text": L["B"][r["id"]]["text"], "A_mode": L["A"][r["id"]]["mode"], "B_mode": L["B"][r["id"]]["mode"]})
    with open(os.path.join(a.dir, "adjudication.jsonl"), "w", encoding="utf-8") as f:
        for r in adj:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    cols = ["id", "stem", "type", "kind", "pos", "canon", "A", "B", "text", "canon_hangul", "A_text", "B_text", "A_mode", "B_mode"]
    with open(os.path.join(a.dir, "adjudication.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols + ["판정"], extrasaction="ignore")
        w.writeheader()
        for r in adj:
            w.writerow(r)
    res["n_adjudication"] = len(adj)

    out = a.out or os.path.join(a.dir, "agreement.json")
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    t = res["time"]
    print("전사 완료 A %d · B %d / %d (판단 불가 A %d · B %d) · 둘 다 %d개"
          % (len(L["A"]), len(L["B"]), len(items), res["n_unclear"]["A"], res["n_unclear"]["B"], len(both)))
    print("1) 발화당 전사 시간 중앙값 %.0f초 · 오디오 1초당 %.1f초" % (t["median_sec_per_utt"], t["median_sec_per_audio_sec"]))
    for kk, vv in t["by"].items():
        print("     %-8s %2d개 · %.0f초 · 재생 %.0f회" % (kk, vv["n"], vv["median_sec"], vv["plays_median"]))
    print("2) 실제 발음 일치 — 자모 일치도 %.3f · 자리 일치율 %.3f · 자모열 완전 일치 %.0f%%"
          % (agree["char_agreement"], agree["site_agreement"], 100 * agree["sequence_exact"]))
    print("3) 오류 위치 — κ %.3f · 양성 일치(Dice) %.3f · B 정밀도 %.2f / 재현율 %.2f (A 기준) · 오류 비율 A %.1f%% · B %.1f%% (%d자리)"
          % (agree["kappa"], agree["dice"], agree["B_precision_vs_A"], agree["B_recall_vs_A"],
             100 * agree["error_rate_A"], 100 * agree["error_rate_B"], agree["sites"]))
    p = res["prefill"]
    print("4) 미리 채움 시간 절감(같은 발화 중앙값) %.0f%% (%d쌍)" % (100 * (p["time_saving_median"] or 0), p["n_pairs"]))
    print("5) 미리 채움 끌림 — 표준발음과 다름 자리: 빈칸 %.1f%% → 미리 채움 %.1f%% (상대 %+.0f%%)"
          % (100 * p["error_site_rate_blank"], 100 * p["error_site_rate_canon"], -100 * (p["pull_relative"] or 0)))
    print("   유형별 불일치율: " + " · ".join("%s %d/%d(%.0f%%)" % (tt, vv["disagree"], vv["flagged"], 100 * vv["disagree_rate"])
                                        for tt, vv in sorted(res["by_type"].items(), key=lambda x: -x[1]["flagged"])))
    h = res["human_vs_pseudo"]
    print("   사람 합의 ↔ 의사라벨 κ %.3f (사람끼리 %.3f) · 오류 비율 사람 %.1f%% · 의사라벨 %.1f%%"
          % (h["kappa"], agree["kappa"], 100 * h["human_error_rate"], 100 * h["pseudo_error_rate"]))
    print("판정 대상 %d건 → adjudication.jsonl · adjudication.csv" % len(adj))
    for kk, vv in v.items():
        print("[판정] %s: %s" % (kk, vv))
    print("[saved] %s" % out)


if __name__ == "__main__":
    main()
