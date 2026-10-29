# -*- coding: utf-8 -*-
"""표준발음을 어디서 만드느냐에 따라 화면 지적이 어떻게 달라지나 — 헛지적(맞게 읽었는데 틀림)과 재현율(진짜 오류를 틀림으로).

발음 API 는 문장(refText)을 받아 실행 중에 KoG2P 로 표준발음을 만든다. KoG2P 는 경음화를 놓치고(먹을걸 · 결정),
띄어쓰기를 무시해 단어 경계 너머로 동화를 건다(일을 누가 → [이를루가]). 학습·평가 라벨(.canon)은 AIHub 표준발음이다.
같은 녹음을 여러 표준발음으로 채점해(`mdd.api_offline` — API 채점 경로 그대로) 화면 지적(errorType)을 사람 전사(TAG)와 맞춘다.

방법(--methods)
  api      지금 API — 문장 전체 KoG2P
  lab      라벨 표준발음(.canon) — 참고 상한(서비스에서는 쓸 수 없다)
  fixed    `mdd.canonical_fix` 결정적 보정(낱말 고쳐쓰기 · 관형형 된소리 · ㅢ · 져 · ㄴ 첨가 과다)
  choose   fixed + 허용 변이를 CTC 우도로 고름 — **라이브러리 기본**(canonical_fix.EXCLUDE_KINDS 를 뺀 종류)
  choose_all  모든 종류(경계 비음화 포함)
  --ablate 를 주면 변이 종류마다 그 종류만 켠 choose 도 잰다(only:<종류>).
  without:<종류>[+<종류>…] = 그 종류를 뺀 choose(예: "without:경계 비음화").
지표(사람 전사 TAG 를 라벨 표준발음 축에 투영해 판정)
  헛지적 = 화면 지적 중 TAG 가 라벨대로인 자리 + 라벨에 없는 자리(표준발음이 더 가진 자리)
  재현율 = TAG 가 라벨과 다른 자리(대치·탈락) 중 화면 지적이 난 비율
  발화 헛지적률 = 헛지적이 하나라도 난 발화 비율
  diff 자리 = KoG2P 와 라벨이 다른 자리(이전 보고와 같은 잣대)
API 말소리 검사(check_audio)에 걸리는 발화는 API 가 거절한다 — 전체와 '통과한 발화만' 두 가지로 낸다.
문항 원문 = 발음챌린지 문항 목록(train_prompts.json — 띄어쓰기 있음, 화자 정보 없음)을 오류 분석 문장 목록
(sentence_analysis.csv)과 글자만 맞춰 잇고, 문항 번호 ↔ 원문은 표준발음 편집거리로 1:1 짝짓기(헝가리안).
발화의 문항 번호는 녹음 파일 이름의 문항 코드(R?###)만 쓴다.

    $PY .../eval_canonical_source.py --per-item 30 --ablate
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import DATA_ROOT, MDD_OUTPUTS

import argparse, collections, csv, json, os, random, re, time, wave

TENSE = {"ㄱ": "ㄲ", "ㄷ": "ㄸ", "ㅂ": "ㅃ", "ㅅ": "ㅆ", "ㅈ": "ㅉ"}
ITEM_RE = re.compile(r"-(R[WSP]\d{3})-")
KINDS = ("경계 경음·격음", "경계 비음화", "경계 유음화", "경계 격음화", "경계 기타", "ㄴ 첨가", "의→에", "ㄷ 탈락", "ㅎ 탈락")


def nice(a, b):
    import edlib
    if not a or not b:
        return None
    return edlib.getNiceAlignment(edlib.align(a, b, task="path"), a, b)


def a2b(a, b):
    """a 의 자리 i → b 의 자리(없으면 None)."""
    al = nice("".join(a), "".join(b))
    if al is None:
        return [None] * len(a)
    out, j = [], 0
    for x, y in zip(al["query_aligned"], al["target_aligned"]):
        if x == "-":
            j += 1; continue
        out.append(j if y != "-" else None)
        j += y != "-"
    return out


def proj(canon, seq):
    al = nice("".join(canon), "".join(seq))
    if al is None:
        return [None] * len(canon)
    return [None if t == "-" else t for q, t in zip(al["query_aligned"], al["target_aligned"]) if q != "-"]


def item_texts(emu, sentences, prompts, codes, lab):
    """문항 번호 → 띄어쓰기 있는 원문."""
    import edlib
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    norm = lambda s: re.sub(r"[^0-9A-Za-z가-힣]", "", s)
    spaced = dict((norm(p), p) for p in prompts)
    texts = [spaced.get(norm(t), t) for t in sentences]
    api_t = ["".join(emu.canonical(t)) for t in texts]
    C = np.array([[edlib.align(api_t[i], lab[k])["editDistance"] / max(1, len(lab[k])) for k in codes] for i in range(len(texts))])
    r_, c_ = linear_sum_assignment(C)
    return dict((codes[j], texts[i]) for i, j in zip(r_, c_) if C[i, j] <= 0.34)


def score(shown, mc, labc, tag_on_lab, diff_lab):
    """한 방법 · 한 발화 → 카운터."""
    c = collections.Counter()
    m = a2b(mc, labc)
    inv = {}
    for q, kl in enumerate(m):
        if kl is not None:
            inv[kl] = q
    for q, s in enumerate(shown):
        if not s:
            continue
        kl = m[q]
        if kl is None:
            c["ff_extra"] += 1
        elif tag_on_lab[kl] == labc[kl]:
            c["ff"] += 1
        else:
            c["tp"] += 1
    for kl in range(len(labc)):
        ok = tag_on_lab[kl] == labc[kl]
        q = inv.get(kl)
        if ok:
            c["n_ok"] += 1
            if kl in diff_lab:
                c["n_ok_diff"] += 1
                c["ff_diff"] += q is not None and shown[q]
        else:
            c["n_err"] += 1
            c["hit"] += q is not None and shown[q]
    c["n_extra"] += sum(1 for kl in m if kl is None)
    c["utt"] += 1
    c["utt_ff"] += (c["ff"] + c["ff_extra"]) > 0
    return c


def rates(c):
    f = lambda x, y: (c[x] / c[y]) if c[y] else float("nan")
    return {"utts": c["utt"],
            "자리_헛지적률": (c["ff"] + c["ff_extra"]) / max(1, c["n_ok"] + c["n_extra"]),
            "발화_헛지적률": f("utt_ff", "utt"),
            "재현율": f("hit", "n_err"),
            "정밀도": c["tp"] / max(1, c["tp"] + c["ff"] + c["ff_extra"]),
            "diff자리_헛지적률": f("ff_diff", "n_ok_diff"),
            "헛지적_수": c["ff"] + c["ff_extra"], "그중_표준발음에만_있는_자리": c["ff_extra"], "맞은_지적": c["tp"],
            "오류_자리": c["n_err"], "맞게_읽은_자리": c["n_ok"]}


def bootstrap(per_item, a, b, key, n=1000, seed=0):
    """문항 단위 재표집 — rates(b)[key] - rates(a)[key] 의 95% 구간."""
    rng = random.Random(seed)
    items = sorted(per_item)
    out = []
    for _ in range(n):
        ca, cb = collections.Counter(), collections.Counter()
        for _k in range(len(items)):
            it = items[rng.randrange(len(items))]
            ca.update(per_item[it].get(a, {})); cb.update(per_item[it].get(b, {}))
        va, vb = rates(ca)[key], rates(cb)[key]
        if va == va and vb == vb:
            out.append(vb - va)
    out.sort()
    if not out:
        return None
    return [out[int(0.025 * len(out))], out[int(0.975 * len(out)) - 1]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=os.path.join(str(MDD_OUTPUTS), "manifest_full_tag_replay10"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--sentences", default=os.path.join(str(MDD_OUTPUTS), "analysis", "mdd_error_2026-09-27",
                                                        "w2vbert_step38000_seen_sentence", "sentence_analysis.csv"))
    ap.add_argument("--prompts", default=os.path.join(str(MDD_OUTPUTS), "analysis", "npu_bench_pack_2026-10-03", "train_prompts.json"))
    ap.add_argument("--ckpt", default=os.path.join(str(DATA_ROOT), "checkpoints", "w2vbert_stage2", "step_38000"))
    ap.add_argument("--scorer", default=os.path.join(str(MDD_OUTPUTS), "score_v2_w2vbert", "score_models_v2_w2vbert.joblib"))
    ap.add_argument("--methods", default="api,lab,fixed,choose")
    ap.add_argument("--ablate", action="store_true", help="변이 종류마다 그 종류만 켠 choose(only:<종류>)도 잰다")
    ap.add_argument("--margin", type=float, default=0.0, help="choose: 변이를 고를 최소 로그우도 차")
    ap.add_argument("--per-item", dest="per_item", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(str(MDD_OUTPUTS), "analysis", "canonical_source_2026-10-11"))
    ap.add_argument("--tag", default="methods", help="출력 파일 이름 꼬리(report_<tag>.json)")
    ap.add_argument("--dump-utts", dest="dump_utts", action="store_true",
                    help="발화마다 말소리 길이·비율·방법별 카운터를 utts_<tag>.jsonl 로 남긴다(말소리 검사 기준을 바꿔 다시 셀 때)")
    a = ap.parse_args()
    import edlib
    import numpy as np
    from mdd.api_offline import ApiEmulator
    from mdd import canonical_fix as cf

    emu = ApiEmulator(a.ckpt, scorer=a.scorer)
    B = cf.CanonicalBuilder(emu.canonical)
    to_id = emu.dec.d.index
    methods = a.methods.split(",")
    if a.ablate:
        methods += ["only:" + k for k in KINDS]
    sentences = [r["sentence"] for r in csv.DictReader(open(a.sentences, encoding="utf-8"))]
    prompts = json.load(open(a.prompts, encoding="utf-8"))["all"] if os.path.exists(a.prompts) else []
    with open(os.path.join(a.manifest, a.split + ".tsv"), encoding="utf-8") as f:
        root = f.readline().rstrip("\n")
        rows = [l.rstrip("\n").split("\t")[0] for l in f]
    rd = lambda ext: open(os.path.join(a.manifest, "%s.%s" % (a.split, ext)), encoding="utf-8").read().splitlines()
    ltr, can = [l.split() for l in rd("ltr")], [l.split() for l in rd("canon")]
    by_item = collections.defaultdict(list)
    for i, p in enumerate(rows):
        m = ITEM_RE.search(p)
        if m and can[i] and not any(len(x) != 1 for x in can[i] + ltr[i]):
            by_item[m.group(1)].append(i)
    codes = sorted(by_item)
    lab = dict((k, collections.Counter(" ".join(can[i]) for i in by_item[k]).most_common(1)[0][0].replace(" ", "")) for k in codes)
    item_text = item_texts(emu, sentences, prompts, codes, lab)
    diff_items = set(k for k, t in item_text.items() if "".join(emu.canonical(t)) != lab[k])
    rng = random.Random(a.seed)
    pick = []
    for k in sorted(item_text):
        pick += [(k, i) for i in rng.sample(by_item[k], min(a.per_item, len(by_item[k])))]
    print("[문항] %d (KoG2P≠라벨 %d) · 발화 %d · 방법 %s" % (len(item_text), len(diff_items), len(pick), ",".join(methods)), flush=True)

    # 문항별 표준발음 후보(오디오와 무관한 것)는 한 번만
    cache = {}
    for k, t in item_text.items():
        base, alts = B.variants(t, exclude=())          # 모든 종류 — 고르는 쪽(choose/only/without)에서 거른다
        cache[k] = {"api": emu.canonical(t), "fixed": B.fixed(t), "base": base, "alts": alts}
    per_item = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))   # 문항 → 방법 → 카운터
    per_item_acc = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))  # API 통과 발화만
    picked_kind = collections.Counter(); picked_toward_lab = collections.Counter()
    mismatch = collections.Counter()
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    fu = open(os.path.join(a.out, "utts_%s.jsonl" % a.tag), "w", encoding="utf-8") if a.dump_utts else None
    for n, (k, i) in enumerate(pick):
        with wave.open(os.path.join(root, rows[i]), "rb") as w:
            x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0
        rejected, ainfo = emu.check_audio(x[: 16000 * 20])
        lp = emu.lprobs(x)
        text, labc, tag = item_text[k], can[i], ltr[i]
        tag_on_lab = proj(labc, tag)
        apic = cache[k]["api"]
        m_api = a2b(apic, labc)
        diff_lab = set(range(len(labc))) - set(kl for ia, kl in enumerate(m_api) if kl is not None and apic[ia] == labc[kl])
        canons = {}
        for meth in methods:
            if meth == "api":
                canons[meth] = apic
            elif meth == "lab":
                canons[meth] = labc
            elif meth == "fixed":
                canons[meth] = cache[k]["fixed"]
            elif meth in ("choose", "choose_all") or meth.startswith("only:") or meth.startswith("without:"):
                if meth == "choose_all":
                    alts = cache[k]["alts"]
                elif meth == "choose":
                    alts = [x_ for x_ in cache[k]["alts"] if x_[3] not in cf.EXCLUDE_KINDS]
                elif meth.startswith("only:"):
                    alts = [x_ for x_ in cache[k]["alts"] if x_[3] == meth[5:]]
                else:
                    drop_k = set(meth[8:].split("+"))
                    alts = [x_ for x_ in cache[k]["alts"] if x_[3] not in drop_k]
                cc, used = cf.choose(lp[0], cache[k]["base"], alts, to_id, blank=emu.dec.blank, margin=a.margin)
                canons[meth] = cc
                if meth == "choose":
                    for s, e, r, kd, g in used:
                        picked_kind[kd] += 1
                        one = list(cache[k]["base"][:s]) + list(r) + list(cache[k]["base"][e:])
                        d0 = edlib.align("".join(cache[k]["base"]), "".join(labc))["editDistance"]
                        d1 = edlib.align("".join(one), "".join(labc))["editDistance"]
                        picked_toward_lab[kd + (" →라벨" if d1 < d0 else " ↛라벨" if d1 > d0 else " =")] += 1
        done, urec = {}, {}
        for meth, mc in canons.items():
            key = "".join(mc)
            if key not in done:
                b, _ = emu.assess(x, text, canon=mc, lp=lp, gate=False)
                sh = ["errorType" in p for p in emu.phonemes(b)]
                done[key] = sh if len(sh) == len(mc) else None
            sh = done[key]
            if sh is None:
                mismatch[meth] += 1; continue
            c = score(sh, mc, labc, tag_on_lab, diff_lab)
            urec[meth] = dict(c)
            per_item[k][meth].update(c)
            if not rejected:
                per_item_acc[k][meth].update(c)
            per_item[k][meth]["rejected"] += rejected
        if a.dump_utts:
            fu.write(json.dumps({"item": k, "stem": os.path.basename(rows[i])[:-4], "text": text, "rejected": bool(rejected),
                                 **ainfo, "sec": round(len(x) / 16000, 2), "methods": urec}, ensure_ascii=False) + "\n")
        if n % 500 == 0:
            print("  %d/%d · %.0fs" % (n + 1, len(pick), time.time() - t0), flush=True)

    if fu is not None:
        fu.close()

    def agg(src, items):
        out = {}
        for meth in methods:
            c = collections.Counter()
            for it in items:
                c.update(src[it].get(meth, {}))
            out[meth] = rates(c)
        return out
    touched = set(k for k in item_text if cache[k]["fixed"] != cache[k]["api"])      # 결정적 보정이 표준발음을 바꾸는 문항
    groups = {"전체": sorted(item_text), "KoG2P≠라벨 문항": sorted(diff_items), "KoG2P=라벨 문항": sorted(set(item_text) - diff_items),
              "보정이 바꾸는 문항": sorted(touched), "보정이 안 바꾸는 문항": sorted(set(item_text) - touched)}
    rep = {"ckpt": a.ckpt, "n_utts": len(pick), "n_items": len(item_text), "n_diff_items": len(diff_items), "margin": a.margin,
           "rejected_by_api_gate": sum(per_item[k]["api"]["rejected"] for k in per_item) / max(1, len(pick)),
           "all": dict((g, agg(per_item, its)) for g, its in groups.items()),
           "accepted_only": dict((g, agg(per_item_acc, its)) for g, its in groups.items()),
           "choose_picked": dict(picked_kind), "choose_picked_vs_lab": dict(picked_toward_lab),
           "len_mismatch": dict(mismatch)}
    ci, ci_g = {}, {}
    for meth in methods:
        if meth == "api":
            continue
        ci[meth] = dict((key, bootstrap(per_item_acc, "api", meth, key)) for key in ("자리_헛지적률", "발화_헛지적률", "재현율"))
        for g in ("보정이 바꾸는 문항", "보정이 안 바꾸는 문항"):
            sub = dict((it, per_item_acc[it]) for it in groups[g] if it in per_item_acc)
            ci_g.setdefault(g, {})[meth] = dict((key, bootstrap(sub, "api", meth, key)) for key in ("발화_헛지적률", "재현율"))
    rep["ci95_vs_api_accepted"] = ci
    rep["ci95_vs_api_accepted_by_group"] = ci_g
    rep["touched_items"] = len(touched)
    os.makedirs(a.out, exist_ok=True)
    json.dump({"report": rep, "item_text": item_text, "diff_items": sorted(diff_items), "touched_items": sorted(touched),
               "per_item": dict((k, dict((m, dict(v)) for m, v in d.items())) for k, d in per_item.items()),
               "per_item_accepted": dict((k, dict((m, dict(v)) for m, v in d.items())) for k, d in per_item_acc.items())},
              open(os.path.join(a.out, "report_%s.json" % a.tag), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    # 요약표
    L = ["# 표준발음 출처별 화면 지적 — %s" % os.path.basename(a.ckpt.rstrip("/")), "",
         "발화 %d · 문항 %d (KoG2P≠라벨 %d) · API 가 거절할 발화 %.1f%%" % (len(pick), len(item_text), len(diff_items), 100 * rep["rejected_by_api_gate"]), ""]
    for scope, title in (("accepted_only", "API 통과 발화만"), ("all", "전체 발화(거절 무시)")):
        for g in groups:
            L += ["## %s · %s" % (title, g), "", "| 방법 | 발화 헛지적률 | 자리 헛지적률 | diff 자리 헛지적률 | 재현율 | 정밀도 |", "|---|---|---|---|---|---|"]
            for meth in methods:
                r = rep[scope][g][meth]
                L.append("| %s | %.1f%% | %.2f%% | %.1f%% | %.1f%% | %.1f%% |" % (meth, 100 * r["발화_헛지적률"], 100 * r["자리_헛지적률"],
                         100 * r["diff자리_헛지적률"], 100 * r["재현율"], 100 * r["정밀도"]))
            L.append("")
    L += ["## api 대비 차이 95% 구간(통과 발화, 문항 재표집)", ""]
    for meth, d in ci.items():
        L.append("- %s: " % meth + " · ".join("%s [%+.2f, %+.2f]%%p" % (k, 100 * v[0], 100 * v[1]) for k, v in d.items() if v))
    for g, dd in ci_g.items():
        L += ["", "### %s (%d문항)" % (g, len(groups[g]))]
        for meth, d in dd.items():
            L.append("- %s: " % meth + " · ".join("%s [%+.2f, %+.2f]%%p" % (k, 100 * v[0], 100 * v[1]) for k, v in d.items() if v))
    L += ["", "choose 가 고른 변이: %s" % json.dumps(rep["choose_picked"], ensure_ascii=False),
          "", "고른 변이가 라벨 쪽으로 갔나: %s" % json.dumps(rep["choose_picked_vs_lab"], ensure_ascii=False)]
    open(os.path.join(a.out, "summary_%s.md" % a.tag), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    main()
