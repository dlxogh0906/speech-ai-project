# -*- coding: utf-8 -*-
"""자음 계열별(ㄱ·ㄷ·ㅂ·ㅅ·ㅈ) 예사·된·거센소리 오류를 화면에서 얼마나 잡나 — 모델이 못 듣나, 들었는데 화면이 숨기나.

같은 녹음을 API 채점 경로(`mdd.api_offline` — app.py 그대로)로 두 구성에 넣는다.
  old  10-17 전 서비스: step_38000 + KoG2P
  new  지금 서비스: 섞은 모델(interp_a0.5_step3000) + canonical_fix(choose, margin 1.0)
첫소리 자리(라벨 표준발음 기준, 뒤가 모음)마다 라벨 자모 L · 사람 전사(TAG) A · API 표준발음 C ·
모델이 들은 자모 H(actual_jamo) · 화면 지적(errorType) · 자리 점수를 남긴다.
말소리 검사는 끈다(gate=False) — 짧은 녹음 기준을 바꾸는 중이라 모델과 화면만 본다.

지표(오류 자리 = A ≠ L 이고 A 가 L 과 같은 계열)
  들은 그대로   H == A        모델이 학습자가 낸 소리를 받아 적었다
  다르게 들음   H ≠ C         표준발음과 다르게 들었다(지적 후보)
  화면 지적     errorType      화면에 틀렸다고 띄웠다
  숨김          H ≠ C 인데 지적 없음(점수 60 이상이라 '애매' 처리)
맞게 읽은 자리(A == L)는 계열 자모별 헛지적률.

    for i in 0 1 2 3 4 5; do $PY .../eval_series_detect.py extract --per-item 120 --shard $i/6 & done; wait
    $PY .../eval_series_detect.py report
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import DATA_ROOT, MDD_OUTPUTS

import argparse, collections, csv, glob, importlib.util, json, os, random, time, wave

SERIES = {"ㄱ": "ㄱ", "ㄲ": "ㄱ", "ㅋ": "ㄱ", "ㄷ": "ㄷ", "ㄸ": "ㄷ", "ㅌ": "ㄷ", "ㅂ": "ㅂ", "ㅃ": "ㅂ", "ㅍ": "ㅂ",
          "ㅅ": "ㅅ", "ㅆ": "ㅅ", "ㅈ": "ㅈ", "ㅉ": "ㅈ", "ㅊ": "ㅈ"}
KIND = {"ㄱ": "예사", "ㄷ": "예사", "ㅂ": "예사", "ㅅ": "예사", "ㅈ": "예사", "ㄲ": "된", "ㄸ": "된", "ㅃ": "된", "ㅆ": "된", "ㅉ": "된",
        "ㅋ": "거센", "ㅌ": "거센", "ㅍ": "거센", "ㅊ": "거센"}
VOWELS = set("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
CONFIGS = ("old", "new")
NAMES = {"old": "10-17 전 서비스", "new": "지금 서비스"}
OUT_DIR = os.path.join(str(MDD_OUTPUTS), "analysis", "series_detect_2026-10-18")


def _sibling(name):
    spec = importlib.util.spec_from_file_location(name, str(_pl.Path(__file__).with_name(name + ".py")))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def heard_probs(l, dec, sites):
    """자리(표준발음 순서 i) → 모델이 그 자리에서 받아 적은 자모의 확률(그 자모가 1위인 프레임들 중 최대). 받아 적은 게 없으면 없음."""
    import numpy as np
    g = l.argmax(-1)
    toks, prev = [], -1
    for t, v in enumerate(g):
        v = int(v)
        if v != dec.blank and v != prev:
            toks.append([v, float(np.exp(l[t, v]))])
        elif v != dec.blank and v == prev:
            toks[-1][1] = max(toks[-1][1], float(np.exp(l[t, v])))
        prev = v
    jam = [p for v, p in toks if 0x3131 <= ord(dec.d[v][0]) <= 0x3163 and len(dec.d[v]) == 1]
    return {s["i"]: round(jam[s["j"]], 3) for s in sites if s.get("j") is not None and s["j"] < len(jam)}


def cmd_extract(a):
    import numpy as np
    import torch
    torch.set_num_threads(a.threads)
    from mdd.api_offline import ApiEmulator
    from mdd import canonical_fix as cf
    ecs = _sibling("eval_canonical_source")
    shard, n_shard = (int(x) for x in a.shard.split("/"))

    names = a.configs.split(",")
    emus = {n: ApiEmulator(ck, scorer=a.scorer) for n, ck in (("old", a.ckpt), ("new", a.mix_ckpt)) if n in names}
    e0 = emus.get("old") or emus["new"]            # 모델과 상관없는 일(KoG2P 표준발음 · 말소리 검사)에 쓴다
    cap = {}
    for _e in emus.values():                       # 화면 응답 만들 때 쓴 자리별 margin 을 옆으로 빼 둔다(빠진 소리의 점수 계산용)
        def _wrap(orig):
            def build(text, canon, sites, margins, **kw):
                cap["margins"] = margins
                return orig(text, canon, sites, margins, **kw)
            return build
        _e.ss.build = _wrap(_e.ss.build)
    B = cf.CanonicalBuilder(e0.canonical)
    with open(os.path.join(a.manifest, "test.tsv"), encoding="utf-8") as f:
        root = f.readline().rstrip("\n")
        rows = [l.rstrip("\n").split("\t")[0] for l in f]
    rd = lambda ext: open(os.path.join(a.manifest, "test.%s" % ext), encoding="utf-8").read().splitlines()
    ltr, can = [l.split() for l in rd("ltr")], [l.split() for l in rd("canon")]
    seg = None
    if a.chunks:                                   # 문단 조각(build_rp_eval_chunks.py) — 조각마다 기준 문장이 다르고 녹음은 원본에서 잘라 쓴다
        texts, seg = rd("text"), [l.split() for l in rd("seg")]
        item_text = {"c%05d" % i: texts[i] for i in range(len(rows)) if can[i] and not any(len(x) != 1 for x in can[i] + ltr[i])}
        pick = [(k, int(k[1:])) for k in sorted(item_text)]
    else:
        sentences = [r["sentence"] for r in csv.DictReader(open(a.sentences, encoding="utf-8"))]
        prompts = json.load(open(a.prompts, encoding="utf-8"))["all"] if os.path.exists(a.prompts) else []
        by_item = collections.defaultdict(list)
        for i, p in enumerate(rows):
            m = ecs.ITEM_RE.search(p)
            if m and can[i] and not any(len(x) != 1 for x in can[i] + ltr[i]):
                by_item[m.group(1)].append(i)
        codes = sorted(by_item)
        lab = dict((k, collections.Counter(" ".join(can[i]) for i in by_item[k]).most_common(1)[0][0].replace(" ", ""))
                   for k in codes)
        item_text = ecs.item_texts(e0, sentences, prompts, codes, lab)
        rng = random.Random(a.seed)
        pick = []
        for k in sorted(item_text):
            pick += [(k, i) for i in rng.sample(by_item[k], min(a.per_item, len(by_item[k])))]
    print("[표본] 문항 %d · 발화 %d" % (len(item_text), len(pick)), flush=True)
    if a.stems:
        want = set(open(a.stems, encoding="utf-8").read().split())
        pick = [(k, i) for k, i in pick if os.path.basename(rows[i])[:-4] in want]
    pick = pick[shard::n_shard]
    cache = {}
    for k in set(k for k, _ in pick):
        base, alts = B.variants(item_text[k])
        cache[k] = {"api": e0.canonical(item_text[k]), "base": base, "alts": alts}

    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, "sites.%d-of-%d.jsonl" % (shard, n_shard))
    done = set()
    if os.path.exists(out):
        done = set(json.loads(l)["stem"] for l in open(out, encoding="utf-8"))
    t0, n_new = time.time(), 0
    with open(out, "a", encoding="utf-8") as fo:
        for n, (k, i) in enumerate(pick, 1):
            stem = os.path.basename(rows[i])[:-4] + ("_%02d" % int(seg[i][2]) if seg else "")
            if stem in done:
                continue
            with wave.open(os.path.join(root, rows[i]), "rb") as w:
                if seg:
                    s0, s1 = int(float(seg[i][0]) * 16000), int(float(seg[i][1]) * 16000)
                    w.setpos(min(s0, w.getnframes()))
                    x = np.frombuffer(w.readframes(max(0, s1 - s0)), np.int16).astype(np.float32) / 32768.0
                else:
                    x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0
            labc, tag = can[i], ltr[i]
            tag_on_lab = ecs.proj(labc, tag)
            _, ainfo = e0.check_audio(x[: 16000 * 20])
            rec = {"stem": stem, "item": ecs.ITEM_RE.search(rows[i]).group(1) if seg else k, **ainfo, "sites": {}}
            if seg:
                rec["text"] = item_text[k]
            for name in a.configs.split(","):
                e = emus[name]
                lp = e.lprobs(x)
                c = cache[k]
                if name == "old" or not c["base"]:
                    canon = c["api"]
                else:
                    canon, _ = cf.choose(lp[0], c["base"], c["alts"], e.dec.d.index, blank=e.dec.blank, margin=a.margin)
                body, raw = e.assess(x, item_text[k], canon=canon, lp=lp, gate=False)
                ph = ApiEmulator.phonemes(body)
                prob = heard_probs(lp[0], e.dec, raw["sites"])
                if len(ph) != len(canon):
                    rec["sites"][name] = None
                    continue
                m = ecs.a2b(canon, labc)
                sites = []
                for q, kl in enumerate(m):
                    if kl is None:
                        continue
                    onset = labc[kl] not in VOWELS and kl + 1 < len(labc) and labc[kl + 1] in VOWELS
                    if not a.all_sites and (labc[kl] not in SERIES or not onset):
                        continue
                    p = ph[q]
                    row = [labc[kl], tag_on_lab[kl], canon[q], p.get("actual_jamo"), int("errorType" in p),
                           p.get("pronunciation"), prob.get(q)]
                    if a.all_sites:
                        row.append("중성" if labc[kl] in VOWELS else ("초성" if onset else "종성"))
                        mg = cap.get("margins") or []
                        key = e.ss.ctx(canon, q, e.vowels) if e.ss.ctx else None
                        row.append(e.ss._score(key, mg[q]) if q < len(mg) else None)   # 소리가 빠졌어도 강제정렬로 잰 점수
                    sites.append(row)
                rec["sites"][name] = sites
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_new += 1
            if n_new % 500 == 0:
                fo.flush()
                el = time.time() - t0
                print("  %d/%d  %.1f발화/s" % (n, len(pick), n_new / el), flush=True)
    print("[끝] %s · %.0f초" % (out, time.time() - t0), flush=True)


def cmd_report(a):
    recs = [json.loads(l) for p in sorted(glob.glob(os.path.join(a.out, "sites.*.jsonl"))) for l in open(p, encoding="utf-8")]
    err = {c: collections.defaultdict(collections.Counter) for c in CONFIGS}
    ok = {c: collections.defaultdict(collections.Counter) for c in CONFIGS}
    for r in recs:
        for c in CONFIGS:
            for L, A, C, H, shown, sc, *_ in r["sites"].get(c) or []:
                if A == L:
                    d = ok[c][L]
                    d["n"] += 1; d["shown"] += shown; d["heard_diff"] += H != C
                elif A is not None and SERIES.get(A) == SERIES[L]:
                    d = err[c][L + A]
                    d["n"] += 1; d["heard_actual"] += H == A; d["heard_diff"] += H != C; d["shown"] += shown
                    d["hidden"] += (H != C) and not shown; d["c_is_l"] += C == L
    pct = lambda d, k: 100.0 * d[k] / d["n"] if d["n"] else float("nan")
    rep = {"n_utt": len(recs), "err": {c: {k: dict(v) for k, v in err[c].items()} for c in CONFIGS},
           "ok": {c: {k: dict(v) for k, v in ok[c].items()} for c in CONFIGS}}
    json.dump(rep, open(os.path.join(a.out, "report.json"), "w"), ensure_ascii=False, indent=1)
    order = lambda k: ("ㄱㄷㅂㅅㅈ".index(SERIES[k[0]]), k)
    L = ["# 자음 계열별 예사·된·거센소리 오류 — 받아쓰기 vs 화면 (2026-10-18)", "",
         "발화 %d · 본 문항(224) · 테스트 화자 · 말소리 검사 끔 · 첫소리 자리만. `eval_series_detect.py`" % len(recs), "",
         "## 1. 오류 자리 — 학습자가 다른 소리를 낸 곳 (지금 서비스 / 10-17 전)", "",
         "| 정답 → 실제 | 자리 수 | 들은 그대로 | 다르게 들음 | **화면 지적** | 들었는데 숨김 | 10-17 전 화면 지적 |",
         "|---|--:|--:|--:|--:|--:|--:|"]
    for k in sorted(set(err["new"]) | set(err["old"]), key=order):
        d, o = err["new"].get(k, collections.Counter()), err["old"].get(k, collections.Counter())
        if d["n"] < a.min_n:
            continue
        L.append("| %s→%s (%s→%s) | %d | %.0f%% | %.0f%% | **%.0f%%** | %.0f%% | %.0f%% |" % (
            k[0], k[1], KIND[k[0]], KIND[k[1]], d["n"], pct(d, "heard_actual"), pct(d, "heard_diff"), pct(d, "shown"),
            pct(d, "hidden"), pct(o, "shown")))
    L += ["", "## 2. 계열별 합계", "", "| 계열 | 오류 자리 | 들은 그대로 | 화면 지적 | 들었는데 숨김 | 맞게 읽은 자리 헛지적 |", "|---|--:|--:|--:|--:|--:|"]
    for s in "ㄱㄷㅂㅅㅈ":
        e_, o_ = collections.Counter(), collections.Counter()
        for k, v in err["new"].items():
            if SERIES[k[0]] == s:
                e_.update(v)
        for k, v in ok["new"].items():
            if SERIES[k] == s:
                o_.update(v)
        L.append("| %s %s | %d | %.0f%% | %.0f%% | %.0f%% | %.2f%% |" % (
            s, "".join(x for x in SERIES if SERIES[x] == s), e_["n"], pct(e_, "heard_actual"), pct(e_, "shown"), pct(e_, "hidden"),
            pct(o_, "shown")))
    L += ["", "## 3. 맞게 읽은 자리의 헛지적 (지금 서비스)", "", "| 자모 | 자리 수 | 헛지적 | 다르게 들음 |", "|---|--:|--:|--:|"]
    for k in sorted(ok["new"], key=lambda k: ("ㄱㄷㅂㅅㅈ".index(SERIES[k]), k)):
        d = ok["new"][k]
        L.append("| %s | %d | %.2f%% | %.2f%% |" % (k, d["n"], pct(d, "shown"), pct(d, "heard_diff")))
    open(os.path.join(a.out, "summary.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--manifest", default=os.path.join(str(MDD_OUTPUTS), "manifest_full_tag_replay10"))
    e.add_argument("--sentences", default=os.path.join(str(MDD_OUTPUTS), "analysis", "mdd_error_2026-09-27",
                                                       "w2vbert_step38000_seen_sentence", "sentence_analysis.csv"))
    e.add_argument("--prompts", default=os.path.join(str(MDD_OUTPUTS), "analysis", "npu_bench_pack_2026-10-03", "train_prompts.json"))
    e.add_argument("--ckpt", default=os.path.join(str(DATA_ROOT), "checkpoints", "w2vbert_stage2", "step_38000"))
    e.add_argument("--mix-ckpt", dest="mix_ckpt",
                   default=os.path.join(str(DATA_ROOT), "checkpoints", "w2vbert_synth_tense_v2", "interp_a0.5_step3000"))
    e.add_argument("--scorer", default=os.path.join(str(MDD_OUTPUTS), "score_v2_w2vbert", "score_models_v2_w2vbert.joblib"))
    e.add_argument("--margin", type=float, default=1.0)
    e.add_argument("--per-item", dest="per_item", type=int, default=120)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--shard", default="0/1")
    e.add_argument("--threads", type=int, default=2)
    e.add_argument("--stems", default=None, help="이 파일의 stem 만(빈칸·줄바꿈 구분)")
    e.add_argument("--configs", default="old,new")
    e.add_argument("--all-sites", dest="all_sites", action="store_true", help="계열 첫소리만이 아니라 모든 자리(초성·중성·종성)를 남긴다")
    e.add_argument("--chunks", action="store_true",
                   help="매니페스트가 문단 조각(build_rp_eval_chunks.py)일 때 — test.text(조각 기준 문장) · test.seg(원본에서 자를 시각)를 쓴다")
    e.add_argument("--out", default=OUT_DIR)
    r = sub.add_parser("report")
    r.add_argument("--out", default=OUT_DIR)
    r.add_argument("--min-n", dest="min_n", type=int, default=20)
    a = ap.parse_args()
    {"extract": cmd_extract, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
