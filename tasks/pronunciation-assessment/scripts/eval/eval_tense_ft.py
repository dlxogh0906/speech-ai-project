# -*- coding: utf-8 -*-
"""세기 쌍 합성 오류 추가 학습(synth_tense) 모델 평가 — 기존 모델과 **같은 발화·같은 서비스 경로**로 잰다.

묶음
  test     manifest_full_tag_replay10 test 에서 고정 무작위 --n-test 발화(학습 문항 · 새 화자):
           JER · MDD 정밀도/재현율/F1 · 맞게 읽은 자리 헛지적 · 들린 그대로/표준으로 덮음 · 세기 쌍별 들린 그대로
  deliberate  일부러 틀린 TTS 녹음(--deliberate, 쉼표로 여럿 — 학습에 없는 목소리. 첫째 = Chatterbox 판
           pair_rescue_2026-10-10/tts_deliberate, 둘째부터 deliberate_extra[폴더 이름]. `make_tts_deliberate.py` 로 만든다):
           틀린 자리를 그 소리로 적었나 / 표준으로 고쳤나 · 맞게 읽은 녹음에서 그 자리 헛지적
  unseen   학습에 없는 문항(manifest_unseen_prompt_rp — 문단 읽기 RP002~004, 학습 화자 제외) 전부: test 와 같은 지표.
           문단이 30~50초라 20초 넘는 녹음은 쉼에서 잘라 받아쓴다(split_at_pauses — 통째로 넣으면 글자를 빠뜨린다).
           test 는 학습 문항이라 '외운 문장' 효과가 섞인다(2026-10-11 리뷰) — 문장 일반화는 이쪽으로 본다
  native   원어민 대화(KsponSpeech) valid 고정 무작위 --n-native 발화: JER (원어민 인식이 망가지지 않았나)
  synthdev 합성 오류 중 학습에서 뺀 문장(manifest_synth_tense_v1/synth_dev): 쌍별 들린 그대로
경로: 무음 다듬기(mdd.vad.trim) → 한 발화씩 CTC → 그리디 → 표준 축 투영(발음 API 와 같은 계산).
--model 이 .pt 면 fairseq 체크포인트(옛 서비스 XLS-R 300m `mdd_tag_all_native.pt`)로 같은 경로를 돈다.
화자 정보(meta)는 읽지 않는다.

    $PY .../eval_tense_ft.py --model $SPEECH_AI_DATA/checkpoints/w2vbert_stage2/step_38000 --name base
    $PY .../eval_tense_ft.py --compare base ft_1000 ft_2000
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import G2P_TOOLS, MDD_OUTPUTS, ensure_kog2p

import argparse, collections, json, os, random, re, time, wave

SR = 16000


def proj(canon, seq):
    import edlib
    c, s = "".join(canon), "".join(seq)
    if not s:
        return [None] * len(c)
    a = edlib.getNiceAlignment(edlib.align(c, s, task="path"), c, s)
    return [None if t == "-" else t for q, t in zip(a["query_aligned"], a["target_aligned"]) if q != "-"]


def edits(ref, hyp):
    import edlib
    if not ref:
        return len(hyp)
    return edlib.align("".join(hyp), "".join(ref))["editDistance"] if hyp else len(ref)


def read_split(mdir, split):
    with open(os.path.join(mdir, split + ".tsv"), encoding="utf-8") as f:
        root = f.readline().rstrip("\n")
        rows = [l.rstrip("\n").split("\t") for l in f]
    rd = lambda ext: [l.split() for l in open(os.path.join(mdir, "%s.%s" % (split, ext)), encoding="utf-8").read().splitlines()] \
        if os.path.exists(os.path.join(mdir, "%s.%s" % (split, ext))) else None
    return [os.path.join(root, r[0]) for r in rows], rd("ltr"), rd("canon")


def split_at_pauses(x, max_sec=15.0, min_pause=0.3):
    """긴 녹음(문단 읽기)을 쉼에서 자른다 — 모델은 20초 이하로 학습해서, 30~50초 문단을 통째로 넣으면 글자를 절반 가까이
    빠뜨린다(2026-10-12 확인: 기준 모델 JER 38.7% → 쉼에서 자르면 24.1%, 30발화). 0.3초 이상 쉼의 가운데를 자를 후보로,
    조각이 max_sec 의 60%를 넘으면 그 쉼에서 자른다. 쉼이 없으면 max_sec 마다 강제로 자른다."""
    from mdd.vad import voiced_mask
    m, _ = voiced_mask(x, SR)
    fr = 0.03
    cand, run = [], 0
    for i, v in enumerate(list(m) + [True]):
        if not v:
            run += 1
        else:
            if run * fr >= min_pause:
                cand.append(i - run // 2 - 1)
            run = 0
    cuts, last = [0], 0
    for c in cand:
        if (c - last) * fr >= max_sec * 0.6:
            cuts.append(c); last = c
    cuts.append(len(m))
    out = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        s0, e0 = int(a * fr * SR), min(len(x), int(b * fr * SR))
        while e0 - s0 > int(max_sec * 1.4 * SR):
            out.append(x[s0:s0 + int(max_sec * SR)]); s0 += int(max_sec * SR)
        if e0 - s0 > SR // 10:
            out.append(x[s0:e0])
    return out


class Runner:
    def __init__(self, model, chunk_over=20.0):
        if model.endswith(".pt"):                     # fairseq 체크포인트(옛 서비스 XLS-R 300m tag_all_native 등) — 같은 경로로 비교
            from shared.fairseq_guard import ensure_fairseq
            ensure_fairseq()
            from mdd import mdd_eval
            mdd_eval.CKPT = model
            os.environ.setdefault("MDD_DATA_DIR", os.path.dirname(model))     # dict.ltr.txt 가 같은 폴더에 있다
            self.dec = mdd_eval.Decoder()
        else:
            from mdd.gop_feats_v2 import HFDecoder
            self.dec = HFDecoder(model)
        self.chunk_over = chunk_over                   # 이보다 긴 녹음은 쉼에서 잘라 조각마다 받아쓰고 잇는다

    def _greedy(self, x):
        from mdd.gop_feats_v2 import forward_lprobs
        lp, Ls = forward_lprobs(self.dec, [x])
        out, prev = [], None
        for v in lp[0, :Ls[0]].argmax(-1).tolist():
            if v != prev and v != self.dec.blank:
                t = self.dec.d[v]
                if len(t) == 1 and 0x3131 <= ord(t) <= 0x3163:
                    out.append(t)
            prev = v
        return out

    def hyp(self, path):
        import numpy as np
        from mdd.vad import trim
        with wave.open(path, "rb") as w:
            x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0
        x, _ = trim(x, SR)
        if self.chunk_over and len(x) > self.chunk_over * SR:
            out = []
            for c in split_at_pauses(x):
                out += self._greedy(c)
            return out
        return self._greedy(x)


def site_stats(canon, tag, pred, acc, fam_of):
    tm, pm = proj(canon, tag), proj(canon, pred)
    for j, c in enumerate(canon):
        t, p = tm[j], pm[j]
        if t == c:
            acc["ok"] += 1; acc["ok_flag"] += p != c
        else:
            acc["err"] += 1; acc["err_det"] += p != c; acc["err_pres"] += p == t; acc["err_rev"] += p == c
            if t is not None and fam_of.get(c) and fam_of.get(c) == fam_of.get(t):
                k = c + t
                acc["pair_n_" + k] += 1; acc["pair_pres_" + k] += p == t; acc["pair_rev_" + k] += p == c


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model")
    ap.add_argument("--name")
    ap.add_argument("--compare", nargs="*")
    ap.add_argument("--suites", default="test,deliberate,native,synthdev")
    ap.add_argument("--n-test", dest="n_test", type=int, default=15000)
    ap.add_argument("--n-native", dest="n_native", type=int, default=2000)
    ap.add_argument("--unseen-manifest", dest="unseen_manifest", default=os.path.join(str(MDD_OUTPUTS), "manifest_unseen_prompt_rp"))
    ap.add_argument("--n-unseen", dest="n_unseen", type=int, default=0, help="0 = 전부")
    ap.add_argument("--merge", action="store_true", help="<name>.json 이 있으면 이번에 돈 묶음만 덮어쓴다(다른 묶음은 둔다)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test-manifest", dest="test_manifest", default=os.path.join(str(MDD_OUTPUTS), "manifest_full_tag_replay10"))
    ap.add_argument("--test-split", dest="test_split", default="test",
                    help="test 묶음의 split 이름(예: manifest_foreign_clean_v2 의 test_unseen — 옛 서비스 모델과 비교한 23,562발화). --n-test 0 = 전부")
    ap.add_argument("--native-manifest", dest="native_manifest", default=os.path.join(str(MDD_OUTPUTS), "manifest_kspon_stage1"))
    ap.add_argument("--synth-manifest", dest="synth_manifest", default=os.path.join(str(MDD_OUTPUTS), "manifest_synth_tense_v1"))
    ap.add_argument("--deliberate", default=os.path.join(str(MDD_OUTPUTS), "analysis", "pair_rescue_2026-10-10", "tts_deliberate"))
    ap.add_argument("--out-dir", dest="out_dir", default=os.path.join(str(MDD_OUTPUTS), "analysis", "synth_tense_ft_2026-10-10"))
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    if a.compare:
        return compare(a)
    from mdd.pair_rescue import FAMILY_OF
    R = Runner(a.model)
    rep = {"model": a.model, "name": a.name}
    out_json = os.path.join(a.out_dir, "%s.json" % a.name)
    if a.merge and os.path.exists(out_json):
        rep = json.load(open(out_json, encoding="utf-8"))
    suites = a.suites.split(",")
    t0 = time.time()

    def tag_suite(mdir, split, n_max, label):
        paths, ltr, can = read_split(mdir, split)
        idx = [i for i in range(len(paths)) if can[i] and not any(len(x) != 1 for x in can[i] + ltr[i])]
        if n_max:
            idx = sorted(random.Random(a.seed).sample(idx, min(n_max, len(idx))))
        acc = collections.Counter()
        for n, i in enumerate(idx):
            h = R.hyp(paths[i])
            acc["edits"] += edits(ltr[i], h); acc["tag_len"] += len(ltr[i])
            site_stats(can[i], ltr[i], h, acc, FAMILY_OF)
            if n % 2000 == 0:
                print("  %s %d/%d · %.0fs" % (label, n + 1, len(idx), time.time() - t0), flush=True)
        tr, fr = acc["err_det"], acc["ok_flag"]
        P, Rc = tr / max(1, tr + fr), tr / max(1, acc["err"])
        out = {"n": len(idx), "jer": acc["edits"] / acc["tag_len"], "precision": P, "recall": Rc,
               "f1": 2 * P * Rc / max(1e-9, P + Rc), "false_flag": fr / acc["ok"],
               "preserve": acc["err_pres"] / acc["err"], "revert": acc["err_rev"] / acc["err"],
               "pairs": dict((k[7:], {"n": acc[k], "pres": acc["pair_pres_" + k[7:]] / acc[k],
                                      "rev": acc["pair_rev_" + k[7:]] / acc[k]})
                             for k in acc if k.startswith("pair_n_"))}
        print("[%s] %s" % (label, json.dumps(dict((k, round(v, 4)) for k, v in out.items() if k != "pairs"))), flush=True)
        return out

    if "test" in suites:
        rep["test"] = tag_suite(a.test_manifest, a.test_split, a.n_test, "test")
        rep["test"]["manifest"] = "%s:%s" % (os.path.basename(a.test_manifest.rstrip("/")), a.test_split)
    if "unseen" in suites:
        rep["unseen"] = tag_suite(a.unseen_manifest, "test", a.n_unseen, "unseen")
        rep["unseen"]["chunked_over_sec"] = R.chunk_over

    if "deliberate" in suites:
        ensure_kog2p()
        argv, _sys.argv = _sys.argv, _sys.argv[:1]
        try:
            from g2p import readRules, graph2prono, ver_info
        finally:
            _sys.argv = argv
        from mdd.jamo_to_kog2p import phones_to_jamo, INV
        ri, ro = readRules(ver_info[0], str(G2P_TOOLS / "KoG2P" / "rulebook.txt"))

        def canonical(s):
            text = re.sub(r"\s+", " ", re.sub(r"[^0-9A-Za-z가-힣\s]", " ", s or "")).strip()
            out = []
            for j in phones_to_jamo([p for p in graph2prono(text, ri, ro).split() if p in INV]):
                out += {"ㅙ": ["ㅗ", "ㅐ"], "ㅞ": ["ㅜ", "ㅔ"]}.get(j, [j])
            return out
        for dn, ddir in enumerate(a.deliberate.split(",")):
            rows = json.load(open(os.path.join(ddir, "manifest.json"), encoding="utf-8"))
            acc = collections.defaultdict(collections.Counter)
            for r in rows:
                cg = canonical(r["ref_text"])
                cb = canonical(r["ref_text"].replace(r["good"], r["bad"]))
                pb = proj(cg, cb)
                tgt = [(j, x, y) for j, (x, y) in enumerate(zip(cg, pb)) if y is not None and x != y and FAMILY_OF.get(x) and FAMILY_OF.get(x) == FAMILY_OF.get(y)]
                if len(tgt) != 1:
                    continue
                j, c, t = tgt[0]
                pm = proj(cg, R.hyp(r["wav"]))
                key = "%s→%s %s/%s" % (c, t, r["good"], r["bad"])
                k = acc[key]
                if r["kind"] == "wrong":
                    k["wrong"] += 1; k["caught"] += pm[j] == t; k["fixed"] += pm[j] == c
                else:
                    k["correct"] += 1; k["flag"] += pm[j] != c
            res = dict((k, dict(v)) for k, v in acc.items())
            if dn == 0:
                rep["deliberate"] = res
            else:
                rep.setdefault("deliberate_extra", {})[os.path.basename(ddir.rstrip("/"))] = res
            tot = sum(acc.values(), collections.Counter())
            print("[deliberate %s] 틀린 녹음 %d 중 그 소리로 %d · 표준으로 고침 %d · 맞게 읽은 %d 중 헛지적 %d" % (
                os.path.basename(ddir.rstrip("/")), tot["wrong"], tot["caught"], tot["fixed"], tot["correct"], tot["flag"]), flush=True)

    if "native" in suites:
        paths, ltr, _ = read_split(a.native_manifest, "valid")
        idx = sorted(random.Random(a.seed).sample(range(len(paths)), min(a.n_native, len(paths))))
        e = n_ = 0
        for i in idx:
            e += edits(ltr[i], R.hyp(paths[i])); n_ += len(ltr[i])
        rep["native"] = {"n": len(idx), "jer": e / max(1, n_)}
        print("[native] JER %.4f" % rep["native"]["jer"], flush=True)

    if "synthdev" in suites and os.path.exists(os.path.join(a.synth_manifest, "synth_dev.tsv")):
        paths, ltr, can = read_split(a.synth_manifest, "synth_dev")
        acc = collections.Counter()
        for i in range(len(paths)):
            site_stats(can[i], ltr[i], R.hyp(paths[i]), acc, FAMILY_OF)
        rep["synthdev"] = {"n": len(paths), "err": acc["err"], "preserve": acc["err_pres"] / max(1, acc["err"]),
                           "revert": acc["err_rev"] / max(1, acc["err"]), "false_flag": acc["ok_flag"] / max(1, acc["ok"])}
        print("[synthdev] %s" % json.dumps(dict((k, round(v, 4)) for k, v in rep["synthdev"].items())), flush=True)

    json.dump(rep, open(out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("[완료] %s · %.0fs" % (a.name, time.time() - t0), flush=True)


def compare(a):
    reps = [json.load(open(os.path.join(a.out_dir, n + ".json"), encoding="utf-8")) for n in a.compare]
    fmt = lambda v: "—" if v is None else ("%.4f" % v if abs(v) < 1 else "%.1f" % v)
    g = lambda r, *ks: (lambda x: x)(_get(r, ks))
    print("| 항목 | " + " | ".join(r["name"] for r in reps) + " |")
    print("|---|" + "---:|" * len(reps))
    for label, ks in (("test JER", ("test", "jer")), ("test 정밀도", ("test", "precision")), ("test 재현율", ("test", "recall")),
                      ("test F1", ("test", "f1")), ("test 맞게 읽은 자리 헛지적", ("test", "false_flag")),
                      ("test 들린 그대로", ("test", "preserve")), ("test 표준으로 덮음", ("test", "revert")),
                      ("unseen JER", ("unseen", "jer")), ("unseen 정밀도", ("unseen", "precision")),
                      ("unseen 재현율", ("unseen", "recall")), ("unseen F1", ("unseen", "f1")),
                      ("unseen 맞게 읽은 자리 헛지적", ("unseen", "false_flag")), ("unseen 들린 그대로", ("unseen", "preserve")),
                      ("unseen 표준으로 덮음", ("unseen", "revert")),
                      ("원어민 JER", ("native", "jer")), ("합성 dev 들린 그대로", ("synthdev", "preserve")),
                      ("합성 dev 헛지적", ("synthdev", "false_flag"))):
        print("| %s | %s |" % (label, " | ".join(fmt(_get(r, ks)) for r in reps)))
    pairs = sorted(set(k for r in reps for k in (r.get("test", {}).get("pairs") or {})),
                   key=lambda k: -reps[0]["test"]["pairs"].get(k, {}).get("n", 0))
    print("\n| 쌍(test) | 자리 | " + " | ".join("%s 들린 그대로" % r["name"] for r in reps) + " |")
    print("|---|---:|" + "---:|" * len(reps))
    for k in pairs:
        n = reps[0]["test"]["pairs"].get(k, {}).get("n", 0)
        print("| %s→%s | %d | %s |" % (k[0], k[1], n, " | ".join(
            "%.1f%%" % (100 * r["test"]["pairs"][k]["pres"]) if k in r["test"]["pairs"] else "—" for r in reps)))
    names = sorted(set(k for r in reps for k in (r.get("deliberate_extra") or {})))
    print("\n| 일부러 틀린 TTS 합계 | " + " | ".join("%s 그 소리로 / 고침 / 맞게 읽은 것 헛지적" % r["name"] for r in reps) + " |")
    print("|---|" + "---:|" * len(reps))
    for nm, get in [("첫 시험셋(Chatterbox)", lambda r: r.get("deliberate"))] + \
            [(n, (lambda n: lambda r: (r.get("deliberate_extra") or {}).get(n))(n)) for n in names]:
        print("| %s | %s |" % (nm, " | ".join(
            "%d / %d / %d (틀린 %d · 맞는 %d)" % (lambda t: (t["caught"], t["fixed"], t["flag"], t["wrong"], t["correct"]))(
                _deliberate_total(get(r))) if get(r) else "—" for r in reps)))
    print("\n| 일부러 틀린 발음(10개씩) | " + " | ".join("%s 그 소리로 / 고침 / 맞게 읽은 것 헛지적" % r["name"] for r in reps) + " |")
    print("|---|" + "---:|" * len(reps))
    for k in (reps[0].get("deliberate") or {}):
        print("| %s | %s |" % (k, " | ".join("%d / %d / %d" % (r["deliberate"][k].get("caught", 0), r["deliberate"][k].get("fixed", 0),
                                                              r["deliberate"][k].get("flag", 0)) if k in r.get("deliberate", {}) else "—"
                                               for r in reps)))


def _deliberate_total(d):
    t = collections.Counter()
    for v in (d or {}).values():
        t.update(v)
    return t


def _get(r, ks):
    x = r
    for k in ks:
        if not isinstance(x, dict) or k not in x:
            return None
        x = x[k]
    return x


if __name__ == "__main__":
    main()
