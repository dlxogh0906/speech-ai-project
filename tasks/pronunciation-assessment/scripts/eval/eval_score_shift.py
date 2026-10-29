# -*- coding: utf-8 -*-
"""표준발음 보정 · 모델 교체가 **점수**를 얼마나 바꾸나 — 지금 서비스 대비 점수 이동과 사람 점수 상관(PCC).

점수 회귀기(`score_models_v2_w2vbert.joblib`)와 화면 점수 보정 파일(calibration · utt_percentile · fluency_rule)은
step_38000 이 뽑은 특징으로 맞췄다. 표준발음이나 모델을 바꾸면 특징이 움직여 **에러 없이 점수만 달라질 수 있다.**
같은 녹음을 아래 구성으로 API 채점 경로(`mdd.api_offline` — app.py 그대로)에 넣어 나란히 잰다.

구성
  A  step_38000 + KoG2P               지금 서비스
  B  step_38000 + canonical_fix(고름)  1단계 — `--canon-fix`
  C  섞은 모델 + canonical_fix(고름)    1+2단계
  D  섞은 모델 + KoG2P                 모델만 바꾼 경우(분해용)
  L  step_38000 + 라벨 표준발음(.canon) 점수 회귀기를 학습한 조건(참고)

점수
  발화 점수 3종(1~5, 회귀기)  pron · comp · flu  → 세션 점수(10발화부터)는 이 값의 평균이다
  화면 점수(0~100)            overall · pronunciation(음소 기반) · fluency(속도·쉼 규칙)
사람 점수는 LAB `EvaluationMetadata`(gop_feats_v2_test.jsonl 에 들어 있는 값). 화자 = stem 앞 5자리.
API 말소리 검사(check_audio)를 통과한 발화만 쓴다(API 는 나머지를 거절한다). 기본은 문장 문항(RS)만 — 단어(RW)는 거의 거절된다.

    for i in 0 1 2 3 4 5; do                                         # GPU · 조각 6개면 약 20분 · 이어 하기 됨
        $PY .../eval_score_shift.py extract --n-spk 800 --per-spk 40 --min-per-spk 15 --shard $i/6 &
    done; wait
    $PY .../eval_score_shift.py report                                 # CPU 수 초
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import DATA_ROOT, MDD_OUTPUTS

import argparse, collections, csv, importlib.util, json, os, random, time, wave

CONFIGS = ("A", "B", "C", "D", "L")
NAMES = {"A": "지금 서비스", "B": "1단계(보정)", "C": "1+2단계(보정+섞은 모델)", "D": "섞은 모델만", "L": "라벨 표준발음(참고)"}
UTT_KEYS = ("pron", "comp", "flu")
SCR_KEYS = ("overall", "pronunciation", "fluency")
HUMAN_OF = {"pron": "pron", "comp": "comp", "flu": "flu", "overall": "pron", "pronunciation": "pron", "fluency": "flu"}
OUT_DIR = os.path.join(str(MDD_OUTPUTS), "analysis", "score_shift_2026-10-16")


def _sibling(name):
    spec = importlib.util.spec_from_file_location(name, str(_pl.Path(__file__).with_name(name + ".py")))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_labels(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["stem"]] = {k: r.get(k, 0) or 0 for k in UTT_KEYS}
    return out


def cmd_extract(a):
    import numpy as np
    import torch
    torch.set_num_threads(a.threads)
    from mdd.api_offline import ApiEmulator
    from mdd import canonical_fix as cf
    ecs = _sibling("eval_canonical_source")
    shard, n_shard = (int(x) for x in a.shard.split("/"))

    labels = load_labels(a.labels)
    print("[사람 점수] %d발화" % len(labels), flush=True)
    emu = ApiEmulator(a.ckpt, scorer=a.scorer)
    emu_m = ApiEmulator(a.mix_ckpt, scorer=a.scorer)
    B = cf.CanonicalBuilder(emu.canonical)

    sentences = [r["sentence"] for r in csv.DictReader(open(a.sentences, encoding="utf-8"))]
    prompts = json.load(open(a.prompts, encoding="utf-8"))["all"] if os.path.exists(a.prompts) else []
    with open(os.path.join(a.manifest, "test.tsv"), encoding="utf-8") as f:
        root = f.readline().rstrip("\n")
        rows = [l.rstrip("\n").split("\t") for l in f]
    rd = lambda ext: open(os.path.join(a.manifest, "test.%s" % ext), encoding="utf-8").read().splitlines()
    ltr, can = [l.split() for l in rd("ltr")], [l.split() for l in rd("canon")]
    by_item = collections.defaultdict(list)
    for i, (p, _) in enumerate(rows):
        m = ecs.ITEM_RE.search(p)
        if m and can[i] and not any(len(x) != 1 for x in can[i] + ltr[i]):
            by_item[m.group(1)].append(i)
    codes = sorted(by_item)
    lab = dict((k, collections.Counter(" ".join(can[i]) for i in by_item[k]).most_common(1)[0][0].replace(" ", ""))
               for k in codes)
    item_text = ecs.item_texts(emu, sentences, prompts, codes, lab)

    # 화자별 후보: 사람 점수가 있고 · 문항 원문을 알고 · 너무 짧지 않은 발화(짧은 단어는 API 가 거절한다)
    by_spk = collections.defaultdict(list)
    for k, idx in by_item.items():
        if k not in item_text or k[:2] not in a.kinds.split(","):
            continue
        for i in idx:
            stem = os.path.basename(rows[i][0])[:-4]
            if stem in labels and int(rows[i][1]) >= a.min_sec * 16000:
                by_spk[stem[:5]].append((k, i, stem))
    rng = random.Random(a.seed)
    spks = sorted(s for s, v in by_spk.items() if len(v) >= a.min_per_spk)
    spks = rng.sample(spks, min(a.n_spk, len(spks)))
    pick = []
    for s in sorted(spks):
        v = sorted(by_spk[s], key=lambda t: t[1])
        pick += rng.sample(v, min(a.per_spk, len(v)))
    print("[표본] 문항 %d · 화자 %d (후보 화자 %d) · 발화 %d" % (len(item_text), len(spks), len(by_spk), len(pick)), flush=True)

    cache = {}
    for k in set(k for k, _, _ in pick):
        base, alts = B.variants(item_text[k])         # 라이브러리 기본(경계 비음화 제외) — app.py --canon-fix 와 같다
        cache[k] = {"api": emu.canonical(item_text[k]), "base": base, "alts": alts}

    os.makedirs(a.out, exist_ok=True)
    pick = pick[shard::n_shard]
    out = os.path.join(a.out, "utts.%d-of-%d.jsonl" % (shard, n_shard))
    done = set()
    if os.path.exists(out):
        done = set(json.loads(l)["stem"] for l in open(out, encoding="utf-8"))
        print("[이어 하기] %d 이미 있음" % len(done), flush=True)
    json.dump(vars(a), open(os.path.join(a.out, "args.%d-of-%d.json" % (shard, n_shard)), "w"), ensure_ascii=False, indent=1)

    def pick_canon(e, lp, k):
        c = cache[k]
        if not c["base"]:
            return c["api"]
        got, _ = cf.choose(lp[0], c["base"], c["alts"], e.dec.d.index, blank=e.dec.blank, margin=a.margin)
        return got

    t0 = time.time()
    n_new = 0
    with open(out, "a", encoding="utf-8") as fo:
        for n, (k, i, stem) in enumerate(pick, 1):
            if stem in done:
                continue
            with wave.open(os.path.join(root, rows[i][0]), "rb") as w:
                x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0
            rec = {"stem": stem, "spk": stem[:5], "item": k, "human": labels[stem]}
            rejected, _ = emu.check_audio(x[: 16000 * 20])
            if rejected:
                rec["rejected"] = True
            else:
                lp_a, lp_m = emu.lprobs(x), emu_m.lprobs(x)
                runs = {"A": (emu, lp_a, cache[k]["api"]), "B": (emu, lp_a, pick_canon(emu, lp_a, k)),
                        "C": (emu_m, lp_m, pick_canon(emu_m, lp_m, k)), "D": (emu_m, lp_m, cache[k]["api"]),
                        "L": (emu, lp_a, can[i])}
                for name, (e, lp, canon) in runs.items():
                    body, raw = e.assess(x, item_text[k], canon=canon, lp=lp, gate=False)
                    r = body["result"]
                    ph = ApiEmulator.phonemes(body)
                    rec[name] = {"utt": raw["scores"], **{s: r.get(s) for s in SCR_KEYS},
                                 "n_flag": sum("errorType" in p for p in ph), "canon": "".join(raw["canon"])}
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_new += 1
            if n_new % 200 == 0:
                fo.flush()
                el = time.time() - t0
                print("  %d/%d  %.1f발화/s  남은 %.0f분" % (n, len(pick), n_new / el, (len(pick) - n) / (n_new / el) / 60), flush=True)
    print("[끝] %s · %.0f초" % (out, time.time() - t0), flush=True)


def _pcc(x, y):
    import numpy as np
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 and x.std() > 0 and y.std() > 0 else float("nan")


def cmd_report(a):
    import numpy as np
    import glob
    recs = [json.loads(l) for p in sorted(glob.glob(os.path.join(a.out, "utts.*.jsonl"))) for l in open(p, encoding="utf-8")]
    ok = [r for r in recs if not r.get("rejected") and all(r[c]["utt"] for c in CONFIGS)]
    print("[발화] 전체 %d · API 통과 %d · 화자 %d" % (len(recs), len(ok), len(set(r["spk"] for r in ok))))

    def val(r, c, key):
        return r[c]["utt"][key] if key in UTT_KEYS else r[c][key]

    keys = UTT_KEYS + SCR_KEYS
    rep = {"n_utt": len(ok), "n_spk": len(set(r["spk"] for r in ok)), "shift": {}, "pcc": {}, "session": {}}

    # ① 발화마다 점수가 얼마나 움직이나 (지금 서비스 A 대비)
    for c in CONFIGS[1:]:
        rep["shift"][c] = {}
        for key in keys:
            xs = [(val(r, "A", key), val(r, c, key)) for r in ok if val(r, "A", key) is not None and val(r, c, key) is not None]
            A_, X_ = np.array([p[0] for p in xs], float), np.array([p[1] for p in xs], float)
            d = X_ - A_
            big = 0.25 if key in UTT_KEYS else 5
            rep["shift"][c][key] = {"n": len(xs), "mean_A": float(A_.mean()), "mean_X": float(X_.mean()),
                                    "mean_d": float(d.mean()), "mean_abs_d": float(np.abs(d).mean()),
                                    "p95_abs_d": float(np.percentile(np.abs(d), 95)), "big": big,
                                    "frac_big": float((np.abs(d) >= big).mean()), "r_vs_A": _pcc(A_, X_),
                                    "same": float((d == 0).mean())}

    # ② 사람 점수와의 상관 — 발화 단위 · 화자 단위(화자의 통과 발화 평균 = 세션 점수와 같은 계산)
    spk_of = collections.defaultdict(list)
    for r in ok:
        spk_of[r["spk"]].append(r)
    spks = sorted(s for s, v in spk_of.items() if len(v) >= a.min_utts)
    rng = np.random.RandomState(0)
    boot = [rng.randint(0, len(spks), len(spks)) for _ in range(a.n_boot)]
    for key in keys:
        hk = HUMAN_OF[key]
        rep["pcc"][key] = {}
        spk_tab = {}
        for c in CONFIGS:
            pu = [(val(r, c, key), r["human"][hk]) for r in ok if r["human"][hk] > 0 and val(r, c, key) is not None]
            m, h = [], []
            for s in spks:
                v = [(val(r, c, key), r["human"][hk]) for r in spk_of[s] if r["human"][hk] > 0 and val(r, c, key) is not None]
                m.append(np.mean([p[0] for p in v]) if len(v) >= a.min_utts else np.nan)
                h.append(np.mean([p[1] for p in v]) if len(v) >= a.min_utts else np.nan)
            m, h = np.array(m), np.array(h)
            spk_tab[c] = (m, h)
            good = ~np.isnan(m)
            rep["pcc"][key][c] = {"utt": _pcc([p[0] for p in pu], [p[1] for p in pu]), "n_utt": len(pu),
                                  "spk": _pcc(m[good], h[good]), "n_spk": int(good.sum())}
        mA, hA = spk_tab["A"]
        for c in CONFIGS[1:]:
            mX, _ = spk_tab[c]
            ds = []
            for b in boot:
                g = ~np.isnan(mA[b]) & ~np.isnan(mX[b])
                ds.append(_pcc(mX[b][g], hA[b][g]) - _pcc(mA[b][g], hA[b][g]))
            ds = np.array([d for d in ds if d == d])
            rep["pcc"][key][c]["d_spk_vs_A"] = float(rep["pcc"][key][c]["spk"] - rep["pcc"][key]["A"]["spk"])
            rep["pcc"][key][c]["d_spk_ci"] = ([float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))] if len(ds)
                                              else [float("nan")] * 2)
            g = ~np.isnan(mA) & ~np.isnan(mX)
            dd = mX[g] - mA[g]
            ra, rx = np.argsort(np.argsort(mA[g])), np.argsort(np.argsort(mX[g]))
            rep["session"].setdefault(c, {})[key] = {"mean_d": float(dd.mean()), "mean_abs_d": float(np.abs(dd).mean()),
                                                     "max_abs_d": float(np.abs(dd).max()), "spearman_vs_A": _pcc(ra, rx)}

    json.dump(rep, open(os.path.join(a.out, "report.json"), "w"), ensure_ascii=False, indent=1)

    L = ["# 점수 이동 — 표준발음 보정 · 섞은 모델 (2026-10-16)", "",
         "API 통과 발화 %d · 화자 %d(화자당 %d발화 이상만 화자 단위 계산) · 본 문항(224) · 테스트 화자. `eval_score_shift.py`" % (
             rep["n_utt"], rep["n_spk"], a.min_utts), "",
         "## 1. 발화마다 점수가 얼마나 움직이나 (지금 서비스 A 대비)", "",
         "| 점수 | 구성 | A 평균 | 바꾼 뒤 평균 | 평균 차 | 평균 |차| | 95% 발화의 |차| 이내 | 크게 바뀐 발화 | 그대로인 발화 | A 와 상관 |",
         "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for key in keys:
        for c in CONFIGS[1:]:
            s = rep["shift"][c][key]
            L.append("| %s | %s %s | %.2f | %.2f | %+.2f | %.2f | %.2f | %.1f%% (≥%s) | %.1f%% | %.3f |" % (
                key, c, NAMES[c], s["mean_A"], s["mean_X"], s["mean_d"], s["mean_abs_d"], s["p95_abs_d"],
                100 * s["frac_big"], s["big"], 100 * s["same"], s["r_vs_A"]))
    L += ["", "pron·comp·flu 는 1~5(회귀기, 세션 점수의 재료) · overall·pronunciation·fluency 는 화면 0~100.", "",
          "## 2. 사람 점수와의 상관 (PCC)", "",
          "| 점수(사람 점수) | 구성 | 발화 단위 | 화자 단위 | 화자 단위 차(A 대비) | 95% 구간 |", "|---|---|--:|--:|--:|---|"]
    for key in keys:
        for c in CONFIGS:
            p = rep["pcc"][key][c]
            extra = ("%+.3f | [%+.3f, %+.3f]" % (p["d_spk_vs_A"], p["d_spk_ci"][0], p["d_spk_ci"][1])) if c != "A" else "— | —"
            L.append("| %s(%s) | %s %s | %.3f | %.3f | %s |" % (key, HUMAN_OF[key], c, NAMES[c], p["utt"], p["spk"], extra))
    L += ["", "화자 단위 = 화자의 통과 발화 평균 vs 사람 점수 평균(세션 점수와 같은 계산). 화자 부트스트랩 %d회." % a.n_boot,
          "화자당 발화 수가 문서값(화자당 ~149발화)보다 적어 절대값은 문서(0.67/0.69/0.76)보다 낮다 — **구성 사이 차이만** 본다.", "",
          "## 3. 세션 점수(화자 평균)가 얼마나 움직이나", "",
          "| 점수 | 구성 | 평균 차 | 평균 |차| | 최대 |차| | 화자 순위 상관(A 대비) |", "|---|---|--:|--:|--:|--:|"]
    for key in keys:
        for c in CONFIGS[1:]:
            s = rep["session"][c][key]
            L.append("| %s | %s %s | %+.3f | %.3f | %.3f | %.3f |" % (key, c, NAMES[c], s["mean_d"], s["mean_abs_d"], s["max_abs_d"], s["spearman_vs_A"]))
    open(os.path.join(a.out, "summary.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--manifest", default=os.path.join(str(MDD_OUTPUTS), "manifest_full_tag_replay10"))
    e.add_argument("--labels", default=os.path.join(str(MDD_OUTPUTS), "score_v2_w2vbert", "gop_feats_v2_test.jsonl"))
    e.add_argument("--sentences", default=os.path.join(str(MDD_OUTPUTS), "analysis", "mdd_error_2026-09-27",
                                                       "w2vbert_step38000_seen_sentence", "sentence_analysis.csv"))
    e.add_argument("--prompts", default=os.path.join(str(MDD_OUTPUTS), "analysis", "npu_bench_pack_2026-10-03", "train_prompts.json"))
    e.add_argument("--ckpt", default=os.path.join(str(DATA_ROOT), "checkpoints", "w2vbert_stage2", "step_38000"))
    e.add_argument("--mix-ckpt", dest="mix_ckpt",
                   default=os.path.join(str(DATA_ROOT), "checkpoints", "w2vbert_synth_tense_v2", "interp_a0.5_step3000"))
    e.add_argument("--scorer", default=os.path.join(str(MDD_OUTPUTS), "score_v2_w2vbert", "score_models_v2_w2vbert.joblib"))
    e.add_argument("--margin", type=float, default=1.0, help="canonical_fix.choose margin (app.py --canon-fix-margin 기본과 같다)")
    e.add_argument("--n-spk", dest="n_spk", type=int, default=400)
    e.add_argument("--per-spk", dest="per_spk", type=int, default=40, help="화자당 후보 발화 수(거절되는 발화가 있어 통과는 이보다 적다)")
    e.add_argument("--min-per-spk", dest="min_per_spk", type=int, default=20)
    e.add_argument("--min-sec", dest="min_sec", type=float, default=1.2)
    e.add_argument("--kinds", default="RS", help="문항 종류(RS 문장 · RW 단어). 단어는 96%% 가 API 말소리 검사에 걸려 기본은 문장만")
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--shard", default="0/1", help="i/n — 표본의 i 번째 조각만(여러 프로세스로 나눠 돌릴 때). CPU 가 병목이다")
    e.add_argument("--threads", type=int, default=2, help="프로세스당 torch CPU 스레드")
    e.add_argument("--out", default=OUT_DIR)
    r = sub.add_parser("report")
    r.add_argument("--out", default=OUT_DIR)
    r.add_argument("--min-utts", dest="min_utts", type=int, default=10, help="화자 단위 계산에 필요한 통과 발화 수(세션 점수 최소와 같다)")
    r.add_argument("--n-boot", dest="n_boot", type=int, default=1000)
    a = ap.parse_args()
    {"extract": cmd_extract, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
