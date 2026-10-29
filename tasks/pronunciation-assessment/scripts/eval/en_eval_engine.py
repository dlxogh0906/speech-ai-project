# -*- coding: utf-8 -*-
"""영어 엔진을 **서버와 같은 경로**로 평가 — 표준발음은 우리 G2P(CMUdict/g2p_en), 앞뒤 무음 맞추기, 응답 변환까지.

`en_fit_phone_scorer.py` 의 수치는 데이터셋이 준 표준발음 위에서 잰 것이다. 실제 서비스는 문장만 받아 G2P 로
표준발음을 만들므로 여기서 다시 잰다.

    so762     speechocean762 test — 단어 점수 PCC(vs 단어 정확도) · 문장 점수 PCC(종합·정확도·유창성·운율)
    l2arctic  한국인 4명 — **단어 단위** 오류 판별: 사람 주석에 대치·삭제가 있는 단어 vs 응답에 errorType 이 있는 단어
              (CC BY-NC · 사내 평가 전용)
    시간      발화당 처리 시간(엔진만) — 다른 작업과 GPU 를 같이 쓰면 늘어난다

    $PY .../en_eval_engine.py --model lv60-espeak
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import DATA_ROOT, MDD_OUTPUTS

import argparse, json, os, re, time

import numpy as np



def norm_word(w):
    return re.sub(r"[^a-z0-9']", "", w.lower())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="lv60-espeak", help="mdd_en.models 의 이름")
    ap.add_argument("--scorer", default=os.path.join(str(MDD_OUTPUTS), "english", "scorer"))
    ap.add_argument("--scorer-dir", dest="scorer_dir", default=None,
                    help="점수기 폴더를 직접 준다(예: <scorer>/synth-d-ko037). 기본: <scorer>/<model>")
    ap.add_argument("--so762", default=os.path.join(str(DATA_ROOT), "english", "speechocean762"))
    ap.add_argument("--l2arctic", default=os.path.join(str(DATA_ROOT), "english", "l2arctic_v5"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None, help="결과 JSON(기본: 점수기 폴더/engine_eval.json)")
    ap.add_argument("--sweep-T", dest="sweep_T", default="", help="L2 단어 판별을 이 지적 문턱들로도(쉼표로, 예: 70,80,85,90,94)")
    ap.add_argument("--policy", default=None, choices=[None, "v1", "v2"],
                    help="지적 규칙. 기본: 점수기 폴더에 flag_policy.json 이 있으면 v2, 없으면 v1")
    ap.add_argument("--skip-so762", dest="skip_so762", action="store_true")
    a = ap.parse_args()
    import soundfile as sf
    import torch, torchaudio
    from mdd_en.audio import trim_edges
    from mdd_en.engine import AlignError, EnglishAssessor
    from mdd_en.l2arctic import load_speakers
    from mdd_en.response import build_response
    from mdd_en.speechocean import load_split
    from mdd_en.models import resolve
    md, kind, vd = resolve(a.model)
    sd = a.scorer_dir or os.path.join(a.scorer, a.model)
    eng = EnglishAssessor(md, kind, sd, vocab_dir=vd, device=a.device, policy=a.policy)
    eng.warmup()

    def load(p):
        x, sr = sf.read(p, dtype="float32", always_2d=True)
        x = x.mean(1)
        if sr != 16000:
            x = torchaudio.functional.resample(torch.from_numpy(x), sr, 16000).numpy()
        return x

    rep = {"model": a.model, "device": eng.rec.device, "flag_T": eng.flag_T, "policy": eng.policy}
    ms = []
    # ── speechocean762 test ──
    rows = [] if a.skip_so762 else load_split(a.so762, "test")
    rows = rows[: a.limit] if a.limit else rows
    wp, wh, sp = [], [], {"overall": [], "pronunciation": [], "fluency": [], "rhythm": []}
    sh = {"total": [], "accuracy": [], "fluency": [], "prosodic": []}
    n_wmis = 0
    n_align_err = 0
    for r in rows:
        t0 = time.time()
        try:
            raw = eng.assess(trim_edges(load(r["wav"])), r["text"])
        except AlignError:
            n_align_err += 1                              # 서버라면 400 too_short — 평가에서 뺀 수를 남긴다
            continue
        ms.append((time.time() - t0) * 1000)
        body = build_response(raw, r["text"], flag_rule=eng.flag_rule)["result"]
        ours = [w for w in body["words"]]
        if len(ours) != len(r["words"]):
            n_wmis += 1
            continue
        for w, h in zip(ours, r["words"]):
            wp.append(w["score"] or 0); wh.append(h["accuracy"])
        for k_o, k_h in (("overall", "total"), ("pronunciation", "accuracy"), ("fluency", "fluency"), ("rhythm", "prosodic")):
            sp[k_o].append(body[k_o] or 0); sh[k_h].append(r[k_h])
    pc = lambda x, y: round(float(np.corrcoef(x, y)[0, 1]), 4) if len(x) > 2 else None
    rep["so762"] = {"n_utts": len(rows), "align_error_utts": n_align_err, "word_count_mismatch_utts": n_wmis,
                    "word_pcc": pc(wp, wh),
                    "overall_vs_total": pc(sp["overall"], sh["total"]),
                    "pronunciation_vs_accuracy": pc(sp["pronunciation"], sh["accuracy"]),
                    "fluency_vs_fluency": pc(sp["fluency"], sh["fluency"]),
                    "rhythm_vs_prosodic": pc(sp["rhythm"], sh["prosodic"])}
    print("[so762] %s" % rep["so762"], flush=True)

    # ── L2-ARCTIC 한국인: 단어 단위 오류 판별 ──
    rows = load_speakers(a.l2arctic)
    rows = rows[: a.limit] if a.limit else rows
    tp = fp = fn = tn = 0
    n_skip = 0
    sweep = [float(x) for x in a.sweep_T.split(",") if x.strip()]
    from mdd_en.metrics import KO_PAIRS, map_positions
    phc = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "cd": 0, "show_fp": 0, "n_ok": 0, "unclear": 0}
    pair_hit = dict(("%s>%s" % k, [0, 0]) for k in KO_PAIRS)          # 사람 c→s 중 우리도 c→s 로 지적 · 사람 c→s 수
    sw = dict((T, [0, 0, 0, 0]) for T in sweep)              # 문턱별 tp · fp · fn · tn
    for r in rows:
        t0 = time.time()
        try:
            raw = eng.assess(trim_edges(load(r["wav"])), r["text"])
        except AlignError:
            n_skip += 1
            continue
        ms.append((time.time() - t0) * 1000)
        body = build_response(raw, r["text"], flag_rule=eng.flag_rule)["result"]
        ours = [(norm_word(w["word"]), any(p.get("errorType") for p in w["phonemes"])) for w in body["words"]]
        human = []
        for k, wt in enumerate(r["words"]):
            ph = [p for p in r["phones"] if p["w"] == k]
            human.append((norm_word(wt), any(p["type"] in ("s", "d") for p in ph)))
        if [w for w, _ in ours] != [w for w, _ in human]:
            n_skip += 1                                   # 단어 나눔이 다르면(숫자·축약 등) 건너뛴다
            continue
        for (_, o), (_, h) in zip(ours, human):
            tp += o and h; fp += o and not h; fn += (not o) and h; tn += (not o) and not h
        # 음소 단위: 우리 표준 자리 ↔ 사람 주석 표준 자리(같은 음소일 때만)
        qs = [q for w in body["words"] for q in w["phonemes"]]
        hp = [p for p in r["phones"] if p["canon"] is not None]
        mp = map_positions([q["alpha"].upper() for q in qs], [p["canon"] for p in hp])
        for i, j in mp.items():
            q, hph = qs[i], hp[j]
            herr = hph["type"] in ("s", "d") and hph["actual"] != hph["canon"]
            flag, shown = bool(q.get("errorType")), bool(q.get("errorType") or q.get("uncertain"))
            phc["unclear"] += q.get("actual_src") == "unclear"
            if herr:
                phc["tp" if flag else "fn"] += 1
                said = (q.get("actual") or "").upper() or None
                if flag and q.get("actual_src") != "unclear" and said == hph["actual"]:
                    phc["cd"] += 1
                k = "%s>%s" % (hph["canon"], hph["actual"])
                if k in pair_hit:
                    pair_hit[k][1] += 1
                    pair_hit[k][0] += flag and said == hph["actual"]
            else:
                phc["n_ok"] += 1
                phc["fp" if flag else "tn"] += 1
                phc["show_fp"] += shown
        for T in sweep:
            bt = build_response(eng.reflag(raw, T), r["text"], flag_rule=dict(eng.flag_rule, T=T))["result"]
            for w, (_, h) in zip(bt["words"], human):
                o = any(p.get("errorType") for p in w["phonemes"])
                c = sw[T]
                c[0] += o and h; c[1] += o and not h; c[2] += (not o) and h; c[3] += (not o) and not h
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    rep["l2arctic_word"] = {"n_utts": len(rows), "skipped_utts": n_skip, "human_err_words": tp + fn,
                            "words": tp + fp + fn + tn, "precision": round(P, 4), "recall": round(R, 4),
                            "f1": round(2 * P * R / max(P + R, 1e-9), 4), "frr": round(fp / max(fp + tn, 1), 4)}
    print("[l2arctic 단어] %s" % rep["l2arctic_word"], flush=True)
    Pp = phc["tp"] / max(phc["tp"] + phc["fp"], 1); Rp = phc["tp"] / max(phc["tp"] + phc["fn"], 1)
    rep["l2arctic_phone"] = {"precision": round(Pp, 4), "recall": round(Rp, 4), "f1": round(2 * Pp * Rp / max(Pp + Rp, 1e-9), 4),
                             "frr": round(phc["fp"] / max(phc["n_ok"], 1), 4),
                             "display_frr": round(phc["show_fp"] / max(phc["n_ok"], 1), 4),
                             "correct_diagnosis_of_flags": round(phc["cd"] / max(phc["tp"], 1), 4),
                             "unclear_sites": phc["unclear"], "counts": phc}
    rep["l2arctic_pairs"] = dict((k, {"n_err": v[1], "flagged_with_right_sound": round(v[0] / max(v[1], 1), 3)})
                                 for k, v in pair_hit.items() if v[1])
    print("[l2arctic 음소] %s" % {k: v for k, v in rep["l2arctic_phone"].items() if k != "counts"}, flush=True)
    print("   [짝별: 사람 오류를 같은 소리로 지적한 비율] %s" % " · ".join(
        "%s %.2f(%d)" % (k, v["flagged_with_right_sound"], v["n_err"]) for k, v in rep["l2arctic_pairs"].items()
        if v["n_err"] >= 10), flush=True)
    if sweep:
        rep["l2arctic_word_sweep"] = {}
        for T, (a_, b_, c_, d_) in sw.items():
            P_, R_ = a_ / max(a_ + b_, 1), a_ / max(a_ + c_, 1)
            rep["l2arctic_word_sweep"][str(T)] = {"precision": round(P_, 4), "recall": round(R_, 4),
                                                  "f1": round(2 * P_ * R_ / max(P_ + R_, 1e-9), 4),
                                                  "frr": round(b_ / max(b_ + d_, 1), 4)}
            print("   [문턱 %g] %s" % (T, rep["l2arctic_word_sweep"][str(T)]), flush=True)
    rep["ms"] = {"median": round(float(np.median(ms)), 1), "p90": round(float(np.percentile(ms, 90)), 1)}
    print("[시간] %s" % rep["ms"], flush=True)
    out = a.out or os.path.join(sd, "engine_eval.json")
    json.dump(rep, open(out, "w"), ensure_ascii=False, indent=1)
    print("[saved] " + out)


if __name__ == "__main__":
    main()
