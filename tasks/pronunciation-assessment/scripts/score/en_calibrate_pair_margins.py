# -*- coding: utf-8 -*-
"""한국인 대표 오류 판정 문턱(규칙별) — **사람 라벨 없이** 037 에서 정한다.

생각: AIHub 037 에서 사람이 조음 점수를 높게 준(≥ --min-art, 기본 4.5) 발화는 대표 오류가 드물 것이다.
그 발화들의 대표 오류 자리에서 LLR(한국인식 − 표준, CTC 우도)을 모아, 규칙마다 상위 --target-fa(기본 5%)만
"한국인식"으로 걸리게 문턱을 잡는다. 표준발음은 서비스와 같은 우리 G2P(발음 후보는 소리로 고른다).

--eval-l2 를 주면 그 문턱으로 L2-ARCTIC 한국인 4명(사람 주석 · 평가 전용)에서 규칙별 정밀도·재현율·헛지적을 잰다.
문턱을 L2 로 고치지 않는다.

산출: --out JSON = {"model", "min_art", "target_fa", "margins": {규칙: 문턱}, "n": {규칙: 표본 수}, "l2": …}

    $PY .../en_calibrate_pair_margins.py --model-dir <체크포인트> --out <json> --eval-l2
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import DATA_ROOT, MDD_OUTPUTS

import argparse, collections, json, os, random, re

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", dest="model_dir", required=True)
    ap.add_argument("--manifest", default=os.path.join(str(MDD_OUTPUTS), "english", "aihub037_ctc", "train.jsonl"))
    ap.add_argument("--min-art", dest="min_art", type=float, default=4.5)
    ap.add_argument("--n", type=int, default=3000, help="037 고득점 발화 수")
    ap.add_argument("--target-fa", dest="target_fa", type=float, default=0.05)
    ap.add_argument("--min-sites", dest="min_sites", type=int, default=30, help="표본이 이보다 적은 규칙은 문턱 없음(판정 안 함)")
    ap.add_argument("--eval-l2", dest="eval_l2", action="store_true")
    ap.add_argument("--l2arctic", default=os.path.join(str(DATA_ROOT), "english", "l2arctic_v5"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import soundfile as sf
    import torch, torchaudio
    from mdd_en.audio import trim_edges
    from mdd_en.g2p import EnglishG2P
    from mdd_en.gop import choose_variants
    from mdd_en.l1_ko_errors import strip
    from mdd_en.l2arctic import annotation_words, load_speakers
    from mdd_en.pseudo import candidates, site_llrs
    from mdd_en.recognizer import PhoneRecognizer

    rec = PhoneRecognizer(a.model_dir, "arpa")
    idx = dict((p, i) for i, p in enumerate(rec.names))
    to_ids = lambda ph: [idx[p] for p in ph]
    g2p = EnglishG2P()

    def load(p):
        x, sr = sf.read(p, dtype="float32", always_2d=True)
        x = x.mean(1)
        if sr != 16000:
            x = torchaudio.functional.resample(torch.from_numpy(x), sr, 16000).numpy()
        return trim_edges(x)

    rows = [json.loads(l) for l in open(a.manifest, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if (r.get("art") or 0) >= a.min_art]
    rows = random.Random(a.seed).sample(rows, min(a.n, len(rows)))
    llrs = collections.defaultdict(list)
    for r in rows:
        words = g2p.words(r["text"])
        if not words:
            continue
        with torch.no_grad():
            post, _ = rec.phone_post(rec.log_probs(load(r["wav"])))
        variants = [[[strip(p) for p in v] for v in w["variants"]] for w in words]
        try:
            pick = choose_variants(post, variants, to_ids)
        except Exception:
            continue
        ws = [{"w": w["word"], "ph": w["variants"][k]} for w, k in zip(words, pick)]
        canon = [strip(p) for w in ws for p in w["ph"]]
        cands = candidates(ws)
        v = site_llrs(post, canon, cands, to_ids) if cands else None
        if v is None:
            continue
        for (pos, name, c, alt), d in zip(cands, v):
            llrs[name].append(d)
    margins, ns = {}, {}
    for name, v in llrs.items():
        ns[name] = len(v)
        if len(v) >= a.min_sites:
            margins[name] = round(float(np.quantile(v, 1 - a.target_fa)), 3)
    rep = {"model": a.model_dir, "min_art": a.min_art, "target_fa": a.target_fa, "n_utts": len(rows),
           "margins": margins, "n": ns}
    print("[margins] %s" % json.dumps(margins, ensure_ascii=False), flush=True)

    if a.eval_l2:
        stat = collections.defaultdict(lambda: [0, 0, 0, 0])      # 양성 맞힘 · 양성 수 · 음성 헛지적 · 음성 수
        for r in load_speakers(a.l2arctic):
            ws = annotation_words(r)
            canon = [strip(p) for w in ws for p in w["ph"]]
            act = [y for w in ws for y in w["act"]]
            cands = [c for c in candidates(ws) if c[3] not in (None, "+AH")]
            if not cands:
                continue
            with torch.no_grad():
                post, _ = rec.phone_post(rec.log_probs(load(r["wav"])))
            v = site_llrs(post, canon, cands, to_ids)
            if v is None:
                continue
            for (pos, name, c, alt), d in zip(cands, v):
                if name not in margins:
                    continue
                hit = d > margins[name]
                if act[pos] == alt:
                    stat[name][0] += hit; stat[name][1] += 1
                elif act[pos] == c:
                    stat[name][2] += hit; stat[name][3] += 1
        l2 = {}
        tot = [0, 0, 0, 0]
        for name, (tp, npos, fp, nneg) in stat.items():
            l2[name] = {"recall": round(tp / max(npos, 1), 3), "n_err": npos, "false_alarm": round(fp / max(nneg, 1), 4),
                        "n_ok": nneg, "precision": round(tp / max(tp + fp, 1), 3)}
            tot = [x + y for x, y in zip(tot, (tp, npos, fp, nneg))]
        l2["_all"] = {"recall": round(tot[0] / max(tot[1], 1), 3), "n_err": tot[1],
                      "false_alarm": round(tot[2] / max(tot[3], 1), 4), "n_ok": tot[3],
                      "precision": round(tot[0] / max(tot[0] + tot[2], 1), 3)}
        rep["l2"] = l2
        print("[l2] 전체 %s" % l2["_all"], flush=True)
        print("     " + " · ".join("%s 재현 %.2f(%d) 헛 %.3f 정밀 %.2f" % (k, v["recall"], v["n_err"], v["false_alarm"], v["precision"])
                                    for k, v in sorted(l2.items(), key=lambda kv: -kv[1]["n_err"]) if k != "_all" and v["n_err"] >= 10),
              flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(rep, open(a.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
