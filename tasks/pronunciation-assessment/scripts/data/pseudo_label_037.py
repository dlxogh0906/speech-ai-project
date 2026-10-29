# -*- coding: utf-8 -*-
"""AIHub 037 한국인 영어에 **한국인 대표 오류 의사라벨**을 붙인다 — 2단계(자가 학습) 학습 데이터.

합성 데이터로 학습한 모델이 037 녹음의 대표 오류 자리(ð · z · θ · f · v · r/l · ɪ/iː · 단어 끝 …)마다
"표준대로 / 한국인식"을 CTC 우도로 비교해 **확실한 자리만** 한국인식으로 바꾼다(`mdd_en.pseudo`).
표준발음은 우리 G2P(서비스와 같은 CMUdict/g2p_en, 발음 후보는 소리로 고른다). 대표 오류가 아닌 자리는 표준대로 둔다.

--check-l2 를 주면 같은 판정을 L2-ARCTIC 한국인(사람 주석)에 돌려 **의사라벨의 정밀도·재현율**을 함께 남긴다
(평가 전용 · 문턱을 맞추는 데 쓰지 않는다).

산출: <out>/train.jsonl  한 줄 = {id, wav, sec, text, phones(의사라벨), canon, n_cand, n_alt, n_amb, alts}
      <out>/stats.json

    PYTHONPATH=/data/speech-ai/tools/hf_py38:/data/speech-ai/tools/en_py38 NLTK_DATA=... HF_HOME=... \\
    $PY tasks/pronunciation-assessment/scripts/data/pseudo_label_037.py --model-dir <체크포인트> --out <폴더>
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import DATA_ROOT, MDD_OUTPUTS

import argparse, collections, json, os, random, re, time

import numpy as np


def to_human(canon, hc):
    """우리 표준열 자리 → 사람 주석 표준열 자리. 정렬해서 같은 음소로 맞은 자리만(G2P 차이가 있는 자리는 뺀다)."""
    from mdd_en.align import align
    sites, _ = align(canon, hc)
    out, j = {}, 0
    for i, s in enumerate(sites):
        if s["actual"] is not None:
            if s["actual"] == canon[i]:
                out[i] = j
            j += 1
        j += len(s["inserted"])
    return out


def label_one(rec, g2p, wav, text, to_ids, margin):
    """→ (단어 목록, 표준 음소열, 판정 목록, 의사라벨 음소열) — 단어 없으면 None."""
    from mdd_en.gop import choose_variants
    from mdd_en.pseudo import apply, candidates, decide
    words = g2p.words(text)
    if not words:
        return None
    lp = rec.log_probs(wav)
    post, _ = rec.phone_post(lp)
    variants = [[[re.sub(r"\d", "", p) for p in v] for v in w["variants"]] for w in words]
    pick = choose_variants(post, variants, to_ids)
    ws = [{"w": w["word"], "ph": w["variants"][k]} for w, k in zip(words, pick)]
    canon = [re.sub(r"\d", "", p) for w in ws for p in w["ph"]]
    dec = decide(post, canon, candidates(ws), to_ids, margin)
    return ws, canon, dec, apply(canon, dec)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", dest="model_dir", required=True, help="ARPAbet 출력층 체크포인트 폴더")
    ap.add_argument("--manifest", default=os.path.join(str(MDD_OUTPUTS), "english", "aihub037_ctc", "train.jsonl"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--margin", type=float, default=1.0, help="로그우도 차 문턱(nat) — 사이 구간은 모름")
    ap.add_argument("--margins-json", dest="margins_json", default=None,
                    help="규칙별 문턱(en_calibrate_pair_margins.py 의 출력) — 주면 --margin 대신 쓴다")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--check-l2", dest="check_l2", action="store_true")
    ap.add_argument("--l2arctic", default=os.path.join(str(DATA_ROOT), "english", "l2arctic_v5"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import soundfile as sf
    import torch, torchaudio
    from mdd_en.audio import trim_edges
    from mdd_en.g2p import EnglishG2P
    from mdd_en.recognizer import PhoneRecognizer

    rec = PhoneRecognizer(a.model_dir, "arpa")
    margin = a.margin
    if a.margins_json:
        margin = json.load(open(a.margins_json, encoding="utf-8"))["margins"]
    g2p = EnglishG2P()
    idx = dict((p, i) for i, p in enumerate(rec.names))
    to_ids = lambda ph: [idx[p] for p in ph]
    os.makedirs(a.out, exist_ok=True)
    stats = {"model": a.model_dir, "margin": a.margins_json or a.margin}

    def load(p):
        x, sr = sf.read(p, dtype="float32", always_2d=True)
        x = x.mean(1)
        if sr != 16000:
            x = torchaudio.functional.resample(torch.from_numpy(x), sr, 16000).numpy()
        return trim_edges(x)

    if a.check_l2:                                            # 사람 주석으로 의사라벨 품질 재기(평가 전용)
        from mdd_en.l2arctic import load_speakers
        tp = fp = fn = amb = 0
        per_rule = collections.defaultdict(lambda: [0, 0, 0])   # 맞음 · 틀림(사람은 표준) · 놓침
        for r in load_speakers(a.l2arctic):
            out = label_one(rec, g2p, load(r["wav"]), r["text"], to_ids, margin)
            if out is None:
                continue
            ws, canon, dec, _ = out
            hc = [p["canon"] for p in r["phones"] if p["canon"] is not None]
            ha = [p["actual"] for p in r["phones"] if p["canon"] is not None]
            hmap = to_human(canon, hc)                         # 우리 표준 자리 → 사람 주석 자리(같은 음소일 때만)
            for pos, name, c, alt, d in dec:
                if alt in (None, "+AH") or hmap.get(pos) is None:
                    continue
                human = ha[hmap[pos]]
                if d is None:
                    amb += 1
                    continue
                if d == "alt" and human == alt:
                    tp += 1; per_rule[name][0] += 1
                elif d == "alt" and human == c:
                    fp += 1; per_rule[name][1] += 1
                elif d == "canon" and human == alt:
                    fn += 1; per_rule[name][2] += 1
        stats["l2_check"] = {"alt_correct": tp, "alt_but_human_canon": fp, "missed": fn, "ambiguous": amb,
                             "precision": round(tp / max(tp + fp, 1), 3), "recall": round(tp / max(tp + fn, 1), 3),
                             "per_rule": dict((k, v) for k, v in per_rule.items())}
        print("[l2 check] %s" % json.dumps(stats["l2_check"], ensure_ascii=False), flush=True)

    rows = [json.loads(l) for l in open(a.manifest, encoding="utf-8") if l.strip()]
    if a.limit:
        rows = random.Random(a.seed).sample(rows, min(a.limit, len(rows)))
    cnt = collections.Counter()
    t0 = time.time()
    with open(os.path.join(a.out, "train.jsonl"), "w", encoding="utf-8") as f:
        for n, r in enumerate(rows):
            try:
                out = label_one(rec, g2p, load(r["wav"]), r["text"], to_ids, margin)
            except Exception as e:                            # 정렬 불가(녹음이 문장보다 짧음) 등
                cnt["error_" + type(e).__name__] += 1
                continue
            if out is None:
                cnt["no_words"] += 1
                continue
            ws, canon, dec, pseudo = out
            alts = [d[1] for d in dec if d[4] == "alt"]
            n_amb = sum(1 for d in dec if d[4] is None)
            cnt["utts"] += 1; cnt["cands"] += len(dec); cnt["alt"] += len(alts); cnt["amb"] += n_amb
            cnt.update("rule_" + x for x in alts)
            f.write(json.dumps({"id": r["id"], "wav": r["wav"], "sec": r["sec"], "text": r["text"],
                                "phones": " ".join(pseudo), "canon": " ".join(canon), "n_cand": len(dec),
                                "n_alt": len(alts), "n_amb": n_amb, "alts": alts, "spk": r.get("spk"),
                                "art": r.get("art")}, ensure_ascii=False) + "\n")
            if (n + 1) % 2000 == 0:
                print("[pseudo] %d/%d · 한국인식 %.1f%% · 모름 %.1f%% · %.0f초" % (
                    n + 1, len(rows), 100 * cnt["alt"] / max(cnt["cands"], 1), 100 * cnt["amb"] / max(cnt["cands"], 1),
                    time.time() - t0), flush=True)
    stats["counts"] = dict(cnt)
    stats["alt_rate"] = round(cnt["alt"] / max(cnt["cands"], 1), 4)
    stats["amb_rate"] = round(cnt["amb"] / max(cnt["cands"], 1), 4)
    json.dump(stats, open(os.path.join(a.out, "stats.json"), "w"), ensure_ascii=False, indent=1)
    print("[done] %s" % json.dumps({k: v for k, v in stats.items() if k != "counts"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
