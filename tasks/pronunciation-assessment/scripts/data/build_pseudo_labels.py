# -*- coding: utf-8 -*-
"""문장 다양성용 **의사라벨** 생성 — 라벨 없는 외국인 발화에 교사 모델의 *실제 발음 전사*를 붙인다.

왜: 문항 hold-out 실험(`results/19`)에서 처음 보는 문장의 헛지적이 3배(4.4 → 12.9%)였고, 학습률·층 고정·마스킹
스크리닝 6개가 전부 효과가 없었다. 남은 원인은 Stage 2 의 문장이 202개뿐이라는 데이터 자체다.
AIHub 131 읽기 발화(`manifest_505_weak_v1` 풀, 문장 4.5만 종)로 문장 수를 늘린다.

원칙
  · **라벨 = 교사 모델의 그리디 CTC 전사(실제로 들은 소리).** 문장 텍스트(KoG2P)를 라벨로 쓰지 않는다 —
    그러면 "틀리게 말해도 정답으로 들어라"를 가르친다(되돌림). 텍스트는 쓰레기 걸러내는 거리 필터에만 쓴다.
  · **hold-out 문항(valid·test)이 부분 문자열로 들어 있는 문장은 전부 뺀다.** 그 말을 의사라벨로 들으면
    "처음 보는 문항" 평가가 오염된다. 한 음절 단어(예: 귀)도 보수적으로 부분 문자열 기준으로 뺀다.
  · 문장 다양성이 목적이라 **문장마다 한 발화부터** 고른다.
  · 교사는 hold-out 문항을 보지 않은 모델(H0)을 쓴다 — 누출 의심을 없애기 위해.

필터: 교사 확신도(내보낸 토큰의 최대 사후확률 평균) 상위 비율 + 전사와 문장 표준발음의 편집거리 비율 상한.

    $PY .../build_pseudo_labels.py --teacher $SPEECH_AI_DATA/checkpoints/holdout_v1_H0/step_4000 \\
        --pool $MDD_OUTPUTS/manifest_505_weak_v1 --holdout $MDD_OUTPUTS/holdout_prompts_v1.json \\
        --n-candidates 45000 --keep-frac 0.7 --max-dist 0.35 --out $MDD_OUTPUTS/pseudo_p5_v1
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import MDD_OUTPUTS

import argparse, collections, json, os, random, re, time, wave
import numpy as np

HANGUL = re.compile(u"[^가-힣]")


def read_wav(path, max_sec=20.0):
    with wave.open(path, "rb") as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2, path
        x = np.frombuffer(w.readframes(min(w.getnframes(), int(max_sec * 16000))), np.int16).astype(np.float32) / 32768.0
    return x


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher", required=True, help="HF 체크포인트 폴더(jamo_vocab.json 포함)")
    ap.add_argument("--pool", required=True, help="pool.{tsv,ltr,meta.jsonl} 가 있는 폴더")
    ap.add_argument("--holdout", required=True, help="select_holdout_prompts.py 의 JSON")
    ap.add_argument("--n-candidates", dest="n_cand", type=int, default=45000)
    ap.add_argument("--per-sentence", dest="per_sentence", type=int, default=1)
    ap.add_argument("--keep-frac", dest="keep_frac", type=float, default=0.7, help="확신도 상위 몇 비율을 남길지")
    ap.add_argument("--max-dist", dest="max_dist", type=float, default=0.35, help="전사↔문장 표준발음 편집거리 비율 상한")
    ap.add_argument("--min-sec", dest="min_sec", type=float, default=0.5)
    ap.add_argument("--max-sec", dest="max_sec", type=float, default=15.0)
    ap.add_argument("--batch-sec", dest="batch_sec", type=float, default=240.0, help="한 배치 오디오 총 길이(초)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.path.exists(a.out) and os.listdir(a.out):
        raise SystemExit("이미 있다(덮어쓰기 금지): %s" % a.out)
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(a.seed)

    # ── 후보 고르기 ──
    H = json.load(open(a.holdout, encoding="utf-8"))
    hold = sorted(set(HANGUL.sub(u"", v.get("norm") or v.get("prompt") or u"") for v in H["items"].values()) - {u""})
    lines = open(os.path.join(a.pool, "pool.tsv"), encoding="utf-8").read().splitlines()
    root, rows = lines[0].strip(), lines[1:]
    ltr = open(os.path.join(a.pool, "pool.ltr"), encoding="utf-8").read().splitlines()
    meta = [json.loads(l) for l in open(os.path.join(a.pool, "pool.meta.jsonl"), encoding="utf-8")]
    assert len(rows) == len(ltr) == len(meta)
    drop = collections.Counter(); hit_items = collections.Counter()
    by_sent = collections.defaultdict(list)
    for i, (r, m) in enumerate(zip(rows, meta)):
        sec = float(m.get("sec") or 0)
        if not (a.min_sec <= sec <= a.max_sec):
            drop["길이"] += 1; continue
        norm = HANGUL.sub(u"", m.get("prompt") or u"")
        hits = [h for h in hold if h in norm]
        if hits:
            drop["hold-out 문항 포함"] += 1
            for h in hits: hit_items[h] += 1
            continue
        by_sent[norm].append(i)
    sents = sorted(by_sent); rng.shuffle(sents)
    perm = dict((s, rng.sample(by_sent[s], len(by_sent[s]))) for s in sents)   # 문장마다 한 번만 섞는다
    cand = []
    for k in range(a.per_sentence):
        for s in sents:
            if k < len(perm[s]):
                cand.append(perm[s][k])
            if len(cand) >= a.n_cand: break
        if len(cand) >= a.n_cand: break
    print("[후보] 풀 %d · 제외 %s · 남은 문장 %d · 후보 %d발화" % (len(rows), dict(drop), len(by_sent), len(cand)), flush=True)

    # ── 교사 전사 ──
    from mdd.gop_feats_v2 import HFDecoder, forward_lprobs
    import torch, edlib
    dec = HFDecoder(a.teacher)
    items = sorted(cand, key=lambda i: int(rows[i].split("\t")[1]))
    res = {}
    t0 = time.time(); b, cur = [], 0
    def flush(batch):
        wavs = [read_wav(os.path.join(root, rows[i].split("\t")[0])) for i in batch]
        with torch.inference_mode():
            lp, Ls = forward_lprobs(dec, wavs)
        for k, i in enumerate(batch):
            l = lp[k, :Ls[k]].numpy()
            g = l.argmax(-1)
            toks, confs, prev = [], [], None
            for t, v in enumerate(g):
                if v != prev and v != dec.blank:
                    s = dec.d[int(v)]
                    if len(s) == 1 and 0x3131 <= ord(s) <= 0x3163:
                        toks.append(s); confs.append(float(np.exp(l[t, v])))
                prev = v
            res[i] = (toks, float(np.mean(confs)) if confs else 0.0, float(np.min(confs)) if confs else 0.0)
    for i in items:
        fr = int(rows[i].split("\t")[1])
        if b and cur + fr > a.batch_sec * 16000:
            flush(b); b, cur = [], 0
            if len(res) % 2000 < len(b) + 64:
                print("   ... %d / %d (%.0f/s)" % (len(res), len(items), len(res) / max(1e-9, time.time() - t0)), flush=True)
        b.append(i); cur += fr
    if b: flush(b)
    print("[전사] %d발화 · %.0fs" % (len(res), time.time() - t0), flush=True)

    # ── 필터 ──
    recs = []
    for i, (toks, mc, mn) in res.items():
        canon = ltr[i].split()
        if not toks or not canon:
            continue
        ed = edlib.align(u"".join(toks), u"".join(canon))["editDistance"]
        recs.append({"i": i, "toks": toks, "conf": mc, "conf_min": mn, "dist": ed / float(len(canon)),
                     "differs": int(toks != canon)})
    ok = [r for r in recs if r["dist"] <= a.max_dist]
    thr = float(np.percentile([r["conf"] for r in ok], 100 * (1 - a.keep_frac))) if ok else 1.0
    keep = [r for r in ok if r["conf"] >= thr]
    keep.sort(key=lambda r: r["i"])

    with open(os.path.join(a.out, "pseudo.tsv"), "w", encoding="utf-8") as ft, \
         open(os.path.join(a.out, "pseudo.ltr"), "w", encoding="utf-8") as fl, \
         open(os.path.join(a.out, "pseudo.meta.jsonl"), "w", encoding="utf-8") as fm:
        ft.write(root + "\n")
        for r in keep:
            i = r["i"]
            ft.write(rows[i] + "\n"); fl.write(u" ".join(r["toks"]) + "\n")
            m = dict(meta[i]); m.update({"pseudo_conf": round(r["conf"], 4), "pseudo_dist": round(r["dist"], 4),
                                         "differs_from_canonical": r["differs"]})
            fm.write(json.dumps(m, ensure_ascii=False) + "\n")
    sents_kept = len(set(HANGUL.sub(u"", meta[r["i"]].get("prompt") or u"") for r in keep))
    stats = {"teacher": os.path.abspath(a.teacher), "pool": os.path.abspath(a.pool), "holdout": os.path.abspath(a.holdout),
             "excluded": dict(drop), "holdout_hits_by_text": dict(hit_items.most_common()),
             "candidates": len(cand), "transcribed": len(res), "after_dist_filter": len(ok),
             "conf_threshold": round(thr, 4), "kept": len(keep), "kept_unique_sentences": sents_kept,
             "kept_speakers": len(set(meta[r["i"]].get("spk") for r in keep)),
             "kept_hours": round(sum(int(rows[r["i"]].split("\t")[1]) for r in keep) / 16000.0 / 3600, 1),
             "kept_differs_from_canonical": round(float(np.mean([r["differs"] for r in keep])) if keep else 0, 4),
             "kept_mean_dist": round(float(np.mean([r["dist"] for r in keep])) if keep else 0, 4),
             "rule": "라벨 = 교사 그리디 전사. 텍스트는 거리 필터에만. hold-out 문항 포함 문장 제외"}
    json.dump(stats, open(os.path.join(a.out, "stats.json"), "w"), ensure_ascii=False, indent=1)
    print("[남김] %d발화 · 문장 %d · 화자 %d · %.1f시간 · 표준발음과 다른 라벨 %.1f%% · 확신도 기준 %.3f"
          % (len(keep), sents_kept, stats["kept_speakers"], stats["kept_hours"], 100 * stats["kept_differs_from_canonical"], thr), flush=True)
    print("[saved] %s" % a.out)


if __name__ == "__main__":
    main()
