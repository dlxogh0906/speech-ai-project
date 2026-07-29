# -*- coding: utf-8 -*-
"""로드맵 PDF가 의도한 MDD F1 — LAR(in-domain) 3자 정렬.

PDF p.15: "MDD 로그 = wav명 · 표준(G2P) · 전사자 · 인식기 예측 · TN/TP/FN/FP · F1 (3자 정렬)"
PDF p.13: "'위치·유형'은 error_tags, '실현형'은 LAR MDD를 각각 정답 소스로"

즉 MDD F1의 '전사자(정답 실현형)' 열 = LAR Speech_lab 전사(=actual_phones). RW/RS error_tags 아님.
따라서 여기선:
  · 표준(G2P)      = target_phones  (manifest_16k/*.canon)
  · 전사자(실현형)  = actual_phones  (manifest_16k/*.ltr)  ← 정답(gold) 열
  · 인식기 예측     = CTC 디코드
3자 정렬(위치단위): C↔R sub/del = 진짜 오발음 위치(gold), C↔H sub/del = 예측 위치 → TP/FP/FN → F1.

주의(정직): 전사자(Speech_lab 실현형)는 인식기 학습 타깃과 '같은 소스'라 약라벨·반순환.
화자분리 test라 누수는 없지만, RW/RS(강라벨·독립)와는 성격이 다름. 두 수치를 나란히 본다.
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
if "--help" in _sys.argv or "-h" in _sys.argv:   # 인자 파서가 없는 스크립트다. --help 로 본체가 실행되면 안 된다.
    print(__doc__ or "설명이 없다 — 소스를 열어볼 것"); raise SystemExit(0)
from shared.paths import MDD_OUTPUTS

import sys, os, json, wave
import numpy as np
from mdd.mdd_eval import Decoder, load_wav, batched
from mdd.mdd_eval_pos import _nw, prf_at_threshold   # 같은 정렬·지표 재사용

MAN = f"{MDD_OUTPUTS}/manifest_16k"
PRED_CACHE = f"{MDD_OUTPUTS}/gold_eval/lar_preds.jsonl"
OUT = f"{MDD_OUTPUTS}/eval_lar_results.json"


def load_split(split):
    paths = []
    with open(f"{MAN}/{split}.tsv") as f:
        f.readline()
        for line in f:
            p, fr = line.rstrip("\n").split("\t")[:2]
            paths.append(("/" + p, int(fr)))
    realized = [l.split() for l in open(f"{MAN}/{split}.ltr").read().splitlines()]   # 전사자(실현형)=gold
    canon = [l.split() for l in open(f"{MAN}/{split}.canon").read().splitlines()]     # 표준(G2P)
    recs = []
    for (p, fr), R, C in zip(paths, realized, canon):
        recs.append({"path": p, "frames": fr, "stem": os.path.basename(p)[:-4],
                     "R": R, "C": C})
    return recs


def decode_all(dec, recs):
    cache = {}
    if os.path.exists(PRED_CACHE):
        for line in open(PRED_CACHE):
            o = json.loads(line); cache[o["stem"]] = o["pred"]
        print(f"[cache] {len(cache):,} 재사용", flush=True)
    todo = [r for r in recs if r["stem"] not in cache]
    if todo:
        items = [(r["frames"], r) for r in todo]
        cf = open(PRED_CACHE, "a"); done = 0
        for bi, idxs in enumerate(batched(items)):
            preds = dec.decode([load_wav(items[i][1]["path"]) for i in idxs])
            for j, i in enumerate(idxs):
                st = items[i][1]["stem"]; cache[st] = preds[j]
                cf.write(json.dumps({"stem": st, "pred": preds[j]}) + "\n"); done += 1
            if bi % 100 == 0:
                cf.flush(); print(f"  [decode] {done:,}/{len(todo):,}", flush=True)
        cf.close()
    for r in recs:
        r["pred"] = cache.get(r["stem"], [])
    return recs


def err_map(C, X):
    """C(표준) 대비 X의 오발음 위치 -> 실현토큰. sub=바뀐음소, del='∅'(탈락)."""
    m = {}
    for op in _nw(C, X):
        if op["kind"] == "sub":
            m[op["tpos"]] = op["actual"]
        elif op["kind"] == "del":
            m[op["tpos"]] = "∅"
    return m


def score(recs):
    pos_pred, pos_gold = [], []
    n_gold_err = n_pos = 0
    tp_diag_ok = tp_total = 0            # 진단(DER): TP위치에서 인식기 실현형 == 전사자 실현형?
    for r in recs:
        C, R, H = r["C"], r["R"], r["pred"]
        if not C:
            continue
        gold = err_map(C, R)   # 전사자(실현형) 기준 진짜 오발음 위치->실현토큰
        pred = err_map(C, H)   # 인식기 예측 위치->실현토큰
        for i in range(len(C)):
            g = 1 if i in gold else 0; p = 1 if i in pred else 0
            pos_gold.append(g); pos_pred.append(p)
            if g and p:        # TP 위치 → 진단 비교(진짜 phone 단위)
                tp_total += 1
                if pred[i] == gold[i]:
                    tp_diag_ok += 1
        n_gold_err += len(gold); n_pos += len(C)
    m = prf_at_threshold(np.array(pos_pred, float), np.array(pos_gold, float), 0.5)
    diag_acc = tp_diag_ok / tp_total if tp_total else 0.0
    return {"detection": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()},
            "diagnosis": {"phone_acc_on_TP": round(diag_acc, 4), "DER": round(1 - diag_acc, 4),
                          "n_TP": tp_total},
            "n_utt": len(recs), "n_positions": n_pos, "n_gold_err_positions": n_gold_err,
            "gold_err_rate": round(n_gold_err / max(n_pos, 1), 4)}


def main():
    recs = load_split("test")
    print(f"[LAR test] {len(recs):,} 발화", flush=True)
    cached = set()
    if os.path.exists(PRED_CACHE):
        cached = {json.loads(l)["stem"] for l in open(PRED_CACHE)}
    dec = Decoder() if any(r["stem"] not in cached for r in recs) else None  # 캐시완비면 GPU 불요
    recs = decode_all(dec, recs)
    res = score(recs)
    json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=2)
    d = res["detection"]
    print("\n" + "=" * 60)
    print(f"[LAR in-domain MDD F1]  F1={d['F1']:.4f}  P={d['P']:.4f}  R={d['R']:.4f}")
    print(f"                        FRR={d['FRR']:.4f}  FAR={d['FAR']:.4f}")
    print(f"  위치 {res['n_positions']:,} · gold오류위치 {res['n_gold_err_positions']:,} "
          f"(오류율 {res['gold_err_rate']*100:.1f}%)")
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
