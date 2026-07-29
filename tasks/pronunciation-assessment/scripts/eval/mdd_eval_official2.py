# -*- coding: utf-8 -*-
"""공식 MDD 체크포인트 채점 — 교정판.

버그수정: gold_eval_spec의 err_pos는 'phones 티어(자모)' 인덱스인데, target_phones는
KoG2P(prompt)라 20% 어긋났음. 여기선 원천 pron_lab JSON을 직접 읽어:
  · canonical = phones 티어 자모열 (= err_pos의 원래 기준 = 모델 출력과 같은 자모공간)
  · err_pos   = error_tag 시간창 ↔ phones 시간창 겹침(s2와 동일 로직)
→ 위치 정합 + 자모 직접비교(KoG2P 변환 없음).
평가 대상 = gold_eval_spec의 held-out stem (화자분리 유지).
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists(): _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.fairseq_guard import ensure_fairseq
ensure_fairseq()
from shared.paths import DATA_ROOT, MDD_OUTPUTS

import sys, os, json, glob, argparse
import numpy as np
from mdd.mdd_eval_official import load_official, load_wav, decode
from mdd.mdd_eval_pos import _nw, prf_at_threshold
from mdd.l1prior.s0_taxonomy import SUPRASEG_FLUENCY

GOLD_SPEC = f"{MDD_OUTPUTS}/gold_eval/gold_eval_spec.jsonl"
GOLD_WAV = f"{DATA_ROOT}/mdd_gold_wav16k"
JSON_ROOTS = [f"{DATA_ROOT}/aihub/european_korean_speech", f"{DATA_ROOT}/aihub/asian_korean_speech",
              f"{DATA_ROOT}/aihub/english_korean_speech", f"{DATA_ROOT}/aihub/cj_korean_speech",
              f"{DATA_ROOT}/aihub"]


def _overlap(a0, a1, b0, b1):
    lo, hi = max(a0, b0), min(a1, b1)
    return max(0.0, hi - lo)


def build_json_index():
    idx = {}
    seen = set()
    for root in JSON_ROOTS:
        if not os.path.isdir(root) or root in seen:
            continue
        seen.add(root)
        for p in glob.glob(os.path.join(root, "**", "*pronunciation_lab*", "**", "*.json"), recursive=True):
            idx.setdefault(os.path.basename(p)[:-5], p)
    return idx


def parse_json(path):
    """→ (canonical 자모열, err_pos[phones티어], err_types)  s2와 동일 매핑."""
    r = json.load(open(path, encoding="utf-8-sig"))
    phon = (r.get("RecordingMetadata", {}) or {}).get("phonemic", {}) or {}
    phones = phon.get("phones") or []
    canon = [p.get("phone") for p in phones]
    tl = [(i, float(p.get("start", 0) or 0), float(p.get("end", 0) or 0)) for i, p in enumerate(phones)]
    err_pos, err_types = [], []
    for e in (phon.get("error_tags") or []):
        tag = e.get("error_tag")
        if not tag:
            continue
        t0, t1 = float(e.get("start", 0) or 0), float(e.get("end", 0) or 0)
        hits = [(i, _overlap(t0, t1, s, en)) for (i, s, en) in tl if _overlap(t0, t1, s, en) > 0]
        if hits:
            hits.sort(key=lambda x: -x[1])
            err_pos.append(hits[0][0]); err_types.append(tag)
    return canon, err_pos, err_types


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=f"{DATA_ROOT}/mdd_official_ckpt/checkpoint_mdd_eu.pt")
    ap.add_argument("--dataset", default="041")
    ap.add_argument("--out", default=f"{MDD_OUTPUTS}/eval_official2_041.json")
    ap.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args()

    print("[index] pron_lab JSON 인덱싱...", flush=True)
    jidx = build_json_index()
    print(f"[index] {len(jidx):,} JSON", flush=True)

    stems = [json.loads(l)["stem"] for l in open(GOLD_SPEC, encoding="utf-8")
             if json.loads(l).get("dataset") == a.dataset]
    recs = []
    miss_j = miss_w = 0
    for s in stems:
        jp = jidx.get(s)
        wp = os.path.join(GOLD_WAV, s[:2], s + ".wav")
        if not jp: miss_j += 1; continue
        if not os.path.exists(wp): miss_w += 1; continue
        recs.append({"stem": s, "json": jp, "wav": wp})
    print(f"[gold {a.dataset}] 대상 {len(recs):,} | JSON없음 {miss_j:,} | wav없음 {miss_w:,}", flush=True)

    model, dic, norm = load_official(a.ckpt)
    cache_path = a.out.replace(".json", "_preds.jsonl")
    cf = open(cache_path, "w", encoding="utf-8")

    utts = []          # (gold_all_set, gold_core_set, pred_set, n_canon)
    n_pos = n_err = n_supra = 0
    B = 16
    for i in range(0, len(recs), B):
        batch = recs[i:i + B]
        parsed = [parse_json(r["json"]) for r in batch]
        hyps = decode(model, dic, [load_wav(r["wav"]) for r in batch], norm)
        for r, (canon, epos, etypes), hyp in zip(batch, parsed, hyps):
            if not canon:
                continue
            cf.write(json.dumps({"stem": r["stem"], "canon": canon, "hyp": hyp,
                                 "epos": epos, "etypes": etypes}) + "\n")
            pred = {op["tpos"] for op in _nw(canon, hyp) if op["kind"] in ("sub", "del")}
            gall, gcore = set(), set()
            for p, t in zip(epos, etypes):
                if 0 <= p < len(canon):
                    gall.add(p)
                    if t in SUPRASEG_FLUENCY: n_supra += 1
                    else: gcore.add(p)
            utts.append((gall, gcore, pred))
            n_pos += len(canon); n_err += len(gall)
        if i % (B * 100) == 0:
            print(f"  {i:,}/{len(recs):,}", flush=True)
    cf.close()

    def tprf(which, tol):
        TP = FP = FN = 0
        for gall, gcore, pred in utts:
            gold = gcore if which == "core" else gall
            gm = sum(1 for g in gold if any(abs(g - p) <= tol for p in pred))
            TP += gm; FN += len(gold) - gm
            FP += sum(1 for p in pred if not any(abs(g - p) <= tol for g in gold))
        P = TP / (TP + FP) if TP + FP else 0.0
        R = TP / (TP + FN) if TP + FN else 0.0
        return {"F1": round(2 * P * R / (P + R), 4) if P + R else 0.0,
                "P": round(P, 4), "R": round(R, 4), "TP": TP, "FP": FP, "FN": FN}

    res = {"dataset": a.dataset, "ckpt": os.path.basename(a.ckpt), "n_utt": len(utts),
           "n_positions": n_pos, "n_err": n_err, "n_supra": n_supra,
           "core": {f"tol{t}": tprf("core", t) for t in (0, 1, 2)},
           "all": {f"tol{t}": tprf("all", t) for t in (0, 1, 2)}}
    json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=2)
    print(f"\n[공식 {a.dataset} · 교정 + 위치허용]  (예측캐시: {cache_path})")
    for t in (0, 1, 2):
        c = res["core"][f"tol{t}"]
        print(f"  CORE ±{t}: F1={c['F1']:.3f}  P={c['P']:.3f}  R={c['R']:.3f}")
    print(f"[saved] {a.out}")


if __name__ == "__main__":
    main()
