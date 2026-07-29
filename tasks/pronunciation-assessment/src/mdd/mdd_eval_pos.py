# -*- coding: utf-8 -*-
"""위치단위 MDD F1 재평가 (계획서 규약 그대로).

발표자료_음소인식기_1A_1B.html:362 = "인식 발음열 ↔ KoG2P ↔ 전문가 전사 — 우리 s5 하네스로 채점"
GPU작업…md:92                     = "MDD 진단 | P·R·F1·FAR·FRR·DER | 인식 발음열 ↔ G2P ↔ 전문가 전사"

공식 팩(align_result.py·calculate_f1.py)은 디스크에 없음(이식 대상). 그 알고리즘 =
  · 3열 정렬  : s1_speechlab_mdd.py 의 _nw() (Needleman-Wunsch)  ← 원문 그대로 복사
  · P/R/F1·FAR/FRR·AP : s5_benchmark.py 의 prf_at_threshold()/average_precision() ← 원문 그대로 복사
  · 유형진단(DER)     : s0_taxonomy.py 의 classify_op()/SUPRASEG_FLUENCY  ← import

채점 단위 = canonical(target_phones)의 '위치'. gold_eval_spec.jsonl 에 이미 있는
err_pos(전문가가 지목한 오류 위치)·err_types(유형)를 정답으로 쓴다.
  · 인식가설 H 를 canonical T 에 NW 정렬 → sub/del 된 tpos = '예측 오발음 위치'
  · gold 분절오류 위치(초분절 제외)와 위치단위로 TP/FP/FN → P/R/F1·FRR·FAR
  · TP 위치에서 classify_op(정렬op) 유형 == gold 유형 ? → 진단정확도·DER
  · 초분절(강세/경계억양/반복/간투사)은 음소MDD 판정불가 → 분절F1에서 제외, 따로 집계

이전(mdd_eval.run_gold_f1)은 '발화 오류유무'를 정규화편집거리 임계값으로 잡은 것 =
위치·유형 정보 폐기. 이 스크립트가 그걸 위치단위로 교정한다. 재학습 불필요.
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists(): _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import DATA_ROOT, MDD_OUTPUTS

import sys, os, json, wave, collections
import numpy as np

from mdd.mdd_eval import Decoder, load_wav, batched      # 학습 criterion과 동일한 CTC 디코더 재사용
from mdd.l1prior.s0_taxonomy import classify_op, SUPRASEG_FLUENCY  # 초분절 태그셋·유형분류(그대로)

GOLD_SPEC = f"{MDD_OUTPUTS}/gold_eval/gold_eval_spec.jsonl"
GOLD_WAV = f"{DATA_ROOT}/mdd_gold_wav16k"
OUT = f"{MDD_OUTPUTS}/eval_pos_results.json"
PRED_CACHE = f"{MDD_OUTPUTS}/gold_eval/gold_preds.jsonl"  # 재채점용(GPU 불요)


# ── s1_speechlab_mdd.py:41-71 _nw() 원문 그대로 (3열 정렬 = align_result.py 동일 알고리즘) ──
def _nw(target, actual, match=0, mismatch=1, gap=1):
    """토큰열 정렬. ops = [{kind,target,actual,tpos,apos}] 반환.
    kind ∈ match|sub|del|ins.  tpos=target 인덱스, apos=actual 인덱스."""
    n, m = len(target), len(actual)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        D[i][0] = i * gap
    for j in range(1, m + 1):
        D[0][j] = j * gap
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = match if target[i - 1] == actual[j - 1] else mismatch
            D[i][j] = min(D[i - 1][j - 1] + c, D[i - 1][j] + gap, D[i][j - 1] + gap)
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            c = match if target[i - 1] == actual[j - 1] else mismatch
            if D[i][j] == D[i - 1][j - 1] + c:
                kind = "match" if target[i - 1] == actual[j - 1] else "sub"
                ops.append({"kind": kind, "target": target[i - 1],
                            "actual": actual[j - 1], "tpos": i - 1, "apos": j - 1})
                i -= 1; j -= 1; continue
        if i > 0 and D[i][j] == D[i - 1][j] + gap:
            ops.append({"kind": "del", "target": target[i - 1],
                        "actual": None, "tpos": i - 1, "apos": None})
            i -= 1; continue
        ops.append({"kind": "ins", "target": None, "actual": actual[j - 1],
                    "tpos": i, "apos": j - 1})
        j -= 1
    ops.reverse()
    return ops


# ── s5_benchmark.py:67-96 원문 그대로 (calculate_f1.py 동일 정의) ──
def average_precision(scores, labels):
    order = np.argsort(-scores)
    y = labels[order]
    P = max(y.sum(), 1e-9)
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    prec = tp / np.maximum(tp + fp, 1e-9)
    rec = tp / P
    ap = 0.0; prev = 0.0
    for k in range(len(y)):
        if y[k] == 1:
            ap += prec[k] * (rec[k] - prev); prev = rec[k]
    return float(ap)


def prf_at_threshold(scores, labels, thr):
    s = np.asarray(scores, float); y = np.asarray(labels, float)
    pred = (s >= thr).astype(float)
    tp = float(((pred == 1) & (y == 1)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    tn = float(((pred == 0) & (y == 0)).sum())
    P = tp / (tp + fp) if (tp + fp) else 0.0
    R = tp / (tp + fn) if (tp + fn) else 0.0
    F1 = 2 * P * R / (P + R) if (P + R) else 0.0
    FRR = fp / (tn + fp) if (tn + fp) else 0.0   # 과지적(정상→오류)
    FAR = fn / (tp + fn) if (tp + fn) else 0.0   # 미검출(오류 놓침)=1-R
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, P=P, R=R, F1=F1, FRR=FRR, FAR=FAR)
# ──────────────────────────────────────────────────────────────────────────


def load_gold():
    """gold_eval_spec + 오디오 존재분만. err_pos/err_types(정답) 함께."""
    recs = []
    miss = 0
    for line in open(GOLD_SPEC, encoding="utf-8"):
        d = json.loads(line); s = d["stem"]
        wp = os.path.join(GOLD_WAV, s[:2], s + ".wav")
        if not os.path.exists(wp):
            miss += 1; continue
        try:
            fr = wave.open(wp, "rb").getnframes()
        except Exception:
            miss += 1; continue
        recs.append({"path": wp, "frames": fr, "stem": s, "spk": d.get("spk"),
                     "L1": d.get("L1"), "target": d.get("target_phones") or [],
                     "has_error": bool(d["has_error"]),
                     "err_pos": d.get("err_pos") or [], "err_types": d.get("err_types") or []})
    print(f"[gold] 오디오존재 {len(recs):,} / 결측 {miss:,}", flush=True)
    return recs


def decode_all(dec, recs):
    """gold 전체 CTC 디코드 → pred 음소열. PRED_CACHE 있으면 재사용(GPU 불요)."""
    cache = {}
    if os.path.exists(PRED_CACHE):
        for line in open(PRED_CACHE, encoding="utf-8"):
            o = json.loads(line); cache[o["stem"]] = o["pred"]
        print(f"[cache] 예측 {len(cache):,}건 재사용", flush=True)
    todo = [r for r in recs if r["stem"] not in cache]
    if todo:
        if dec is None:
            raise SystemExit(f"예측 캐시에 {len(todo):,}건 없음 + 디코더 없음. --decode 로 실행 필요.")
        items = [(r["frames"], r) for r in todo]
        cf = open(PRED_CACHE, "a", encoding="utf-8")
        done = 0
        for bi, idxs in enumerate(batched(items)):
            wavs = [load_wav(items[i][1]["path"]) for i in idxs]
            preds = dec.decode(wavs)
            for j, i in enumerate(idxs):
                st = items[i][1]["stem"]; cache[st] = preds[j]
                cf.write(json.dumps({"stem": st, "pred": preds[j]}) + "\n"); done += 1
            if bi % 200 == 0:
                cf.flush(); print(f"  [decode] {done:,}/{len(todo):,}", flush=True)
        cf.close()
    for r in recs:
        r["pred"] = cache.get(r["stem"], [])
    return recs


def score(recs):
    """위치단위 채점. canonical 각 위치를 gold(분절오류)·pred(sub/del)로 이진표시 → 혼동행렬."""
    pos_pred, pos_gold = [], []          # 전체 위치 스트림(prf/AP 입력)
    per_err = per_len = 0                # gold PER(정답음소열 대비 인식 편집거리)
    tp_diag_ok = tp_total = 0            # 진단(유형) 정확도
    by_type = collections.defaultdict(lambda: {"gold": 0, "hit": 0, "diag_ok": 0})  # 유형별 recall·진단
    by_L1 = collections.defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
    supra_gold = supra_hit = 0           # 초분절(분절F1 제외, 참고집계)
    n_pos_none = n_pos_oob = 0
    import editdistance

    for r in recs:
        T, H = r["target"], r["pred"]
        per_err += editdistance.eval(H, T); per_len += len(T)
        ops = _nw(T, H)
        pred_seg = {op["tpos"] for op in ops if op["kind"] in ("sub", "del")}
        op_at = {op["tpos"]: op for op in ops if op["kind"] in ("sub", "del", "match")}

        # gold 위치 → 분절/초분절 분리
        gold_seg = {}   # tpos -> gold err_type
        for p, t in zip(r["err_pos"], r["err_types"]):
            if t in SUPRASEG_FLUENCY:
                supra_gold += 1
                # 초분절은 위치정렬 대상 아님 — 인식기가 그 위치에서 sub/del 냈으면 '탐지'로 참고집계
                if p is not None and 0 <= p < len(T) and p in pred_seg:
                    supra_hit += 1
                continue
            if p is None:
                n_pos_none += 1; continue
            if not (0 <= p < len(T)):
                n_pos_oob += 1; continue
            gold_seg[p] = t

        L1 = r["L1"]
        for i in range(len(T)):
            g = 1 if i in gold_seg else 0
            pr = 1 if i in pred_seg else 0
            pos_gold.append(g); pos_pred.append(pr)
            cell = ("tp" if (g and pr) else "fp" if (pr and not g)
                    else "fn" if (g and not pr) else "tn")
            by_L1[L1][cell] += 1
            if g:
                gt = gold_seg[i]; by_type[gt]["gold"] += 1
                if pr:  # TP 위치 → 진단(유형) 비교
                    by_type[gt]["hit"] += 1; tp_total += 1
                    op = op_at.get(i)
                    if op is not None:
                        op2 = {"kind": op["kind"], "target": op["target"],
                               "actual": op["actual"],
                               "target_next": T[i + 1] if i + 1 < len(T) else None}
                        pred_type = classify_op(op2)[0]
                        if pred_type == gt:
                            tp_diag_ok += 1; by_type[gt]["diag_ok"] += 1

    pos_pred = np.array(pos_pred, float); pos_gold = np.array(pos_gold, float)
    m = prf_at_threshold(pos_pred, pos_gold, 0.5)   # 하드결정(임계값 불요) — s5 함수 그대로
    ap = average_precision(pos_pred, pos_gold)
    diag_acc = tp_diag_ok / tp_total if tp_total else 0.0

    per_type = {t: {"support": v["gold"], "recall": round(v["hit"] / v["gold"], 4) if v["gold"] else 0.0,
                    "diag_acc": round(v["diag_ok"] / v["hit"], 4) if v["hit"] else 0.0}
                for t, v in sorted(by_type.items(), key=lambda kv: -kv[1]["gold"])}
    per_L1 = {}
    for L1, c in by_L1.items():
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        P = tp / (tp + fp) if tp + fp else 0.0; R = tp / (tp + fn) if tp + fn else 0.0
        per_L1[L1] = {"F1": round(2 * P * R / (P + R), 4) if P + R else 0.0,
                      "P": round(P, 4), "R": round(R, 4), "n_err_pos": tp + fn}
    per_L1 = dict(sorted(per_L1.items(), key=lambda kv: -kv[1]["n_err_pos"]))

    return {
        "detection": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()},
        "diagnosis": {"type_acc_on_TP": round(diag_acc, 4), "DER": round(1 - diag_acc, 4),
                      "n_TP": tp_total},
        "AP_position": round(ap, 4),
        "gold_per": round(per_err / max(per_len, 1) * 100, 3),
        "n_utt": len(recs), "n_positions": int(len(pos_gold)),
        "n_err_positions_seg": int(pos_gold.sum()),
        "err_pos_dropped": {"none": n_pos_none, "out_of_range": n_pos_oob},
        "suprasegmental": {"gold": supra_gold, "detected_at_pos": supra_hit,
                           "note": "음소MDD 판정불가 — 분절 F1에서 제외, 참고용"},
        "by_error_type": per_type,
        "by_L1_top": dict(list(per_L1.items())[:15]),
    }


def main():
    decode = "--no-decode" not in sys.argv   # 캐시만으로 채점하려면 --no-decode
    recs = load_gold()
    dec = Decoder() if decode else None
    recs = decode_all(dec, recs)
    res = score(recs)
    json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=2)
    d = res["detection"]
    print("\n" + "=" * 60)
    print(f"[위치단위 MDD 탐지]  F1={d['F1']:.4f}  P={d['P']:.4f}  R={d['R']:.4f}")
    print(f"                     FRR(과지적)={d['FRR']:.4f}  FAR(미검출)={d['FAR']:.4f}")
    print(f"[진단(유형)]         정확도={res['diagnosis']['type_acc_on_TP']:.4f}  "
          f"DER={res['diagnosis']['DER']:.4f}  (TP {res['diagnosis']['n_TP']:,}위치)")
    print(f"[위치 AP]            {res['AP_position']:.4f}")
    print(f"[gold PER]           {res['gold_per']:.2f}%   위치 {res['n_positions']:,} "
          f"(오류위치 {res['n_err_positions_seg']:,})")
    print(f"[초분절]             gold {res['suprasegmental']['gold']:,} (분절F1 제외)")
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
