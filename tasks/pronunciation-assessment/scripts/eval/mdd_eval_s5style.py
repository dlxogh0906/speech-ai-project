# -*- coding: utf-8 -*-
"""s5 방식(발화 × 오류유형) 채점 — 인식기 출력 기준.

발표자료: "인식기 출력을 s5 하네스에 넣어 채점". s5는 (발화,유형) 단위로
'이 발화에 유형 T가 있나'를 채점(위치 무관). 여기선 예측자 = 인식기:
  · gold_types  = 전문가 error_tags 유형집합(CORE)
  · pred_types  = 인식기 출력을 canonical에 정렬 → sub/del op → classify_op 유형집합
→ 유형별 P/R/F1 + micro/macro. 041 캐시(예측) 재사용, 재디코드 없음.
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists(): _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import MDD_OUTPUTS

import sys, json, argparse, collections
import numpy as np
from mdd.l1prior.s0_symbol_map import jamo_seq_to_kog2p
from mdd.l1prior.s0_taxonomy import classify_op, CORE_TAGS, SUPRASEG_FLUENCY
from mdd.mdd_eval_pos import _nw


def pred_types_from(canon_jamo, hyp_jamo):
    """canon↔hyp(자모) → KoG2P 정렬 → sub/del op classify → 유형집합."""
    ck = [x[0] for x in jamo_seq_to_kog2p([j if j else "?" for j in canon_jamo])]
    hk = [x[0] for x in jamo_seq_to_kog2p([j if j else "?" for j in hyp_jamo])]
    types = set()
    ops = _nw(ck, hk)
    for i, op in enumerate(ops):
        if op["kind"] not in ("sub", "del", "ins"):
            continue
        tpos = op["tpos"]
        tn = ck[tpos + 1] if (op["kind"] != "ins" and tpos + 1 < len(ck)) else None
        t = classify_op({"kind": op["kind"], "target": op["target"],
                         "actual": op["actual"], "target_next": tn})[0]
        types.add(t)
    return types


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=f"{MDD_OUTPUTS}/eval_official2_041_preds.jsonl")
    a = ap.parse_args()

    per = {t: {"tp": 0, "fp": 0, "fn": 0, "support": 0} for t in CORE_TAGS}
    n = 0
    for line in open(a.cache, encoding="utf-8"):
        d = json.loads(line)
        canon, hyp = d["canon"], d["hyp"]
        if not canon:
            continue
        n += 1
        gold = {t for t in d["etypes"] if t in CORE_TAGS}
        pred = pred_types_from(canon, hyp) & set(CORE_TAGS)
        for t in CORE_TAGS:
            g = t in gold; p = t in pred
            if g: per[t]["support"] += 1
            if g and p: per[t]["tp"] += 1
            elif p and not g: per[t]["fp"] += 1
            elif g and not p: per[t]["fn"] += 1

    print(f"[s5 방식 · 발화×유형] {n:,} 발화\n")
    print(f"{'유형':<10}{'support':>8}{'P':>7}{'R':>7}{'F1':>7}")
    TP = FP = FN = 0; f1s = []
    for t in CORE_TAGS:
        c = per[t]; tp, fp, fn = c["tp"], c["fp"], c["fn"]
        P = tp / (tp + fp) if tp + fp else 0.0
        R = tp / (tp + fn) if tp + fn else 0.0
        F1 = 2 * P * R / (P + R) if P + R else 0.0
        print(f"{t:<10}{c['support']:>8}{P:>7.3f}{R:>7.3f}{F1:>7.3f}")
        TP += tp; FP += fp; FN += fn
        if c["support"] > 0: f1s.append(F1)
    P = TP / (TP + FP) if TP + FP else 0.0
    R = TP / (TP + FN) if TP + FN else 0.0
    micro = 2 * P * R / (P + R) if P + R else 0.0
    macro = float(np.mean(f1s)) if f1s else 0.0
    print(f"\n{'micro':<10}{'':>8}{P:>7.3f}{R:>7.3f}{micro:>7.3f}")
    print(f"{'macro-F1':<10}{'':>8}{'':>7}{'':>7}{macro:>7.3f}")


if __name__ == "__main__":
    main()
