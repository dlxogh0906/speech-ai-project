# -*- coding: utf-8 -*-
"""record 잣대(약라벨)로 현재 모델을 채점 — 참고 지표.

정답 = record의 actual_phones(맞춤법 전사→KoG2P), 기준 = target_phones(프롬프트→KoG2P).
공식 MDD 채점법(3자 정렬 + weighted F1)을 그대로 쓰되 정답지만 record로 바꾼 것.

⚠️ 이 수치는 헤드라인이 아니다. record는 발음 오류가 라벨에서 지워진 약라벨이라,
   모델이 정직하게 전사할수록 감점되는 반순환 잣대다. 보고용 대표 수치는 TAG F1.
   구 record 학습 모델(0.80) 대비 우위를 보이는 보조 근거로만 사용할 것.

입력: 예측 캐시(jsonl, {stem, pred}) — GPU 불필요.
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists(): _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.fairseq_guard import ensure_fairseq
ensure_fairseq()
from shared.paths import MDD_ANALYSIS, MDD_OUTPUTS

import sys, json, argparse
import numpy as np
import editdistance
from sklearn.metrics import f1_score

from mdd.jamo_to_kog2p import INV                      # KoG2P 심볼 → 자모
from mdd.mdd_reproduce_official import align, fixup    # 공식 3자 정렬

ap = argparse.ArgumentParser()
ap.add_argument("--preds", default=f"{MDD_OUTPUTS}/word_skip_preds_lar.jsonl")
ap.add_argument("--records", default=f"{MDD_ANALYSIS}/040_041_042_043_both/records.jsonl")
ap.add_argument("--out", default=f"{MDD_OUTPUTS}/eval_record_f1.json")
A = ap.parse_args()

H = lambda c: 0xAC00 <= ord(c) <= 0xD7A3
clean = lambda s: "".join(c for c in s if H(c) or c == " ").strip()


def score(canon, manual, pred):
    """공식 방식: 표준음·전사자·인식기 3자 정렬 → 위치별 이진 라벨."""
    n1, n2, n3 = fixup(*align("".join(canon), "".join(manual), "".join(pred) or "-"))
    L = min(len(n1), len(n2), len(n3))
    return ([int(n1[k] != n2[k]) for k in range(L)],
            [int(n1[k] != n3[k]) for k in range(L)])


preds = {}
for line in open(A.preds, encoding="utf-8"):
    d = json.loads(line); preds[d["stem"]] = d["pred"]
print(f"[preds] {len(preds):,}", flush=True)

# clean = 전사가 프롬프트와 완전히 일치(=LAR 적응 학습에 쓰인 발화) / err = 그 외(비학습)
X = {"all": [], "clean": [], "err": []}
Y = {"all": [], "clean": [], "err": []}
per = {"all": [], "clean": [], "err": []}
n = 0
for line in open(A.records, encoding="utf-8"):
    r = json.loads(line); stem = r["file"][:-5]
    if stem not in preds:
        continue
    tgt = [INV[p] for p in r.get("target_phones", []) if p in INV]   # 표준음(canonical)
    act = [INV[p] for p in r.get("actual_phones", []) if p in INV]   # record 라벨(약)
    if not tgt or not act:
        continue
    g = "clean" if clean(r["prompt"]).replace(" ", "") == clean(r["orthographic_clean"]).replace(" ", "") else "err"
    p = preds[stem]
    per[g].append(editdistance.eval(p, act) / max(1, len(act))); per["all"].append(per[g][-1])
    x, y = score(tgt, act, p)
    X[g] += x; Y[g] += y; X["all"] += x; Y["all"] += y
    n += 1
    if n % 50000 == 0:
        print(f"  {n:,}", flush=True)

res = {"n": n, "preds_file": A.preds}
print(f"\n========== record 잣대 채점 (n={n:,}) ==========")
for g, name in (("all", "전체"), ("clean", "무오류(적응학습에 사용)"), ("err", "오류포함(비학습·공정)")):
    f1 = float(f1_score(X[g], Y[g], average="weighted"))
    res[g] = {"n": len(per[g]), "PER": round(float(np.mean(per[g])), 4), "f1_weighted": round(f1, 4)}
    print(f"{name:22} n={len(per[g]):7,}  PER={np.mean(per[g]):.1%}  record식 F1={f1:.4f}")
json.dump(res, open(A.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"[saved] {A.out}")
