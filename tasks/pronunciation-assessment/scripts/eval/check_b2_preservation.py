# -*- coding: utf-8 -*-
"""B-2 검증: 오발음이 '정답으로 복원되지 않고' 그대로 전사되는가 + PER.

MDD 모델의 최대 실패 양식은 **자동 교정**이다. 학습이 표준발음 쪽으로 쏠리면 모델은
학습자가 틀리게 말해도 맞는 음소를 뱉는다. 그러면 F1은 정밀도만 높고 오류를 못 잡는다.

세 가지 PER을 나란히 본다 (모두 71음소 공간, 코퍼스 단위 = 총편집거리/총길이):
  PER(pred‖TAG)    모델 vs 전문가가 실제로 들은 것   ← 낮을수록 좋음(전사 정확도)
  PER(pred‖canon)  모델 vs 표준발음                 ← 학습자가 틀린 만큼 벌어져야 정상
  PER(TAG‖canon)   전문가 vs 표준발음               ← 실제 오발음의 양 (기준값)

  보존율 = PER(pred‖canon) / PER(TAG‖canon)
     1.0 ≈ 사람이 벗어난 만큼 모델도 벗어남  → 오발음 보존 ✅
     0에 가까움 = 모델이 표준발음으로 되돌림 → 자동 교정 ❌
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.fairseq_guard import ensure_fairseq
ensure_fairseq()
from shared.paths import MDD_OUTPUTS

import sys, os, json, collections, argparse
import edlib

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", default=f"{MDD_OUTPUTS}/manifest_tag_all_lar")
ap.add_argument("--preds", default=f"{MDD_OUTPUTS}/eval_tag_all_lar_preds.jsonl")
ap.add_argument("--out", default=f"{MDD_OUTPUTS}/check_b2_preservation.json")
A = ap.parse_args()
M = A.manifest

from mdd.jamo_to_kog2p import jamo_to_phones, ALL_PHONES, VOWEL, ONSET, CODA
JAMO = set(VOWEL) | set(ONSET) | set(CODA)
ENC = {p: chr(0xE000 + i) for i, p in enumerate(ALL_PHONES)}
enc = lambda toks: "".join(ENC[p] for p in jamo_to_phones(toks))

rel = [l.split("\t") for l in open(f"{M}/test.tsv").read().splitlines()[1:]]
canon = [l.split() for l in open(f"{M}/test.canon", encoding="utf-8").read().splitlines()]
manual = [l.split() for l in open(f"{M}/test.ltr", encoding="utf-8").read().splitlines()]
N = len(rel)
P = [None] * N
for line in open(A.preds, encoding="utf-8"):
    d = json.loads(line); P[d["i"]] = d["pred"]
print(f"[test] {N:,}발화 · 예측 {sum(p is not None for p in P):,}\n")

ed = lambda a, b: edlib.align(a, b, mode="NW", task="distance")["editDistance"] if (a or b) else 0
acc = collections.defaultdict(lambda: [0, 0])     # key → [편집거리합, 길이합]
per_utt = []
for i in range(N):
    p = enc([t for t in (P[i] or []) if t in JAMO])
    c, m = enc(canon[i]), enc(manual[i])
    if not c or not m:
        continue
    d_pm, d_pc, d_mc = ed(p, m), ed(p, c), ed(m, c)
    grp = "err" if d_mc > 0 else "clean"           # 학습자가 실제로 틀린 발화 / 표준대로 말한 발화
    for k, (d, L) in (("pm", (d_pm, len(m))), ("pc", (d_pc, len(c))), ("mc", (d_mc, len(c)))):
        acc[("all", k)][0] += d; acc[("all", k)][1] += L
        acc[(grp, k)][0] += d; acc[(grp, k)][1] += L
    per_utt.append((grp, d_pc / max(1, len(c)), d_mc / max(1, len(c))))

res = {}
print(f"{'구간':22}{'PER(pred‖TAG)':>16}{'PER(pred‖canon)':>18}{'PER(TAG‖canon)':>17}{'보존율':>10}")
for grp, label in (("all", "전체"), ("err", "실제 오발음 있는 발화"), ("clean", "표준대로 말한 발화")):
    if not acc[(grp, "pm")][1]:
        continue
    pm, pc, mc = (acc[(grp, k)][0] / acc[(grp, k)][1] for k in ("pm", "pc", "mc"))
    keep = pc / mc if mc > 0 else float("nan")
    res[grp] = {"per_pred_tag": round(pm, 4), "per_pred_canon": round(pc, 4),
                "per_tag_canon": round(mc, 4), "preservation": round(keep, 3) if mc > 0 else None,
                "n_positions": acc[(grp, "pm")][1]}
    print(f"{label:22}{pm:>16.4f}{pc:>18.4f}{mc:>17.4f}{(f'{keep:.3f}' if mc > 0 else '—'):>10}")

ne = sum(1 for g, _, _ in per_utt if g == "err")
print(f"\n발화 구성: 오발음 있음 {ne:,} ({ne/len(per_utt):.1%}) · 표준대로 {len(per_utt)-ne:,}")
print(f"전사 정확도: PER(pred‖TAG) = {res['all']['per_pred_tag']:.4f} "
      f"→ 전문가가 들은 음소의 {1-res['all']['per_pred_tag']:.1%}를 그대로 받아씀")
k = res["err"]["preservation"]
print(f"오발음 보존율(오발음 발화) = {k:.3f}  "
      f"→ {'✅ 보존 (자동 교정 아님)' if k >= 0.85 else ('⚠️ 부분 교정' if k >= 0.6 else '❌ 자동 교정 의심')}")
json.dump(res, open(A.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"[saved] {A.out}")
