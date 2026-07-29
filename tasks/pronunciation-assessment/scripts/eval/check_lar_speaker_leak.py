# -*- coding: utf-8 -*-
"""LAR 도메인 적응 때 섞어 넣은 무오류 LAR 3.9만 중 일부가 **test 화자의 발화**였다.
그 화자들의 TAG 성적이 나머지보다 부풀었는지 확인한다.

누출 형태: train 에 `00025-...-LAR010-...`, test 에 `00025-...-RS025-...`
  → 발화는 서로 다르지만 **화자가 같다**. 발음 습관을 미리 본 셈이므로 유리할 수 있다.

대조군이 깔끔하다: 같은 test, 같은 모델, 같은 채점 코드. 화자 집합만 둘로 가른다.
tag_all(LAR 미포함=누출 없음)과 tag_all_lar(누출 발생) 둘 다 재면
'누출 때문인지' vs '원래 그 화자들이 쉬운지'까지 분리된다.
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

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", default=f"{MDD_OUTPUTS}/manifest_tag_all_lar")
ap.add_argument("--out", default=f"{MDD_OUTPUTS}/check_lar_leak.json")
A = ap.parse_args()
M = A.manifest
MODELS = [("tag_all", f"{MDD_OUTPUTS}/eval_tag_all_preds.jsonl", "LAR 미포함 (누출 없음)"),
          ("tag_all_lar", f"{MDD_OUTPUTS}/eval_tag_all_lar_preds.jsonl", "LAR 포함 (누출 발생)")]

rel = [l.split("\t") for l in open(f"{M}/test.tsv").read().splitlines()[1:]]
canon = [l.split() for l in open(f"{M}/test.canon", encoding="utf-8").read().splitlines()]
manual = [l.split() for l in open(f"{M}/test.ltr", encoding="utf-8").read().splitlines()]
N = len(rel)

tr_spk = collections.defaultdict(set)                 # 화자 → train 에 있던 과제종류
for l in open(f"{M}/train.tsv").read().splitlines()[1:]:
    b = os.path.basename(l.split("\t")[0])[:-4].split("-")
    if len(b) > 5:
        tr_spk[b[0]].add(b[5][:2])                    # LA(R) / RS / RW
leaked = {s for s, k in tr_spk.items() if k}          # train 에 등장한 모든 화자

te_spk = [os.path.basename(r[0])[:-4].split("-")[0] for r in rel]
is_leak = [s in leaked for s in te_spk]
lk = {s for s in te_spk if s in leaked}
print(f"[test] 발화 {N:,} · 화자 {len(set(te_spk)):,}")
print(f"  train 에도 있는 화자 {len(lk):,}명 → test 발화 {sum(is_leak):,}건 ({sum(is_leak)/N:.1%})")
kinds = collections.Counter(k for s in lk for k in tr_spk[s])
print(f"  그 화자들이 train 에서 가진 과제: {dict(kinds)}   (LA=LAR 문장낭독)\n")

from mdd.mdd_reproduce_official import align, fixup
from mdd.jamo_to_kog2p import VOWEL, ONSET, CODA
from sklearn.metrics import f1_score
JAMO = set(VOWEL) | set(ONSET) | set(CODA)


def score(c_str, m_str, p_str):
    n1, n2, n3 = fixup(*align(c_str, m_str, p_str or "-"))
    L = min(len(n1), len(n2), len(n3))
    return ([int(n1[k] != n2[k]) for k in range(L)], [int(n1[k] != n3[k]) for k in range(L)])


res = {}
print(f"{'모델':14}{'비고':22}{'누출 화자':>11}{'깨끗한 화자':>13}{'차이':>9}{'전체':>10}")
for name, preds, note in MODELS:
    if not os.path.exists(preds) or sum(1 for _ in open(preds)) != N:
        print(f"{name:14}{note:22}  예측 캐시 없음/줄수 불일치 — 건너뜀"); continue
    P = [None] * N
    for line in open(preds, encoding="utf-8"):
        d = json.loads(line); P[d["i"]] = d["pred"]
    G = {True: ([], []), False: ([], [])}
    XA, YA = [], []
    for i in range(N):
        p = [t for t in (P[i] or []) if t in JAMO]
        x, y = score("".join(canon[i]), "".join(manual[i]), "".join(p))
        G[is_leak[i]][0].extend(x); G[is_leak[i]][1].extend(y)
        XA += x; YA += y
    f_lk = f1_score(G[True][0], G[True][1], average="weighted")
    f_cl = f1_score(G[False][0], G[False][1], average="weighted")
    f_all = f1_score(XA, YA, average="weighted")
    res[name] = {"leaked": round(f_lk, 4), "clean": round(f_cl, 4),
                 "gap": round(f_lk - f_cl, 4), "all": round(f_all, 4)}
    print(f"{name:14}{note:22}{f_lk:>11.4f}{f_cl:>13.4f}{f_lk-f_cl:>+9.4f}{f_all:>10.4f}")

if len(res) == 2:
    d = res["tag_all_lar"]["gap"] - res["tag_all"]["gap"]
    print(f"\n누출로 인한 순수 이득 = (LAR포함 격차) − (LAR미포함 격차) = "
          f"{res['tag_all_lar']['gap']:+.4f} − {res['tag_all']['gap']:+.4f} = **{d:+.4f}**")
    print("  → tag_all 은 그 화자들의 LAR을 본 적 없으므로, 거기서 이미 벌어진 격차는 "
          "'원래 쉬운 화자'라는 뜻이고 누출과 무관하다.")
json.dump({"n_test": N, "n_leaked_spk": len(lk), "n_leaked_utt": sum(is_leak), "models": res},
          open(A.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"[saved] {A.out}")
