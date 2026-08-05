# -*- coding: utf-8 -*-
"""A안 검증: TAG combined test(10.4만) 디코드 → 같은 예측을
① 자모 공간(기존 0.9607 방식) ② KoG2P 71음소 변환 후 공간 — 두 번 채점해 F1 비교.

예측은 eval_tag_combined_preds.jsonl 로 저장(이전 평가는 미저장이라 재디코드) →
다음부터 재채점은 CPU만으로 가능. 이미 파일이 있고 줄 수가 맞으면 디코드 스킵.
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
from shared.paths import DATA_ROOT, MDD_OUTPUTS

import sys, os, json, time, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", default=f"{MDD_OUTPUTS}/manifest_tag_combined")
ap.add_argument("--ckpt", default=f"{DATA_ROOT}/mdd_ckpt_tag_combined/checkpoint_best.pt")
ap.add_argument("--preds", default=f"{MDD_OUTPUTS}/eval_tag_combined_preds.jsonl")
ap.add_argument("--out", default=f"{MDD_OUTPUTS}/eval_tag71_combined.json")
ap.add_argument("--min_group", type=int, default=1000, help="언어코드별 F1 표기 최소 발화 수")
A = ap.parse_args()
M, CKPT, PREDS, OUT = A.manifest, A.ckpt, A.preds, A.out

rel = [l.split("\t") for l in open(f"{M}/test.tsv").read().splitlines()[1:]]
canon = [l.split() for l in open(f"{M}/test.canon", encoding="utf-8").read().splitlines()]
manual = [l.split() for l in open(f"{M}/test.ltr", encoding="utf-8").read().splitlines()]
N = len(rel); assert N == len(canon) == len(manual), (N, len(canon), len(manual))

# ---------- 1) 예측 확보 ----------
def have_preds():
    return os.path.exists(PREDS) and sum(1 for _ in open(PREDS)) == N

if have_preds():
    P = [None] * N
    for line in open(PREDS, encoding="utf-8"):
        d = json.loads(line); P[d["i"]] = d["pred"]
    print(f"[preds] loaded {N:,} from {PREDS}", flush=True)
else:
    # 최상위 `mdd_eval` 모듈은 없다 — 패키지 모듈은 `mdd.mdd_eval` 이다.
    # (2026-07-30 감사에서 지적된 결함. 캐시가 늘 있어서 이 분기가 한 번도 안 돌았다)
    # Decoder.__init__ 이 모듈 전역 CKPT 를 읽으므로 인스턴스 생성 전에 덮어쓴다.
    from mdd import mdd_eval
    mdd_eval.CKPT = CKPT
    from mdd.mdd_eval import Decoder, load_wav, batched
    dec = Decoder()
    P = [None] * N
    items = [(int(rel[i][1]), i) for i in range(N)]   # (frames, orig_idx) — batched()는 [0]로 정렬
    t0 = time.time(); done = 0
    with open(PREDS + ".tmp", "w", encoding="utf-8") as f:
        for bi, idxs in enumerate(batched(items)):
            orig = [items[j][1] for j in idxs]
            outs = dec.decode([load_wav("/" + rel[o][0]) for o in orig])
            for o, p in zip(orig, outs):
                P[o] = p
                f.write(json.dumps({"i": o, "pred": p}, ensure_ascii=False) + "\n")
            done += len(orig)
            if bi % 40 == 0:
                print(f"  decode {done:,}/{N:,} ({done/max(1e-9,time.time()-t0):.0f}/s)", flush=True)
    os.replace(PREDS + ".tmp", PREDS)
    print(f"[preds] decoded+saved {N:,} in {time.time()-t0:.0f}s", flush=True)

# ---------- 2) 이중 채점 ----------
from mdd.mdd_reproduce_official import align, fixup
from mdd.jamo_to_kog2p import jamo_to_phones, ALL_PHONES, VOWEL, ONSET, CODA
from sklearn.metrics import f1_score

JAMO = set(VOWEL) | set(ONSET) | set(CODA)
ENC = {p: chr(0xE000 + i) for i, p in enumerate(ALL_PHONES)}   # 음소 1개 = 유니크 1글자 (edlib용)

def enc71(toks):
    return "".join(ENC[p] for p in jamo_to_phones(toks))

def score(c_str, m_str, p_str):
    n1, n2, n3 = fixup(*align(c_str, m_str, p_str or "-"))
    L = min(len(n1), len(n2), len(n3))
    x = [int(n1[k] != n2[k]) for k in range(L)]   # true error (canonical vs 전사자)
    y = [int(n1[k] != n3[k]) for k in range(L)]   # pred error (canonical vs 인식기)
    return x, y

import collections
codes = []
for r in rel:                                   # basename 4번째 필드 = 화자 L1 코드 (예: EN/ZH/JA)
    b = os.path.basename(r[0]).split("-")
    codes.append(b[3] if len(b) > 3 else "??")

Xj, Yj, X7, Y7 = [], [], [], []
Xg = collections.defaultdict(list); Yg = collections.defaultdict(list); Ng = collections.Counter()
diff_utts = 0; specials = 0
t0 = time.time()
for i in range(N):
    p_raw = P[i] or []
    p = [t for t in p_raw if t in JAMO]
    if len(p) != len(p_raw):
        specials += 1
    xj, yj = score("".join(canon[i]), "".join(manual[i]), "".join(p))
    x7, y7 = score(enc71(canon[i]), enc71(manual[i]), enc71(p))
    tp = lambda x, y: (sum(a & b for a, b in zip(x, y)), sum(y) - sum(a & b for a, b in zip(x, y)), sum(x))
    if tp(xj, yj) != tp(x7, y7):
        diff_utts += 1
    Xj += xj; Yj += yj; X7 += x7; Y7 += y7
    Xg[codes[i]] += xj; Yg[codes[i]] += yj; Ng[codes[i]] += 1
    if i % 20000 == 0:
        print(f"  score {i:,}/{N:,}", flush=True)

def stats(X, Y):
    TP = sum(a & b for a, b in zip(X, Y)); FP = sum(Y) - TP; FN = sum(X) - TP
    pr = TP / max(1, TP + FP); rc = TP / max(1, TP + FN)
    return {"f1_weighted": round(f1_score(X, Y, average="weighted"), 4),
            "err_f1": round(2 * pr * rc / max(1e-9, pr + rc), 4),
            "precision": round(pr, 4), "recall": round(rc, 4),
            "positions": len(X), "true_err": sum(X), "pred_err": sum(Y)}

sj, s7 = stats(Xj, Yj), stats(X7, Y7)
groups = {}
for c in sorted(Ng, key=lambda k: -Ng[k]):
    if Ng[c] >= A.min_group:
        groups[c] = {"n_utts": Ng[c], "f1_weighted": round(f1_score(Xg[c], Yg[c], average="weighted"), 4)}
res = {"n": N, "ckpt": CKPT, "jamo": sj, "kog2p71": s7, "by_lang": groups,
       "utts_scoring_differs": diff_utts, "preds_with_special_tokens": specials}
json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

print(f"\n================ A안 검증 결과 (n={N:,}) ================")
print(f"{'':14}{'자모(기존)':>14}{'KoG2P 71':>14}")
print(f"{'weighted F1':14}{sj['f1_weighted']:>14}{s7['f1_weighted']:>14}")
print(f"{'error-F1':14}{sj['err_f1']:>14}{s7['err_f1']:>14}")
print(f"{'precision':14}{sj['precision']:>14}{s7['precision']:>14}")
print(f"{'recall':14}{sj['recall']:>14}{s7['recall']:>14}")
print(f"{'positions':14}{sj['positions']:>14,}{s7['positions']:>14,}")
print(f"채점 결과 달라진 발화: {diff_utts:,}/{N:,} | special 토큰 포함 예측: {specials:,}")
if groups:
    print(f"\n── 언어권별 weighted F1 (n≥{A.min_group}) ──")
    for c, g in groups.items():
        print(f"  {c:4} n={g['n_utts']:>7,}  F1={g['f1_weighted']}")
print(f"[saved] {OUT} · preds {PREDS}")
