# -*- coding: utf-8 -*-
"""고정 정답지 채점 — 정답(true error)을 모델 예측과 무관하게 만든다.

기존 `mdd_eval_tag71.py` 의 문제
--------------------------------
`mdd_reproduce_official.align(canon, manual, pred)` 는 3자 정렬이다. 54~55행에서
canon↔manual 축과 canon↔pred 축을 **합쳐서** canonical 축을 넓히므로, 축의 길이와
그 위에서 계산되는 정답 `x = (canon != manual)` 이 **평가 대상 모델에 따라 달라진다.**
실측(2026-07-30 감사): 같은 test 17.8만 발화인데 채점 위치가 1,799,189~1,801,203,
정답 오류 개수가 74,428~74,471 로 모델마다 다르다. 고정 정답지라면 있을 수 없다.

이 스크립트의 채점 방식
-----------------------
축을 **canonical 음소 위치**로 고정한다 (길이 = len(canon), 모델과 무관).

    x[j] = 1  if  manual 이 canonical 위치 j 에서 canon 과 다르다   ← 2자 정렬(canon,manual)
    y[j] = 1  if  pred   이 canonical 위치 j 에서 canon 과 다르다   ← 2자 정렬(canon,pred)

x 는 데이터셋(TAG 라벨)만으로 정해지므로 어떤 모델을 평가하든 동일하다.
모델이 바뀌면 y 만 바뀐다. 이것이 "정답 만들 때 pred 를 뺀다" 의 구현이다.

알려진 한계
-----------
화자/모델이 **삽입**한 음소(canonical 에 대응 위치가 없는 것)는 축에서 빠진다.
기존 방식은 `fixup()` 으로 길이를 억지로 맞춰 삽입을 부분 반영했으나, 그 보정이
정답까지 흔드는 원인이었다. 여기서는 치환·삭제만 채점한다(canonical-anchored MDD).
따라서 이 수치는 기존 수치와 **직접 비교 대상이 아니라 나란히 놓고 보는 값**이다.

사용
----
    $PY tasks/pronunciation-assessment/scripts/eval/mdd_eval_fixedref.py \
        --manifest $MDD_OUTPUTS/manifest_tag_all_lar \
        --preds    $MDD_OUTPUTS/eval_tag_all_lar_preds.jsonl \
        --out      tasks/pronunciation-assessment/outputs/fixedref_tag_all_lar.json

예측 캐시만 읽으므로 **GPU 불필요**(CPU 수 분). 캐시가 없으면 실행하지 않는다.
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import MDD_OUTPUTS

import os, json, time, argparse, collections
import edlib
from sklearn.metrics import f1_score

from mdd.jamo_to_kog2p import jamo_to_phones, ALL_PHONES, VOWEL, ONSET, CODA

JAMO = set(VOWEL) | set(ONSET) | set(CODA)
ENC = {p: chr(0xE000 + i) for i, p in enumerate(ALL_PHONES)}   # 음소 1개 = 1글자 (edlib용)


def enc71(toks):
    return "".join(ENC[p] for p in jamo_to_phones(toks))


def diff_vs_canon(c_str, s_str):
    """canonical 각 위치에서 s 가 canon 과 다른가. 반환 길이 == len(c_str).

    s 의 삽입(canonical 에 '-' 가 오는 열)은 canonical 위치가 없으므로 건너뛴다.
    s 의 삭제는 s 쪽이 '-' 이므로 '다름'(1)으로 잡힌다.
    """
    if not c_str:
        return []
    if not s_str:
        return [1] * len(c_str)
    r = edlib.align(c_str, s_str, task="path")
    a = edlib.getNiceAlignment(r, c_str, s_str)
    cq, st = a["query_aligned"], a["target_aligned"]
    out = []
    for ch_c, ch_s in zip(cq, st):
        if ch_c == "-":          # s 의 삽입 → canonical 위치 없음
            continue
        out.append(int(ch_c != ch_s))
    assert len(out) == len(c_str), (len(out), len(c_str))
    return out


def stats(X, Y):
    TP = sum(a & b for a, b in zip(X, Y))
    FP = sum(Y) - TP
    FN = sum(X) - TP
    pr = TP / max(1, TP + FP)
    rc = TP / max(1, TP + FN)
    return {"f1_weighted": round(f1_score(X, Y, average="weighted"), 4),
            "err_f1": round(2 * pr * rc / max(1e-9, pr + rc), 4),
            "precision": round(pr, 4), "recall": round(rc, 4),
            "positions": len(X), "true_err": sum(X), "pred_err": sum(Y)}


def main():
    ap = argparse.ArgumentParser(description="고정 정답지(canonical 축) MDD 채점")
    ap.add_argument("--manifest", default=f"{MDD_OUTPUTS}/manifest_tag_all_lar")
    ap.add_argument("--preds", required=True, help="예측 캐시 jsonl (필수 — 디코드는 하지 않는다)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_group", type=int, default=1000)
    A = ap.parse_args()

    M = A.manifest
    rel = [l.split("\t") for l in open(f"{M}/test.tsv").read().splitlines()[1:]]
    canon = [l.split() for l in open(f"{M}/test.canon", encoding="utf-8").read().splitlines()]
    manual = [l.split() for l in open(f"{M}/test.ltr", encoding="utf-8").read().splitlines()]
    N = len(rel)
    assert N == len(canon) == len(manual), (N, len(canon), len(manual))

    if not os.path.exists(A.preds):
        raise SystemExit(f"예측 캐시가 없습니다: {A.preds}\n"
                         f"이 스크립트는 재채점 전용이라 디코드하지 않습니다. "
                         f"먼저 mdd_eval_tag71.py 로 캐시를 만드십시오.")
    n_lines = sum(1 for _ in open(A.preds))
    if n_lines != N:
        raise SystemExit(f"캐시 줄 수 불일치: preds {n_lines:,} vs manifest {N:,}\n"
                         f"--manifest 가 이 캐시와 짝이 맞는지 확인하십시오.")

    P = [None] * N
    for line in open(A.preds, encoding="utf-8"):
        d = json.loads(line)
        P[d["i"]] = d["pred"]
    print(f"[preds] {N:,} loaded from {os.path.basename(A.preds)}", flush=True)

    codes = []
    for r in rel:                      # basename 4번째 필드 = 화자 L1 코드
        b = os.path.basename(r[0]).split("-")
        codes.append(b[3] if len(b) > 3 else "??")

    Xj, Yj, X7, Y7 = [], [], [], []
    Xg = collections.defaultdict(list); Yg = collections.defaultdict(list)
    Ng = collections.Counter()
    specials = 0
    t0 = time.time()
    for i in range(N):
        p_raw = P[i] or []
        p = [t for t in p_raw if t in JAMO]
        if len(p) != len(p_raw):
            specials += 1

        cj, mj, pj = "".join(canon[i]), "".join(manual[i]), "".join(p)
        xj = diff_vs_canon(cj, mj)
        yj = diff_vs_canon(cj, pj)

        c7, m7, p7 = enc71(canon[i]), enc71(manual[i]), enc71(p)
        x7 = diff_vs_canon(c7, m7)
        y7 = diff_vs_canon(c7, p7)

        Xj += xj; Yj += yj; X7 += x7; Y7 += y7
        Xg[codes[i]] += xj; Yg[codes[i]] += yj; Ng[codes[i]] += 1
        if i % 20000 == 0:
            print(f"  score {i:,}/{N:,}", flush=True)

    sj, s7 = stats(Xj, Yj), stats(X7, Y7)
    groups = {}
    for c in sorted(Ng, key=lambda k: -Ng[k]):
        if Ng[c] >= A.min_group:
            groups[c] = {"n_utts": Ng[c],
                         "f1_weighted": round(f1_score(Xg[c], Yg[c], average="weighted"), 4)}

    res = {"_방식": "고정 정답지(canonical 축). 정답 x 는 pred 와 무관 — 모델이 바뀌어도 동일.",
           "_한계": "삽입은 채점하지 않는다(치환·삭제만). 기존 3자정렬 수치와 직접 비교 불가.",
           "n": N, "preds": A.preds, "manifest": M,
           "jamo": sj, "kog2p71": s7, "by_lang": groups,
           "preds_with_special_tokens": specials}
    os.makedirs(os.path.dirname(os.path.abspath(A.out)), exist_ok=True)
    json.dump(res, open(A.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print(f"\n===== 고정 정답지 채점 (n={N:,}, {time.time()-t0:.0f}s) =====")
    print(f"{'':14}{'자모':>12}{'KoG2P 71':>12}")
    for k in ("f1_weighted", "err_f1", "precision", "recall"):
        print(f"{k:14}{sj[k]:>12}{s7[k]:>12}")
    print(f"{'채점 위치':14}{sj['positions']:>12,}{s7['positions']:>12,}")
    print(f"{'정답(오류)':14}{sj['true_err']:>12,}{s7['true_err']:>12,}")
    print(f"→ {A.out}")


if __name__ == "__main__":
    main()
