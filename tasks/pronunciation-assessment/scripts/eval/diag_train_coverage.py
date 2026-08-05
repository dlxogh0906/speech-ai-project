# -*- coding: utf-8 -*-
"""근본원인 진단 — 학습셋이 각 오발음 대비를 몇 번이나 보여주는가.

묻는 것
-------
모델이 `ㅂ→ㅃ` 를 96% 놓친다. 두 가지 설명이 가능하다.

    (a) 결정 경계 문제  — 배웠는데 argmax 에서 진다      → 디코딩/임계값으로 고친다
    (b) 학습 신호 부족  — 애초에 볼 기회가 거의 없었다   → 데이터로만 고친다

(b) 라면 후처리로는 절대 못 고친다. 그래서 **학습셋(train)** 에서 canonical→realized
쌍이 각각 몇 번 나오는지 직접 센다. test 에서 잰 되돌림률과 대조하면 어느 쪽인지 갈린다.

세는 방법
---------
train 의 canon/ltr 을 2자 정렬해 canonical 위치마다 (표준음소, 실제발음) 쌍을 만든다.
평가 스크립트와 같은 축이라 test 결과와 바로 대조된다.

    $PY tasks/pronunciation-assessment/scripts/eval/diag_train_coverage.py \\
        --diag <diag_overcorrection_*.json> --out <파일.json>

GPU 불필요. train 155만 발화라 CPU 몇 분.
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import MDD_OUTPUTS

import os, json, argparse, collections
import edlib
from mdd.jamo_to_kog2p import VOWEL, ONSET, CODA

JAMO = set(VOWEL) | set(ONSET) | set(CODA)


def pairs_of(c_str, m_str):
    """canonical 위치별 (표준, 실제) 쌍. 삽입 열은 버린다(평가 축과 동일)."""
    if not c_str:
        return []
    if not m_str:
        return [(ch, "-") for ch in c_str]
    a = edlib.getNiceAlignment(edlib.align(c_str, m_str, task="path"), c_str, m_str)
    return [(cc, ss) for cc, ss in zip(a["query_aligned"], a["target_aligned"]) if cc != "-"]


def main():
    ap = argparse.ArgumentParser(description="학습셋의 오발음 대비 노출 횟수")
    ap.add_argument("--manifest", default=f"{MDD_OUTPUTS}/manifest_tag_all_lar")
    ap.add_argument("--split", default="train")
    ap.add_argument("--diag", help="test 되돌림률 파일(diag_overcorrection_*.json). 주면 대조표를 낸다")
    ap.add_argument("--out", required=True)
    A = ap.parse_args()

    M = A.manifest
    canon = [l.split() for l in open(f"{M}/{A.split}.canon", encoding="utf-8").read().splitlines()]
    manual = [l.split() for l in open(f"{M}/{A.split}.ltr", encoding="utf-8").read().splitlines()]
    assert len(canon) == len(manual), (len(canon), len(manual))
    N = len(canon)

    pair = collections.Counter()          # (표준, 실제) → 횟수
    seen_canon = collections.Counter()     # 표준음소 총 등장
    n_err_tok = n_tok = 0
    n_utt_with_err = 0

    for i in range(N):
        c = "".join(t for t in canon[i] if t in JAMO)
        m = "".join(t for t in manual[i] if t in JAMO)
        if not c:
            continue
        has = False
        for cc, ss in pairs_of(c, m):
            n_tok += 1
            seen_canon[cc] += 1
            if cc != ss:
                pair[(cc, ss)] += 1
                n_err_tok += 1
                has = True
        if has:
            n_utt_with_err += 1
        if i % 200000 == 0:
            print(f"  {i:,}/{N:,}", flush=True)

    res = {
        "_설명": "학습셋이 각 오발음 대비(표준→실제)를 몇 번 보여주는지 센다.",
        "split": A.split, "n_utt": N,
        "요약": {
            "총_음소위치": n_tok, "오류_음소위치": n_err_tok,
            "오류_비율": round(n_err_tok / max(1, n_tok), 4),
            "오류포함_발화": n_utt_with_err,
            "오류포함_발화비율": round(n_utt_with_err / max(1, N), 4)},
        "대비별_노출횟수": [{"표준": k[0], "실제": k[1], "n": v}
                       for k, v in pair.most_common(60)],
    }

    # test 되돌림률과 대조
    if A.diag and os.path.exists(A.diag):
        d = json.load(open(A.diag, encoding="utf-8"))
        rows = []
        for key in ("되돌림률_높은_오발음유형", "되돌림률_낮은_오발음유형"):
            for r in d.get(key, []):
                k = (r["표준"], r["실제발음"])
                rows.append({"표준": k[0], "실제": k[1],
                             "train_노출": pair.get(k, 0),
                             "train_대비_표준음소총수": seen_canon.get(k[0], 0),
                             "노출비율_ppm": round(pair.get(k, 0) / max(1, seen_canon.get(k[0], 1)) * 1e6),
                             "test_n": r["n"], "되돌림률": r["되돌림률"]})
        rows.sort(key=lambda x: -x["되돌림률"])
        res["대조표_되돌림률_vs_학습노출"] = rows

        # 상관
        try:
            import statistics
            xs = [r["train_노출"] for r in rows]
            ys = [r["되돌림률"] for r in rows]
            if len(xs) > 3:
                mx, my = statistics.mean(xs), statistics.mean(ys)
                cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
                sx = sum((a - mx) ** 2 for a in xs) ** .5
                sy = sum((b - my) ** 2 for b in ys) ** .5
                res["상관_학습노출_vs_되돌림률"] = round(cov / max(1e-9, sx * sy), 4)
        except Exception:
            pass

    os.makedirs(os.path.dirname(os.path.abspath(A.out)), exist_ok=True)
    json.dump(res, open(A.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    s = res["요약"]
    print(f"\n===== 학습셋 노출 진단 ({A.split}, n={N:,}) =====")
    print(f"  총 음소 위치 {s['총_음소위치']:,} 중 오류 {s['오류_음소위치']:,} ({s['오류_비율']*100:.2f}%)")
    print(f"  오류 포함 발화 {s['오류포함_발화']:,} ({s['오류포함_발화비율']*100:.1f}%)")
    if "대조표_되돌림률_vs_학습노출" in res:
        print(f"\n{'표준→실제':12}{'train 노출':>12}{'test n':>9}{'되돌림률':>10}")
        for r in res["대조표_되돌림률_vs_학습노출"]:
            print(f"  {r['표준']}→{r['실제']:<8}{r['train_노출']:>12,}{r['test_n']:>9,}{r['되돌림률']*100:>9.1f}%")
        if "상관_학습노출_vs_되돌림률" in res:
            print(f"\n  상관계수(학습노출 vs 되돌림률) = {res['상관_학습노출_vs_되돌림률']}")
    print(f"→ {A.out}")


if __name__ == "__main__":
    main()
