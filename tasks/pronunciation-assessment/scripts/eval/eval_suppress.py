# -*- coding: utf-8 -*-
"""삭제형 억제 정책의 효과를 레슨 단위로 측정한다.

`src/mdd/suppress.py` 의 정책을 적용했을 때 사용자 체감 지표가 어떻게 변하는지 잰다.
후처리이므로 재학습·GPU 가 필요 없고, 어떤 체크포인트에도 적용된다.

측정 단위는 음소 위치가 아니라 **레슨(기본 30발화)** 이다. 사용자는 음소를 세지 않고
"또 틀렸다고 하네"를 세기 때문이다.

    $PY tasks/pronunciation-assessment/scripts/eval/eval_suppress.py \
        --preds $MDD_OUTPUTS/eval_tag_all_lar_preds.jsonl --out <파일.json>

한계는 diag_overcorrection.py 와 같다(삽입 제외, 치환·삭제 축).
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import MDD_OUTPUTS

import os, json, argparse, collections, statistics
import edlib
from mdd.jamo_to_kog2p import VOWEL, ONSET, CODA

JAMO = set(VOWEL) | set(ONSET) | set(CODA)


def project(c, s):
    if not c:
        return []
    if not s:
        return ["-"] * len(c)
    a = edlib.getNiceAlignment(edlib.align(c, s, task="path"), c, s)
    return [ss for cc, ss in zip(a["query_aligned"], a["target_aligned"]) if cc != "-"]


def load_preds(path, N):
    if not os.path.exists(path):
        raise SystemExit(f"예측 캐시가 없습니다: {path}")
    P, seen = [None] * N, set()
    for line in open(path, encoding="utf-8"):
        d = json.loads(line); i = d["i"]
        if not (0 <= i < N): raise SystemExit(f"캐시 인덱스 범위 밖: {i}")
        if i in seen: raise SystemExit(f"캐시 중복 인덱스: {i}")
        seen.add(i); P[i] = d["pred"]
    if len(seen) != N: raise SystemExit(f"캐시 불완전: {len(seen):,}/{N:,}")
    return P


def main():
    ap = argparse.ArgumentParser(description="삭제형 억제 정책 효과 측정")
    ap.add_argument("--manifest", default=f"{MDD_OUTPUTS}/manifest_tag_all_lar")
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lesson", type=int, default=30)
    A = ap.parse_args()

    M = A.manifest
    rel = [l.split("\t") for l in open(f"{M}/test.tsv").read().splitlines()[1:]]
    canon = [l.split() for l in open(f"{M}/test.canon", encoding="utf-8").read().splitlines()]
    manual = [l.split() for l in open(f"{M}/test.ltr", encoding="utf-8").read().splitlines()]
    N = len(rel)
    P = load_preds(A.preds, N)

    # 발화별로 (억울한 지적: 전체 / 치환형만, 정확 포착: 전체 / 치환형만) 을 센다
    rows = []
    tot = dict(fa_all=0, fa_sub=0, catch_all=0, catch_sub=0, err=0, ok=0)
    for i in range(N):
        p = [t for t in (P[i] or []) if t in JAMO]
        c = "".join(t for t in canon[i] if t in JAMO)
        if not c:
            continue
        mv = project(c, "".join(t for t in manual[i] if t in JAMO))
        pv = project(c, "".join(p))
        fa_a = fa_s = ca = cs = 0
        for j, ch in enumerate(c):
            if mv[j] == ch:                       # 정상 자리
                tot["ok"] += 1
                if pv[j] != ch:
                    fa_a += 1
                    if pv[j] != "-":              # 삭제형이 아니면 남는다
                        fa_s += 1
            else:                                 # 진짜 오류 자리
                tot["err"] += 1
                if pv[j] == mv[j]:
                    ca += 1
                    if pv[j] != "-":              # 삭제로 잡은 포착은 억제되면 사라진다
                        cs += 1
        tot["fa_all"] += fa_a; tot["fa_sub"] += fa_s
        tot["catch_all"] += ca; tot["catch_sub"] += cs
        rows.append((os.path.basename(rel[i][0])[:5], fa_a, fa_s, ca, cs))

    # 레슨 단위 집계 — 화자별로 lesson 개씩 묶는다
    by = collections.defaultdict(list)
    for spk, fa_a, fa_s, ca, cs in rows:
        by[spk].append((fa_a, fa_s, ca, cs))
    L = {k: [] for k in ("fa_all", "fa_sub", "catch_all", "catch_sub")}
    for spk, us in by.items():
        for k in range(0, len(us) - A.lesson + 1, A.lesson):
            blk = us[k:k + A.lesson]
            L["fa_all"].append(sum(b[0] for b in blk))
            L["fa_sub"].append(sum(b[1] for b in blk))
            L["catch_all"].append(sum(b[2] for b in blk))
            L["catch_sub"].append(sum(b[3] for b in blk))
    nL = len(L["fa_all"])
    if not nL:
        raise SystemExit(f"레슨을 만들 수 없습니다(화자당 {A.lesson}발화 미만)")

    def clean(v): return sum(1 for x in v if x == 0) / nL
    res = {
        "_설명": "삭제형 지적을 숨겼을 때의 사용자 체감 변화. 후처리이므로 재학습 없음.",
        "_한계": "삽입 제외(치환·삭제 축). diag_overcorrection.py 와 동일.",
        "preds": A.preds, "n_utt": N, "n_레슨": nL, "레슨_발화수": A.lesson,
        "현행": {
            "레슨당_억울한지적": round(statistics.mean(L["fa_all"]), 2),
            "깨끗한레슨": round(clean(L["fa_all"]), 4),
            "레슨당_정확포착": round(statistics.mean(L["catch_all"]), 2),
            "정상자리_오탐률": round(tot["fa_all"] / max(1, tot["ok"]), 4)},
        "삭제형_억제후": {
            "레슨당_억울한지적": round(statistics.mean(L["fa_sub"]), 2),
            "깨끗한레슨": round(clean(L["fa_sub"]), 4),
            "레슨당_정확포착": round(statistics.mean(L["catch_sub"]), 2),
            "정상자리_오탐률": round(tot["fa_sub"] / max(1, tot["ok"]), 4)},
    }
    a, b = res["현행"], res["삭제형_억제후"]
    d_fa = a["레슨당_억울한지적"] - b["레슨당_억울한지적"]
    d_ca = a["레슨당_정확포착"] - b["레슨당_정확포착"]
    res["교환비"] = {"억울한지적_감소": round(d_fa, 2), "정확포착_감소": round(d_ca, 2),
                   "비율": round(d_fa / d_ca, 2) if d_ca else None}

    os.makedirs(os.path.dirname(os.path.abspath(A.out)), exist_ok=True)
    json.dump(res, open(A.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print(f"\n===== 삭제형 억제 · {os.path.basename(A.preds)} =====")
    print(f"레슨 {nL:,}개 ({A.lesson}발화 단위)")
    print(f"{'':22}{'현행':>10}{'억제후':>10}")
    print(f"{'레슨당 억울한 지적':22}{a['레슨당_억울한지적']:>10}{b['레슨당_억울한지적']:>10}")
    print(f"{'깨끗한 레슨':22}{a['깨끗한레슨']*100:>9.1f}%{b['깨끗한레슨']*100:>9.1f}%")
    print(f"{'레슨당 정확 포착':22}{a['레슨당_정확포착']:>10}{b['레슨당_정확포착']:>10}")
    print(f"{'정상자리 오탐률':22}{a['정상자리_오탐률']*100:>9.2f}%{b['정상자리_오탐률']*100:>9.2f}%")
    print(f"\n교환비: 억울한 지적 {d_fa:.2f} 감소 ↔ 정확 포착 {d_ca:.2f} 감소 = {res['교환비']['비율']}배")
    print(f"→ {A.out}")


if __name__ == "__main__":
    main()
