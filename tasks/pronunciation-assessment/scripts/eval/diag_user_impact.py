# -*- coding: utf-8 -*-
"""사용자 체감 단위 진단 — 음소 위치가 아니라 발화·레슨 단위로 센다.

왜 다시 재는가
--------------
`diag_overcorrection.py` 는 음소 위치당 비율을 낸다(오탐 1.7% vs 2.6%). 그런데 사용자는
음소 위치를 세지 않는다. **"또 틀렸다고 하네"** 를 센다. 한 발화에 오탐이 3개 몰려 있어도
사용자에겐 나쁜 경험 1회다. 반대로 오탐이 여러 발화에 흩어지면 3회다.
따라서 모델 교체 판단(놓침 ↓ vs 오탐 ↑)은 이 단위에서 해야 한다.

세는 것
-------
발화 단위
    억울한 지적 발생 발화   정상 자리인데 모델이 오류라 한 위치가 1개 이상
    놓친 오류 발생 발화     진짜 오발음인데 모델이 표준으로 되돌린 위치가 1개 이상

레슨 단위 (기본 30발화 = 제품이 점수를 내는 단위)
    화자별로 발화를 LESSON 개씩 묶어 레슨을 만들고, 레슨당 평균/분포를 낸다.
    **깨끗한 레슨 비율** = 억울한 지적이 0회인 레슨 — 제품 체감에 가장 가깝다.

한계는 diag_overcorrection.py 와 동일하다: 삽입은 축에 없어 제외된다(치환·삭제형만).

    $PY tasks/pronunciation-assessment/scripts/eval/diag_user_impact.py \
        --preds $MDD_OUTPUTS/eval_tag_all_lar_preds.jsonl --out <파일.json>

예측 캐시만 읽는다(GPU 불필요).
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


def project(c_str, s_str):
    """canonical 각 위치에 s 의 대응 기호를 올린다. 길이 == len(c_str). 삽입 열은 버린다."""
    if not c_str:
        return []
    if not s_str:
        return ["-"] * len(c_str)
    a = edlib.getNiceAlignment(edlib.align(c_str, s_str, task="path"), c_str, s_str)
    return [ch_s for ch_c, ch_s in zip(a["query_aligned"], a["target_aligned"]) if ch_c != "-"]


def load_preds(path, N):
    if not os.path.exists(path):
        raise SystemExit(f"예측 캐시가 없습니다: {path}")
    P = [None] * N
    seen = set()
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        i = d["i"]
        if not (0 <= i < N):
            raise SystemExit(f"캐시 인덱스 범위 밖: i={i}, N={N}")
        if i in seen:
            raise SystemExit(f"캐시 중복 인덱스: i={i}")
        seen.add(i)
        P[i] = d["pred"]
    if len(seen) != N:
        raise SystemExit(f"캐시 불완전: {len(seen):,}/{N:,}")
    return P


def main():
    ap = argparse.ArgumentParser(description="사용자 체감(발화·레슨) 단위 진단")
    ap.add_argument("--manifest", default=f"{MDD_OUTPUTS}/manifest_tag_all_lar")
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lesson", type=int, default=30, help="레슨 1회를 이루는 발화 수")
    A = ap.parse_args()

    M = A.manifest
    rel = [l.split("\t") for l in open(f"{M}/test.tsv").read().splitlines()[1:]]
    canon = [l.split() for l in open(f"{M}/test.canon", encoding="utf-8").read().splitlines()]
    manual = [l.split() for l in open(f"{M}/test.ltr", encoding="utf-8").read().splitlines()]
    N = len(rel)
    P = load_preds(A.preds, N)

    # 발화별 집계
    utt = []            # (spk, n_false, n_missed, n_caught, n_real_err)
    for i in range(N):
        p = [t for t in (P[i] or []) if t in JAMO]
        c = "".join(canon[i])
        if not c:
            continue
        mv = project(c, "".join(manual[i]))
        pv = project(c, "".join(p))
        n_false = n_missed = n_caught = n_err = 0
        for j, ch_c in enumerate(c):
            if mv[j] != ch_c:
                n_err += 1
                if pv[j] == ch_c:
                    n_missed += 1
                elif pv[j] == mv[j]:
                    n_caught += 1
            elif pv[j] != ch_c:
                n_false += 1
        spk = os.path.basename(rel[i][0])[:5]      # 화자 ID = 파일명 앞 5자리
        utt.append((spk, n_false, n_missed, n_caught, n_err))

    nU = len(utt)
    u_false_any = sum(1 for u in utt if u[1] > 0)
    u_missed_any = sum(1 for u in utt if u[2] > 0)
    u_has_err = sum(1 for u in utt if u[4] > 0)

    # 레슨 단위 — 화자별로 LESSON 개씩 묶는다
    by_spk = collections.defaultdict(list)
    for u in utt:
        by_spk[u[0]].append(u)
    lessons = []
    for spk, us in by_spk.items():
        for k in range(0, len(us) - A.lesson + 1, A.lesson):   # 자투리는 버린다
            blk = us[k:k + A.lesson]
            lessons.append((sum(b[1] for b in blk), sum(b[2] for b in blk),
                            sum(b[3] for b in blk), sum(b[4] for b in blk)))
    nL = len(lessons)
    if nL == 0:
        raise SystemExit(f"레슨을 만들 수 없습니다(화자당 {A.lesson}발화 미만)")
    f_per = [l[0] for l in lessons]
    m_per = [l[1] for l in lessons]
    c_per = [l[2] for l in lessons]

    res = {
        "_설명": "음소 위치가 아니라 발화·레슨 단위로 센 사용자 체감 지표.",
        "_한계": "삽입 제외(치환·삭제형만) — diag_overcorrection.py 와 동일한 축.",
        "preds": A.preds, "n_utt": nU, "레슨_발화수": A.lesson, "n_레슨": nL,
        "발화_단위": {
            "전체": nU,
            "억울한지적_발생": u_false_any, "억울한지적_발생률": round(u_false_any / nU, 4),
            "놓친오류_발생": u_missed_any, "놓친오류_발생률": round(u_missed_any / nU, 4),
            "진짜오류_포함발화": u_has_err, "진짜오류_포함률": round(u_has_err / nU, 4),
        },
        "레슨_단위": {
            "레슨당_억울한지적_평균": round(statistics.mean(f_per), 2),
            "레슨당_억울한지적_중앙값": statistics.median(f_per),
            "깨끗한레슨_억울한지적0회": round(sum(1 for x in f_per if x == 0) / nL, 4),
            "레슨당_놓친오류_평균": round(statistics.mean(m_per), 2),
            "레슨당_정확포착_평균": round(statistics.mean(c_per), 2),
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(A.out)), exist_ok=True)
    json.dump(res, open(A.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    u, L = res["발화_단위"], res["레슨_단위"]
    print(f"\n===== 사용자 체감 진단 · {os.path.basename(A.preds)} =====")
    print(f"발화 {nU:,}개 / 레슨 {nL:,}개 ({A.lesson}발화 단위)")
    print(f"  억울한 지적이 1회 이상인 발화   {u['억울한지적_발생']:>7,}  {u['억울한지적_발생률']*100:5.1f}%")
    print(f"  놓친 오류가 있는 발화           {u['놓친오류_발생']:>7,}  {u['놓친오류_발생률']*100:5.1f}%")
    print(f"  레슨당 억울한 지적 (평균/중앙)  {L['레슨당_억울한지적_평균']:>7} / {L['레슨당_억울한지적_중앙값']}")
    print(f"  억울한 지적 0회인 깨끗한 레슨   {L['깨끗한레슨_억울한지적0회']*100:>6.1f}%")
    print(f"  레슨당 놓친 오류 / 정확 포착    {L['레슨당_놓친오류_평균']:>7} / {L['레슨당_정확포착_평균']}")
    print(f"→ {A.out}")


if __name__ == "__main__":
    main()
