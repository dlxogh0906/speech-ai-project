# -*- coding: utf-8 -*-
"""오발음 '교정' 진단 — 모델이 틀린 발음을 표준발음으로 되돌려 버리는가.

서비스 관점의 질문
------------------
사용자가 일부러 틀리게 발음했는데 모델이 표준발음으로 받아써 버리면, 진단기는
"잘 했다"고 답한다. 제품 신뢰를 직접 깨는 실패다. 그게 실제로 얼마나 일어나는지 센다.

세는 방법
---------
canonical(표준발음) 축에 canon / manual(전문가가 실제로 들은 것) / pred(모델) 를 올린다.
**치환·삭제형 오발음 자리** = manual[j] != canon[j] 인 위치 j. 그 자리에서 모델 행동을 3분류:

    되돌림(교정)  pred[j] == canon[j]    ← 사용자가 틀렸는데 표준으로 받아씀. 놓친다.
    포착          pred[j] == manual[j]   ← 전문가와 같게 들음. 정확히 잡는다.
    제3의 반응    그 외                   ← 다른 음소로 들었거나 그 자리를 삭제함

정상 자리(manual == canon)에서의 오탐률도 같이 낸다.

⚠️ 반드시 알아야 할 한계 (2026-07-30 교차감사에서 지적됨)
--------------------------------------------------------
1. **삽입은 세지 않는다.** canonical 에 대응 위치가 없는 편집(화자/모델의 음소 삽입)은
   축에서 빠진다. all_lar 기준 참조 편집연산 74,293 중 삽입 6,311(5,270발화)이 제외되어
   분모 67,982 는 전체의 **91.5%** 다. 따라서 "모든 오발음" 이 아니라
   **"치환·삭제형 오발음"** 이라고 불러야 한다.
2. **되돌림률 = 1 − recall.** 이 값은 `mdd_eval_fixedref.py` 의 recall 과 수학적으로
   같은 정보다(실측: recall 0.4099 ↔ 되돌림률 0.5901). 새 지표가 아니라 **FN 을
   "표준으로 되돌림 / 제3의 반응" 으로 쪼갠 것**이 이 스크립트의 기여다.
3. **"제3의 반응" 에는 삭제가 섞인다.** 모델별 801~886건(해당 범주의 15~18%)은
   음소가 아니라 그 자리를 지운 것이다.
4. 분모 67,982 의 정확한 값은 edlib 타이브레이크에 의존한다(등가 최적 경로 범위
   67,977~68,279). **모델 간 동일하다는 성질만** 정렬 정책과 무관한 불변량이다.

    $PY tasks/pronunciation-assessment/scripts/eval/diag_overcorrection.py \
        --preds $MDD_OUTPUTS/eval_tag_all_lar_preds.jsonl --out <파일.json>

예측 캐시만 읽는다(GPU 불필요, CPU 약 30초).
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


def project(c_str, s_str):
    """canonical 각 위치에 s 의 대응 기호를 올린다. 반환 길이 == len(c_str).
    s 가 그 자리를 삭제했으면 '-'. s 의 삽입 열은 canonical 자리가 없으므로 버린다."""
    if not c_str:
        return []
    if not s_str:
        return ["-"] * len(c_str)
    a = edlib.getNiceAlignment(edlib.align(c_str, s_str, task="path"), c_str, s_str)
    out = []
    for ch_c, ch_s in zip(a["query_aligned"], a["target_aligned"]):
        if ch_c == "-":
            continue
        out.append(ch_s)
    return out


def main():
    ap = argparse.ArgumentParser(description="오발음 교정(과잉정정) 진단")
    ap.add_argument("--manifest", default=f"{MDD_OUTPUTS}/manifest_tag_all_lar")
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--examples", type=int, default=12, help="되돌림 사례 표본 수")
    A = ap.parse_args()

    M = A.manifest
    rel = [l.split("\t") for l in open(f"{M}/test.tsv").read().splitlines()[1:]]
    canon = [l.split() for l in open(f"{M}/test.canon", encoding="utf-8").read().splitlines()]
    manual = [l.split() for l in open(f"{M}/test.ltr", encoding="utf-8").read().splitlines()]
    N = len(rel)

    if not os.path.exists(A.preds):
        raise SystemExit(f"예측 캐시가 없습니다: {A.preds}")
    P = [None] * N
    seen = set()
    for line in open(A.preds, encoding="utf-8"):
        d = json.loads(line)
        i = d["i"]
        if not (0 <= i < N):
            raise SystemExit(f"캐시 인덱스가 범위 밖입니다: i={i}, N={N}. --manifest 가 짝이 맞습니까?")
        if i in seen:
            raise SystemExit(f"캐시에 중복 인덱스가 있습니다: i={i}")
        seen.add(i)
        P[i] = d["pred"]
    if len(seen) != N:                      # 누락을 빈 시퀀스로 조용히 삼키지 않는다
        raise SystemExit(f"캐시가 불완전합니다: {len(seen):,}/{N:,} 행. "
                         f"누락분이 빈 예측으로 처리되면 되돌림률이 왜곡됩니다.")

    # 진짜 오발음 자리에서의 모델 행동
    revert = caught = third = 0
    # 정상 자리에서의 모델 행동
    ok_keep = ok_false = 0
    # 오발음 유형별 되돌림률 (canon→manual 쌍)
    by_pair = collections.defaultdict(lambda: [0, 0])   # [되돌림, 전체]
    # 화자 L1 별
    by_lang = collections.defaultdict(lambda: [0, 0])
    samples = []

    for i in range(N):
        p = [t for t in (P[i] or []) if t in JAMO]
        c = "".join(canon[i]); m = "".join(manual[i]); q = "".join(p)
        if not c:
            continue
        mv = project(c, m)
        pv = project(c, q)
        lang = os.path.basename(rel[i][0]).split("-")[3] if len(os.path.basename(rel[i][0]).split("-")) > 3 else "??"
        for j, ch_c in enumerate(c):
            if mv[j] != ch_c:                      # ── 진짜 오발음 자리
                by_pair[(ch_c, mv[j])][1] += 1
                by_lang[lang][1] += 1
                if pv[j] == ch_c:
                    revert += 1
                    by_pair[(ch_c, mv[j])][0] += 1
                    by_lang[lang][0] += 1
                    if len(samples) < A.examples and 4 <= len(c) <= 11:
                        samples.append({"utt": os.path.basename(rel[i][0]),
                                        "canon": " ".join(canon[i]),
                                        "manual": " ".join(manual[i]),
                                        "pred": " ".join(p),
                                        "위치": j, "표준": ch_c, "실제발음": mv[j]})
                elif pv[j] == mv[j]:
                    caught += 1
                else:
                    third += 1
            else:                                  # ── 정상 자리
                if pv[j] == ch_c:
                    ok_keep += 1
                else:
                    ok_false += 1

    tot_err = revert + caught + third
    tot_ok = ok_keep + ok_false
    pairs = sorted(((k, v) for k, v in by_pair.items() if v[1] >= 300),
                   key=lambda kv: -kv[1][0] / max(1, kv[1][1]))
    res = {
        "_설명": "치환·삭제형 오발음 자리에서 모델이 표준발음으로 되돌리는 비율(=놓침)을 센다.",
        "_한계": ("삽입은 축에 없어 제외된다 — 분모는 참조 편집연산의 약 91.5%(치환+삭제)다. "
                "되돌림률은 mdd_eval_fixedref 의 (1-recall) 과 동일한 정보이며, 이 스크립트의 "
                "기여는 FN 을 되돌림/제3의반응으로 쪼갠 것이다. "
                "'제3의 반응'에는 삭제가 15~18% 섞여 있다. "
                "분모의 정확한 값은 edlib 타이브레이크 의존(등가 경로 범위 67,977~68,279)."),
        "preds": A.preds, "n_utt": N,
        "오발음_자리_치환삭제만": {
            "전체": tot_err,
            "되돌림_표준으로": revert, "되돌림률": round(revert / max(1, tot_err), 4),
            "포착_전문가와동일": caught, "포착률": round(caught / max(1, tot_err), 4),
            "제3의반응_음소또는삭제": third, "제3률": round(third / max(1, tot_err), 4)},
        "정상_자리": {
            "전체": tot_ok, "유지": ok_keep,
            "오탐": ok_false, "오탐률": round(ok_false / max(1, tot_ok), 4)},
        "되돌림률_높은_오발음유형": [
            {"표준": k[0], "실제발음": k[1], "n": v[1], "되돌림률": round(v[0] / v[1], 4)}
            for k, v in pairs[:15]],
        "되돌림률_낮은_오발음유형": [
            {"표준": k[0], "실제발음": k[1], "n": v[1], "되돌림률": round(v[0] / v[1], 4)}
            for k, v in pairs[-10:]],
        "화자L1별_되돌림률": {k: {"n": v[1], "되돌림률": round(v[0] / max(1, v[1]), 4)}
                          for k, v in sorted(by_lang.items(), key=lambda kv: -kv[1][1]) if v[1] >= 1000},
        "사례": samples,
    }
    os.makedirs(os.path.dirname(os.path.abspath(A.out)), exist_ok=True)
    json.dump(res, open(A.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    e = res["오발음_자리_치환삭제만"]; o = res["정상_자리"]
    print(f"\n===== 오발음 교정 진단 (n={N:,} 발화) =====")
    print(f"치환·삭제형 오발음 {e['전체']:,}자리에서 모델은:  (삽입 제외 — 참조 편집연산의 91.5%)")
    print(f"  표준으로 되돌림(놓침)  {e['되돌림_표준으로']:>9,}  {e['되돌림률']*100:5.1f}%")
    print(f"  전문가와 같게 포착     {e['포착_전문가와동일']:>9,}  {e['포착률']*100:5.1f}%")
    print(f"  제3의 반응(음소/삭제)  {e['제3의반응_음소또는삭제']:>9,}  {e['제3률']*100:5.1f}%")
    print(f"정상 자리 {o['전체']:,}개 중 오탐 {o['오탐']:,} ({o['오탐률']*100:.1f}%)")
    print(f"→ {A.out}")


if __name__ == "__main__":
    main()
