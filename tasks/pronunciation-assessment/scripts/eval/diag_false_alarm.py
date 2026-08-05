# -*- coding: utf-8 -*-
"""오탐(억울한 지적) 해부 — 정상 발음을 틀렸다고 하는 경우가 무엇으로 이루어져 있는가.

왜 필요한가
-----------
제품 실패 모드가 둘이다. "틀렸는데 넘어감"(놓침)과 "맞았는데 틀렸다고 함"(오탐).
2026-07-30 실측: 레슨(30발화)당 억울한 지적 4.82회, 억울한 지적이 0회인 깨끗한 레슨 16.5%.
오류 검출을 늘리는 방향(errup)은 이 값을 7.38회 / 5.0% 로 악화시켜 제품 판단상 기각됐다.
따라서 **현역 모델의 오탐을 먼저 줄인다.** 그러려면 오탐의 구성부터 알아야 한다.

분류
----
정상 자리(manual == canon)인데 모델이 canon 과 다르게 낸 위치를 다음으로 쪼갠다.

    삭제형   pred == '-'      모델이 그 음소를 아예 못 냈다. "틀렸다"가 아니라 "못 알아들음"
                              → UI 에서 지적으로 표시하지 않는 선택지가 있다(무비용)
    치환형   pred == 다른 음소  모델이 다른 음소로 들었다. 진짜 오탐

추가로: 음소별 오탐률, 혼동쌍(canon→pred) 상위, 발화 길이별, 화자 L1 별,
발화당 오탐 개수 분포(1개짜리가 많은지 뭉쳐 있는지)를 낸다.

한계는 diag_overcorrection.py 와 동일(삽입 제외, 치환·삭제 축).

    $PY tasks/pronunciation-assessment/scripts/eval/diag_false_alarm.py \
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

import os, json, argparse, collections
import edlib
from mdd.jamo_to_kog2p import VOWEL, ONSET, CODA

JAMO = set(VOWEL) | set(ONSET) | set(CODA)


def project(c_str, s_str):
    if not c_str:
        return []
    if not s_str:
        return ["-"] * len(c_str)
    a = edlib.getNiceAlignment(edlib.align(c_str, s_str, task="path"), c_str, s_str)
    return [ch_s for ch_c, ch_s in zip(a["query_aligned"], a["target_aligned"]) if ch_c != "-"]


def load_preds(path, N):
    if not os.path.exists(path):
        raise SystemExit(f"예측 캐시가 없습니다: {path}")
    P, seen = [None] * N, set()
    for line in open(path, encoding="utf-8"):
        d = json.loads(line); i = d["i"]
        if not (0 <= i < N): raise SystemExit(f"캐시 인덱스 범위 밖: i={i}, N={N}")
        if i in seen: raise SystemExit(f"캐시 중복 인덱스: i={i}")
        seen.add(i); P[i] = d["pred"]
    if len(seen) != N: raise SystemExit(f"캐시 불완전: {len(seen):,}/{N:,}")
    return P


def main():
    ap = argparse.ArgumentParser(description="오탐 해부")
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

    fa_del = fa_sub = 0                       # 오탐: 삭제형 / 치환형
    ok_total = 0
    caught = missed = third = 0               # 진짜 오류 자리 (참고용, 억제 효과 계산에 씀)
    caught_del = 0                            # '포착' 중 모델이 삭제로 맞춘 것
    pair = collections.Counter()              # 혼동쌍 canon→pred (치환형 오탐)
    by_phone = collections.defaultdict(lambda: [0, 0])   # 음소별 [오탐, 정상자리]
    by_lang = collections.defaultdict(lambda: [0, 0])
    by_len = collections.defaultdict(lambda: [0, 0])     # 발화 길이 구간별
    per_utt_fa = collections.Counter()        # 발화당 오탐 개수 분포
    utt_rows = []

    for i in range(N):
        p = [t for t in (P[i] or []) if t in JAMO]
        c = "".join(canon[i])
        if not c:
            continue
        mv = project(c, "".join(manual[i]))
        pv = project(c, "".join(p))
        lang = os.path.basename(rel[i][0]).split("-")[3] if len(os.path.basename(rel[i][0]).split("-")) > 3 else "??"
        L = len(c)
        bucket = "1-5" if L <= 5 else "6-10" if L <= 10 else "11-20" if L <= 20 else "21+"
        n_fa_sub = n_fa_del = 0
        for j, ch_c in enumerate(c):
            if mv[j] == ch_c:                              # ── 정상 자리
                ok_total += 1
                by_phone[ch_c][1] += 1; by_lang[lang][1] += 1; by_len[bucket][1] += 1
                if pv[j] != ch_c:
                    by_phone[ch_c][0] += 1; by_lang[lang][0] += 1; by_len[bucket][0] += 1
                    if pv[j] == "-":
                        fa_del += 1; n_fa_del += 1
                    else:
                        fa_sub += 1; n_fa_sub += 1
                        pair[(ch_c, pv[j])] += 1
            else:                                          # ── 진짜 오류 자리
                if pv[j] == ch_c:
                    missed += 1
                elif pv[j] == mv[j]:
                    caught += 1
                    if pv[j] == "-":
                        caught_del += 1
                else:
                    third += 1
        per_utt_fa[n_fa_sub + n_fa_del] += 1
        utt_rows.append((os.path.basename(rel[i][0])[:5], n_fa_sub, n_fa_del))

    fa_total = fa_del + fa_sub

    # 삭제형 오탐을 지적에서 빼면 레슨당 억울한 지적이 얼마나 되나
    import statistics
    by_spk = collections.defaultdict(list)
    for spk, s, d in utt_rows:
        by_spk[spk].append((s, d))
    now, sup = [], []
    for spk, us in by_spk.items():
        for k in range(0, len(us) - A.lesson + 1, A.lesson):
            blk = us[k:k + A.lesson]
            now.append(sum(s + d for s, d in blk))
            sup.append(sum(s for s, d in blk))
    clean_now = sum(1 for x in now if x == 0) / max(1, len(now))
    clean_sup = sum(1 for x in sup if x == 0) / max(1, len(sup))

    ph = sorted(((k, v) for k, v in by_phone.items() if v[1] >= 2000),
                key=lambda kv: -kv[1][0] / max(1, kv[1][1]))
    res = {
        "_설명": "정상 발음을 틀렸다고 하는 오탐의 구성. 삭제형(못 알아들음) vs 치환형(다른 음소로 들음).",
        "_한계": "삽입 제외(치환·삭제 축) — diag_overcorrection.py 와 동일.",
        "preds": A.preds, "n_utt": N,
        "오탐_구성": {
            "정상자리_전체": ok_total, "오탐_전체": fa_total,
            "오탐률": round(fa_total / max(1, ok_total), 4),
            "삭제형_못알아들음": fa_del, "삭제형_비중": round(fa_del / max(1, fa_total), 4),
            "치환형_다른음소": fa_sub, "치환형_비중": round(fa_sub / max(1, fa_total), 4)},
        "삭제형을_지적에서_빼면": {
            "레슨당_억울한지적": [round(statistics.mean(now), 2), round(statistics.mean(sup), 2)],
            "깨끗한레슨_비율": [round(clean_now, 4), round(clean_sup, 4)],
            "잃는것_진짜오류중_삭제로포착": caught_del,
            "잃는것_비중_포착대비": round(caught_del / max(1, caught), 4)},
        "진짜오류_자리_참고": {"포착": caught, "놓침": missed, "제3": third},
        "혼동쌍_상위_치환형오탐": [{"표준": k[0], "모델이들은것": k[1], "n": v}
                              for k, v in pair.most_common(15)],
        "음소별_오탐률_상위": [{"음소": k, "정상자리": v[1], "오탐": v[0],
                            "오탐률": round(v[0] / v[1], 4)} for k, v in ph[:12]],
        "발화길이별": {k: {"정상자리": v[1], "오탐률": round(v[0] / max(1, v[1]), 4)}
                    for k, v in sorted(by_len.items())},
        "화자L1별": {k: {"정상자리": v[1], "오탐률": round(v[0] / max(1, v[1]), 4)}
                   for k, v in sorted(by_lang.items(), key=lambda kv: -kv[1][1])[:10]},
        "발화당_오탐개수_분포": {str(k): per_utt_fa[k] for k in sorted(per_utt_fa)[:8]},
    }
    os.makedirs(os.path.dirname(os.path.abspath(A.out)), exist_ok=True)
    json.dump(res, open(A.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    f = res["오탐_구성"]; s = res["삭제형을_지적에서_빼면"]
    print(f"\n===== 오탐 해부 · {os.path.basename(A.preds)} =====")
    print(f"정상 자리 {f['정상자리_전체']:,} 중 오탐 {f['오탐_전체']:,} ({f['오탐률']*100:.2f}%)")
    print(f"  삭제형(못 알아들음)  {f['삭제형_못알아들음']:>8,}  {f['삭제형_비중']*100:5.1f}%")
    print(f"  치환형(다른 음소)    {f['치환형_다른음소']:>8,}  {f['치환형_비중']*100:5.1f}%")
    print(f"\n삭제형을 지적에서 빼면:")
    print(f"  레슨당 억울한 지적   {s['레슨당_억울한지적'][0]} → {s['레슨당_억울한지적'][1]}")
    print(f"  깨끗한 레슨          {s['깨끗한레슨_비율'][0]*100:.1f}% → {s['깨끗한레슨_비율'][1]*100:.1f}%")
    print(f"  대가: 삭제로 포착한 진짜 오류 {s['잃는것_진짜오류중_삭제로포착']:,}건 상실 "
          f"(포착 전체의 {s['잃는것_비중_포착대비']*100:.1f}%)")
    print(f"→ {A.out}")


if __name__ == "__main__":
    main()
