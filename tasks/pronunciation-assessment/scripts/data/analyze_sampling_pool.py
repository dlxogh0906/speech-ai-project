# -*- coding: utf-8 -*-
"""Stage 2 학습 풀의 **오발음 구성**을 센다 — 재학습 전에 sampling 실험을 설계하기 위한 자료.

학습 매니페스트(`build_full_tag_replay.py`)는 오발음 발화를 복제해 정상:오발음 = 60:40 을 맞춘다.
비율을 더 올리려면 **복제가 더 늘어난다** — 그때 같은 화자·같은 문항이 몇 번까지 반복되는지를
먼저 알아야 한다(모델이 문장을 외우는 문제가 이미 관측됐다: `results/08_문장암기_일반화.md`).

이 스크립트는 파싱 캐시(`cache/foreign_tag_lab_parsed.jsonl.gz`)만 읽는다 — GPU·오디오 불필요.
  · 정상/오발음 발화 수, 오류 유형별(대치·삭제·삽입) 발화 수
  · 화자·문항별 오발음 발화 분포(과대표집 위험)
  · 지정한 혼동쌍(되돌림이 심한 쌍)을 포함한 발화 수 — hard example oversampling 후보 규모
  · 목표 비율(50:50·40:60 등)에 필요한 복제 배수와 그때의 화자·문항 최대 반복 수 추정

    $PY .../analyze_sampling_pool.py --out $MDD_OUTPUTS/analysis/mdd_error_2026-09-27/sampling_pool \\
        --hard-pairs "ㅅ>ㅆ" "ㅈ>ㅊ" "ㄱ>ㅋ" "ㄸ>ㄷ" "ㄴ>ㅇ" "ㅉ>ㅈ"
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import MDD_OUTPUTS

import argparse, collections, csv, gzip, json, os
import edlib
import numpy as np

VOWELS = set(u"ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")


def ops(canon, tag):
    """canonical 자리별 (대치·삭제) 와 삽입 수. eval 쪽 정렬과 같은 edlib 경로."""
    c, t = u"".join(canon), u"".join(tag)
    if not c:
        return [], 0
    if not t:
        return [(i, ch, None) for i, ch in enumerate(canon)], 0
    a = edlib.getNiceAlignment(edlib.align(c, t, task="path"), c, t)
    out, ins, i = [], 0, 0
    for qc, ts in zip(a["query_aligned"], a["target_aligned"]):
        if qc == "-":
            ins += 1
            continue
        if ts != qc:
            out.append((i, qc, None if ts == "-" else ts))
        i += 1
    return out, ins


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=os.path.join(str(MDD_OUTPUTS), "cache", "foreign_tag_lab_parsed.jsonl.gz"))
    ap.add_argument("--exclude-speakers-from", dest="excl", nargs="*", default=[],
                    help="<매니페스트>:<split> — 평가 화자를 빼고 학습 풀만 보려면 지정")
    ap.add_argument("--hard-pairs", dest="hard_pairs", nargs="*", default=[],
                    help='"표준>실제" 형식. 되돌림이 심한 쌍을 넣는다')
    ap.add_argument("--targets", nargs="*", type=float, default=[0.4, 0.5, 0.6],
                    help="오발음 목표 비율(외국인 행 기준). 현재 매니페스트는 0.4")
    ap.add_argument("--out", required=True, help="산출 접두어(.json/.csv)")
    a = ap.parse_args()

    keep_out = set()
    for spec in a.excl:
        mdir, split = spec.rsplit(":", 1)
        with open(os.path.join(mdir, "%s.tsv" % split), encoding="utf-8") as f:
            next(f)
            for line in f:
                keep_out.add(os.path.basename(line.split("\t")[0])[:5])
    hard = set()
    for s in a.hard_pairs:
        if ">" in s:
            c, t = s.split(">", 1)
            hard.add((c.strip(), t.strip()))

    n = clean = mis = 0
    has = collections.Counter()                 # sub/del/ins 를 가진 발화 수
    mis_by_spk, mis_by_item = collections.Counter(), collections.Counter()
    all_by_spk = collections.Counter()
    hard_utts, hard_sites = 0, collections.Counter()
    site_types = collections.Counter()
    dropped = 0
    with gzip.open(a.cache, "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("dropped"):
                dropped += 1
                continue
            if d.get("spk") in keep_out:
                continue
            ltr, canon = d.get("ltr") or [], d.get("canon") or []
            if not ltr or not canon:
                continue
            n += 1
            all_by_spk[d["spk"]] += 1
            diff, ins = ops(canon, ltr)
            if not diff and not ins:
                clean += 1
                continue
            mis += 1
            mis_by_spk[d["spk"]] += 1
            mis_by_item[d.get("item") or "?"] += 1
            if any(t is None for _, _, t in diff):
                has["deletion"] += 1
            if any(t is not None for _, _, t in diff):
                has["substitution"] += 1
            if ins:
                has["insertion"] += 1
            hit = False
            for i, c, t in diff:
                site_types["deletion" if t is None else "substitution"] += 1
                if (c, t or u"∅") in hard or (c, t) in hard:
                    hard_sites[(c, t or u"∅")] += 1
                    hit = True
            site_types["insertion"] += ins
            if hit:
                hard_utts += 1
            if n % 200000 == 0:
                print("   ... %d 발화" % n, flush=True)

    q = lambda c: {"n_keys": len(c), "mean": round(float(np.mean(list(c.values()))), 2) if c else 0,
                   "p50": int(np.percentile(list(c.values()), 50)) if c else 0,
                   "p95": int(np.percentile(list(c.values()), 95)) if c else 0,
                   "max": max(c.values()) if c else 0}
    cur_ratio = mis / float(n) if n else 0
    proj = []
    for tgt in a.targets:
        # 정상 발화는 전부 쓰고 오발음만 복제한다고 가정(현재 빌더와 같은 방식)
        need = clean * tgt / max(1e-9, 1.0 - tgt)
        rep = need / float(mis) if mis else 0
        proj.append({"target_mis_ratio": tgt, "mis_rows_needed": int(round(need)),
                     "repeat_factor": round(rep, 2), "total_rows": int(round(clean + need)),
                     "max_copies_of_one_utt": int(np.ceil(rep)),
                     "max_rows_from_one_speaker": int(round(q(mis_by_spk)["max"] * rep)),
                     "max_rows_from_one_item": int(round(q(mis_by_item)["max"] * rep))})

    out = {"cache": os.path.abspath(a.cache), "excluded_speaker_sets": a.excl,
           "utterances": n, "dropped_in_cache": dropped, "clean": clean, "mispronounced": mis,
           "mis_ratio": round(cur_ratio, 4),
           "utts_containing": dict(has), "sites": dict(site_types),
           "site_share": {k: round(v / float(max(1, sum(site_types.values()))), 4) for k, v in site_types.items()},
           "mis_utts_per_speaker": q(mis_by_spk), "mis_utts_per_item": q(mis_by_item),
           "all_utts_per_speaker": q(all_by_spk),
           "hard_pairs": [u"%s>%s" % p for p in sorted(hard)],
           "hard_pair_utterances": hard_utts,
           "hard_pair_sites": {u"%s>%s" % k: v for k, v in hard_sites.most_common()},
           "projection": proj,
           "note": "복제는 같은 오디오를 그대로 반복한다 — 배수가 커질수록 화자·문항 편중이 커진다"}
    json.dump(out, open(a.out + ".json", "w"), ensure_ascii=False, indent=1)
    with open(a.out + "_projection.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(proj[0].keys()))
        w.writeheader()
        for r in proj:
            w.writerow(r)
    print("[pool] 발화 %d · 정상 %d · 오발음 %d (%.1f%%)" % (n, clean, mis, 100 * cur_ratio))
    print("[sites] %s" % out["site_share"])
    print("[risk] 오발음 발화/화자 p95 %d max %d · /문항 p95 %d max %d"
          % (out["mis_utts_per_speaker"]["p95"], out["mis_utts_per_speaker"]["max"],
             out["mis_utts_per_item"]["p95"], out["mis_utts_per_item"]["max"]))
    print("[saved] %s.json" % a.out)


if __name__ == "__main__":
    main()
