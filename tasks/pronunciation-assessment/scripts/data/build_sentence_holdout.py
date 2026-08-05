# -*- coding: utf-8 -*-
"""문장 홀드아웃 매니페스트 생성 — 암기 가설을 실측으로 확정/기각하기 위한 것.

배경
----
2026-07-30 진단: 학습셋 155만 발화가 **287종 문장**뿐이고 문장당 평균 5,398회 반복된다.
모델이 문장을 외워 표준발음을 읊는 정황이 잡혔다.

    같은 소리(TAG=ㅃ)인데 표준발음이 ㅃ면 89.9%, ㅂ면 3.6% 로 맞힌다 (25배)
    문장 내 위치가 뒤로 갈수록 되돌림률 상승 (앞 20% 62.2% → 뒤 20% 74.2%)
    test 256종이 train 287종에 전부 포함 → 지금 수치로는 일반화를 알 수 없다

이 스크립트는 **일부 문장을 학습에서 완전히 빼고** 그 문장 전용 평가셋을 만든다.
여기서 이어학습한 모델의 되돌림률이 크게 오르면 암기 가설이 확정된다.

설계
----
- **화자는 일부러 겹치게 둔다.** 홀드아웃 문장을 읽은 화자가 다른 문장으로 train 에
  남아 있어야 이번 실험이 **문장 변수만** 분리해 잰다. 화자까지 빼면 두 효과가 섞인다.
- 홀드아웃 문장 선택은 **시드 고정 + 발화 수 기준**으로 결정적이다.
- 원본 매니페스트는 건드리지 않고 새 디렉터리에 쓴다.

    $PY tasks/pronunciation-assessment/scripts/data/build_sentence_holdout.py \\
        --out $MDD_OUTPUTS/manifest_tag_holdout30 --n_holdout 30

생성물: {train,valid,test}.{tsv,canon,ltr} + dict.ltr.txt + holdout_meta.json
  train  홀드아웃 문장을 뺀 학습셋
  valid  원본 valid 에서 홀드아웃 문장을 뺀 것 (학습 중 모니터링용)
  test   **홀드아웃 문장만** — 이게 미학습 문장 평가셋이다
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import MDD_OUTPUTS

import os, json, random, argparse, collections, shutil


def read_split(M, s):
    tsv = open(f"{M}/{s}.tsv").read().splitlines()
    root, rows = tsv[0], tsv[1:]
    canon = open(f"{M}/{s}.canon", encoding="utf-8").read().splitlines()
    ltr = open(f"{M}/{s}.ltr", encoding="utf-8").read().splitlines()
    n = min(len(rows), len(canon), len(ltr))
    if not (len(rows) == len(canon) == len(ltr)):
        raise SystemExit(f"{s}: 줄 수 불일치 tsv={len(rows)} canon={len(canon)} ltr={len(ltr)}")
    return root, rows[:n], canon[:n], ltr[:n]


def write_split(out, s, root, rows, canon, ltr):
    with open(f"{out}/{s}.tsv", "w") as f:
        f.write(root + "\n")
        f.write("\n".join(rows) + ("\n" if rows else ""))
    for name, data in (("canon", canon), ("ltr", ltr)):
        with open(f"{out}/{s}.{name}", "w", encoding="utf-8") as f:
            f.write("\n".join(data) + ("\n" if data else ""))


def main():
    ap = argparse.ArgumentParser(description="문장 홀드아웃 매니페스트 생성")
    ap.add_argument("--manifest", default=f"{MDD_OUTPUTS}/manifest_tag_all_lar")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_holdout", type=int, default=30, help="학습에서 뺄 문장 수")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_utt", type=int, default=300,
                    help="홀드아웃 후보 문장의 최소 발화 수 (평가 표본 확보용)")
    A = ap.parse_args()

    M = A.manifest
    splits = {s: read_split(M, s) for s in ("train", "valid", "test")}

    # 문장별 총 발화 수 (전 split 합산)
    total = collections.Counter()
    for s in splits:
        for c in splits[s][2]:
            total[c] += 1

    # 후보: 발화 수가 충분한 문장. 그중 시드 고정 무작위로 n_holdout 개
    cands = sorted([c for c, v in total.items() if v >= A.min_utt])
    if len(cands) < A.n_holdout:
        raise SystemExit(f"후보 문장이 부족합니다: {len(cands)} < {A.n_holdout} "
                         f"(--min_utt 을 낮추십시오)")
    rng = random.Random(A.seed)
    hold = set(rng.sample(cands, A.n_holdout))

    os.makedirs(A.out, exist_ok=True)

    # train / valid: 홀드아웃 문장 제거   test: 홀드아웃 문장만 (전 split 에서 끌어모음)
    kept = {}
    ho_rows, ho_canon, ho_ltr = [], [], []
    root = None
    for s in ("train", "valid", "test"):
        r, rows, canon, ltr = splits[s]
        root = root or r
        if r != root:
            raise SystemExit(f"{s}: 오디오 루트가 다릅니다. 수동 확인이 필요합니다")
        keep_i = [i for i in range(len(rows)) if canon[i] not in hold]
        ho_i = [i for i in range(len(rows)) if canon[i] in hold]
        kept[s] = ([rows[i] for i in keep_i], [canon[i] for i in keep_i], [ltr[i] for i in keep_i])
        ho_rows += [rows[i] for i in ho_i]
        ho_canon += [canon[i] for i in ho_i]
        ho_ltr += [ltr[i] for i in ho_i]

    write_split(A.out, "train", root, *kept["train"])
    write_split(A.out, "valid", root, *kept["valid"])
    write_split(A.out, "test", root, ho_rows, ho_canon, ho_ltr)
    shutil.copy(f"{M}/dict.ltr.txt", f"{A.out}/dict.ltr.txt")

    def spk(rows):
        return {os.path.basename(r.split("\t")[0])[:5] for r in rows}

    tr_spk, ho_spk = spk(kept["train"][0]), spk(ho_rows)
    err_utt = sum(1 for c, l in zip(ho_canon, ho_ltr) if c != l)

    meta = {
        "_설명": "문장 홀드아웃 — 학습에서 뺀 문장만으로 test 를 구성해 암기 여부를 잰다.",
        "_설계": "화자는 일부러 겹치게 둔다. 문장 변수만 분리하기 위함.",
        "원본": M, "seed": A.seed, "n_holdout": A.n_holdout, "min_utt": A.min_utt,
        "홀드아웃_문장수": len(hold),
        "train": {"발화": len(kept["train"][0]), "문장": len(set(kept["train"][1])),
                  "화자": len(tr_spk)},
        "valid": {"발화": len(kept["valid"][0]), "문장": len(set(kept["valid"][1]))},
        "test_홀드아웃": {"발화": len(ho_rows), "문장": len(set(ho_canon)),
                       "화자": len(ho_spk),
                       "화자_train과겹침": len(ho_spk & tr_spk),
                       "오류포함_발화": err_utt,
                       "오류포함_비율": round(err_utt / max(1, len(ho_rows)), 4)},
        "검증": {
            "test문장이_train에_없음": len(set(ho_canon) & set(kept["train"][1])) == 0,
            "test문장이_valid에_없음": len(set(ho_canon) & set(kept["valid"][1])) == 0,
        },
    }
    json.dump(meta, open(f"{A.out}/holdout_meta.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    print(f"\n===== 문장 홀드아웃 매니페스트 =====")
    print(f"  출력  {A.out}")
    print(f"  train  발화 {meta['train']['발화']:>9,}  문장 {meta['train']['문장']:>4}  화자 {meta['train']['화자']:,}")
    print(f"  valid  발화 {meta['valid']['발화']:>9,}  문장 {meta['valid']['문장']:>4}")
    print(f"  test   발화 {meta['test_홀드아웃']['발화']:>9,}  문장 {meta['test_홀드아웃']['문장']:>4}"
          f"  화자 {meta['test_홀드아웃']['화자']:,} (train과 {meta['test_홀드아웃']['화자_train과겹침']:,}명 겹침 — 의도된 것)")
    print(f"  test 중 오류 포함 발화 {err_utt:,} ({meta['test_홀드아웃']['오류포함_비율']*100:.1f}%)")
    v = meta["검증"]
    print(f"  검증  test문장 ∉ train: {v['test문장이_train에_없음']}   ∉ valid: {v['test문장이_valid에_없음']}")
    if not (v["test문장이_train에_없음"] and v["test문장이_valid에_없음"]):
        raise SystemExit("홀드아웃 누수가 있습니다. 중단합니다.")


if __name__ == "__main__":
    main()
