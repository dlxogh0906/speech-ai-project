# -*- coding: utf-8 -*-
"""지름길 차단 매니페스트 — 문장 반복을 줄이고 문장 종류를 늘린다.

왜 필요한가 (2026-07-31 홀드아웃 실험 결과)
--------------------------------------------
현행 학습셋은 155만 발화가 **287종 문장**뿐이고 문장당 평균 5,398회 반복된다.
모델은 음향을 듣는 대신 **어느 문장인지 식별해 외운 표준발음을 출력하는 지름길**을 배웠다.

    본 문장 231종   되돌림 61.8%  오탐 1.9%   weighted F1 0.9509
    안 본 문장 30종 되돌림 22.1%  오탐 25.3%  weighted F1 0.8218
                    ↑ 외울 게 없으면 실제로 듣는다

데이터 양은 원인이 아니다 — LAR 21만 발화(문장 30종)를 추가했을 때 되돌림은
58.9% → 59.0% 로 **변화가 없었다**. 반면 원어민 5.4만 발화(문장 25,854종)를 3.5%만
섞은 `tag_all_native` 는 56.0% 로 내려갔다. **움직이는 것은 발화 수가 아니라 문장 종류다.**

이 스크립트가 하는 일
---------------------
1. **문장당 발화 수 상한** — 화자 라운드로빈으로 골라 화자 다양성을 최대한 지킨다
2. **원어민 데이터 병합** — 25,854종 문장. KoG2P 71음소 → 자모 46 로 변환해 기호를 맞춘다
3. valid/test 는 손대지 않는다 (홀드아웃 실험과 직접 비교하기 위해)

    $PY tasks/pronunciation-assessment/scripts/data/build_diverse_manifest.py \\
        --src $MDD_OUTPUTS/manifest_tag_holdout30 \\
        --native $MDD_OUTPUTS/manifest_native \\
        --out $MDD_OUTPUTS/manifest_diverse_cap1000 --cap 1000
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import MDD_OUTPUTS

import os, json, shutil, argparse, collections
from mdd.jamo_to_kog2p import phones_to_jamo


def spk_of(row):
    return os.path.basename(row.split("\t")[0])[:5]


def cap_by_sentence(rows, canon, ltr, cap):
    """문장별 상한. 화자 라운드로빈으로 골라 화자 다양성을 지킨다."""
    by = collections.defaultdict(list)
    for i, c in enumerate(canon):
        by[c].append(i)
    keep = []
    for c, idx in by.items():
        if len(idx) <= cap:
            keep += idx
            continue
        per_spk = collections.defaultdict(list)
        for i in idx:
            per_spk[spk_of(rows[i])].append(i)
        ks = sorted(per_spk)
        picked, ptr = [], 0
        while len(picked) < cap:
            advanced = False
            for k in ks:
                if ptr < len(per_spk[k]):
                    picked.append(per_spk[k][ptr]); advanced = True
                    if len(picked) >= cap: break
            if not advanced: break
            ptr += 1
        keep += picked
    return sorted(keep)


def main():
    ap = argparse.ArgumentParser(description="지름길 차단 매니페스트 생성")
    ap.add_argument("--src", default=f"{MDD_OUTPUTS}/manifest_tag_holdout30",
                    help="학습자 매니페스트 (자모 46). 홀드아웃이 이미 빠져 있어야 한다")
    ap.add_argument("--native", default=f"{MDD_OUTPUTS}/manifest_native",
                    help="원어민 매니페스트 (KoG2P 71). train 만 쓴다")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap", type=int, default=1000, help="문장당 발화 수 상한")
    ap.add_argument("--no_native", action="store_true", help="원어민 병합 없이 상한만 적용")
    A = ap.parse_args()

    S = A.src
    tsv = open(f"{S}/train.tsv").read().splitlines()
    root, rows = tsv[0], tsv[1:]
    canon = open(f"{S}/train.canon", encoding="utf-8").read().splitlines()
    ltr = open(f"{S}/train.ltr", encoding="utf-8").read().splitlines()
    if not (len(rows) == len(canon) == len(ltr)):
        raise SystemExit(f"src 줄 수 불일치: {len(rows)}/{len(canon)}/{len(ltr)}")

    keep = cap_by_sentence(rows, canon, ltr, A.cap)
    o_rows = [rows[i] for i in keep]
    o_canon = [canon[i] for i in keep]
    o_ltr = [ltr[i] for i in keep]
    n_learner, n_sent_learner = len(o_rows), len(set(o_canon))

    n_native = 0
    if not A.no_native:
        N = A.native
        ntsv = open(f"{N}/train.tsv").read().splitlines()
        nroot, nrows = ntsv[0], ntsv[1:]
        if nroot != root:
            raise SystemExit(f"오디오 루트가 다릅니다: '{root}' vs '{nroot}'. 수동 확인 필요")
        ncanon = open(f"{N}/train.canon", encoding="utf-8").read().splitlines()
        nltr = open(f"{N}/train.ltr", encoding="utf-8").read().splitlines()
        bad = 0
        for i in range(min(len(nrows), len(ncanon), len(nltr))):
            try:                                   # KoG2P 71 → 자모 46
                c = " ".join(phones_to_jamo(ncanon[i].split()))
                l = " ".join(phones_to_jamo(nltr[i].split()))
            except Exception:
                bad += 1; continue
            if not c or not l:
                bad += 1; continue
            o_rows.append(nrows[i]); o_canon.append(c); o_ltr.append(l); n_native += 1
        if bad:
            print(f"[native] 변환 실패로 건너뜀 {bad:,}건")

    os.makedirs(A.out, exist_ok=True)
    open(f"{A.out}/train.tsv", "w").write(root + "\n" + "\n".join(o_rows) + "\n")
    open(f"{A.out}/train.canon", "w", encoding="utf-8").write("\n".join(o_canon) + "\n")
    open(f"{A.out}/train.ltr", "w", encoding="utf-8").write("\n".join(o_ltr) + "\n")
    for s in ("valid", "test"):                    # 비교 가능성 유지 — 손대지 않는다
        for ext in ("tsv", "canon", "ltr"):
            shutil.copy(f"{S}/{s}.{ext}", f"{A.out}/{s}.{ext}")
    shutil.copy(f"{S}/dict.ltr.txt", f"{A.out}/dict.ltr.txt")

    spk = {spk_of(r) for r in o_rows}
    err = sum(1 for c, l in zip(o_canon, o_ltr) if c != l)
    meta = {
        "_설명": "문장당 상한 + 원어민 병합으로 문장 암기 지름길을 차단한다.",
        "src": S, "native": None if A.no_native else A.native, "cap": A.cap,
        "학습자": {"발화": n_learner, "문장": n_sent_learner},
        "원어민": {"발화": n_native},
        "합계": {"발화": len(o_rows), "문장": len(set(o_canon)), "화자": len(spk),
               "오류포함_발화": err,
               "문장당_평균": round(len(o_rows) / max(1, len(set(o_canon))), 1)},
    }
    json.dump(meta, open(f"{A.out}/diverse_meta.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    m = meta["합계"]
    print(f"\n===== 지름길 차단 매니페스트 (cap={A.cap}) =====")
    print(f"  출력  {A.out}")
    print(f"  학습자 {n_learner:>9,} 발화 / {n_sent_learner:>6,} 문장")
    print(f"  원어민 {n_native:>9,} 발화")
    print(f"  ─────────────────────────────────────────")
    print(f"  합계   {m['발화']:>9,} 발화 / {m['문장']:>6,} 문장 / 화자 {m['화자']:,}")
    print(f"  문장당 평균 {m['문장당_평균']}회   (현행 5,398회)")
    print(f"  오류 포함 발화 {err:,}")


if __name__ == "__main__":
    main()
