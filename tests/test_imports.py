# -*- coding: utf-8 -*-
"""저장소 전체 import 스모크 — 리팩터링이 코드를 깨뜨렸는지 판정하는 1차 방어선.

이 저장소에는 실행 테스트를 붙일 수 없는 스크립트가 많다(GPU·26GB 체크포인트·수시간).
그래서 최소한 **모든 모듈이 import 되는지**와 **모든 스크립트가 파싱되는지**는 항상 본다.

    $PY tests/test_imports.py            # 전체
    $PY tests/test_imports.py --quick    # torch/fairseq 없이 가능한 것만
"""
import argparse
import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), *(str(p) for p in sorted(ROOT.glob("tasks/*/src")))]

# 라이브러리 모듈 — 다른 코드가 import 하므로 반드시 살아 있어야 한다
LIBS = [
    "shared.paths", "shared.fairseq_guard", "shared.bootstrap",
    "country_classifier", "country_classifier.audio", "country_classifier.dataset",
    "country_classifier.metrics", "country_classifier.utils",
    "country_classifier.baseline_common", "country_classifier.make_speaker_split",
    "country_classifier.make_country40_probe",
    "mdd", "mdd.jamo_to_kog2p", "mdd.l1prior", "mdd.l1prior.s0_taxonomy",
    "mdd.l1prior.s0_symbol_map",
]
# torch/fairseq/GPU 가 있어야 import 되는 것들
LIBS_HEAVY = [
    "country_classifier.model", "mdd.mdd_eval", "mdd.pipeline",
    "mdd.gop_feats_v2", "mdd.mdd_eval_pos", "mdd.mdd_reproduce_official",
]


def check_syntax():
    """모든 .py 가 파싱되는가 (archive 포함 — 아카이브도 깨진 채 두지 않는다)."""
    bad = []
    n = 0
    for dp, dn, fns in os.walk(ROOT):
        if ".git" in dp or "__pycache__" in dp:
            continue
        for f in sorted(fns):
            if not f.endswith(".py"):
                continue
            p = Path(dp) / f
            n += 1
            try:
                compile(p.read_text(encoding="utf-8"), str(p), "exec")
            except SyntaxError as e:
                bad.append((p.relative_to(ROOT), f"line {e.lineno}: {e.msg}"))
    return n, bad


def check_paths():
    """경로 상수가 실제로 존재하는가. 없어도 치명적이진 않지만 알려준다."""
    from shared import paths
    rows = []
    for name in ("DATA_ROOT", "MODEL_ROOT", "FAIRSEQ_ROOT", "NAT_OUTPUTS", "NAT_DATA",
                 "MDD_OUTPUTS", "MDD_ANALYSIS", "KOG2P_DIR", "MDD_CKPT"):
        try:
            v = str(getattr(paths, name))
            rows.append((name, v, Path(v).exists()))
        except RuntimeError:
            rows.append((name, "(미설정 — .env 참고)", False))
    return rows


# 코드가 깨진 것과 '환경이 아직 안 갖춰진 것'은 다르다. 후자는 실패로 세지 않는다.
_ENV_MARKERS = ("가 설정되지 않았습니다", "No module named 'g2p'", "No module named 'torch'",
                "No module named 'fairseq'", "No module named 'edlib'")


def check_imports(mods):
    ok, fail, skipped = [], [], []
    for m in mods:
        try:
            importlib.import_module(m)
            ok.append(m)
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            (skipped if any(k in str(e) for k in _ENV_MARKERS) else fail).append((m, msg))
    return ok, fail, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="torch/fairseq 가 필요한 모듈은 건너뛴다")
    a = ap.parse_args()

    print("[1] 문법 검사")
    n, bad = check_syntax()
    print(f"    {n}개 파일 · 실패 {len(bad)}")
    for p, msg in bad:
        print(f"      ✗ {p} | {msg}")

    print("\n[2] 경로 상수")
    try:
        rows = check_paths()
    except Exception as e:
        rows = []
        print(f"    ✗ 경로 모듈 로드 실패: {type(e).__name__}: {e}")
    for name, v, exists in rows:
        print(f"    {'✓' if exists else '✗'} {name:14} {v}")

    print("\n[3] 라이브러리 import (경량)")
    ok, fail, skip = check_imports(LIBS)
    print(f"    {len(ok)}/{len(LIBS)} 성공" + (f" · 환경 미구성 {len(skip)}" if skip else ""))
    for m, e in fail:
        print(f"      ✗ {m} | {e}")
    for m, e in skip:
        print(f"      · {m} (환경 미구성 — .env / 외부 도구) | {e.split(chr(10))[0]}")

    fail_heavy = []
    if not a.quick:
        print("\n[4] 라이브러리 import (torch/fairseq 필요)")
        ok2, fail_heavy, skip2 = check_imports(LIBS_HEAVY)
        print(f"    {len(ok2)}/{len(LIBS_HEAVY)} 성공" + (f" · 환경 미구성 {len(skip2)}" if skip2 else ""))
        for m, e in fail_heavy:
            print(f"      ✗ {m} | {e}")
        for m, e in skip2:
            print(f"      · {m} (환경 미구성) | {e.split(chr(10))[0]}")

    total_fail = len(bad) + len(fail) + len(fail_heavy)
    print(f"\n{'통과' if total_fail == 0 else f'실패 {total_fail}건'}"
          "   (환경 미구성은 코드 결함이 아니므로 실패로 세지 않는다)")
    return 1 if total_fail else 0


if __name__ == "__main__":
    sys.exit(main())
