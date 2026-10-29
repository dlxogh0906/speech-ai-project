# -*- coding: utf-8 -*-
"""원어민 낭독(Zeroth train)을 이어 학습용 행으로 만든다 — 라벨 = 표준발음(맞게 읽었다고 본다).

왜: 학습 문장이 287개뿐이라 처음 보는 문장에서 원어민도 틀렸다고 나온다(2026-10-19 점검: 낭독 100음절당 지적 6.9).
학습 데이터의 원어민 행은 Kspon 대화(8.7%)뿐이고, 여러 문장을 또박또박 읽은 원어민 녹음은 없었다.

라벨 = `canonical_fix.CanonicalBuilder.fixed`(KoG2P + 보정). 단어 경계처럼 두 발음이 다 맞는 곳은 검증 모델(--check-model)
로그우도로 녹음에 맞는 쪽을 고른다(`choose`, margin 1.0 — 발음 API 와 같다). 받아쓴 자모열과 라벨의 차이율이
--max-cer 를 넘으면 버린다(전사가 녹음과 안 맞는 줄 · 잡음 녹음).
Zeroth test 화자(test_data_01)는 쓰지 않는다 — 원어민 헛지적 시험용으로 남긴다.

출력: <out>/rows.tsv(첫 줄 = 루트) · rows.ltr · rows.meta.jsonl

    PYTHONPATH=/data/speech-ai/tools/hf_py38:$PYTHONPATH $PY .../build_native_rows.py --shard 0/4
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import DATA_ROOT, MDD_OUTPUTS

import argparse, glob, itertools, json, os, time, wave


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zeroth", default=os.path.join(str(DATA_ROOT), "zeroth", "wav16k", "train_data_01"))
    ap.add_argument("--check-model", dest="check_model", default=os.path.join(str(DATA_ROOT), "checkpoints", "w2vbert_stage2", "step_38000"))
    ap.add_argument("--max-cer", dest="max_cer", type=float, default=0.25)
    ap.add_argument("--max-sec", dest="max_sec", type=float, default=20.0)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(str(MDD_OUTPUTS), "native_rows_zeroth"))
    a = ap.parse_args()
    import numpy as np
    import torch
    import editdistance
    torch.set_num_threads(a.threads)
    from mdd.api_offline import ApiEmulator
    from mdd import canonical_fix as cf
    si, sn = (int(x) for x in a.shard.split("/"))
    e = ApiEmulator(a.check_model)
    B = cf.CanonicalBuilder(e.canonical)
    items = []
    for t in sorted(glob.glob(os.path.join(a.zeroth, "*", "*", "*.trans.txt"))):
        for l in open(t, encoding="utf-8"):
            if " " not in l.strip():
                continue
            k, s = l.strip().split(" ", 1)
            p = os.path.join(os.path.dirname(t), k + ".wav")
            if os.path.exists(p):
                items.append((p, s))
    items = items[si::sn]
    os.makedirs(a.out, exist_ok=True)
    ft = open(os.path.join(a.out, "rows.%d-of-%d.tsv" % (si, sn)), "w", encoding="utf-8")
    fl = open(os.path.join(a.out, "rows.%d-of-%d.ltr" % (si, sn)), "w", encoding="utf-8")
    fm = open(os.path.join(a.out, "rows.%d-of-%d.meta.jsonl" % (si, sn)), "w", encoding="utf-8")
    drop = {"long": 0, "empty": 0, "unk": 0, "cer": 0}
    t0 = time.time()
    for n, (p, s) in enumerate(items, 1):
        with wave.open(p, "rb") as w:
            frames = w.getnframes()
            x = np.frombuffer(w.readframes(frames), np.int16).astype(np.float32) / 32768.0
        if frames > a.max_sec * 16000:
            drop["long"] += 1; continue
        base, alts = B.variants(s)
        if not base:
            drop["empty"] += 1; continue
        lp = e.lprobs(x)
        lab, _ = cf.choose(lp[0], base, alts, e.dec.d.index, blank=e.dec.blank, margin=1.0)
        if e.dec.d.unk() in [e.dec.d.index(j) for j in lab]:
            drop["unk"] += 1; continue
        g = [e.dec.d[int(v)] for v, _ in itertools.groupby(lp[0].argmax(-1)) if int(v) != e.dec.blank]
        g = [j for j in g if len(j) == 1 and 0x3131 <= ord(j) <= 0x3163]
        cer = editdistance.eval(g, lab) / max(1, len(lab))
        if cer > a.max_cer:
            drop["cer"] += 1; continue
        ft.write("%s\t%d\n" % (p.lstrip("/"), frames))
        fl.write(" ".join(lab) + "\n")
        fm.write(json.dumps({"path": p, "text": s, "cer": round(cer, 3)}, ensure_ascii=False) + "\n")
        if n % 1000 == 0:
            print("  %d/%d · %.0fs · 버림 %s" % (n, len(items), time.time() - t0, drop), flush=True)
    for f in (ft, fl, fm):
        f.close()
    print("[끝] 조각 %d/%d · 후보 %d · 버림 %s · %.0fs" % (si, sn, len(items), drop, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
