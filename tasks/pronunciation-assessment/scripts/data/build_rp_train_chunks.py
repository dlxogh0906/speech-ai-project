# -*- coding: utf-8 -*-
"""학습자 문단 읽기(RP) 실제 녹음을 쉼에서 잘라 이어 학습용 행으로 만든다 — 라벨 = 사람 전사(TAG, 실제 발음).

왜(2026-10-19): 학습 문항이 75문장 + 150낱말뿐이라 모델이 문장을 외운다. AIHub 040~043 의 문단 읽기(RP002~004)는
학습에 안 썼고, 학습 화자의 RP 녹음 약 9,000개는 '처음 보는 문항' 평가셋(`build_unseen_prompt_eval.py`)에서도
화자 겹침 때문에 빠져 그대로 남아 있었다. 실제 학습자 · 새 글 · 사람 라벨이 다 있는 유일한 데이터다.

  - **RP004 는 쓰지 않는다** — 화자 상관없이 통째로 '처음 보는 문단' 시험용으로 남긴다. 기본은 RP002 · RP003.
  - 학습 화자(--speakers-from, 기본 replay10 train)의 녹음만 쓴다 → 기존 평가셋(학습 화자 제외)과 화자가 안 겹친다.
  - 자르기: TAG phones 티어의 시각으로 쉼(빈 구간 또는 틈 ≥ --min-gap)에서 자른다. 조각 길이 --min-chunk ~ --max-chunk 초.
    쉼 없이 --max-chunk 를 넘는 곳이 있으면 그 녹음은 버린다. spn·<unk> 가 든 조각도 버린다.
  - 조각 소리 = 첫 소리 시작 −0.1초 ~ 끝 소리 끝 +0.1초(쉼 가운데를 넘지 않게).

출력: <wav-out>/<stem>_<k>.wav · <out>/rows.tsv(첫 줄 = /) · rows.ltr · rows.meta.jsonl · stats.json

    $PY .../build_rp_train_chunks.py --max-recs 4000
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import DATA_ROOT, MDD_OUTPUTS

import argparse, collections, glob, importlib.util, io, json, os, random, re, wave

ITEM = re.compile(r"-(R[WSP]\d{3})-")
SR = 16000


def _load(name):
    spec = importlib.util.spec_from_file_location(name, str(_pl.Path(__file__).with_name(name + ".py")))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_phones(path):
    raw = open(path, "rb").read()
    txt = None
    for enc in ("utf-8", "utf-16"):
        try:
            txt = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if txt is None:
        return None
    m = re.search(r'name = "phones".*?(?=name = "|\Z)', txt, re.S)
    if not m:
        return None
    return [(float(a), float(b), t.strip()) for a, b, t in re.findall(r'xmin = ([\d.]+)\s*xmax = ([\d.]+)\s*text = "([^"]*)"', m.group(0))]


def is_jamo(c):
    return 0x3131 <= ord(c) <= 0x3163


def chunks_of(iv, min_gap, min_chunk, max_chunk):
    """phones 구간 → [(시작, 끝, [자모…], 앞쉼 가운데, 뒷쉼 가운데)] · 쉼 없이 너무 길면 None."""
    sp = [(a, b, t) for a, b, t in iv if t not in ("", ".", "-")]
    if not sp:
        return []
    breaks = []                                     # (앞 소리 번호, 쉼 가운데 시각)
    for k in range(len(sp) - 1):
        gap = sp[k + 1][0] - sp[k][1]
        if gap >= min_gap:
            breaks.append((k, (sp[k][1] + sp[k + 1][0]) / 2))
    out, s = [], 0
    lo = max(0.0, sp[0][0] - 0.1)
    bi = 0
    while s < len(sp):
        cand = None
        while bi < len(breaks) and breaks[bi][0] < s:
            bi += 1
        j = bi
        while j < len(breaks):
            k, mid = breaks[j]
            dur = sp[k][1] - sp[s][0]
            if dur > max_chunk:
                break
            if dur >= min_chunk:
                cand = j
                break
            j += 1
        if cand is None:
            dur = sp[-1][1] - sp[s][0]
            if dur > max_chunk:
                return None
            e = len(sp) - 1; hi = sp[-1][1] + 0.1
        else:
            e, hi = breaks[cand][0], breaks[cand][1]
        out.append((max(lo, sp[s][0] - 0.1), min(hi, sp[e][1] + 0.1), sp[s:e + 1]))
        lo = hi
        s = e + 1
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", nargs="+", default=["RP002", "RP003"])
    ap.add_argument("--speakers-from", dest="speakers_from", default=os.path.join(str(MDD_OUTPUTS), "manifest_full_tag_replay10") + ":train")
    ap.add_argument("--max-recs", dest="max_recs", type=int, default=4000, help="쓸 녹음 수 상한(무작위)")
    ap.add_argument("--min-gap", dest="min_gap", type=float, default=0.15)
    ap.add_argument("--min-chunk", dest="min_chunk", type=float, default=4.0)
    ap.add_argument("--max-chunk", dest="max_chunk", type=float, default=15.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wav-out", dest="wav_out", default=os.path.join(str(DATA_ROOT), "rp_train_chunks"))
    ap.add_argument("--out", default=os.path.join(str(MDD_OUTPUTS), "rp_train_chunks"))
    a = ap.parse_args()
    import numpy as np
    rpb = _load("build_unseen_prompt_eval")
    spk = rpb.load_speakers(a.speakers_from)
    want = set(a.items)
    assert "RP004" not in want, "RP004 는 처음 보는 문단 시험용으로 남긴다"
    tags = {}
    for root in glob.glob(os.path.join(rpb.AIHUB, "0[0-9][0-9]*", "01-1.정식개방데이터", "*", "02.라벨링데이터", "*_pronunciation_tag")):
        for p in glob.glob(os.path.join(root, "*.TextGrid")):
            b = os.path.basename(p)[:-9]
            m = ITEM.search(b)
            if m and m.group(1) in want and b[:5] in spk:
                tags[b] = p
    stems = sorted(tags)
    random.Random(a.seed).shuffle(stems)
    print("[후보] 학습 화자의 %s 녹음 %d" % ("·".join(sorted(want)), len(stems)), flush=True)
    os.makedirs(a.wav_out, exist_ok=True); os.makedirs(a.out, exist_ok=True)
    ft = open(os.path.join(a.out, "rows.tsv"), "w", encoding="utf-8"); ft.write("/\n")
    fl = open(os.path.join(a.out, "rows.ltr"), "w", encoding="utf-8")
    fm = open(os.path.join(a.out, "rows.meta.jsonl"), "w", encoding="utf-8")
    drop, used, n_chunk, sec = collections.Counter(), 0, 0, 0.0
    for stem in stems:
        if used >= a.max_recs:
            break
        wp = rpb.wavpath(stem)
        if not os.path.exists(wp):
            drop["wav없음"] += 1; continue
        iv = read_phones(tags[stem])
        if not iv:
            drop["라벨파싱"] += 1; continue
        ch = chunks_of(iv, a.min_gap, a.min_chunk, a.max_chunk)
        if ch is None:
            drop["쉼없이 김"] += 1; continue
        with wave.open(wp, "rb") as w:
            x = np.frombuffer(w.readframes(w.getnframes()), np.int16)
        ok = 0
        for k, (t0, t1, ph) in enumerate(ch):
            labs = "".join(t for _, _, t in ph)
            if "spn" in labs or "unk" in labs:
                drop["spn 조각"] += 1; continue
            jam = [c for c in labs if is_jamo(c)]
            if len(jam) < 8:
                drop["짧은 조각"] += 1; continue
            seg = x[int(t0 * SR): int(t1 * SR)]
            if len(seg) < SR * 1.0:
                drop["짧은 조각"] += 1; continue
            dst = os.path.join(a.wav_out, "%s_%02d.wav" % (stem, k))
            with wave.open(dst, "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(seg.tobytes())
            ft.write("%s\t%d\n" % (dst.lstrip("/"), len(seg)))
            fl.write(" ".join(jam) + "\n")
            fm.write(json.dumps({"stem": stem, "k": k, "t0": round(t0, 2), "t1": round(t1, 2), "item": ITEM.search(stem).group(1)},
                                ensure_ascii=False) + "\n")
            ok += 1; n_chunk += 1; sec += len(seg) / SR
        used += ok > 0
    for f in (ft, fl, fm):
        f.close()
    st = {"items": sorted(want), "recordings": used, "chunks": n_chunk, "hours": round(sec / 3600, 2), "dropped": dict(drop),
          "speakers_from": a.speakers_from, "note": "RP004 는 시험용으로 남김 · 학습 화자만"}
    json.dump(st, open(os.path.join(a.out, "stats.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(st, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
