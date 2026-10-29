# -*- coding: utf-8 -*-
"""AIHub 040~043 발음평가 wav(RW/RS/RP) → 16kHz mono int16 → $SPEECH_AI_DATA/mdd_gold_wav16k.

왜 필요한가: fairseq audio_finetuning 은 16kHz 가 아니면 예외로 멈추고(raw_audio_dataset.py),
XLS-R 은 16kHz 로 사전학습됐다. 받은 원본은 48kHz 가 대부분이지만 44.1k·24k·16k·96k 가 섞여 있다.

입력: `extract_aihub_zips.py` 로 푼 AIHub 구조
    <root>/<번호>.<데이터셋명>/01-1.정식개방데이터/{Training,Validation}/01.원천데이터/*_pronunciation_sound*/*.wav
출력: <out>/<화자ID 앞 2자리>/<stem>.wav   (기존 학습 서버 뱅크와 같은 배치 — 매니페스트 호환)
      <out>/_index.tsv    stem · 원본 샘플레이트 · 원본 채널 · 16k 샘플 수(= 매니페스트 frames) · 데이터셋 · split
      <out>/_errors.tsv   실패·너무 짧음(<400 샘플) — 조용히 버리지 않는다

변환식은 기존 `mdd_gold_extract_full.py`·`rebuild_tag_manifest.py` 와 같다 (wave 로 읽기 · 채널 평균 ·
scipy resample_poly · clip × 32767 → int16). 같은 폴더를 두 스크립트가 함께 쓰므로 바이트까지 같아야 한다
(2026-09-14 무작위 20개 대조 20/20 동일).

`rebuild_tag_manifest.py --build` 와의 차이: 그쪽은 매니페스트 필터(길이 8,000~240,000 · 라벨 존재)를 통과한
발화만 변환한다. 이건 필터 없이 전부 변환하고 원본 샘플레이트·길이 인덱스를 남긴다.

멱등: 출력은 임시 파일에 쓰고 rename 하므로, 있는 파일은 완성본이다 → 건너뛰고 헤더에서 샘플 수만 읽는다.

    $PY tasks/pronunciation-assessment/scripts/data/resample_gold_16k.py --limit 2000      # 시험 (속도 측정)
    $PY tasks/pronunciation-assessment/scripts/data/resample_gold_16k.py --workers 14      # 전체 — tmux 에서
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import DATA_ROOT

import argparse, collections, glob, os, time, wave
from math import gcd
from multiprocessing import Pool

import numpy as np

SR = 16000
MIN_SAMPLES = 400        # 기존 뱅크와 같은 하한(25 ms)


def find_sources(root):
    """[(src, stem, 데이터셋, split)] — stem 이 전역 유일한지 확인한다."""
    pat = os.path.join(root, "*", "01-1.정식개방데이터", "*", "01.원천데이터", "*_pronunciation_sound*")
    out, seen, dup = [], {}, []
    for d in sorted(glob.glob(pat)):
        if not os.path.isdir(d):                    # 옆에 있는 원본 zip 도 패턴에 걸린다
            continue
        parts = d.split(os.sep)
        ds, split = parts[-5].split(".")[0], parts[-3]     # <번호>.<이름>/01-1.정식개방데이터/<split>/01.원천데이터/<zip폴더>
        for e in os.scandir(d):
            if not e.name.endswith(".wav"):
                continue
            stem = e.name[:-4]
            if stem in seen:
                dup.append((stem, seen[stem], e.path))
                continue
            seen[stem] = e.path
            out.append((e.path, stem, ds, split))
    return out, dup


def to16k(src):
    """→ (int16 배열, 원본 sr, 원본 채널). 식은 rebuild_tag_manifest.resample_to 와 한 글자도 다르지 않게 둔다 —
    두 스크립트가 같은 폴더에 쓰므로 누가 만들든 같은 바이트여야 한다."""
    from scipy.signal import resample_poly
    with wave.open(src, "rb") as w:
        sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    a = np.frombuffer(raw, {1: np.uint8, 2: np.int16, 4: np.int32}[sw]).astype(np.float32)
    a = (a - 128) / 128 if sw == 1 else a / {2: 32768.0, 4: 2147483648.0}[sw]
    if ch > 1:
        a = a.reshape(-1, ch).mean(1)
    if sr != SR:
        g = gcd(int(sr), SR)
        a = resample_poly(a, SR // g, sr // g)
    return (np.clip(a, -1, 1) * 32767).astype(np.int16), sr, ch


def work(job):
    src, stem, ds, split, out = job
    dst = os.path.join(out, stem[:2], stem + ".wav")
    if os.path.exists(dst):
        try:
            with wave.open(dst) as w:
                n = w.getnframes()
            with wave.open(src) as w:                # 인덱스의 원본 샘플레이트·채널은 헤더만 읽어 채운다
                sr, ch = w.getframerate(), w.getnchannels()
            return ("skip", stem, sr, ch, n, ds, split, "")
        except Exception:                           # 깨진 기존 파일이면 다시 만든다
            os.remove(dst)
    try:
        pcm, sr, ch = to16k(src)
        if pcm.size < MIN_SAMPLES:
            return ("short", stem, sr, ch, pcm.size, ds, split, f"{pcm.size} samples")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = f"{dst}.{os.getpid()}.tmp"            # rebuild_tag_manifest.py 가 같은 폴더에 dst+".tmp" 로 쓴다 — 이름을 겹치지 않게
        with wave.open(tmp, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
        os.replace(tmp, dst)
        return ("ok", stem, sr, ch, pcm.size, ds, split, "")
    except Exception as e:
        return ("error", stem, "", "", 0, ds, split, f"{type(e).__name__}: {e}".replace("\t", " ")[:300])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=f"{DATA_ROOT}/aihub", help="extract_aihub_zips.py 로 푼 AIHub 폴더")
    ap.add_argument("--out", default=f"{DATA_ROOT}/mdd_gold_wav16k")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--limit", type=int, default=0, help="앞에서 N개만 (시험용). 0 = 전체")
    a = ap.parse_args()

    t0 = time.time()
    src, dup = find_sources(a.root)
    print(f"[index] wav {len(src):,}개 ({time.time() - t0:.0f}s) · 중복 stem {len(dup):,}", flush=True)
    if not src:
        raise SystemExit(f"{a.root} 아래에서 *_pronunciation_sound*/*.wav 를 못 찾았다")
    if dup:
        for d in dup[:5]:
            print(f"  중복: {d}", flush=True)
        raise SystemExit("stem 이 겹친다 — 출력 파일명이 충돌하므로 멈춘다")
    if a.limit:
        src = src[:a.limit]
    os.makedirs(a.out, exist_ok=True)

    cnt, srs = collections.Counter(), collections.Counter()
    rows, errs, frames = [], [], 0
    t1 = time.time()
    jobs = [(p, s, ds, sp, a.out) for p, s, ds, sp in src]
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(work, jobs, chunksize=64), 1):
            status, stem, sr, ch, n, ds, split, msg = r
            cnt[status] += 1
            if status in ("ok", "skip"):
                rows.append((stem, sr, ch, n, ds, split)); frames += n
                if sr:
                    srs[sr] += 1
            else:
                errs.append((stem, status, ds, split, msg))
            if i % 50000 == 0 or i == len(jobs):
                el = time.time() - t1
                print(f"  {i:,}/{len(jobs):,}  {i / el:.0f}개/s  남은 {(len(jobs) - i) / (i / el) / 60:.0f}분  {dict(cnt)}", flush=True)

    # 인덱스는 이번 실행분 전체를 새로 쓴다(--limit 시험이면 _index.limit.tsv)
    suffix = ".limit" if a.limit else ""
    with open(os.path.join(a.out, f"_index{suffix}.tsv"), "w", encoding="utf-8") as f:
        f.write("stem\tsrc_sr\tsrc_ch\tframes16k\tdataset\tsplit\n")
        for r in sorted(rows):
            f.write("\t".join(map(str, r)) + "\n")
    with open(os.path.join(a.out, f"_errors{suffix}.tsv"), "w", encoding="utf-8") as f:
        f.write("stem\tstatus\tdataset\tsplit\tmessage\n")
        for r in sorted(errs):
            f.write("\t".join(map(str, r)) + "\n")

    print(f"[done] {dict(cnt)} · {frames / SR / 3600:,.1f} h · {time.time() - t0:.0f}s", flush=True)
    print(f"       원본 샘플레이트(새로 변환한 것): {dict(srs.most_common())}", flush=True)
    if errs:
        print(f"       ⚠️ 실패·짧음 {len(errs):,}건 → {a.out}/_errors{suffix}.tsv", flush=True)
    return 1 if cnt["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
