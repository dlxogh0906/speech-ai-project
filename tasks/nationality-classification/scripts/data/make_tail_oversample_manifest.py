# -*- coding: utf-8 -*-
"""꼬리 클래스 오버샘플링 학습 목록 — 같은 행을 반복하거나, 131 녹음을 웹마이크 음질로 바꾼 사본으로 늘린다 (2026-10-20).

왜: 꼬리 국가(학습 화자 ≤15)는 학습 행의 절반 남짓이 131(낭독 녹음)인데, 꼬리 성적을 재는 test 는 웹마이크다.
    같은 행을 더 자주 보여주기만 하면 그 몇 사람을 더 외울 뿐이다(손실 가중 ×17 이 꼬리를 오히려 낮췄다 — results/27).
    131 행을 웹마이크처럼 바꾼 사본으로 늘리면 녹음 환경 지름길이 끊기고, 반복할 때마다 소리가 달라진다.

만드는 것 (--src = 행 균등 목록 폴더, make_capped_manifest.py --cap min 의 결과)
  - 꼬리 클래스(--full_src 학습 목록에서 화자 ≤ --tail_max_spk): 행을 --factor 배로 늘린다.
    그 클래스의 다른 학습 행(행 균등 목록에 없던 것)을 먼저 쓰고, 모자라면 행을 돌아가며 반복한다.
  - --aug web131 이면
    · 늘린 꼬리 행 중 131 행은 모두 웹마이크 음질 사본으로 바꾼다(사본마다 다른 무작위 체인)
    · 모든 클래스의 원래 행 중 131 행은 --aug_frac 확률로 사본으로 바꾼다 — 증강 흔적이 꼬리 클래스와만 엮이지 않게
    · --aug_extra_domains zeroth,kspon 이면 그 도메인의 원래 행도 같은 확률로 바꾼다 — 한국어에는 131 행이 없어서
      이게 없으면 "증강 없는 깨끗한 낭독 = 한국어 · 증강 낭독 = 외국인" 이 된다. 따로 뽑으므로 131 선택(과 이미 만든 사본)은 그대로다.
  - valid · test 는 그대로 복사한다.
  증강 체인은 한국어 채널 증강(results/11, novelty/jobs/build_kor_aug.py)과 같다:
  대역 제한(80~150 Hz 고역 · 2.5~4.5 kHz 저역) · 잔향 · 컴프레서 · 잡음 억제 · 볼륨 · 코덱(opus 16~32k / aac 48k / 없음).
  단 배경 잡음(KsponSpeech 다른 발화)은 넣지 않는다 — 외국인 발화에 한국인 말소리를 섞지 않으려고.
  사본은 앞 --max_sec 초만 만든다(학습은 앞 8초만 읽는다).

    $PY scripts/data/make_tail_oversample_manifest.py --dst <폴더>                    # 같은 행 반복
    $PY scripts/data/make_tail_oversample_manifest.py --dst <폴더> --aug web131       # 131 웹마이크 음질 사본
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import NAT_OUTPUTS

import argparse, collections, csv, json, os, random, shutil, subprocess, time
from multiprocessing import Pool
from pathlib import Path

COPY = ("valid.csv", "test.csv", "test_koraug_zeroth.csv", "speaker_split.json", "prep_meta.json", "restore_meta.json",
        "split_meta.json")


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        return r.fieldnames, list(r)


def chain(rng):
    """웹마이크 음질 체인 하나 → (ffmpeg -af 문자열, 코덱 인자 또는 None). build_kor_aug.py 와 같은 분포."""
    af = [f"highpass=f={rng.uniform(80, 150):.0f}", f"lowpass=f={rng.uniform(2500, 4500):.0f}"]
    if rng.random() < 0.6:
        af.append(f"aecho=0.8:{rng.uniform(0.3, 0.7):.2f}:{rng.randint(20, 60)}:{rng.uniform(0.15, 0.35):.2f}")
    if rng.random() < 0.7:
        af.append(f"acompressor=threshold={rng.uniform(-25, -15):.0f}dB:ratio={rng.uniform(3, 6):.1f}")
    if rng.random() < 0.5:
        af.append(f"afftdn=nf={rng.uniform(-30, -20):.0f}")
    af.append(f"volume={rng.uniform(0.4, 1.2):.2f}")
    codec = rng.choice([["-c:a", "libopus", "-b:a", f"{rng.choice([16, 24, 32])}k", "-f", "webm"],
                        ["-c:a", "aac", "-b:a", "48k", "-f", "mp4"], None])
    return ",".join(af), codec


def render(job):
    """(원본, 출력, 시드, 앞 몇 초, ffmpeg) → 출력 경로 또는 None. 이미 있으면 다시 만들지 않는다."""
    src, dst, seed, max_sec, ff = job
    if os.path.exists(dst):
        return dst
    filt, codec = chain(random.Random(seed))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    part = dst[:-4] + ".part.wav"
    head = [ff, "-y", "-hide_banner", "-loglevel", "error", "-t", f"{max_sec}", "-i", src, "-af", filt]
    to16k = ["-ac", "1", "-ar", "16000", "-sample_fmt", "s16"]
    if codec:
        enc = dst[:-4] + (".webm" if "webm" in codec else ".m4a")
        p = subprocess.run(head + codec + [enc], capture_output=True)
        if p.returncode == 0:
            p = subprocess.run([ff, "-y", "-hide_banner", "-loglevel", "error", "-i", enc, *to16k, part], capture_output=True)
        if os.path.exists(enc):
            os.remove(enc)
    else:
        p = subprocess.run(head + to16k + [part], capture_output=True)
    if p.returncode != 0 or not os.path.exists(part):
        return None
    os.replace(part, dst)
    return dst


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=f"{NAT_OUTPUTS}/webmic_kd/lab131full_zeroth_r16k_spk811_capmin", help="행 균등 목록 폴더")
    ap.add_argument("--full_src", default=f"{NAT_OUTPUTS}/webmic_kd/lab131full_zeroth_r16k_spk811",
                    help="전체 학습 목록 폴더 — 꼬리 판정(화자 수)과 꼬리 클래스의 다른 행을 여기서 가져온다")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--tail_max_spk", type=int, default=15, help="학습 화자가 이 수 이하인 클래스를 꼬리로 본다")
    ap.add_argument("--factor", type=int, default=3, help="꼬리 클래스 행을 몇 배로")
    ap.add_argument("--aug", choices=("none", "web131"), default="none")
    ap.add_argument("--aug_frac", type=float, default=0.5, help="원래 행 중 131 행을 사본으로 바꿀 확률 (모든 클래스)")
    ap.add_argument("--aug_extra_domains", default="", help="131 말고도 같은 확률로 바꿀 도메인 (쉼표, 예: zeroth,kspon)")
    ap.add_argument("--wav_dir", default=f"{NAT_OUTPUTS}/webmic_kd/aug_web131", help="사본 wav 를 쓸 곳")
    ap.add_argument("--max_sec", type=float, default=8.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    fields, base = read_csv(Path(a.src) / "train.csv")
    _, full = read_csv(Path(a.full_src) / "train.csv")
    spk = collections.defaultdict(set)
    for r in full:
        spk[r["label_en"]].add(r["speaker"])
    tail = sorted((c for c in spk if len(spk[c]) <= a.tail_max_spk), key=lambda c: (len(spk[c]), c))
    rng = random.Random(a.seed)
    in_base = {r["audio_path"] for r in base}

    out = [dict(r, _orig=1) for r in base]
    added = {}
    for c in tail:
        have = [r for r in base if r["label_en"] == c]
        others = [r for r in full if r["label_en"] == c and r["audio_path"] not in in_base]
        rng.shuffle(others)
        need = (a.factor - 1) * len(have)
        extra = others[:need]
        pool, i = have + others, 0
        while len(extra) < need:
            extra.append(pool[i % len(pool)]); i += 1
        out += [dict(r, _orig=0) for r in extra]
        added[c] = {"rows_before": len(have), "rows_after": len(have) + len(extra), "unique_new": min(need, len(others)),
                    "repeated": max(0, need - len(others))}

    n_aug, failed = 0, 0
    if a.aug == "web131":
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        jobs, where = [], []
        extra = set(filter(None, a.aug_extra_domains.split(",")))
        rng2 = random.Random(a.seed + 1)
        for k, r in enumerate(out):
            if r["domain"] == "p505":
                if r["_orig"] and rng.random() >= a.aug_frac:
                    continue
            elif r["domain"] not in extra or (r["_orig"] and rng2.random() >= a.aug_frac):
                continue
            stem = Path(r["audio_path"]).stem
            jobs.append((r["audio_path"], f"{a.wav_dir}/{stem[:2]}/{stem}_wa{k}.wav", a.seed * 1000003 + k, a.max_sec, ff))
            where.append(k)
        t0 = time.time()
        print(f"[aug] 131 행 {len(jobs):,}개를 웹마이크 음질 사본으로 · 작업자 {a.workers}", flush=True)
        with Pool(a.workers) as p:
            for j, (k, dst) in enumerate(zip(where, p.imap(render, jobs, chunksize=16))):
                if dst is None:
                    failed += 1
                    continue
                r = out[k]
                out[k] = dict(r, audio_path=dst, domain=r["domain"] + "_webaug",
                              duration_sec=f"{min(float(r['duration_sec'] or 0), a.max_sec):.2f}")
                n_aug += 1
                if j % 2000 == 0:
                    print(f"  {j:,}/{len(jobs):,} · {time.time() - t0:.0f}s", flush=True)
        if failed > 0.01 * max(1, len(jobs)):
            raise SystemExit(f"사본 실패가 1% 를 넘는다 ({failed:,}/{len(jobs):,}) — 아무것도 쓰지 않는다")

    dst = Path(a.dst)
    dst.mkdir(parents=True, exist_ok=True)
    with open(dst / "train.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(out)
    for fn in COPY:
        if (Path(a.src) / fn).is_file():
            shutil.copy2(Path(a.src) / fn, dst / fn)
    per = collections.Counter(r["label_en"] for r in out)
    meta = {"src": a.src, "full_src": a.full_src, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tail_max_spk": a.tail_max_spk, "factor": a.factor, "aug": a.aug, "aug_frac": a.aug_frac,
            "aug_extra_domains": a.aug_extra_domains, "seed": a.seed,
            "rows": {"src": len(base), "dst": len(out)}, "tail": added, "augmented": n_aug, "aug_failed_kept_original": failed,
            "rows_per_class": {"min": min(per.values()), "max": max(per.values())},
            "note": "꼬리 = 전체 학습 목록에서 화자 ≤ tail_max_spk. 사본은 domain <원래>_webaug. valid · test 는 원본 복사."}
    (dst / "oversample_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[done] 학습 {len(base):,} → {len(out):,}행 · 꼬리 {len(tail)}개 ×{a.factor} · 증강 사본 {n_aug:,} (실패 {failed}) · "
          f"클래스당 {meta['rows_per_class']['min']:,}~{meta['rows_per_class']['max']:,}행 → {dst}", flush=True)


if __name__ == "__main__":
    main()
