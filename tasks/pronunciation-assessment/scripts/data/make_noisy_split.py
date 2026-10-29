# -*- coding: utf-8 -*-
"""매니페스트 split 에 웅성거림(babble) 소음을 섞은 사본을 만든다 — 전시장·교실 같은 현장 소음에서의 강건성 점검용.

소음 = 다른 코퍼스(기본 KsponSpeech) 발화 --n_talkers 개를 겹친 것. 발화 전체 RMS 기준으로 목표 SNR(dB)에 맞춰 더한다.
SNR 마다 <out>/snr<값>/ 에 같은 이름의 매니페스트(.tsv · .ltr · .canon · .meta.jsonl · dict)와 wav 를 쓴다 →
`extract_phone_feats.py --manifest <out>/snr10 --split <split>` 로 그대로 돌릴 수 있다.

    $PY tasks/pronunciation-assessment/scripts/data/make_noisy_split.py --manifest $MDD_OUTPUTS/manifest_foreign_clean_v2 \
        --split valid_unseen --snr 20 10 5 --out /data/speech-ai/out/novelty/noisy
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import MDD_OUTPUTS

import argparse, os, random, shutil, wave
from multiprocessing import Pool

import numpy as np

_G = {}


def read_wav(p):
    with wave.open(p, "rb") as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2, p
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0


def work(job):
    i, src, dsts, seed = job
    rng = random.Random(seed)
    x = read_wav(src)
    n = len(x); noise = np.zeros(n, np.float32)
    for _ in range(_G["talkers"]):
        y = read_wav(rng.choice(_G["noise"]))
        while len(y) < n:
            y = np.concatenate([y, read_wav(rng.choice(_G["noise"]))])
        s = rng.randrange(0, len(y) - n + 1); noise += y[s:s + n] * rng.uniform(0.6, 1.0)
    px, pn = float((x ** 2).mean()) + 1e-10, float((noise ** 2).mean()) + 1e-10
    for snr, dst in dsts:
        g = np.sqrt(px / (pn * 10 ** (snr / 10.0)))
        z = x + g * noise
        z = z / max(1.0, float(np.abs(z).max()) / 0.99)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with wave.open(dst, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes((z * 32767).astype(np.int16).tobytes())
    return i


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True); ap.add_argument("--split", required=True)
    ap.add_argument("--noise_manifest", default=str(MDD_OUTPUTS / "manifest_kspon_stage1")); ap.add_argument("--noise_split", default="valid")
    ap.add_argument("--snr", type=float, nargs="+", default=[20, 10, 5]); ap.add_argument("--n_talkers", type=int, default=6)
    ap.add_argument("--max_utts", type=int, default=0); ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=12); ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    with open(os.path.join(a.manifest, f"{a.split}.tsv"), encoding="utf-8") as f:
        root = f.readline().rstrip("\n"); rows = [l.rstrip("\n").split("\t") for l in f]
    if a.max_utts:
        rows = rows[:a.max_utts]
    with open(os.path.join(a.noise_manifest, f"{a.noise_split}.tsv"), encoding="utf-8") as f:
        nroot = f.readline().rstrip("\n"); noise = [os.path.join(nroot, l.split("\t")[0]) for l in f]
    _G.update(noise=noise, talkers=a.n_talkers)
    jobs = []
    for i, (p, n) in enumerate(rows):
        dsts = [(s, os.path.join(a.out, f"snr{int(s)}", "wav", p)) for s in a.snr]
        jobs.append((i, os.path.join(root, p), dsts, a.seed * 1000003 + i))
    with Pool(a.workers) as pool:
        for k, _ in enumerate(pool.imap_unordered(work, jobs, chunksize=64)):
            if k % 1000 == 0:
                print(f"  {k:,}/{len(jobs):,}", flush=True)
    for s in a.snr:
        d = os.path.join(a.out, f"snr{int(s)}")
        with open(os.path.join(d, f"{a.split}.tsv"), "w", encoding="utf-8") as f:
            f.write(os.path.join(d, "wav") + "\n")
            for p, n in rows:
                f.write(f"{p}\t{n}\n")
        for ext in ("ltr", "canon", "meta.jsonl"):
            src = os.path.join(a.manifest, f"{a.split}.{ext}")
            if os.path.exists(src):
                lines = open(src, encoding="utf-8").read().splitlines()[:len(rows)]
                open(os.path.join(d, f"{a.split}.{ext}"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
        shutil.copy(os.path.join(a.manifest, "dict.ltr.txt"), os.path.join(d, "dict.ltr.txt"))
    print(f"[완료] {a.out} · {len(rows):,}발화 × SNR {a.snr} · 소음원 {len(noise):,}발화({a.n_talkers}명 겹침)", flush=True)


if __name__ == "__main__":
    main()
