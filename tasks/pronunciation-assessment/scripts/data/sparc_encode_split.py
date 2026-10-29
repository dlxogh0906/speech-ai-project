# -*- coding: utf-8 -*-
"""매니페스트 split 의 발화를 SPARC(조음 복원)로 인코딩한다 — 조음 피드백 가능성 검증용.

SPARC(Speech Articulatory Coding, MIT)는 음성에서 조음기관 궤적을 복원한다:
    EMA 12차원(혀뿌리 TD · 혀몸 TB · 혀끝 TT · 아래앞니 LI(턱) · 윗입술 UL · 아랫입술 LL 의 x,y) + loudness · pitch · periodicity, 50 Hz
50 Hz 는 우리 XLS-R 프레임(20 ms)과 같아 강제정렬 구간에 그대로 맞물린다.

⚠️ **별도 환경에서 돈다** — `speech-articulatory-coding` 은 최신 torch 를 끌고 와 기존 환경(torch 2.1.2 · fairseq 핀)을 깬다.
    /data/speech-ai/envs/sparc/bin/python .../sparc_encode_split.py --manifest <dir> --split valid_unseen --out <dir> --shard 0/3
   HF 캐시는 HF_HOME 로 큰 디스크에 둔다. 이 스크립트는 저장소 모듈을 import 하지 않는다(그 환경엔 fairseq 가 없다).

출력 <out>/<split>.shard<k>.npz : frames(ΣT×15 float16: ema12·loudness·pitch·periodicity) · offsets · idx(매니페스트 행) · spk_emb
"""
import argparse, os, time

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="multi", help="en | multi(한국어 KSS 포함) | en+")
    ap.add_argument("--shard", default="0/1", help="k/n — n 개로 나눠 k 번째만")
    ap.add_argument("--max_utts", type=int, default=0)
    a = ap.parse_args()

    import torch
    from sparc import load_model
    k, n = map(int, a.shard.split("/"))
    with open(os.path.join(a.manifest, f"{a.split}.tsv"), encoding="utf-8") as f:
        root = f.readline().rstrip("\n")
        paths = [os.path.join(root, l.split("\t")[0]) for l in f]
    idx = list(range(len(paths)))[: a.max_utts or None][k::n]
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    coder = load_model(a.model, device=dev)
    os.makedirs(a.out, exist_ok=True)
    frames, offsets, spk, t0, fail = [], [0], [], time.time(), 0
    for c, i in enumerate(idx):
        try:
            code = coder.encode(paths[i])
            T = int(code["ema"].shape[0])
            x = np.concatenate([code["ema"][:T], code["loudness"][:T], code["pitch"][:T], code["periodicity"][:T]], -1)
            spk.append(np.asarray(code["spk_emb"], np.float32))
        except Exception as e:                       # 너무 짧은 발화 등 — 빈 구간으로 남긴다
            fail += 1; x = np.zeros((0, 15), np.float32); spk.append(np.zeros(64, np.float32))
            if fail <= 5:
                print(f"  [fail] {paths[i]}: {type(e).__name__}: {e}", flush=True)
        frames.append(x.astype(np.float16)); offsets.append(offsets[-1] + len(x))
        if c % 500 == 0:
            print(f"  shard {k}/{n}: {c:,}/{len(idx):,} ({c / max(1e-9, time.time() - t0):.1f}/s) fail {fail}", flush=True)
    out = os.path.join(a.out, f"{a.split}.shard{k}.npz")
    np.savez(out, frames=np.concatenate(frames) if frames else np.zeros((0, 15), np.float16), offsets=np.array(offsets, np.int64),
             idx=np.array(idx, np.int64), spk_emb=np.stack(spk) if spk else np.zeros((0, 64), np.float32))
    print(f"[완료] {out} · {len(idx):,}발화 · 실패 {fail} · {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
