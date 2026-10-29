# -*- coding: utf-8 -*-
"""Compare full-utterance and local-crop CTC evidence for Korean sibilants.

Exploratory only: samples are balanced within prompt/canonical/actual groups;
their precision is not a service precision estimate. Cropping uses canonical
CTC alignment, never the actual-phone label. No serving files are changed.
"""
import sys
from pathlib import Path

_root = Path(__file__).resolve()
while not (_root / "pyproject.toml").exists():
    if _root.parent == _root:
        raise RuntimeError("repository root not found")
    _root = _root.parent
sys.path[:0] = [str(_root), str(_root / "tasks/pronunciation-assessment/src")]

import argparse
import collections
import glob
import json
import os
import random
import time
import wave

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sites", required=True, help="directory of sites.*.jsonl")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-group", type=int, default=12)
    ap.add_argument("--seed", type=int, default=231023)
    ap.add_argument("--batch-size", type=int, default=12)
    args = ap.parse_args()
    import torch
    from mdd.gop_feats_v2 import HFDecoder, forward_lprobs, viterbi_align
    from mdd.vad import trim as trim_silence

    torch.set_num_threads(2)
    if os.path.exists(args.out):
        raise FileExistsError(args.out)
    groups = collections.defaultdict(list)
    for path in sorted(glob.glob(os.path.join(args.sites, "sites.*.jsonl"))):
        with open(path, encoding="utf-8") as stream:
            for line in stream:
                r = json.loads(line)
                if r["canon"] in ("ㅅ", "ㅆ") and r["actual"] in ("ㅅ", "ㅆ"):
                    groups[(r["item"], r["canon"], r["actual"])].append(r)
    rng = random.Random(args.seed)
    selected = []
    for key in sorted(groups):
        selected.extend(rng.sample(groups[key], min(args.per_group, len(groups[key]))))
    by_stem = collections.defaultdict(list)
    for r in selected:
        by_stem[r["stem"]].append(r)
    utterances = {}
    with open(os.path.join(args.manifest, "test.tsv"), encoding="utf-8") as tsv, \
            open(os.path.join(args.manifest, "test.canon"), encoding="utf-8") as canonical:
        audio_root = next(tsv).strip()
        for row, c in zip(tsv, canonical):
            rel = row.split("\t")[0]
            stem = Path(rel).stem
            if stem in by_stem:
                utterances[stem] = (os.path.join(audio_root, rel), c.split())
    print("selected sites=%d utterances=%d" % (len(selected), len(utterances)), flush=True)
    dec = HFDecoder(args.ckpt)
    ids = [dec.d.index(c) for c in ("ㅅ", "ㅆ")]
    sizes = (0.6, 1.0, 1.6)
    pending, completed = [], []
    start = time.monotonic()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    def evidence(l, a, b):
        lo, hi = max(0, int(a)), min(len(l), int(b))
        if hi <= lo:
            return None
        return float(l[lo:hi, ids[1]].max() - l[lo:hi, ids[0]].max())

    with open(args.out, "w", encoding="utf-8") as output:
        def flush():
            if not pending:
                return
            lp, lens = forward_lprobs(dec, [p[0] for p in pending])
            for k, (x, rec, key, target) in enumerate(pending):
                l = lp[k, :lens[k]].numpy()
                frame = len(x) / 16000.0 / len(l)
                rec["scores"][key] = evidence(l, round(target[0] / frame), round(target[1] / frame))
                rec["wide_scores"][key] = evidence(l, round((target[0] - 0.08) / frame),
                                                   round((target[1] + 0.08) / frame))
            pending[:] = []
            keep = []
            for rec in completed:
                if len(rec["scores"]) == 1 + len(sizes):
                    output.write(json.dumps(rec, ensure_ascii=False) + "\n")
                else:
                    keep.append(rec)
            completed[:] = keep
            output.flush()

        for n, stem in enumerate(sorted(utterances)):
            path, canon = utterances[stem]
            with wave.open(path, "rb") as w:
                if w.getframerate() != 16000 or w.getnchannels() != 1 or w.getsampwidth() != 2:
                    raise ValueError("expected PCM16 mono 16 kHz: " + path)
                x = np.frombuffer(w.readframes(20 * 16000), np.int16).astype(np.float32) / 32768.0
            wt, _ = trim_silence(x, 16000)
            if len(wt) < 4800:
                continue
            lp, lens = forward_lprobs(dec, [wt])
            l = lp[0, :lens[0]].numpy()
            segs = viterbi_align(l, [dec.d.index(c) for c in canon], blank=dec.blank)
            if segs is None:
                continue
            frame = len(wt) / 16000.0 / len(l)
            for r in by_stem[stem]:
                q = r["q"]
                if q >= len(segs) or not segs[q] or canon[q] != r["canon"]:
                    continue
                a, b = max(0, segs[q][0] - 1), min(len(l), segs[q][-1] + 3)
                t0, t1 = a * frame, b * frame
                center = (segs[q][0] + segs[q][-1] + 1) * frame / 2
                rec = {k: r[k] for k in ("stem", "spk", "item", "q", "canon", "actual", "flag", "heard")}
                rec.update(next=canon[q + 1] if q + 1 < len(canon) else None,
                           prev=canon[q - 1] if q else "^", center=center,
                           cached_d1=r["f"]["d1"], scores={"full": evidence(l, a, b)},
                           wide_scores={"full": evidence(l, round(a - 0.08 / frame), round(b + 0.08 / frame))})
                completed.append(rec)
                for size in sizes:
                    lo = max(0, int((center - size / 2) * 16000))
                    hi = min(len(wt), int((center + size / 2) * 16000))
                    chunk = wt[lo:hi]
                    # Padding is fixed and label-independent, preventing feature
                    # extraction failures at utterance boundaries.
                    pad = 1600
                    chunk = np.pad(chunk, (pad, pad))
                    offset = lo / 16000.0 - pad / 16000.0
                    pending.append((chunk, rec, "crop_%.1f" % size, (t0 - offset, t1 - offset)))
                    if len(pending) >= args.batch_size:
                        flush()
            if n % 100 == 0:
                print("%d/%d elapsed=%.1fs" % (n, len(utterances), time.monotonic() - start), flush=True)
        flush()
        if completed:
            raise RuntimeError("incomplete crop records")
    meta = {"args": vars(args), "selected_sites": len(selected), "selected_utterances": len(utterances),
            "elapsed_s": time.monotonic() - start, "crop_padding_each_side_s": 0.1,
            "score": "max logp(ss) - max logp(s), at the original canonical-aligned target window; half-open intervals",
            "wide_score": "same window expanded by 80 ms on each side, for both full and cropped audio",
            "limitations": ["class-stratified sample; precision is not deployment precision",
                            "base encoder has seen these prompts; only diagnostic of context dependence",
                            "canonical CTC alignment may have timing errors"]}
    with open(args.out + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("done", meta["elapsed_s"], flush=True)


if __name__ == "__main__":
    main()
