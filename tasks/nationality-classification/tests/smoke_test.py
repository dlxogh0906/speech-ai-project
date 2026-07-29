"""End-to-end smoke test for the country-classification baseline.

Validates, on a handful of samples:
  1. manifest -> CSV materialisation
  2. audio path resolution
  3. WAV loading via country_classifier.audio
  4. label map
  5. model forward (one batch)
  6. cross-entropy loss
  7. top-1 / top-5 metrics

It uses REAL AIHub audio when it is present (after `extract_audio.py`); otherwise it
falls back to SYNTHETIC wavs so the full code path is still exercised on a machine
that does not yet have the corpus extracted. Pass --synthetic to force synthetic mode,
or --backend tiny to skip the heavy XLS-R 2B checkpoint entirely.

Examples:
    python scripts/smoke_test.py --manifest_dir <set> --backend tiny --batch_size 2 --max_sec 5
    python scripts/smoke_test.py --manifest_dir <set> --backend fairseq_xlsr \
        --xlsr_ckpt $SPEECH_AI_MODELS/xlsr2_2B_1000k.pt --batch_size 1 --max_sec 5
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.fairseq_guard import ensure_fairseq
ensure_fairseq()
from shared.paths import FAIRSEQ_ROOT, MODEL_ROOT


import argparse
import sys
import tempfile
from pathlib import Path

from country_classifier import baseline_common as bc


def log(step, msg):
    print(f"[smoke] {step:<22s} {msg}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest_dir", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--backend", default="tiny", choices=["tiny", "fairseq_xlsr"])
    ap.add_argument("--xlsr_ckpt", default=f"{MODEL_ROOT}/xlsr2_2B_1000k.pt")
    ap.add_argument("--fairseq_root", default=str(FAIRSEQ_ROOT))
    ap.add_argument("--audio_root", default=bc.DEFAULT_AUDIO_ROOT)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--max_sec", type=float, default=5.0)
    ap.add_argument("--n_samples", type=int, default=8)
    ap.add_argument("--sample_rate", type=int, default=16000)
    ap.add_argument("--synthetic", action="store_true",
                    help="force synthetic audio even if real audio exists")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader

    from country_classifier.dataset import AudioCountryDataset, collate_audio_batch
    from country_classifier.metrics import TopKAccuracy
    from country_classifier.model import build_model
    from country_classifier.utils import set_seed

    set_seed(0)
    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    log("device", device)

    labels, label2id, id2label = bc.build_label_maps(args.manifest_dir)
    num_classes = len(labels)
    log("labels", f"{num_classes} classes: {labels}")

    workdir = Path(tempfile.mkdtemp(prefix="smoke_"))
    csv_path = workdir / "smoke.csv"

    use_synth = args.synthetic
    if not use_synth:
        audio_index = bc.build_audio_index(args.audio_root)
        if audio_index:
            log("audio index", f"{len(audio_index):,} wavs")
        bc.prepare_split_csv(args.manifest_dir, args.split, csv_path, label2id,
                             limit=args.n_samples, audio_root=args.audio_root,
                             audio_index=audio_index or None)
        present, checked = bc.audio_exists_fraction(csv_path, sample=args.n_samples)
        log("audio probe", f"{present}/{checked} real wavs present")
        if present == 0:
            log("fallback", "no real audio found -> using SYNTHETIC wavs")
            use_synth = True
    if use_synth:
        csv_path = bc.generate_synthetic_csv(workdir, labels, label2id,
                                             n_per_split=max(args.n_samples, args.batch_size * 2),
                                             sample_rate=args.sample_rate,
                                             seconds=min(args.max_sec, 4.0))

    dataset = AudioCountryDataset(csv_path, sample_rate=args.sample_rate,
                                  max_duration_sec=args.max_sec, skip_errors=True)
    log("dataset", f"{len(dataset)} rows from {csv_path}")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_audio_batch)
    batch = None
    for batch in loader:
        if batch is not None:
            break
    if batch is None:
        sys.exit("[smoke] FAIL: every sample was skipped (audio unreadable/too short)")
    log("batch", f"waveforms={tuple(batch['waveforms'].shape)} "
                 f"labels={batch['labels'].tolist()} skipped={len(batch['skipped'])}")

    log("build model", f"backend={args.backend}")
    if args.backend == "tiny":
        model = build_model(backend="tiny", num_classes=num_classes)
    else:
        model = build_model(backend="fairseq_xlsr", num_classes=num_classes,
                            checkpoint_path=args.xlsr_ckpt, fairseq_root=args.fairseq_root,
                            freeze_encoder=True)
    model.to(device)

    waveforms = batch["waveforms"].to(device)
    padding_mask = batch["padding_mask"].to(device)
    targets = batch["labels"].to(device)

    model.train()
    logits = model(waveforms, padding_mask)
    log("forward", f"logits={tuple(logits.shape)}")
    assert logits.shape == (waveforms.shape[0], num_classes)

    loss = torch.nn.functional.cross_entropy(logits, targets)
    loss.backward()
    log("loss+backward", f"loss={loss.item():.4f}")

    meter = TopKAccuracy(num_classes)
    meter.update(logits.detach(), targets)
    m = meter.compute()
    log("metrics", f"top1={m['top1_acc']:.3f} top5={m['top5_acc']:.3f} n={m['num_samples']}")

    print("\n[smoke] PASS (synthetic)" if use_synth else "\n[smoke] PASS (real audio)")


if __name__ == "__main__":
    main()
