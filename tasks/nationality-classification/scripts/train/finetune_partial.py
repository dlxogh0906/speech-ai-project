"""Partial fine-tuning of XLS-R 2B (top-N transformer layers + head) for the fine split.

Frozen-feature probing hit a ceiling (macro top-5 ~0.50). This unfreezes only the top N
transformer layers (layer-wise LR: tiny on encoder, normal on head) and trains end-to-end
on audio. To keep epochs feasible we cap utterances per class (undersample the dominant
classes; keep all minority utts -- safer than oversampling, which we showed hurts).

Eval metric = MACRO top-5 (per-class top-5 averaged = real deployment number), plus micro.

    python scripts/finetune_partial.py --manifest_dir data/aihub_nationality_fine_811 \
        --out_dir runs/ft_top8_fine811 --unfreeze_top_layers 8 --cap_per_class 2000 \
        --batch_size 8 --grad_accum_steps 4 --epochs 4 --max_sec 8 --enc_lr 1e-5 --head_lr 1e-3
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists(): _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.fairseq_guard import ensure_fairseq
ensure_fairseq()
from shared.paths import FAIRSEQ_ROOT, MODEL_ROOT, NAT_DATA


import argparse
import csv
import re
import time
from pathlib import Path

import numpy as np

from country_classifier import baseline_common as bc


def cap_csv(src, dst, cap, seed=42):
    """Write dst keeping at most `cap` rows per label_id (random subset of majors)."""
    rng = np.random.default_rng(seed)
    rows = list(csv.DictReader(open(src, encoding="utf-8")))
    by = {}
    for r in rows:
        by.setdefault(r["label_id"], []).append(r)
    out = []
    for lid, rs in by.items():
        if cap and len(rs) > cap:
            idx = rng.choice(len(rs), cap, replace=False)
            rs = [rs[i] for i in idx]
        out.extend(rs)
    with open(dst, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=bc.CSV_FIELDS)
        w.writeheader(); w.writerows(out)
    return len(out)


def unfreeze_top_layers(encoder, n):
    """Freeze all encoder params, then unfreeze the top `n` transformer layers."""
    for p in encoder.parameters():
        p.requires_grad = False
    layer_ids = set()
    for name, _ in encoder.named_parameters():
        m = re.search(r"layers\.(\d+)\.", name)
        if m:
            layer_ids.add(int(m.group(1)))
    if not layer_ids:
        raise RuntimeError("could not locate transformer layers in encoder")
    keep = set(sorted(layer_ids)[-n:])
    n_un = 0
    for name, p in encoder.named_parameters():
        m = re.search(r"layers\.(\d+)\.", name)
        if (m and int(m.group(1)) in keep) or name.startswith("layer_norm"):
            p.requires_grad = True
            n_un += p.numel()
    return keep, n_un


def disable_activation_checkpointing(encoder):
    """Restore original (non-checkpointed) forward so AMP works and backward is ~2x faster.

    fairseq's checkpoint_wrapper monkeypatches layer.forward and saves the original as
    `precheckpoint_forward`; we restore it. (The activation-checkpointing recompute is what
    clashed with autocast -> 'Half vs Float' in layer_norm.)
    """
    n = 0
    for m in encoder.modules():
        if hasattr(m, "precheckpoint_forward"):
            m.forward = m.precheckpoint_forward
            n += 1
    return n


def macro_micro_top5(logits, y, num_classes):
    idx = np.argpartition(-logits, 4, axis=1)[:, :5]
    hit = np.array([yi in r for yi, r in zip(y, idx)])
    micro = float(hit.mean()) if len(y) else 0.0
    present = sorted(set(y.tolist()))
    macro = float(np.mean([hit[y == c].mean() for c in present])) if present else 0.0
    return micro, macro, len(present)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest_dir", default=f"{NAT_DATA}/aihub_nationality_fine_811")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--xlsr_ckpt", default=f"{MODEL_ROOT}/xlsr2_2B_1000k.pt")
    ap.add_argument("--fairseq_root", default=str(FAIRSEQ_ROOT))
    ap.add_argument("--audio_root", default=bc.DEFAULT_AUDIO_ROOT)
    ap.add_argument("--unfreeze_top_layers", type=int, default=8)
    ap.add_argument("--cap_per_class", type=int, default=8000)
    ap.add_argument("--valid_cap", type=int, default=600, help="cap valid utts/class (selection)")
    ap.add_argument("--test_cap", type=int, default=1500, help="cap test utts/class (report)")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--grad_accum_steps", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max_sec", type=float, default=8.0)
    ap.add_argument("--enc_lr", type=float, default=1e-5)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit_smoke", type=int, default=0, help=">0: tiny smoke run")
    args = ap.parse_args()

    import torch
    from country_classifier.dataset import build_dataloader
    from country_classifier.model import build_model
    from country_classifier.utils import set_seed

    set_seed(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    dev = "cuda"

    labels, label2id, id2label = bc.build_label_maps(args.manifest_dir)
    C = len(labels)

    md = Path(args.manifest_dir)
    train_csv = out_dir / "train_capped.csv"
    valid_csv = out_dir / "valid_capped.csv"
    test_csv = out_dir / "test_capped.csv"
    n_tr = cap_csv(md / "train.csv", train_csv, args.cap_per_class if not args.limit_smoke else 0, args.seed)
    n_va = cap_csv(md / "valid.csv", valid_csv, args.valid_cap, args.seed)
    n_te = cap_csv(md / "test.csv", test_csv, args.test_cap, args.seed)
    if args.limit_smoke:
        for p in (train_csv, valid_csv, test_csv):
            rows = list(csv.DictReader(open(p, encoding="utf-8")))[: args.limit_smoke]
            with open(p, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=bc.CSV_FIELDS); w.writeheader(); w.writerows(rows)
        n_tr = n_va = n_te = args.limit_smoke
    print(f"[data] train={n_tr:,} (cap {args.cap_per_class}/cls)  valid={n_va:,}  test={n_te:,}", flush=True)

    train_loader = build_dataloader(train_csv, 16000, args.batch_size, shuffle=True, seed=args.seed,
                                    max_duration_sec=args.max_sec, num_workers=args.num_workers)
    valid_loader = build_dataloader(valid_csv, 16000, args.batch_size, shuffle=False, seed=args.seed,
                                    max_duration_sec=args.max_sec, num_workers=args.num_workers)
    test_loader = build_dataloader(test_csv, 16000, args.batch_size, shuffle=False, seed=args.seed,
                                   max_duration_sec=args.max_sec, num_workers=args.num_workers)

    print("[model] loading XLS-R 2B ...", flush=True)
    t0 = time.time()
    model = build_model(backend="fairseq_xlsr", num_classes=C, checkpoint_path=args.xlsr_ckpt,
                        fairseq_root=args.fairseq_root, freeze_encoder=False)  # grad flows
    keep, n_un = unfreeze_top_layers(model.encoder, args.unfreeze_top_layers)
    n_ckpt = disable_activation_checkpointing(model.encoder)
    model.to(dev)
    try:
        model.encoder.encoder.layerdrop = 0.0
    except Exception:
        pass
    print(f"[model] loaded {time.time()-t0:.0f}s; unfrozen top layers {sorted(keep)}, "
          f"{n_un/1e6:.1f}M enc params + head; checkpointing off on {n_ckpt} modules", flush=True)

    enc_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("encoder")]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("encoder")]
    opt = torch.optim.AdamW([
        {"params": enc_params, "lr": args.enc_lr},
        {"params": head_params, "lr": args.head_lr},
    ], weight_decay=args.weight_decay)
    # AMP is safe now that activation checkpointing is disabled (no recompute dtype clash).
    scaler = torch.cuda.amp.GradScaler()

    def evaluate(loader):
        model.eval()
        L, Y = [], []
        with torch.no_grad():
            for b in loader:
                if b is None:
                    continue
                with torch.autocast("cuda"):
                    out = model(b["waveforms"].to(dev), b["padding_mask"].to(dev))
                L.append(out.float().cpu().numpy()); Y.extend(b["labels"].tolist())
        return macro_micro_top5(np.concatenate(L), np.array(Y), C)

    log = out_dir / "ft_log.csv"
    log.write_text("epoch,train_loss,valid_macro,valid_micro,test_macro,test_micro,sec\n", encoding="utf-8")
    best_valid = -1; best_report = None
    for ep in range(1, args.epochs + 1):
        model.train()
        for p in model.encoder.parameters():  # keep frozen layers in eval-like (no update anyway)
            pass
        opt.zero_grad(set_to_none=True)
        t1 = time.time(); run = 0.0; seen = 0; step = 0
        for b in train_loader:
            if b is None:
                continue
            try:
                with torch.autocast("cuda"):
                    out = model(b["waveforms"].to(dev), b["padding_mask"].to(dev))
                    loss = torch.nn.functional.cross_entropy(out, b["labels"].to(dev)) / args.grad_accum_steps
                scaler.scale(loss).backward(); step += 1
                if step % args.grad_accum_steps == 0:
                    scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("[OOM] reduce --batch_size or --max_sec, raise --grad_accum_steps", flush=True)
                    torch.cuda.empty_cache()
                raise
            run += loss.item() * args.grad_accum_steps * b["labels"].size(0); seen += b["labels"].size(0)
            if step % 100 == 0:
                print(f"  e{ep} step{step} loss={run/max(1,seen):.3f} seen={seen} "
                      f"({seen/(time.time()-t1):.1f} utt/s)", flush=True)
        vmi, vma, _ = evaluate(valid_loader)        # selection (no test leakage)
        tmi, tma, pres = evaluate(test_loader)       # report only
        dt = time.time() - t1
        print(f"[epoch {ep}] train_loss={run/max(1,seen):.4f} | VALID macro={vma:.4f} micro={vmi:.4f} "
              f"| TEST macro={tma:.4f} micro={tmi:.4f} (present {pres}) {dt:.0f}s", flush=True)
        with log.open("a") as f:
            f.write(f"{ep},{run/max(1,seen):.4f},{vma:.4f},{vmi:.4f},{tma:.4f},{tmi:.4f},{dt:.0f}\n")
        if vma > best_valid:
            best_valid = vma; best_report = (ep, tma, tmi)
            torch.save({"classifier": model.classifier.state_dict(), "label2id": label2id,
                        "unfrozen": sorted(keep), "epoch": ep, "valid_macro": vma,
                        "test_macro": tma, "test_micro": tmi}, out_dir / "best_head.pt")
    ep, tma, tmi = best_report
    print(f"\n[done] best epoch={ep} (by valid macro) -> TEST macro top-5={tma:.4f} micro top-5={tmi:.4f}"
          f"  | baseline frozen: macro 0.498 / micro 0.931  -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
