"""Partial fine-tune XLS-R 2B for 41-class COUNTRY output (boss wants per-country).
Adds the two levers that pair with fine-tuning:
  * logit-adjustment loss (balanced softmax)  -> macro / long-tail
  * SpecAugment on encoder features (on-the-fly) -> generalization, esp. small classes
Top-N transformer layers unfrozen with layer-wise LR; majority capped via --cap_per_class.
Logs BOTH top-1 and top-5 (top-1 is the product metric). Saves head only.

Example (real run, in tmux):
  PY=$HOME/miniconda3/envs/fairseq_xlsr/bin/python
  $PY finetune_country41.py --out_dir $NATIONALITY_OUTPUTS/ft_country41 \
      --unfreeze_top_layers 6 --cap_per_class 6000 --batch_size 16 --grad_accum_steps 2 \
      --epochs 8 --max_sec 12 --enc_lr 1e-5 --head_lr 1e-3 --logit_adjust_tau 1.0 --specaug 1
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists(): _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.fairseq_guard import ensure_fairseq
ensure_fairseq()
from shared.paths import FAIRSEQ_ROOT, MODEL_ROOT, NAT_DATA

import argparse, csv, re, sys, time
from pathlib import Path
import numpy as np
from country_classifier import baseline_common as bc


def cap_csv(src, dst, cap, seed=42):
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
        w = csv.DictWriter(f, fieldnames=bc.CSV_FIELDS); w.writeheader(); w.writerows(out)
    return out


def unfreeze_top_layers(encoder, n):
    for p in encoder.parameters():
        p.requires_grad = False
    layer_ids = set()
    for name, _ in encoder.named_parameters():
        m = re.search(r"layers\.(\d+)\.", name)
        if m:
            layer_ids.add(int(m.group(1)))
    if not layer_ids:
        raise RuntimeError("could not locate transformer layers")
    keep = set(sorted(layer_ids)[-n:]); n_un = 0
    for name, p in encoder.named_parameters():
        m = re.search(r"layers\.(\d+)\.", name)
        if (m and int(m.group(1)) in keep) or name.startswith("layer_norm"):
            p.requires_grad = True; n_un += p.numel()
    return keep, n_un


def disable_activation_checkpointing(encoder):
    n = 0
    for m in encoder.modules():
        if hasattr(m, "precheckpoint_forward"):
            m.forward = m.precheckpoint_forward; n += 1
    return n


def metrics(logits, y, C):
    p1 = logits.argmax(1); i5 = np.argpartition(-logits, 4, axis=1)[:, :5]
    h1 = (p1 == y); h5 = np.array([yi in r for yi, r in zip(y, i5)])
    pres = sorted(set(y.tolist()))
    f = lambda h: (float(h.mean()), float(np.mean([h[y == c].mean() for c in pres])))
    mi1, ma1 = f(h1); mi5, ma5 = f(h5)
    return {"micro1": mi1, "macro1": ma1, "micro5": mi5, "macro5": ma5, "present": len(pres)}


def spec_augment(x, n_freq, F, n_time, T):
    """x: (B, Tt, D) encoder features. Zero random freq/time bands (shared across batch)."""
    import torch
    B, Tt, D = x.shape
    x = x.clone()
    for _ in range(n_freq):
        if F > 0 and D > 1:
            w = int(torch.randint(0, F + 1, (1,))); f0 = int(torch.randint(0, max(1, D - w), (1,)))
            x[:, :, f0:f0 + w] = 0
    for _ in range(n_time):
        if T > 0 and Tt > 1:
            w = int(torch.randint(0, min(T, Tt) + 1, (1,))); t0 = int(torch.randint(0, max(1, Tt - w), (1,)))
            x[:, t0:t0 + w, :] = 0
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest_dir", default=f"{NAT_DATA}/aihub_country41_811")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--xlsr_ckpt", default=f"{MODEL_ROOT}/xlsr2_2B_1000k.pt")
    ap.add_argument("--fairseq_root", default=str(FAIRSEQ_ROOT))
    ap.add_argument("--unfreeze_top_layers", type=int, default=6)
    ap.add_argument("--cap_per_class", type=int, default=6000)
    ap.add_argument("--valid_cap", type=int, default=600)
    ap.add_argument("--test_cap", type=int, default=1500)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--grad_accum_steps", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--max_sec", type=float, default=12.0)
    ap.add_argument("--enc_lr", type=float, default=1e-5)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--logit_adjust_tau", type=float, default=1.0, help="0 disables")
    ap.add_argument("--specaug", type=int, default=1)
    ap.add_argument("--specaug_freq", type=int, default=2)
    ap.add_argument("--specaug_F", type=int, default=160)
    ap.add_argument("--specaug_time", type=int, default=2)
    ap.add_argument("--specaug_T", type=int, default=40)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit_smoke", type=int, default=0)
    args = ap.parse_args()

    import torch
    from country_classifier.model import build_model
    from country_classifier.dataset import build_dataloader
    from country_classifier.utils import set_seed
    set_seed(args.seed); dev = "cuda"
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    labels, label2id, id2label = bc.build_label_maps(args.manifest_dir)
    C = len(labels); md = Path(args.manifest_dir)
    tr_csv, va_csv, te_csv = out_dir / "train_capped.csv", out_dir / "valid_capped.csv", out_dir / "test_capped.csv"
    tr_rows = cap_csv(md / "train.csv", tr_csv, args.cap_per_class if not args.limit_smoke else 0, args.seed)
    cap_csv(md / "valid.csv", va_csv, args.valid_cap, args.seed)
    cap_csv(md / "test.csv", te_csv, args.test_cap, args.seed)
    if args.limit_smoke:
        for p in (tr_csv, va_csv, te_csv):
            rows = list(csv.DictReader(open(p, encoding="utf-8")))[: args.limit_smoke]
            with open(p, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=bc.CSV_FIELDS); w.writeheader(); w.writerows(rows)

    # logit-adjustment prior from (capped) train distribution
    cnt = np.ones(C)
    for r in csv.DictReader(open(tr_csv, encoding="utf-8")):
        cnt[int(r["label_id"])] += 1
    prior = torch.tensor(np.log(cnt / cnt.sum()), dtype=torch.float32, device=dev)
    print(f"[data] train={len(tr_rows):,} cap {args.cap_per_class}/cls | C={C} | "
          f"logit_adjust_tau={args.logit_adjust_tau} specaug={args.specaug}", flush=True)

    mk = lambda p, sh: build_dataloader(p, 16000, args.batch_size, shuffle=sh, seed=args.seed,
                                        max_duration_sec=args.max_sec, num_workers=args.num_workers)
    train_loader, valid_loader, test_loader = mk(tr_csv, True), mk(va_csv, False), mk(te_csv, False)

    print("[model] loading XLS-R 2B ...", flush=True); t0 = time.time()
    model = build_model(backend="fairseq_xlsr", num_classes=C, checkpoint_path=args.xlsr_ckpt,
                        fairseq_root=args.fairseq_root, freeze_encoder=False)
    keep, n_un = unfreeze_top_layers(model.encoder, args.unfreeze_top_layers)
    disable_activation_checkpointing(model.encoder); model.to(dev)
    try: model.encoder.encoder.layerdrop = 0.0
    except Exception: pass
    print(f"[model] {time.time()-t0:.0f}s; unfrozen {sorted(keep)} ({n_un/1e6:.1f}M)", flush=True)

    enc_p = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("encoder")]
    hd_p = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("encoder")]
    opt = torch.optim.AdamW([{"params": enc_p, "lr": args.enc_lr},
                             {"params": hd_p, "lr": args.head_lr}], weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler()

    def fwd(b, train):
        wav = b["waveforms"].to(dev); pm = b["padding_mask"].to(dev)
        feats, fmask = model._extract_features(wav, pm)
        if train and args.specaug:
            feats = spec_augment(feats, args.specaug_freq, args.specaug_F, args.specaug_time, args.specaug_T)
        return model.classifier(model._pool(feats, fmask))

    def evaluate(loader):
        model.eval(); L, Y = [], []
        with torch.no_grad():
            for b in loader:
                if b is None: continue
                with torch.autocast("cuda"):
                    out = fwd(b, False)
                L.append(out.float().cpu().numpy()); Y.extend(b["labels"].tolist())
        return metrics(np.concatenate(L), np.array(Y), C)

    log = out_dir / "ft_log.csv"
    log.write_text("epoch,loss,v_top1,v_top5,t_top1,t_macro1,t_top5,t_macro5,sec\n", encoding="utf-8")
    best_v = -1
    for ep in range(1, args.epochs + 1):
        model.train(); opt.zero_grad(set_to_none=True); t1 = time.time(); run = seen = step = 0
        for b in train_loader:
            if b is None: continue
            with torch.autocast("cuda"):
                out = fwd(b, True); y = b["labels"].to(dev)
                logit = out + args.logit_adjust_tau * prior if args.logit_adjust_tau else out
                loss = torch.nn.functional.cross_entropy(logit, y) / args.grad_accum_steps
            scaler.scale(loss).backward(); step += 1
            if step % args.grad_accum_steps == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            run += loss.item() * args.grad_accum_steps * y.size(0); seen += y.size(0)
            if step % 100 == 0:
                print(f"  e{ep} s{step} loss={run/max(1,seen):.3f} ({seen/(time.time()-t1):.1f} utt/s)", flush=True)
        v = evaluate(valid_loader); t = evaluate(test_loader); dt = time.time() - t1
        print(f"[ep {ep}] loss={run/max(1,seen):.4f} | VALID top1={v['micro1']:.4f} top5={v['micro5']:.4f} "
              f"| TEST top1={t['micro1']:.4f} macro1={t['macro1']:.4f} top5={t['micro5']:.4f} macro5={t['macro5']:.4f} "
              f"(present {t['present']}) {dt:.0f}s", flush=True)
        with log.open("a") as f:
            f.write(f"{ep},{run/max(1,seen):.4f},{v['micro1']:.4f},{v['micro5']:.4f},"
                    f"{t['micro1']:.4f},{t['macro1']:.4f},{t['micro5']:.4f},{t['macro5']:.4f},{dt:.0f}\n")
        if v["micro1"] > best_v:
            best_v = v["micro1"]
            torch.save({"classifier": model.classifier.state_dict(), "label2id": label2id,
                        "unfrozen": sorted(keep), "epoch": ep, **{f"test_{k}": t[k] for k in t}},
                       out_dir / "best_head.pt")
            print(f"   * saved best (valid top1={best_v:.4f})", flush=True)
    print(f"[done] -> {out_dir} (baseline frozen country41: top1 0.833 macro1 0.356)", flush=True)


if __name__ == "__main__":
    main()
