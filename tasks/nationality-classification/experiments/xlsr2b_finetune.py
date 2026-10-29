# -*- coding: utf-8 -*-
"""XLS-R 2B(≤31층) 국적 분류기 미세조정 — ATOM-Max 같은 NPU 서버에 올릴 '큰 모델' 후보.

왜: 2B 동결 + 헤드(0702/0704)는 같은 test 에서 300m 미세조정(KD 인코더)보다 낮고(top-5 0.941 vs 0.958), 웹마이크·원어민 낭독을 못 봤다.
   NPU 용량을 쓰려면 2B 도 run5·Zeroth 데이터로 상위 층을 학습해야 한다. 구조·풀링·표준화 규약은 0704 헤드와 같다
   (`extract_features(..., layer=31)` → layer_results[31] 마스크 평균 → nan_to_num·clip ±50 → (x−mu)/sd → Linear(1920, 41)).

  학습:   $PY experiments/xlsr2b_finetune.py --data_dir $W/lab131full_zeroth --out_dir $W/run_2b [--unfreeze_from 24] [--smoke]
  평가:   $PY experiments/xlsr2b_finetune.py --eval_only --ckpt $W/run_2b/ep0.pt --test <csv> --out <dir> [--name 2b]
          → <dir>/logits.npz (ok · L_<name>, webmic_kd_eval.py 와 같은 형식) + report.json

체크포인트는 학습한 층(unfreeze_from~31)과 헤드만 담는다(`encoder_partial`). 나머지 층은 원본 xlsr2_2B_1000k.pt 에서 온다.
"""
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src"))), str(_pl.Path(__file__).resolve().parent)]
from shared.paths import NAT_OUTPUTS
import argparse, collections, csv, json, math, os, time
import numpy as np

WD = _pl.Path(NAT_OUTPUTS) / "webmic_kd"
XLSR2B = "/data/speech-ai/models/xlsr2_2B_1000k.pt"
HEAD2B = "/data/speech-ai/weights/nationality/0704_country41_sent043_en/head.pt"
HEAD2B_EN = "/data/speech-ai/weights/nationality/0702_country41_sent_best/head.pt"     # labels ↔ labels_en 대응표
LAYER = 31


def load_2b(xlsr2b, device):
    import torch
    from shared.fairseq_guard import ensure_fairseq
    ensure_fairseq()
    from fairseq import checkpoint_utils
    models, _ = checkpoint_utils.load_model_ensemble([str(xlsr2b)])
    enc = models[0]
    del enc.encoder.layers[LAYER + 1:]
    enc.encoder.layerdrop = 0.0                     # layer_results 인덱스가 밀리지 않게
    return enc.to(device)


def head_from_0704(class_order):
    """0704 헤드(한국어 라벨 순서)를 학습 클래스 순서(영문)로 재배열. (W, b, mu, sd) numpy."""
    import torch
    h = torch.load(HEAD2B, map_location="cpu")
    ref = torch.load(HEAD2B_EN, map_location="cpu")
    ko2en = dict(zip([str(x) for x in ref["labels"]], [str(x) for x in ref["labels_en"]]))
    en = [ko2en.get(str(x), str(x)) for x in h["labels"]]           # 0704 는 이미 영문 라벨, 0702 는 한글 → 영문 대응
    missing = [c for c in class_order if c not in en]
    assert not missing, f"0704 헤드에 없는 클래스: {missing}"
    perm = [en.index(c) for c in class_order]
    W = h["head"]["weight"].numpy()[perm]; b = h["head"]["bias"].numpy()[perm]
    return W.astype(np.float32), b.astype(np.float32), np.asarray(h["mu"], np.float32).reshape(-1), np.asarray(h["sd"], np.float32).reshape(-1)


def build(xlsr2b, ckpt, class_order, device, unfreeze_from):
    """(enc, head, mu, sd, labels). ckpt 가 있으면 부분 상태를 얹고, 없으면 0704 헤드로 초기화."""
    import torch, torch.nn as nn
    enc = load_2b(xlsr2b, device)
    if ckpt:
        ck = torch.load(ckpt, map_location="cpu")
        labels = [str(x) for x in ck["labels"]]
        missing, unexpected = enc.load_state_dict(ck["encoder_partial"], strict=False)
        assert not unexpected, unexpected[:5]
        W, b, mu, sd = ck["head"]["weight"].numpy(), ck["head"]["bias"].numpy(), np.asarray(ck["mu"], np.float32), np.asarray(ck["sd"], np.float32)
    else:
        labels = list(class_order)
        W, b, mu, sd = head_from_0704(labels)
    head = nn.Linear(W.shape[1], W.shape[0])
    head.load_state_dict({"weight": torch.from_numpy(W), "bias": torch.from_numpy(b)})
    return enc, head.to(device), torch.from_numpy(mu).to(device), torch.from_numpy(sd).to(device), labels


def save_partial(path, enc, head, mu, sd, labels, unfreeze_from, **extra):
    import torch
    keep = {k: v.detach().cpu() for k, v in enc.state_dict().items()
            if any(k.startswith(f"encoder.layers.{i}.") for i in range(unfreeze_from, LAYER + 1))}
    torch.save({"encoder_partial": keep, "head": {k: v.detach().cpu() for k, v in head.state_dict().items()},
                "mu": mu.cpu().numpy(), "sd": sd.cpu().numpy(), "labels": list(labels), "layer": LAYER,
                "unfreeze_from": unfreeze_from, **extra}, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default=str(WD / "lab131full_zeroth"))
    ap.add_argument("--xlsr2b", default=XLSR2B)
    ap.add_argument("--out_dir", default=str(WD / "run_2b"))
    ap.add_argument("--ckpt", default="", help="이어서/평가할 부분 체크포인트")
    ap.add_argument("--unfreeze_from", type=int, default=24, help="이 층부터 31층까지 학습")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--head_lr", type=float, default=5e-5)
    ap.add_argument("--min_lr_ratio", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--max_sec", type=float, default=8.0)
    ap.add_argument("--max_batch_sec", type=float, default=48.0)
    ap.add_argument("--max_batch", type=int, default=16)
    ap.add_argument("--eval_batch_sec", type=float, default=120.0)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--eval_every", type=int, default=1500)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="학습 행 상한(0 = 전부). 무작위 표본")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--test", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--name", default="2b")
    A = ap.parse_args()
    if A.smoke:
        A.epochs, A.eval_every, A.warmup, A.limit = 1, 30, 5, 3000
        A.out_dir = A.out_dir.rstrip("/") + "_smoke"
    os.makedirs(A.out_dir, exist_ok=True)

    import torch, torch.nn.functional as F
    from torch.utils.data import DataLoader
    from country_classifier.dataset import AudioCountryDataset, collate_audio_batch
    from country_classifier.kd_student import LengthBucketBatchSampler, forward_pooled, write_loader_csv
    from webmic_kd_train import read_rows, score
    torch.manual_seed(A.seed)
    dev = torch.device("cuda")

    sig = np.load(os.path.join(A.data_dir, "teacher_train.npz"), allow_pickle=True)
    labels = [str(x) for x in sig["labels"]]
    c2i = {c: i for i, c in enumerate(labels)}
    enc, head, mu, sd, labels_ck = build(A.xlsr2b, A.ckpt, labels, dev, A.unfreeze_from)
    assert labels_ck == labels, "클래스 순서 불일치"

    def run_rows(rows, tag, batch_sec):
        """rows → logits (N,41), seen. autocast fp16."""
        enc.eval(); head.eval()
        tmp = write_loader_csv(rows, os.path.join(A.out_dir, f"_{tag}_loader.csv"))
        ds = AudioCountryDataset(tmp, 16000, min_duration_sec=0.25, max_duration_sec=12.0)
        bs = LengthBucketBatchSampler([float(r["duration_sec"]) for r in rows], 12.0, batch_sec, 64, shuffle=False)
        ld = DataLoader(ds, batch_sampler=bs, num_workers=A.workers, collate_fn=collate_audio_batch, pin_memory=True)
        L = np.zeros((len(rows), len(labels)), np.float32); seen = np.zeros(len(rows), bool); t0 = time.time()
        with torch.no_grad():
            for bi, bt in enumerate(ld):
                if bt is None: continue
                idx = np.array([int(s) for s in bt["speaker_ids"]])
                with torch.autocast("cuda"):
                    pooled = forward_pooled(enc, bt["waveforms"].to(dev), bt["padding_mask"].to(dev), LAYER)
                L[idx] = head((pooled - mu) / sd).float().cpu().numpy(); seen[idx] = True
                if bi % 200 == 0: print(f"    [{tag}] {seen.sum():,}/{len(rows):,} ({time.time()-t0:.0f}s)", flush=True)
        enc.train(); head.train()
        return L, seen

    if A.eval_only:
        rows = read_rows(A.test); os.makedirs(A.out, exist_ok=True)
        L, seen = run_rows(rows, "test", A.eval_batch_sec)
        np.savez(os.path.join(A.out, "logits.npz"), ok=seen, **{f"L_{A.name}": L})
        r = score(rows, L, seen, c2i, None)
        json.dump({"results": {A.name: r}}, open(os.path.join(A.out, "report.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"[{A.name}] " + " · ".join(f"{k} " + "/".join(str(v) for kk, v in d.items() if kk != "n") for k, d in r.items()), flush=True)
        return

    rows_tr = read_rows(os.path.join(A.data_dir, "train.csv"))
    if A.limit and A.limit < len(rows_tr):
        rng = np.random.default_rng(A.seed); rows_tr = [rows_tr[i] for i in sorted(rng.choice(len(rows_tr), A.limit, replace=False))]
    rows_va = read_rows(os.path.join(A.data_dir, "valid.csv"))
    if A.smoke:
        rows_va = rows_va[::max(1, len(rows_va) // 600)]
    Y = torch.from_numpy(np.array([c2i[r["label_en"]] for r in rows_tr], np.int64)).to(dev)
    print(f"[data] train {len(rows_tr):,} · valid {len(rows_va):,} · 클래스 {len(labels)}", flush=True)

    for p in enc.parameters():
        p.requires_grad = False
    for li in range(A.unfreeze_from, LAYER + 1):
        for p in enc.encoder.layers[li].parameters():
            p.requires_grad = True
    enc_params = [p for p in enc.parameters() if p.requires_grad]
    print(f"[model] 2B ≤{LAYER}층 · 학습 파라미터 인코더 {sum(p.numel() for p in enc_params)/1e6:.0f}M (층 {A.unfreeze_from}–{LAYER}) + 헤드 · "
          f"전체 {sum(p.numel() for p in enc.parameters())/1e6:.0f}M", flush=True)
    opt = torch.optim.AdamW([{"params": enc_params, "lr": A.lr}, {"params": list(head.parameters()), "lr": A.head_lr}], weight_decay=A.wd)
    base_lrs = [A.lr, A.head_lr]
    scaler = torch.cuda.amp.GradScaler()
    ema_src = enc_params + list(head.parameters())
    ema_buf = [p.detach().clone() for p in ema_src] if A.ema > 0 else None

    def ema_update(step):
        d = min(A.ema, (1 + step) / (10 + step))
        with torch.no_grad():
            torch._foreach_mul_(ema_buf, d); torch._foreach_add_(ema_buf, [p.detach() for p in ema_src], alpha=1 - d)

    def ema_swap():
        if ema_buf is None: return
        with torch.no_grad():
            for e, p in zip(ema_buf, ema_src):
                tmp = p.detach().clone(); p.copy_(e); e.copy_(tmp)

    def evaluate(tag):
        L, seen = run_rows(rows_va, "valid", A.eval_batch_sec)
        r = score(rows_va, L, seen, c2i, None)
        print(f"    [{tag}] " + " · ".join(f"{k} " + "/".join(str(v) for kk, v in d.items() if kk != "n") for k, d in r.items()), flush=True)
        return r

    tmp_tr = write_loader_csv(rows_tr, os.path.join(A.out_dir, "_train_loader.csv"))
    ds_tr = AudioCountryDataset(tmp_tr, 16000, min_duration_sec=0.25, max_duration_sec=A.max_sec)
    bs_tr = LengthBucketBatchSampler([float(r["duration_sec"]) for r in rows_tr], A.max_sec, A.max_batch_sec, A.max_batch, shuffle=True, seed=A.seed)
    steps_per_ep = math.ceil(len(bs_tr) / A.grad_accum); total = steps_per_ep * A.epochs
    print(f"[sched] 배치 {len(bs_tr):,}/epoch · 옵티마이저 스텝 {steps_per_ep:,}/epoch · 총 {total:,}", flush=True)

    def set_lr(step):
        f = (step + 1) / A.warmup if step < A.warmup else A.min_lr_ratio + (1 - A.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, (step - A.warmup) / max(1, total - A.warmup))))
        for g, b in zip(opt.param_groups, base_lrs): g["lr"] = b * f

    hist = {"cfg": vars(A), "evals": []}; best = -1.0
    r0 = evaluate("valid 시작(0704 헤드 · 2B 동결 = 규약 검증)" if not A.ckpt else f"valid 시작({os.path.basename(A.ckpt)})")
    hist["evals"].append({"step": 0, **r0})
    step = 0; enc.train(); head.train()
    for ep in range(A.epochs):
        bs_tr.set_epoch(ep)
        ld = DataLoader(ds_tr, batch_sampler=bs_tr, num_workers=A.workers, collate_fn=collate_audio_batch, pin_memory=True)
        t0 = time.time(); done = 0; agg = collections.defaultdict(float)
        opt.zero_grad(set_to_none=True)
        for bi, bt in enumerate(ld):
            if bt is None: continue
            idx = torch.tensor([int(s) for s in bt["speaker_ids"]], device=dev); y = Y[idx]
            with torch.autocast("cuda"):
                pooled = forward_pooled(enc, bt["waveforms"].to(dev, non_blocking=True), bt["padding_mask"].to(dev, non_blocking=True), LAYER)
            z = head((pooled - mu) / sd).float()
            loss = F.cross_entropy(z, y)
            scaler.scale(loss / A.grad_accum).backward()
            if (bi + 1) % A.grad_accum == 0:
                set_lr(step); scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(ema_src, 5.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); step += 1
                if ema_buf is not None: ema_update(step)
                if A.eval_every and step % A.eval_every == 0:
                    ema_swap(); r = evaluate(f"valid ep{ep} step{step}"); hist["evals"].append({"step": step, **r})
                    crit = r["web"]["top5"] + r["web"]["macro5"]
                    if crit > best:
                        best = crit; save_partial(os.path.join(A.out_dir, "best.pt"), enc, head, mu, sd, labels, A.unfreeze_from, cfg=vars(A), step=step, valid=r)
                        print(f"    ✓ best 갱신 (step {step}, web top5+macro5 {crit:.4f})", flush=True)
                    ema_swap(); json.dump(hist, open(os.path.join(A.out_dir, "history.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            B = len(idx); done += B
            with torch.no_grad():
                agg["loss"] += float(loss) * B; agg["acc"] += float((z.argmax(1) == y).sum())
            if bi % 200 == 0:
                sp = done / max(time.time() - t0, 1e-9)
                print(f"  ep{ep} {done:,}/{len(rows_tr):,} step {step} ({sp:.0f} utt/s · 잔여 ~{(len(rows_tr)-done)/max(sp,1e-9)/60:.0f}분) "
                      f"loss {agg['loss']/max(done,1):.3f} · acc {agg['acc']/max(done,1):.3f} · lr {opt.param_groups[0]['lr']:.2e} · "
                      f"GPU {torch.cuda.max_memory_allocated()/2**30:.1f} GB", flush=True)
        ema_swap(); r = evaluate(f"valid ep{ep} 끝"); hist["evals"].append({"step": step, "end": True, **r})
        save_partial(os.path.join(A.out_dir, f"ep{ep}.pt"), enc, head, mu, sd, labels, A.unfreeze_from, cfg=vars(A), step=step, valid=r)
        crit = r["web"]["top5"] + r["web"]["macro5"]
        if crit > best:
            best = crit; save_partial(os.path.join(A.out_dir, "best.pt"), enc, head, mu, sd, labels, A.unfreeze_from, cfg=vars(A), step=step, valid=r)
        json.dump(hist, open(os.path.join(A.out_dir, "history.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
