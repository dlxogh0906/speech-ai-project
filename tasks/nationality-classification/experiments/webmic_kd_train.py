# -*- coding: utf-8 -*-
"""웹마이크 재학습 — 배포 KD 모델에서 출발해 040~043 을 배우고, 옛 지식은 자기 증류로 지킨다.

구조는 배포 모델과 같다(CPU 서빙 그대로): XLS-R 300m ≤17층 → 마스크 평균풀링 → (x−mu)/sd → 선형 헤드(41).
층 --unfreeze_from ~17 만 학습(기본 12, kd_encoder_train.py 와 같다). mu/sd 는 고정.

손실 (행 i, 교사 = 학습 전 배포 모델, 신호는 webmic_kd_signals.py)
  라벨 행      (1 − w_i)·CE_i + w_i·T²·KL_i        w_i = α · margin_i · [교사 top1 == 라벨]
               ← 교사가 틀린 행(040 미국 → 유럽 국가 등)은 증류를 끄고 정답만 배운다.
  라벨 없는 행  λ_u·T²·KL_i + β_u·(1 − cos(ẑs_i, ẑt_i))      ẑ = (pooled − mu)/sd
  (131 = 505 도메인)  ← 정답을 모르니 옛 모델의 출력·표현을 그대로 따라가게 해서 보존한다.
배치: 길이 버킷(패딩 최소화). 입력은 앞 --max_sec 초(평가는 12초 — 서빙과 같다).
선택: 매 평가마다 valid 로 기록, 기준(웹 micro5 + macro5)이 오르면 best 저장. epoch 끝마다 ep{n}.pt 도 저장.
      --ema 를 주면 평가·저장 모두 EMA 가중치다(스텝 간 흔들림 완화).
배치 섞기: 길이 버킷은 과제·도메인과 상관이 커서(단어 ≈ 짧은 배치, 131 ≈ 8초 배치) 스텝마다 구성이 쏠린다.
      --grad_accum 을 키우고 --max_batch_sec 를 줄이면 한 스텝에 서로 다른 길이 배치가 여러 개 섞인다.

    $PY .../webmic_kd_train.py --smoke        # 소량 파이프라인 확인
    $PY .../webmic_kd_train.py --epochs 4     # 본학습 (GPU)
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import NAT_OUTPUTS, XLSR_300M

import argparse, collections, csv, json, math, os, time
import numpy as np

WD = NAT_OUTPUTS / "webmic_kd"


def read_rows(path, limit=0):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    return rows[:limit] if limit else rows


def score(rows, logits, seen, c2i, t_logits=None):
    """valid/test 요약: 웹(라벨) micro/macro · 데이터셋별 · 과제군별 · 131 교사 일치 · 한국어."""
    dom = np.array([r["domain"] for r in rows], dtype=object)
    y = np.array([c2i.get(r["label_en"], -1) for r in rows])
    task = np.array(["word" if r["task"] == "RW" else "sent" for r in rows], dtype=object)
    top5 = np.argsort(-logits, 1)[:, :5]
    h1 = top5[:, 0] == y
    h5 = (top5 == y[:, None]).any(1)
    web = seen & np.array([d.startswith("web") for d in dom])
    out = {}
    if web.any():
        cls = sorted(set(y[web].tolist()))
        out["web"] = {"n": int(web.sum()), "top1": round(float(h1[web].mean()), 4),
                      "top5": round(float(h5[web].mean()), 4),
                      "macro5": round(float(np.mean([h5[web & (y == c)].mean() for c in cls])), 4)}
        for d in sorted({d for d in dom[web]}):
            m = web & (dom == d)
            out[d] = {"n": int(m.sum()), "top1": round(float(h1[m].mean()), 4), "top5": round(float(h5[m].mean()), 4)}
        for g in ["sent", "word"]:
            m = web & (task == g)
            if m.any():
                out["web_" + g] = {"n": int(m.sum()), "top1": round(float(h1[m].mean()), 4),
                                   "top5": round(float(h5[m].mean()), 4)}
    p505 = seen & np.array([d.startswith("p505") for d in dom])
    if (p505 & (y >= 0)).any():                                  # 131 을 라벨과 함께 쓸 때
        m = p505 & (y >= 0)
        cls = sorted(set(y[m].tolist()))
        out["p505"] = {"n": int(m.sum()), "top1": round(float(h1[m].mean()), 4), "top5": round(float(h5[m].mean()), 4),
                       "macro5": round(float(np.mean([h5[m & (y == c)].mean() for c in cls])), 4)}
    elif p505.any() and t_logits is not None:
        t1 = t_logits.argmax(1)
        out["p505"] = {"n": int(p505.sum()), "agree_top1": round(float((top5[p505, 0] == t1[p505]).mean()), 4),
                       "teacher_in_top5": round(float((top5[p505] == t1[p505][:, None]).any(1).mean()), 4)}
    kor = seen & np.array([d in ("kspon", "kor001") for d in dom])
    if kor.any():
        out["korean"] = {"n": int(kor.sum()), "top1": round(float(h1[kor].mean()), 4)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default=str(WD))
    ap.add_argument("--init", default=str(WD / "init_kd_from_onnx.pt"))
    ap.add_argument("--xlsr300", default=str(XLSR_300M))
    ap.add_argument("--out_dir", default=str(WD / "run1"))
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--unfreeze_from", type=int, default=12)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--T", type=float, default=2.0)
    ap.add_argument("--lambda_u", type=float, default=1.0, help="라벨 없는 행 KL 가중")
    ap.add_argument("--beta_u", type=float, default=0.5, help="라벨 없는 행 표현(cos) 가중")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head_lr", type=float, default=2e-4)
    ap.add_argument("--min_lr_ratio", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--max_sec", type=float, default=8.0)
    ap.add_argument("--max_batch_sec", type=float, default=128.0)
    ap.add_argument("--max_batch", type=int, default=48)
    ap.add_argument("--eval_batch_sec", type=float, default=240.0, help="평가 배치의 오디오 총량(초) — GPU 공유 시 줄인다")
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--eval_every", type=int, default=1500, help="옵티마이저 스텝 간격 (0 = epoch 끝만)")
    ap.add_argument("--ema", type=float, default=0.0, help="가중치 이동평균 감쇠(예 0.999). 켜면 평가·저장을 EMA 가중치로")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--aug", type=float, default=0.0, help="학습 행에 채널 증강(country_classifier.wav_augment)을 거는 확률. 라벨과 무관하게 모든 행")
    ap.add_argument("--aug_profile", default="channel", choices=["channel", "booth"], help="booth = 채널 증강 + 군중 잡음·폰 대역·짧은 발화 (부스 시연 조건)")
    ap.add_argument("--spk_balance", action="store_true",
                    help="국가 비율(N_c/N)은 그대로 두고, 국가 안에서는 화자마다 같은 비중(1/S_c), 화자 안에서는 발화를 고르게(1/n_s) "
                         "복원 추출한다. 발화가 많은 몇 명의 목소리를 외우는 것을 줄이려는 실험(2026-10-01)")
    ap.add_argument("--epoch_size", type=int, default=0, help="--spk_balance 일 때 epoch 당 뽑을 행 수 (0 = 학습 행 수와 같게)")
    ap.add_argument("--focal_gamma", type=float, default=0.0, help="라벨 CE 를 focal loss (1-p)^γ·CE 로 바꾼다 (0 = 끔, 기존과 같음)")
    ap.add_argument("--extra_labels", default="",
                    help="교사·초기 체크포인트에 없는 새 국가(쉼표 구분, train.csv 의 label_en 값). 헤드를 늘리고(기존 행 그대로 · 새 행 0) "
                         "새 국가 행은 교사 신호가 없어 CE 로만 배운다. 교사 확률은 새 국가 칸을 0 으로 채운다(2026-10-01 확장 파일럿)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    A = ap.parse_args()
    if A.smoke:
        A.epochs, A.eval_every, A.warmup = 1, 20, 5
        A.out_dir = A.out_dir.rstrip("/") + "_smoke"
    os.makedirs(A.out_dir, exist_ok=True)

    import torch, torch.nn.functional as F
    from torch.utils.data import DataLoader
    from country_classifier.dataset import AudioCountryDataset, collate_audio_batch
    from country_classifier.kd_student import (LengthBucketBatchSampler, build_student, forward_pooled,
                                               save_student, write_loader_csv)
    torch.manual_seed(A.seed)
    dev = torch.device("cuda")

    # ── 데이터·교사 신호 ──
    lim_tr, lim_ev = (3000, 600) if A.smoke else (0, 0)
    rows_tr = read_rows(os.path.join(A.data_dir, "train.csv"), lim_tr)
    rows_va = read_rows(os.path.join(A.data_dir, "valid.csv"))
    if A.smoke:
        rows_va = rows_va[::max(1, len(rows_va) // lim_ev)]
    sig = np.load(os.path.join(A.data_dir, "teacher_train.npz"), allow_pickle=True)
    sig_labels = [str(x) for x in sig["labels"]]
    extra = [x.strip() for x in A.extra_labels.split(",") if x.strip() and x.strip() not in sig_labels]
    labels = sig_labels + extra
    c2i = {c: i for i, c in enumerate(labels)}
    y_tr = np.array([c2i[r["label_en"]] if r["label_en"] else -1 for r in rows_tr], np.int64)
    r2s = np.full(len(rows_tr), -1, np.int64)
    keep = sig["row"] < len(rows_tr)
    r2s[sig["row"][keep]] = np.nonzero(keep)[0]
    TL = torch.from_numpy(sig["logits"]).to(dev)
    if extra:                                                     # 교사는 새 국가를 모른다 — 확률 0 (softmax 전 -1e4)
        TL = torch.cat([TL, torch.full((TL.shape[0], len(extra)), -1e4, dtype=TL.dtype, device=dev)], 1)
    TP = torch.from_numpy(sig["pooled"].astype(np.float32)).to(dev)
    TM = torch.from_numpy(sig["margin"].astype(np.float32)).to(dev)
    R2S = torch.from_numpy(r2s).to(dev)
    Y = torch.from_numpy(y_tr).to(dev)
    t_va = None                                                   # 131 라벨이 없을 때만 쓰는 '배포 모델 일치율' 기준
    tv_path = os.path.join(A.data_dir, "teacher_valid.npz")
    if not A.smoke and os.path.exists(tv_path):
        tv = np.load(tv_path, allow_pickle=True)
        t_va = np.zeros((len(rows_va), len(labels)), np.float32)
        t_va[tv["row"], :tv["logits"].shape[1]] = tv["logits"]
    n_lab = int((y_tr >= 0).sum())
    print(f"[data] train {len(rows_tr):,} (라벨 {n_lab:,} · 라벨없음 {len(rows_tr)-n_lab:,} · 교사신호 {(r2s>=0).sum():,}) "
          f"· valid {len(rows_va):,} · 증강 p={A.aug} ({A.aug_profile})", flush=True)

    # ── 모델 ──
    enc, head, mu, sd, labels_ck, layer = build_student(A.xlsr300, A.init, dev)
    assert labels_ck == sig_labels, "교사 신호와 초기 체크포인트의 클래스 순서가 다르다"
    if extra:                                                     # 헤드 확장: 기존 41행은 그대로, 새 행은 가중치 0 · 편향은 기존 평균
        import torch.nn as nn
        big = nn.Linear(head.in_features, len(labels)).to(dev)
        with torch.no_grad():
            big.weight.zero_(); big.bias.fill_(float(head.bias.mean()))
            big.weight[:len(labels_ck)] = head.weight; big.bias[:len(labels_ck)] = head.bias
        head = big
        print(f"[model] 헤드 확장 {len(labels_ck)} → {len(labels)} (새 국가: {', '.join(extra)})", flush=True)
    for p in enc.parameters():
        p.requires_grad = False
    for li in range(A.unfreeze_from, layer + 1):
        for p in enc.encoder.layers[li].parameters():
            p.requires_grad = True
    enc_params = [p for p in enc.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": A.lr},
                             {"params": list(head.parameters()), "lr": A.head_lr}], weight_decay=A.wd)
    base_lrs = [A.lr, A.head_lr]
    scaler = torch.cuda.amp.GradScaler()
    ema_src = enc_params + list(head.parameters())
    ema_buf = [p.detach().clone() for p in ema_src] if A.ema > 0 else None

    def ema_update(step):
        d = min(A.ema, (1 + step) / (10 + step))
        with torch.no_grad():
            torch._foreach_mul_(ema_buf, d)
            torch._foreach_add_(ema_buf, [p.detach() for p in ema_src], alpha=1 - d)

    def ema_swap():
        """학습 가중치 ↔ EMA 가중치 맞바꾸기 (평가·저장 전후로 두 번 부른다)."""
        if ema_buf is None:
            return
        with torch.no_grad():
            for e, p in zip(ema_buf, ema_src):
                tmp = p.detach().clone()
                p.copy_(e)
                e.copy_(tmp)
    print(f"[model] 학습 파라미터: 인코더 {sum(p.numel() for p in enc_params)/1e6:.1f}M (층 {A.unfreeze_from}–{layer}) + 헤드",
          flush=True)

    tmp_tr = write_loader_csv(rows_tr, os.path.join(A.out_dir, "_train_loader.csv"))
    tmp_va = write_loader_csv(rows_va, os.path.join(A.out_dir, "_valid_loader.csv"))
    class _AugDataset(AudioCountryDataset):
        """학습 행에 확률 A.aug 로 채널 증강. 교사 신호는 원음 기준이지만 표현 일치(KD)는 그대로 둔다 — 증강에 불변인 표현을 배우게 한다."""
        _calls = 0

        def __getitem__(self, idx):
            it = super().__getitem__(idx)
            if A.aug > 0 and "_skip" not in it:
                _AugDataset._calls += 1
                rng = np.random.default_rng([A.seed, idx, _AugDataset._calls, os.getpid()])
                if rng.random() < A.aug:
                    from country_classifier import wav_augment as WA
                    if A.aug_profile == "booth":
                        if not WA._BABBLE["paths"]:
                            WA.set_babble_pool(_babble_paths)
                        it["waveform"] = torch.from_numpy(WA.augment_booth(it["waveform"].numpy(), rng, max_sec=A.max_sec))
                    else:
                        it["waveform"] = torch.from_numpy(WA.augment(it["waveform"].numpy(), rng, max_sec=A.max_sec))
            return it

    _babble_paths = [r["audio_path"] for r in rows_tr if r.get("domain") == "kspon"][:3000]      # 군중 잡음 재료 (부스 프로필)
    ds_tr = _AugDataset(tmp_tr, 16000, min_duration_sec=0.25, max_duration_sec=A.max_sec)
    ds_va = AudioCountryDataset(tmp_va, 16000, min_duration_sec=0.25, max_duration_sec=12.0)
    bs_tr = LengthBucketBatchSampler([float(r["duration_sec"]) for r in rows_tr], A.max_sec, A.max_batch_sec,
                                     A.max_batch, shuffle=True, seed=A.seed)
    if A.spk_balance:
        class _SpkBalancedBatches:
            """행 r 의 추출 확률 ∝ (N_c / N) · (1 / S_c) · (1 / n_s). epoch 마다 다시 뽑고 길이 버킷으로 묶는다."""

            def __init__(self, rows, size, seed):
                cls = np.array([r["label_en"] for r in rows]); spk = np.array([r["speaker"] for r in rows])
                _, ci, n_c = np.unique(cls, return_inverse=True, return_counts=True)
                _, si, n_s = np.unique(spk, return_inverse=True, return_counts=True)
                s_c = np.zeros(len(n_c), np.int64)
                for c in range(len(n_c)):
                    s_c[c] = len(np.unique(si[ci == c]))
                w = n_c[ci] / s_c[ci] / n_s[si]
                self.p = w / w.sum()
                self.d = np.array([float(r["duration_sec"]) for r in rows])
                self.size, self.seed, self.epoch = (size or len(rows)), seed, 0
                print(f"[spk_balance] 국가 {len(n_c)} · 화자 {len(n_s):,} · epoch 당 {self.size:,}행 복원 추출", flush=True)
                self._make()

            def set_epoch(self, ep):
                self.epoch = ep
                self._make()

            def _make(self):
                rng = np.random.default_rng([self.seed, 7, self.epoch])
                self.idx = rng.choice(len(self.p), size=self.size, replace=True, p=self.p)
                inner = LengthBucketBatchSampler(self.d[self.idx], A.max_sec, A.max_batch_sec, A.max_batch,
                                                 shuffle=True, seed=self.seed)
                inner.set_epoch(self.epoch)
                self.batches = [[int(self.idx[i]) for i in b] for b in inner._batches()]

            def __len__(self):
                return len(self.batches)

            def __iter__(self):
                return iter(self.batches)
        bs_tr = _SpkBalancedBatches(rows_tr, A.epoch_size, A.seed)
    bs_va = LengthBucketBatchSampler([float(r["duration_sec"]) for r in rows_va], 12.0, A.eval_batch_sec, 128,
                                     shuffle=False)
    steps_per_ep = math.ceil(len(bs_tr) / A.grad_accum)
    total = steps_per_ep * A.epochs
    print(f"[sched] 배치 {len(bs_tr):,}/epoch · 옵티마이저 스텝 {steps_per_ep:,}/epoch · 총 {total:,}", flush=True)

    def set_lr(step):
        if step < A.warmup:
            f = (step + 1) / A.warmup
        else:
            prog = min(1.0, (step - A.warmup) / max(1, total - A.warmup))
            f = A.min_lr_ratio + (1 - A.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * prog))
        for g, b in zip(opt.param_groups, base_lrs):
            g["lr"] = b * f

    @torch.no_grad()
    def evaluate(tag):
        enc.eval(); head.eval()
        ld = DataLoader(ds_va, batch_sampler=bs_va, num_workers=A.workers, collate_fn=collate_audio_batch,
                        pin_memory=True)
        L = np.zeros((len(rows_va), len(labels)), np.float32)
        seen = np.zeros(len(rows_va), bool)
        for bt in ld:
            if bt is None:
                continue
            idx = np.array([int(s) for s in bt["speaker_ids"]])
            with torch.autocast("cuda"):
                pooled = forward_pooled(enc, bt["waveforms"].to(dev), bt["padding_mask"].to(dev), layer)
            L[idx] = head((pooled - mu) / sd).float().cpu().numpy()
            seen[idx] = True
        enc.train(); head.train()
        r = score(rows_va, L, seen, c2i, t_va)
        print(f"    [{tag}] " + " · ".join(
            f"{k} " + "/".join(f"{v}" for kk, v in d.items() if kk != "n") for k, d in r.items()), flush=True)
        return r

    hist = {"cfg": vars(A), "evals": []}
    best = -1.0
    r0 = evaluate(f"valid 시작(초기 가중치 {os.path.basename(A.init)})")
    hist["evals"].append({"step": 0, "ep": 0, **r0})
    T2 = A.T * A.T
    step = 0
    enc.train(); head.train()
    for ep in range(A.epochs):
        bs_tr.set_epoch(ep)
        ld = DataLoader(ds_tr, batch_sampler=bs_tr, num_workers=A.workers, collate_fn=collate_audio_batch,
                        pin_memory=True, persistent_workers=False)
        t0 = time.time(); done = 0
        agg = collections.defaultdict(float)
        opt.zero_grad(set_to_none=True)
        for bi, bt in enumerate(ld):
            if bt is None:
                continue
            idx = torch.tensor([int(s) for s in bt["speaker_ids"]], device=dev)
            si = R2S[idx]
            y = Y[idx]
            with torch.autocast("cuda"):
                pooled = forward_pooled(enc, bt["waveforms"].to(dev, non_blocking=True),
                                        bt["padding_mask"].to(dev, non_blocking=True), layer)
            zn = (pooled - mu) / sd
            z = head(zn).float()
            B = len(idx)
            lab = y >= 0
            hs = si >= 0
            sic = si.clamp_min(0)
            t_log = TL[sic]
            kl = F.kl_div(F.log_softmax(z / A.T, 1), F.softmax(t_log / A.T, 1), reduction="none").sum(1) * T2
            ce = F.cross_entropy(z, y.clamp_min(0), reduction="none")
            ce_l = (1 - torch.exp(-ce)).pow(A.focal_gamma) * ce if A.focal_gamma > 0 else ce   # focal: (1-p_t)^γ·CE
            t_ok = (t_log.argmax(1) == y) & hs
            w = A.alpha * TM[sic] * t_ok.float()
            loss_lab = torch.where(lab, (1 - w) * ce_l + w * kl, torch.zeros_like(ce))
            tzn = (TP[sic] - mu) / sd
            cos = F.cosine_similarity(zn, tzn, dim=1)
            unl = (~lab) & hs
            loss_unl = torch.where(unl, A.lambda_u * kl + A.beta_u * (1 - cos), torch.zeros_like(ce))
            loss = (loss_lab.sum() + loss_unl.sum()) / B
            scaler.scale(loss / A.grad_accum).backward()
            if (bi + 1) % A.grad_accum == 0:
                set_lr(step)
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(enc_params + list(head.parameters()), 5.0)
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True)
                step += 1
                if ema_buf is not None:
                    ema_update(step)
                if A.eval_every and step % A.eval_every == 0:
                    ema_swap()
                    r = evaluate(f"valid ep{ep} step{step}")
                    hist["evals"].append({"step": step, "ep": ep, **r})
                    crit = r["web"]["top5"] + r["web"]["macro5"]
                    if crit > best:
                        best = crit
                        save_student(os.path.join(A.out_dir, "best.pt"), enc, head, mu, sd, labels, layer,
                                     cfg=vars(A), step=step, valid=r)
                        print(f"    ✓ best 갱신 (step {step}, web top5+macro5 {crit:.4f})", flush=True)
                    ema_swap()
                    json.dump(hist, open(os.path.join(A.out_dir, "history.json"), "w", encoding="utf-8"),
                              ensure_ascii=False, indent=1)
            done += B
            with torch.no_grad():
                agg["loss"] += float(loss) * B
                agg["ce"] += float(ce[lab].sum())
                agg["n_lab"] += float(lab.sum())
                agg["kl_u"] += float(kl[unl].sum())
                agg["cos_u"] += float(cos[unl].sum())
                agg["n_unl"] += float(unl.sum())
                agg["acc"] += float(((z.argmax(1) == y) & lab).sum())
                agg["w"] += float(w.sum())
            if bi % 300 == 0:
                sp = done / max(time.time() - t0, 1e-9)
                nl, nu = max(agg["n_lab"], 1), max(agg["n_unl"], 1)
                print(f"  ep{ep} {done:,}/{len(rows_tr):,} step {step} ({sp:.0f} utt/s · 잔여 ~{(len(rows_tr)-done)/max(sp,1e-9)/60:.0f}분) "
                      f"loss {agg['loss']/max(done,1):.3f} · ce {agg['ce']/nl:.3f} · acc {agg['acc']/nl:.3f} · "
                      f"kd가중 {agg['w']/nl:.3f} · KLu {agg['kl_u']/nu:.3f} · cosu {agg['cos_u']/nu:.4f} · "
                      f"lr {opt.param_groups[0]['lr']:.2e}", flush=True)
        ema_swap()
        r = evaluate(f"valid ep{ep} 끝")
        hist["evals"].append({"step": step, "ep": ep, "end": True, **r})
        crit = r["web"]["top5"] + r["web"]["macro5"]
        save_student(os.path.join(A.out_dir, f"ep{ep}.pt"), enc, head, mu, sd, labels, layer,
                     cfg=vars(A), step=step, valid=r)
        if crit > best:
            best = crit
            save_student(os.path.join(A.out_dir, "best.pt"), enc, head, mu, sd, labels, layer,
                         cfg=vars(A), step=step, valid=r)
            print(f"    ✓ best 갱신 (ep{ep} 끝, web top5+macro5 {crit:.4f})", flush=True)
        ema_swap()
        hist.setdefault("epoch_sec", []).append(round(time.time() - t0))
        json.dump(hist, open(os.path.join(A.out_dir, "history.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
    print(f"[done] best {best:.4f} → {A.out_dir}/best.pt")


if __name__ == "__main__":
    main()
