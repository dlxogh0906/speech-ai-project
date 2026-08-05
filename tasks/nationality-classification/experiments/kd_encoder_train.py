# -*- coding: utf-8 -*-
"""인코더 증류 본학습 — 300m 상위층을 2B 교사에 맞춰 학습.

헤드 KD 의 종착(꼬리 동률 −0.4%p)을 넘으려면 표현 자체를 바꿔야 한다.
변형 실험의 결론을 이식한다:

  손실 = (1−w_i)·CE + w_i·T²·KL(z_s/T ‖ z_t/T) + β·(1−cos(proj(pool17), L31_t))
  w_i  = α · margin_i          ← 게이팅: 교사가 모르는 행(043 등)은 스스로 강등
  (DKD·특징KD 단독은 기각됐지만, 특징 항은 인코더가 '학습 가능'해진 지금은 의미가 다르다)

구조:
  XLS-R 300m → layer17 출력 → 마스크 평균풀링 → (x−mu)/sd → 선형 헤드(41)
  - 18층 이상은 사용하지 않으므로 **절단** (메모리·속도 절약)
  - 하위층(0..unfreeze 미만)은 동결 — 표현 붕괴 방지 + 속도
  - mu/sd 는 헤드 KD 때의 train 통계로 고정, 헤드는 CE-baseline 가중치로 초기화

평가: 고정 speaker_split(json) 의 valid 로 epoch 선택, test 는 최종 1회.
비교선: baseline 0.7510/0.9331/0.6170 · 헤드KD 최고 0.7696/0.9412/0.6356.

    $PY .../kd_encoder_train.py --smoke            # 500행 1에폭 파이프라인 검증
    $PY .../kd_encoder_train.py                    # 본학습 (~수 시간/epoch, GPU)
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), str(_r / "tasks/nationality-classification/experiments"),
                 *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import DATA_ROOT, NAT_TASK, NAT_OUTPUTS, FAIRSEQ_ROOT

import argparse, csv, json, math, os, time
from collections import defaultdict
import numpy as np

ENC_DIR = NAT_TASK / "outputs" / "kd_encoder"


def load_manifest(path, classes):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    c2i = {c: i for i, c in enumerate(classes)}
    return rows, np.array([c2i[r["label_en"]] for r in rows], np.int64)


def loader_csv(rows, dst):
    """행번호를 speaker_id 칸에 실어 로더 호환 CSV 생성 (스킵 복원용)."""
    with open(dst, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["audio_path", "json_path", "speaker_id", "label_name", "label_id", "duration_sec"])
        for i, r in enumerate(rows):
            w.writerow([r["audio_path"], "", str(i), r["label_en"], 0, r["duration_sec"]])
    return dst


def macro5_of(hits5, y):
    pres = sorted(set(y.tolist()))
    return float(np.mean([hits5[y == c].mean() for c in pres]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--unfreeze_from", type=int, default=12, help="이 층부터 17층까지 학습 (그 아래 동결)")
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--T", type=float, default=2.0)
    ap.add_argument("--beta", type=float, default=0.5, help="특징(cos) 손실 가중")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--head_lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_sec", type=float, default=8.0, help="학습 크롭 (평가는 12s)")
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--xlsr300", default=os.path.expanduser("~/models/xlsr2_300m.pt"))
    ap.add_argument("--signals", default=str(ENC_DIR / "teacher_signals.npz"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out_dir", default=str(ENC_DIR / "run1"))
    A = ap.parse_args()
    # 기본 체크포인트 경로 정리 (models 디렉터리 우선 탐색)
    import glob as _g
    if not os.path.isfile(A.xlsr300):
        for cand in [os.path.expanduser("~/models/xlsr2_300m.pt"), f"{DATA_ROOT}/models/xlsr2_300m.pt"]:
            if os.path.isfile(cand): A.xlsr300 = cand; break
    os.makedirs(A.out_dir, exist_ok=True)

    import torch, torch.nn as nn, torch.nn.functional as F
    from shared.fairseq_guard import ensure_fairseq
    ensure_fairseq()
    from fairseq import checkpoint_utils
    from country_classifier.dataset import build_dataloader

    # ── 데이터·교사 신호 ──
    sig = np.load(A.signals, allow_pickle=True)
    classes = [str(c) for c in sig["classes"]]
    rows_tr, y_tr = load_manifest(ENC_DIR / "train_subset.csv", classes)
    rows_va, y_va = load_manifest(ENC_DIR / "valid.csv", classes)
    rows_te, y_te = load_manifest(ENC_DIR / "test.csv", classes)
    trow = sig["row"]; TLOG = sig["logits"].astype(np.float32)
    TFEAT = sig["pooled31"]; TMARG = sig["margin"].astype(np.float32)
    row2sig = np.full(len(rows_tr), -1, np.int64)
    row2sig[trow] = np.arange(len(trow))
    print(f"[data] train {len(rows_tr):,} (교사신호 {len(trow):,}) · valid {len(rows_va):,} · test {len(rows_te):,}", flush=True)
    if A.smoke:
        rows_tr, y_tr = rows_tr[:500], y_tr[:500]
        rows_va, y_va = rows_va[:800], y_va[:800]
        rows_te, y_te = rows_te[:800], y_te[:800]
        A.epochs = 1

    # ── 학생 모델: 300m 절단(≤17층) + 풀링 + 헤드 ──
    print(f"[model] 300m 로드: {A.xlsr300}", flush=True)
    models, _cfg = checkpoint_utils.load_model_ensemble([A.xlsr300])
    m300 = models[0]
    enc = m300                                            # wav2vec2 모델 (encoder.layers)
    LAYER = 17
    del enc.encoder.layers[LAYER + 1:]                    # 절단 — 18층 이상 미사용
    for p in enc.parameters(): p.requires_grad = False
    for li in range(A.unfreeze_from, LAYER + 1):
        for p in enc.encoder.layers[li].parameters(): p.requires_grad = True

    ce_head = torch.load(NAT_TASK / "outputs/kd_variants/kd_heads/ce_baseline_head.pt",
                         map_location="cpu") if (NAT_TASK / "outputs/kd_variants/kd_heads/ce_baseline_head.pt").exists() \
        else torch.load(NAT_TASK / "outputs/kd_no043/kd_heads/ce_baseline_head.pt", map_location="cpu")
    assert list(ce_head["labels"]) == classes, "헤드 클래스 순서 불일치"
    mu = torch.from_numpy(np.asarray(ce_head["mu"], np.float32)).cuda()
    sd = torch.from_numpy(np.asarray(ce_head["sd"], np.float32)).cuda()
    head = nn.Linear(1024, len(classes))
    head.load_state_dict(ce_head["head"])
    proj = nn.Linear(1024, 1920)                          # 특징 손실용 투영
    enc.cuda(); head.cuda(); proj.cuda()

    params = [{"params": [p for p in enc.parameters() if p.requires_grad], "lr": A.lr},
              {"params": list(head.parameters()) + list(proj.parameters()), "lr": A.head_lr}]
    opt = torch.optim.AdamW(params, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    n_train = sum(p.numel() for p in enc.parameters() if p.requires_grad)
    print(f"[model] 학습 파라미터: 인코더 {n_train/1e6:.1f}M (층 {A.unfreeze_from}–{LAYER}) + 헤드/투영", flush=True)

    def fwd(wav, pm, train=True):
        with torch.autocast("cuda"):
            res = enc.extract_features(source=wav, padding_mask=pm, mask=False)
            x = res["layer_results"][LAYER][0].transpose(0, 1)       # (B,T,C)
            fp = res.get("padding_mask")
            if fp is None: pooled = x.mean(1)
            else:
                v = (~fp).to(x.dtype).unsqueeze(-1)
                pooled = (x * v).sum(1) / v.sum(1).clamp_min(1.0)
        pooled = torch.nan_to_num(pooled.float()).clamp(-50, 50)
        z = head((pooled - mu) / sd)
        return z, pooled

    @torch.no_grad()
    def evaluate(rows, y, tag, max_n=0):
        enc.eval(); head.eval()
        tmp = loader_csv(rows, f"{A.out_dir}/_eval_tmp.csv")
        ld = build_dataloader(tmp, 16000, 24, shuffle=False, seed=0,
                              max_duration_sec=12.0, num_workers=8)
        hits1 = np.zeros(len(rows), bool); hits5 = np.zeros(len(rows), bool); seen = np.zeros(len(rows), bool)
        for bt in ld:
            if bt is None: continue
            idx = np.array([int(s) for s in bt["speaker_ids"]])
            z, _ = fwd(bt["waveforms"].cuda(), bt["padding_mask"].cuda(), train=False)
            L = z.cpu().numpy()
            t5 = np.argpartition(-L, 4, 1)[:, :5]
            hits1[idx] = L.argmax(1) == y[idx]
            hits5[idx] = [a in r for a, r in zip(y[idx], t5)]
            seen[idx] = True
            if max_n and seen.sum() >= max_n: break
        m = seen
        r = {"n": int(m.sum()), "micro1": round(float(hits1[m].mean()), 4),
             "micro5": round(float(hits5[m].mean()), 4),
             "macro5": round(macro5_of(hits5[m], y[m]), 4)}
        print(f"    [{tag}] {r}", flush=True)
        return r

    # ── 학습 ──
    T0, a0, beta = A.T, A.alpha, A.beta
    TT = torch.from_numpy(TLOG).cuda()
    TM = torch.from_numpy(TMARG).cuda()
    TF = torch.from_numpy(TFEAT.astype(np.float32)).cuda()
    R2S = torch.from_numpy(row2sig).cuda()
    hist = {"cfg": vars(A), "epochs": []}
    best = {"vm": -1}
    tmp_tr = loader_csv(rows_tr, f"{A.out_dir}/_train_tmp.csv")
    for ep in range(A.epochs):
        enc.train(); head.train()
        ld = build_dataloader(tmp_tr, 16000, A.batch_size, shuffle=True, seed=ep,
                              max_duration_sec=A.max_sec, num_workers=8)
        t0 = time.time(); done = 0; ls = defaultdict(float); nb = 0
        opt.zero_grad()
        for bi, bt in enumerate(ld):
            if bt is None: continue
            idx = torch.tensor([int(s) for s in bt["speaker_ids"]], device="cuda")
            si = R2S[idx]
            has = si >= 0
            z, pooled = fwd(bt["waveforms"].cuda(), bt["padding_mask"].cuda())
            yb = torch.from_numpy(y_tr[idx.cpu().numpy()]).cuda()
            ce = F.cross_entropy(z, yb, reduction="none")
            loss = ce.mean() * 0
            w = torch.zeros_like(ce)
            if has.any():
                sih = si[has]
                kl = F.kl_div(F.log_softmax(z[has] / T0, 1),
                              F.softmax(TT[sih] / T0, 1), reduction="none").sum(1)
                w_h = a0 * TM[sih]
                ce_h = (1 - w_h) * ce[has] + w_h * (T0 * T0) * kl
                feat = 1 - F.cosine_similarity(proj(pooled[has]), TF[sih], dim=1)
                loss = loss + (ce_h + beta * feat).sum() / len(ce)
            if (~has).any():
                loss = loss + ce[~has].sum() / len(ce)
            scaler.scale(loss / A.grad_accum).backward()
            if (bi + 1) % A.grad_accum == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad()
            done += len(ce); ls["loss"] += float(loss) * len(ce); nb += len(ce)
            if bi % 200 == 0:
                sp = done / max(time.time() - t0, 1e-9)
                print(f"  ep{ep} {done:,}/{len(rows_tr):,} ({sp:.0f} utt/s · loss {ls['loss']/max(nb,1):.3f} · "
                      f"잔여 ~{(len(rows_tr)-done)/max(sp,1e-9)/60:.0f}분)", flush=True)
        va = evaluate(rows_va, y_va, f"valid ep{ep}", max_n=20000 if not A.smoke else 0)
        hist["epochs"].append({"ep": ep, "valid": va, "elapsed": round(time.time() - t0, 1)})
        if va["macro5"] > best["vm"]:
            best = {"vm": va["macro5"], "ep": ep}
            torch.save({"encoder": enc.state_dict(), "head": head.state_dict(),
                        "mu": mu.cpu().numpy(), "sd": sd.cpu().numpy(), "labels": classes,
                        "layer": LAYER, "unfreeze_from": A.unfreeze_from,
                        "alpha": a0, "T": T0, "beta": beta},
                       f"{A.out_dir}/encoder_kd_best.pt")
            print(f"    ✓ best 갱신 (ep{ep}, valid macro5 {va['macro5']})", flush=True)
        json.dump(hist, open(f"{A.out_dir}/history.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # ── 최종 test (best 로드) ──
    ck = torch.load(f"{A.out_dir}/encoder_kd_best.pt", map_location="cuda")
    enc.load_state_dict(ck["encoder"]); head.load_state_dict(ck["head"])
    te = evaluate(rows_te, y_te, "test(final)")
    hist["test"] = te
    hist["reference"] = {"baseline": {"micro1": 0.7510, "micro5": 0.9331, "macro5": 0.6170},
                         "head_kd_best": {"micro1": 0.7696, "micro5": 0.9412, "macro5": 0.6356}}
    json.dump(hist, open(f"{A.out_dir}/history.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[saved] {A.out_dir}/history.json · encoder_kd_best.pt")


if __name__ == "__main__":
    main()
