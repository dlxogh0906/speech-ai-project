# -*- coding: utf-8 -*-
"""w2v-BERT 2.0 (HuggingFace) 자모 CTC 학습 — XLS-R fairseq 레시피를 그대로 옮긴 것.

같은 것: 매니페스트(tsv/ltr) · 자모 46종 사전 · tri_stage lr · fp16 · 2단계 구성.
⚠️ 마스킹(SpecAugment)은 **꺼져 있었다**(2026-09-28 확인). w2v-BERT 2.0 원본 config 가 apply_spec_augment=False 이고
이 스크립트는 mask 확률만 넘겼다 — Stage 1·2(step_38000)·hold-out H0/H4 모두 마스킹 없이 학습됐다.
fairseq XLS-R 레시피(mask 0.5)와 이 점이 다르다. 켜려면 --spec-augment.
다른 것: fairseq 가 아니라 transformers, 파형이 아니라 필터뱅크(80×2, 50Hz) 입력.
        w2v-BERT 는 conv 특징추출기가 없어 `freeze_finetune_updates` 대신 초기 N 스텝 동안 인코더를 얼린다.

    $PY .../train_w2vbert_ctc.py --manifest $MDD_OUTPUTS/manifest_kspon_native_v1 \\
        --dict $MDD_OUTPUTS/manifest_kspon_native_v1/dict.ltr.txt \\
        --save-dir $SPEECH_AI_DATA/checkpoints/w2vbert_stage1 --max-update 10000

transformers 는 학습 환경(버전 핀)에 넣지 않는다 — 별도 폴더를 얹어 실행한다:
    PYTHONPATH=/data/speech-ai/tools/hf_py38 HF_HOME=... $PY ...
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import argparse, json, math, os, random, time, wave

import numpy as np

BLANK = 0          # fairseq 와 같게 0 번을 blank 로 둔다(<s>/<pad> 자리)


def weighted_ctc(model, logits, am, lab, w):
    """HF Wav2Vec2BertForCTC 의 loss 와 같은 식에 행 가중만 얹는다.

    HF(reduction='mean') = mean_i( ctc_i / len_i ).  여기서는 Σ w_i·(ctc_i/len_i) / Σ w_i.
    가중치가 모두 1 이면 두 값이 같다(GPU 에서 실제 배치로 확인 — 결과는 18/19번 문서).
    """
    import torch
    input_lengths = model._get_feat_extract_output_lengths(am.sum([-1])).to(torch.long)
    mask = lab >= 0
    tgt_len = mask.sum(-1)
    flat = lab.masked_select(mask)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)
    with torch.backends.cudnn.flags(enabled=False):
        per = torch.nn.functional.ctc_loss(log_probs, flat, input_lengths, tgt_len, blank=model.config.pad_token_id,
                                           reduction="none", zero_infinity=model.config.ctc_zero_infinity)
    per = per / tgt_len.clamp(min=1).to(per.dtype)
    return (w * per).sum() / w.sum()


def read_split(mdir, split):
    with open(os.path.join(mdir, split + ".tsv"), encoding="utf-8") as f:
        root = f.readline().rstrip("\n")
        rows = [l.rstrip("\n").split("\t") for l in f]
    ltr = [l.split() for l in open(os.path.join(mdir, split + ".ltr"), encoding="utf-8").read().splitlines()]
    assert len(rows) == len(ltr), (len(rows), len(ltr))
    return [(os.path.join(root, p), int(fr)) for p, fr in rows], ltr


def load_vocab(dict_path):
    toks = [l.split()[0] for l in open(dict_path, encoding="utf-8") if l.strip()]
    return {t: i + 4 for i, t in enumerate(toks)}      # 0~3 = fairseq 특수토큰 자리(blank 포함)


def read_wav(path, max_sec):
    with wave.open(path, "rb") as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1, path
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0
    return x[: int(16000 * max_sec)]


def batches_by_frames(items, frames, shuffle_seed=None):
    """길이 비슷한 것끼리 묶는다(fairseq max_tokens 와 같은 취지)."""
    order = sorted(range(len(items)), key=lambda i: items[i][1])
    out, cur, n = [], [], 0
    for i in order:
        if cur and (n + items[i][1] > frames or len(cur) >= 64):
            out.append(cur); cur, n = [], 0
        cur.append(i); n += items[i][1]
    if cur:
        out.append(cur)
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(out)
    return out


def tri_stage(step, total, peak, ratios=(0.1, 0.4, 0.5), final_scale=0.05, init_scale=0.01):
    w, h = int(total * ratios[0]), int(total * ratios[1])
    if step < w:
        return peak * (init_scale + (1 - init_scale) * step / max(1, w))
    if step < w + h:
        return peak
    p = (step - w - h) / max(1, total - w - h)
    return peak * math.exp(math.log(final_scale) * p)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--train-split", dest="train_split", default="train")
    ap.add_argument("--valid-split", dest="valid_split", default="dev")
    ap.add_argument("--dict", dest="dict_path", default=None, help="기본: <manifest>/dict.ltr.txt")
    ap.add_argument("--model", default="facebook/w2v-bert-2.0", help="HF 이름 또는 이전 단계 저장 폴더")
    ap.add_argument("--save-dir", dest="save_dir", required=True)
    ap.add_argument("--max-update", dest="max_update", type=int, default=10000)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch-frames", dest="batch_frames", type=int, default=1_600_000, help="배치당 오디오 샘플 수")
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--freeze-updates", dest="freeze_updates", type=int, default=2000)
    ap.add_argument("--mask-prob", dest="mask_prob", type=float, default=0.5,
                    help="(--spec-augment 없이는 효과 없음 — 예전 실행과 같은 config 값을 남기려고 둔다)")
    ap.add_argument("--spec-augment", dest="spec_augment", action="store_true",
                    help="SpecAugment 를 실제로 켠다(apply_spec_augment=True). 확률은 아래 두 값")
    ap.add_argument("--mask-time-prob", dest="mask_time_prob", type=float, default=0.05)
    ap.add_argument("--mask-feature-prob", dest="mask_feature_prob", type=float, default=0.0)
    ap.add_argument("--encoder-lr-scale", dest="encoder_lr_scale", type=float, default=1.0,
                    help="인코더 학습률 배수(CTC 헤드는 1.0 그대로)")
    ap.add_argument("--freeze-layers", dest="freeze_layers", type=int, default=0,
                    help="아래쪽 인코더 층 N 개 + 입력 투영을 고정한다(0=고정 없음)")
    ap.add_argument("--stop-at", dest="stop_at", type=int, default=0,
                    help="이 step 에서 저장하고 멈춘다. lr 스케줄은 --max-update 기준 그대로 — "
                         "짧은 스크리닝을 전체 학습의 앞부분과 같은 조건으로 만든다")
    ap.add_argument("--save-interval", dest="save_interval", type=int, default=1000)
    ap.add_argument("--valid-n", dest="valid_n", type=int, default=2000, help="검증 표본 수")
    ap.add_argument("--max-sec", dest="max_sec", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=6, help="특징 추출 병렬 로더 수")
    ap.add_argument("--weights", default=None,
                    help="train 행마다 loss 가중치 한 줄씩(train.tsv 와 같은 순서). 주면 CTC loss 를 직접 계산해 "
                         "Σ(w·l)/Σw 로 합친다 — 가중치가 모두 1 이면 기존 'mean' 과 같은 식. "
                         "비교 실험은 가중 없는 기준선도 이 경로(전부 1)로 돌릴 것")
    a = ap.parse_args()

    import torch
    from torch.cuda.amp import GradScaler, autocast
    from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertForCTC

    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    os.makedirs(a.save_dir, exist_ok=True)
    vocab = load_vocab(a.dict_path or os.path.join(a.manifest, "dict.ltr.txt"))
    inv = {i: t for t, i in vocab.items()}
    n_out = max(vocab.values()) + 1

    tr_items, tr_ltr = read_split(a.manifest, a.train_split)
    tr_w = None
    if a.weights:
        tr_w = np.loadtxt(a.weights, dtype=np.float32, ndmin=1)
        assert len(tr_w) == len(tr_items), (len(tr_w), len(tr_items))
        assert np.all(tr_w > 0), "가중치는 양수여야 한다"
        print(f"[weights] {a.weights} · 평균 {tr_w.mean():.4f} · 최대 {tr_w.max():.3f} · "
              f"1 이 아닌 행 {(tr_w != 1).mean() * 100:.2f}%", flush=True)
    va_items, va_ltr = read_split(a.manifest, a.valid_split)
    if len(va_items) > a.valid_n:
        keep = random.Random(a.seed).sample(range(len(va_items)), a.valid_n)
        va_items = [va_items[i] for i in keep]; va_ltr = [va_ltr[i] for i in keep]
    print(f"[data] train {len(tr_items):,} · valid {len(va_items):,} · vocab {n_out}", flush=True)

    fe = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
    extra = {}
    if a.spec_augment:
        extra = {"apply_spec_augment": True, "mask_time_prob": a.mask_time_prob, "mask_time_length": 10,
                 "mask_feature_prob": a.mask_feature_prob}
    kw = dict(vocab_size=n_out, ctc_loss_reduction="mean", ctc_zero_infinity=True, pad_token_id=BLANK,
              mask_time_prob=a.mask_prob, mask_feature_prob=a.mask_prob, mask_feature_length=64,
              layerdrop=0.1, activation_dropout=0.1)
    kw.update(extra)                                   # --spec-augment 이면 마스킹 값을 덮어쓴다
    model = Wav2Vec2BertForCTC.from_pretrained(a.model, **kw).cuda()
    print(f"[aug] SpecAugment {'켬 ' + str(extra) if a.spec_augment else '꺼짐(apply_spec_augment=False)'}", flush=True)

    def apply_layer_freeze():
        if a.freeze_layers > 0:
            enc = model.wav2vec2_bert
            enc.feature_projection.requires_grad_(False)
            for layer in enc.encoder.layers[:a.freeze_layers]:
                layer.requires_grad_(False)
    apply_layer_freeze()
    if a.freeze_layers:
        n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[freeze] 아래 {a.freeze_layers}층 + 입력 투영 고정 · 학습 파라미터 {n_tr/1e6:.0f}M", flush=True)
    print(f"[model] {a.model} · {sum(p.numel() for p in model.parameters())/1e6:.0f}M", flush=True)

    enc_ids = set(id(p) for p in model.wav2vec2_bert.parameters())
    groups = [{"params": [p for p in model.parameters() if id(p) in enc_ids], "lr_scale": a.encoder_lr_scale},
              {"params": [p for p in model.parameters() if id(p) not in enc_ids], "lr_scale": 1.0}]
    opt = torch.optim.AdamW(groups, lr=a.lr, betas=(0.9, 0.98), eps=1e-8, weight_decay=0.0)
    if a.encoder_lr_scale != 1.0:
        print(f"[lr] 인코더 × {a.encoder_lr_scale} · 헤드 × 1.0", flush=True)
    scaler = GradScaler()

    def collate_cpu(idx, items, ltr):
        xs = [read_wav(items[i][0], a.max_sec) for i in idx]
        enc = fe(xs, sampling_rate=16000, return_tensors="pt", padding=True)
        ys = [[vocab[t] for t in ltr[i] if t in vocab] for i in idx]
        L = max(len(y) for y in ys)
        lab = torch.full((len(ys), L), -100, dtype=torch.long)
        for k, y in enumerate(ys):
            lab[k, :len(y)] = torch.tensor(y)
        return enc["input_features"], enc["attention_mask"], lab, ys

    def collate(idx, items, ltr):
        f, m, l, ys = collate_cpu(idx, items, ltr)
        return f.cuda(), m.cuda(), l.cuda(), ys

    class _Batches(torch.utils.data.Dataset):      # 배치 목록을 받아 워커에서 필터뱅크까지 만든다
        def __init__(self, batches): self.b = batches
        def __len__(self): return len(self.b)
        def __getitem__(self, k):
            f, m, l, ys = collate_cpu(self.b[k], tr_items, tr_ltr)
            w = torch.from_numpy(tr_w[self.b[k]]) if tr_w is not None else torch.empty(0)
            return f, m, l, ys, w



    def loader(epoch):
        bs = batches_by_frames(tr_items, a.batch_frames, shuffle_seed=a.seed + epoch)
        return torch.utils.data.DataLoader(_Batches(bs), batch_size=None, shuffle=False,
                                           num_workers=a.workers, prefetch_factor=4, persistent_workers=False)

    @torch.inference_mode()
    def validate():
        model.eval()
        import edlib
        err = tot = 0
        for idx in batches_by_frames(va_items, a.batch_frames // 2):
            feats, am, _, ys = collate(idx, va_items, va_ltr)
            with autocast():
                logits = model(input_features=feats, attention_mask=am).logits.float()
            olen = model._get_feat_extract_output_lengths(am.sum(-1)).tolist() if hasattr(
                model, "_get_feat_extract_output_lengths") else [logits.shape[1]] * len(idx)
            for k, y in enumerate(ys):
                toks = logits[k, : int(olen[k])].argmax(-1).unique_consecutive().tolist()
                hyp = "".join(inv[t] for t in toks if t != BLANK and t in inv)
                ref = "".join(inv[t] for t in y)
                err += edlib.align(hyp, ref)["editDistance"]; tot += len(ref)
        model.train()
        return err / max(1, tot)

    frozen = a.freeze_updates > 0
    if frozen:
        model.wav2vec2_bert.requires_grad_(False)            # 처음에는 CTC 헤드만 학습
    step, t0, losses, best = 0, time.time(), [], 1e9
    log = open(os.path.join(a.save_dir, "train.log"), "a", encoding="utf-8")
    model.train()
    for epoch in range(1000):
        for bi, (feats, am, lab, _, bw) in enumerate(loader(epoch)):
            if frozen and step >= a.freeze_updates:           # fairseq freeze_finetune_updates 대응
                model.wav2vec2_bert.requires_grad_(True); frozen = False
                apply_layer_freeze()
                print("[unfreeze] 인코더 학습 시작", flush=True)
            feats, am, lab = feats.cuda(non_blocking=True), am.cuda(non_blocking=True), lab.cuda(non_blocking=True)
            with autocast():
                if tr_w is None:
                    loss = model(input_features=feats, attention_mask=am, labels=lab).loss / a.accum
                else:
                    logits = model(input_features=feats, attention_mask=am).logits
                    loss = weighted_ctc(model, logits, am, lab, bw.cuda(non_blocking=True)) / a.accum
            scaler.scale(loss).backward()
            losses.append(float(loss) * a.accum)
            if (bi + 1) % a.accum:
                continue
            for g in opt.param_groups:
                g["lr"] = tri_stage(step, a.max_update, a.lr) * g.get("lr_scale", 1.0)
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 100 == 0:
                m = {"step": step, "loss": round(sum(losses[-400:]) / len(losses[-400:]), 3),
                     "lr": round(opt.param_groups[0]["lr"], 8), "wall": int(time.time() - t0)}
                print(json.dumps(m), flush=True); log.write(json.dumps(m) + "\n"); log.flush()
            if step % a.save_interval == 0 or step == a.max_update or (a.stop_at and step == a.stop_at):
                uer = validate()
                m = {"step": step, "valid_uer": round(uer * 100, 3), "wall": int(time.time() - t0)}
                print(json.dumps(m), flush=True); log.write(json.dumps(m) + "\n"); log.flush()
                d = os.path.join(a.save_dir, f"step_{step}")
                model.save_pretrained(d); fe.save_pretrained(d)
                json.dump({"vocab": vocab, "blank": BLANK}, open(os.path.join(d, "jamo_vocab.json"), "w"), ensure_ascii=False)
                if uer < best:
                    best = uer
                    json.dump({"best_step": step, "valid_uer": uer}, open(os.path.join(a.save_dir, "best.json"), "w"))
            if step >= a.max_update or (a.stop_at and step >= a.stop_at):
                print(f"[done] {step} 스텝 · best valid UER {best*100:.3f}% · {int(time.time()-t0)}s", flush=True)
                return


if __name__ == "__main__":
    main()
