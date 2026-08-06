# -*- coding: utf-8 -*-
"""MDD 평가: ⑨ PER(외국인/원어민 분리) + ⑩ gold MDD F1.

- fairseq wav2vec_ctc 체크포인트 로드 → CTC greedy 디코드(학습 criterion과 동일)
- ⑨ PER: test.tsv(외국인) / native test / gold별 = editdistance(pred, target)/len(target)
- ⑩ F1: gold 오디오 디코드 → 정답음소열(target_phones)과 정규화 편집거리 = 오발음 점수
        → gold has_error 와 P/R/F1 (임계값 dev로 튜닝, test에서 보고) + AP
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
from shared.paths import DATA_ROOT, MDD_OUTPUTS

import sys, os, json, wave, collections
import numpy as np
import torch
import torch.nn.functional as F
import editdistance
from fairseq import checkpoint_utils

CKPT = f"{DATA_ROOT}/mdd_ckpt/checkpoint_best.pt"
GOLD_SPEC = f"{MDD_OUTPUTS}/gold_eval/gold_eval_spec.jsonl"
GOLD_WAV = f"{DATA_ROOT}/mdd_gold_wav16k"
OUT = f"{MDD_OUTPUTS}/eval_results.json"
MAX_SAMPLES_PER_BATCH = 1_600_000     # GPU 배치 예산(16k 샘플)


def load_wav(path):
    with wave.open(path, "rb") as w:
        n = w.getnframes(); raw = w.readframes(n)
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0


class Decoder:
    """GPU 가 있으면 fp16 GPU, 없으면 fp32 CPU 로 올린다.

    예전에는 `.cuda().half()` 가 무조건 걸려 GPU 없는 머신에서 적재 자체가 죽었다.
    CPU 에서 fp16 은 느리거나 미지원이라 CPU 일 때만 fp32 로 간다 —
    **GPU 경로는 이전과 완전히 동일하다.**
    `MDD_FORCE_CPU=1` 로 GPU 가 있어도 CPU 를 강제할 수 있다(속도 측정·검증용).
    """

    def __init__(self):
        models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([CKPT])
        m = models[0].eval()
        use_cuda = torch.cuda.is_available() and os.environ.get("MDD_FORCE_CPU") != "1"
        self.model = m.cuda().half() if use_cuda else m.float()
        self.device = "cuda" if use_cuda else "cpu"
        self.task = task
        self.d = task.target_dictionary
        self.blank = self.d.index(task.blank_symbol) if hasattr(task, "blank_symbol") else 0
        self.normalize = cfg.task.normalize
        print(f"[model] loaded | blank_idx={self.blank} normalize={self.normalize} vocab={len(self.d)}", flush=True)

    @torch.inference_mode()
    def decode(self, wavs):
        lens = [len(w) for w in wavs]; mx = max(lens)
        src = torch.zeros(len(wavs), mx); pm = torch.zeros(len(wavs), mx, dtype=torch.bool)
        for i, w in enumerate(wavs):
            t = torch.from_numpy(w)
            if self.normalize:
                t = F.layer_norm(t, t.shape)
            src[i, :len(w)] = t; pm[i, len(w):] = True
        q = next(self.model.parameters())
        src = src.to(device=q.device, dtype=q.dtype); pm = pm.to(q.device)
        net = self.model(source=src, padding_mask=pm)
        lprobs = self.model.get_normalized_probs(net, log_probs=True).transpose(0, 1).float().cpu()  # (B,T,C)
        opm = net.get("padding_mask")
        out = []
        for b in range(len(wavs)):
            lp = lprobs[b]
            if opm is not None:
                L = int((~opm[b]).sum()); lp = lp[:L]
            toks = lp.argmax(-1).unique_consecutive()
            out.append([self.d[i] for i in toks.tolist() if i != self.blank])
        return out


def batched(items):  # items: list of (wav_path, ...) → yield batches of indices by length
    order = sorted(range(len(items)), key=lambda i: items[i][0])  # by frames (items[i][0]=frames)
    batch, cur = [], 0
    for i in order:
        fr = items[i][0]
        if batch and cur + fr > MAX_SAMPLES_PER_BATCH:
            yield batch; batch, cur = [], 0
        batch.append(i); cur += fr
    if batch:
        yield batch


def run_per(dec, tsv, ltr, name):
    """tsv/ltr 로 PER 측정."""
    paths = []
    with open(tsv) as f:
        f.readline()
        for line in f:
            p, fr = line.rstrip("\n").split("\t")[:2]
            paths.append(("/" + p, int(fr)))
    tgts = [l.split() for l in open(ltr).read().splitlines()]
    items = [(fr, p, t) for (p, fr), t in zip(paths, tgts)]
    c_err = c_len = 0; n = 0
    for bi, idxs in enumerate(batched(items)):
        wavs = [load_wav(items[i][1]) for i in idxs]
        preds = dec.decode(wavs)
        for j, i in enumerate(idxs):
            c_err += editdistance.eval(preds[j], items[i][2]); c_len += len(items[i][2]); n += 1
        if bi % 200 == 0:
            print(f"  [{name}] {n:,}/{len(items):,} PER={c_err/max(c_len,1)*100:.2f}%", flush=True)
    per = c_err / max(c_len, 1) * 100
    print(f"[PER {name}] {per:.2f}%  ({n:,} 발화)", flush=True)
    return {"per": round(per, 3), "n": n}


def run_gold_f1(dec):
    """gold 오디오 디코드 → 정규화 편집거리(오발음 점수) → has_error 와 F1."""
    recs = []
    for line in open(GOLD_SPEC, encoding="utf-8"):
        d = json.loads(line); s = d["stem"]
        wp = os.path.join(GOLD_WAV, s[:2], s + ".wav")
        if not os.path.exists(wp):
            continue
        try:
            fr = wave.open(wp, "rb").getnframes()
        except Exception:
            continue
        recs.append({"path": wp, "frames": fr, "target": d["target_phones"] or [],
                     "has_error": 1 if d["has_error"] else 0,
                     "L1": d.get("L1"), "stem": s})
    print(f"[gold] {len(recs):,} 발화 디코드 시작", flush=True)
    items = [(r["frames"], r) for r in recs]
    scores = np.zeros(len(recs)); labels = np.zeros(len(recs)); per_err = per_len = 0
    done = 0
    idx_of = {id(r): k for k, r in enumerate(recs)}
    for bi, idxs in enumerate(batched(items)):
        wavs = [load_wav(items[i][1]["path"]) for i in idxs]
        preds = dec.decode(wavs)
        for j, i in enumerate(idxs):
            r = items[i][1]; tgt = r["target"]
            ed = editdistance.eval(preds[j], tgt)
            scores[i] = ed / max(len(tgt), 1)      # 정규화 편집거리 = 오발음 점수
            labels[i] = r["has_error"]
            per_err += ed; per_len += len(tgt); done += 1
        if bi % 200 == 0:
            print(f"  [gold] {done:,}/{len(recs):,}", flush=True)
    # dev/test 분할(해시)로 임계값 튜닝
    import hashlib
    is_test = np.array([int(hashlib.md5(r["stem"].encode()).hexdigest()[:8], 16) % 1000 < 300 for r in recs])
    def prf(sc, la, thr):
        pred = (sc >= thr).astype(float)
        tp = ((pred == 1) & (la == 1)).sum(); fp = ((pred == 1) & (la == 0)).sum()
        fn = ((pred == 0) & (la == 1)).sum(); tn = ((pred == 0) & (la == 0)).sum()
        P = tp / (tp + fp) if tp + fp else 0.0; R = tp / (tp + fn) if tp + fn else 0.0
        F1 = 2 * P * R / (P + R) if P + R else 0.0
        return dict(P=float(P), R=float(R), F1=float(F1),
                    FRR=float(fp / (tn + fp)) if tn + fp else 0.0,
                    FAR=float(fn / (tp + fn)) if tp + fn else 0.0, thr=float(thr))
    dev_s, dev_l = scores[~is_test], labels[~is_test]
    best = max((prf(dev_s, dev_l, t) for t in np.unique(np.round(dev_s, 4))), key=lambda x: x["F1"])
    test = prf(scores[is_test], labels[is_test], best["thr"])
    # AP (threshold-free)
    o = np.argsort(-scores[is_test]); y = labels[is_test][o]; Pn = y.sum()
    ap = float(np.sum((np.cumsum(y) / np.maximum(np.arange(1, len(y) + 1), 1)) * y) / max(Pn, 1))
    gold_per = per_err / max(per_len, 1) * 100
    print(f"\n[MDD F1] test F1={test['F1']:.3f} (P={test['P']:.3f} R={test['R']:.3f}) | AP={ap:.3f} | 임계값={best['thr']:.3f}", flush=True)
    print(f"[gold PER] {gold_per:.2f}% (정답음소열 대비, 발음과제)", flush=True)
    return {"test": test, "dev_best": best, "AP": round(ap, 4), "gold_per": round(gold_per, 3),
            "n_total": len(recs), "n_test": int(is_test.sum()), "pos_rate": float(labels.mean())}


def main():
    dec = Decoder()
    res = {}
    print("\n===== ⑨ PER (분리 측정) =====", flush=True)
    res["per_foreign"] = run_per(dec, f"{MDD_OUTPUTS}/manifest_16k/test.tsv",
                                 f"{MDD_OUTPUTS}/manifest_16k/test.ltr", "외국인")
    res["per_native"] = run_per(dec, f"{MDD_OUTPUTS}/manifest_native/test.tsv",
                                f"{MDD_OUTPUTS}/manifest_native/test.ltr", "원어민")
    print("\n===== ⑩ gold MDD F1 =====", flush=True)
    res["mdd_f1"] = run_gold_f1(dec)
    json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=2)
    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    main()
