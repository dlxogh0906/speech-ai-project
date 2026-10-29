# -*- coding: utf-8 -*-
"""국적 분류 백본 비교 — XLS-R 300M(현재 배포 백본) vs wav2vec 2.0 Large(LV-60) · WavLM Large · EAT Large (· XLS-R 2B 참고).

같은 데이터·같은 분할·같은 헤드로 "표현 자체"만 비교한다(SUPERB 식 동결 프로브):
  백본 동결 → 층마다 마스크 평균풀링 → 표준화 → 선형 헤드(41) → valid 로 층 선택 → test 1회.

데이터: 배포 주 모델 run_zeroth 의 학습 목록(lab131full_zeroth/{train,valid,test}.csv)에서
  지금 디스크에 음성이 남아 있는 행만 쓴다(국적 전용 음성은 2026-10-11 삭제됨).
  - web040~043: 원본은 지워졌고 발음평가용 16 kHz 사본(DATA_ROOT/mdd_gold_wav16k)을 파일명으로 찾아 쓴다.
  - 131(p505): 남은 일부만. kor001: 없음. kor_aug(한국어 증강본): 비교에서 뺀다(원본만 쓴다).
  - train 은 클래스당 --cap 행까지, 화자를 번갈아 뽑아 화자 다양성을 최대로. valid/test 는 남은 전부.
  분할(화자 비중복)은 run_zeroth 그대로다.

    $PY .../backbone_compare.py prep
    PYTHONPATH=<transformers 4.46 (py3.8)> $PY .../backbone_compare.py extract --backbone wavlm_large
    $PY .../backbone_compare.py probe --backbone wavlm_large
    $PY .../backbone_compare.py report
산출물: NAT_OUTPUTS/backbone_compare_20261017/
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import argparse
import collections
import csv
import json
import math
import os
import random
import time

import numpy as np

from shared import paths as P

OUT_NAME = "backbone_compare_20261017"
SRC_NAME = "webmic_kd/lab131full_zeroth"
SR = 16000
BACKBONES = {
    # name: (종류, 위치, 저장할 층, 폭)
    "xlsr300": ("fairseq", "XLSR_300M", list(range(1, 25)), 1024),
    "w2v2_lv60": ("hf_w2v2", "facebook/wav2vec2-large-lv60", list(range(1, 25)), 1024),
    "wavlm_large": ("hf_wavlm", "microsoft/wavlm-large", list(range(1, 25)), 1024),
    "eat_large": ("eat", "worstchan/EAT-large_epoch20_pretrain", list(range(1, 25)), 1024),   # 층마다 패치평균 + CLS
    "xlsr2b": ("fairseq", "XLSR_2B", [8, 12, 16, 20, 24, 28, 31, 34, 38, 42, 46, 48], 1920),
}


def out_dir():
    d = P.NAT_OUTPUTS / OUT_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


# ───────────────────────────── prep ─────────────────────────────
def cmd_prep(a):
    src = P.NAT_OUTPUTS / SRC_NAME
    gold = P.DATA_ROOT / "mdd_gold_wav16k"
    stem2path = {}
    for d in os.scandir(str(gold)):
        if d.is_dir():
            for f in os.scandir(d.path):
                if f.name.endswith(".wav"):
                    stem2path[f.name[:-4]] = f.path
    print("[prep] 16 kHz 사본 %d개" % len(stem2path), flush=True)
    rnd = random.Random(a.seed)
    stats = {}
    for split in ("train", "valid", "test"):
        rows, miss = [], collections.Counter()
        for r in csv.DictReader(open(str(src / ("%s.csv" % split)), encoding="utf-8")):
            dom = r["domain"]
            if dom == "kor_aug_train":
                continue
            if dom.startswith("web"):
                path = stem2path.get(_pl.Path(r["audio_path"]).stem)
            else:
                path = r["audio_path"] if os.path.exists(r["audio_path"]) else None
            if path is None:
                miss[dom] += 1
                continue
            rows.append({"audio_path": path, "label": r["label_en"], "speaker": r["speaker"], "domain": dom,
                         "task": r["task"], "duration_sec": r["duration_sec"]})
        if split == "train":
            by_cls = collections.defaultdict(lambda: collections.defaultdict(list))
            for r in rows:
                by_cls[r["label"]][r["speaker"]].append(r)
            picked = []
            for lab in sorted(by_cls):
                spk = by_cls[lab]
                for v in spk.values():
                    rnd.shuffle(v)
                order = sorted(spk)
                rnd.shuffle(order)
                take, i = [], 0
                while len(take) < a.cap and any(spk[s] for s in order):    # 화자를 번갈아 하나씩
                    s = order[i % len(order)]
                    if spk[s]:
                        take.append(spk[s].pop())
                    i += 1
                picked += take
            rows = picked
        rows.sort(key=lambda r: float(r["duration_sec"]))                   # 길이순 → 배치 패딩 최소
        with open(str(out_dir() / ("manifest_%s.csv" % split)), "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        cls = collections.Counter(r["label"] for r in rows)
        stats[split] = {"rows": len(rows), "speakers": len({r["speaker"] for r in rows}), "classes": len(cls),
                        "per_class": dict(sorted(cls.items())), "missing_audio_by_domain": dict(miss),
                        "hours": round(sum(min(float(r["duration_sec"]), a.max_sec) for r in rows) / 3600, 1)}
        print("[prep] %s 행 %d · 화자 %d · 클래스 %d · %.1f 시간(앞 %ds 기준) · 음성 없음 %s" % (
            split, len(rows), stats[split]["speakers"], len(cls), stats[split]["hours"], a.max_sec, dict(miss)), flush=True)
    stats["args"] = {"cap": a.cap, "seed": a.seed, "max_sec": a.max_sec, "source": str(src)}
    (out_dir() / "manifest_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")


# ───────────────────────────── extract ─────────────────────────────
def _load(path, max_sec):
    import soundfile as sf
    import torch
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = wav.mean(1)
    if sr != SR:
        import torchaudio.functional as AF
        wav = AF.resample(torch.from_numpy(wav), sr, SR).numpy()
    wav = wav[: int(max_sec * SR)]
    if len(wav) < SR // 2:                                                  # 0.5초 미만은 0.5초로 채운다
        wav = np.pad(wav, (0, SR // 2 - len(wav)))
    return np.nan_to_num(wav)


class _Clips(object):
    def __init__(self, rows, kind, max_sec):
        self.rows, self.kind, self.max_sec = rows, kind, max_sec

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        import torch
        try:
            wav = _load(self.rows[i]["audio_path"], self.max_sec)
        except Exception as e:                                              # 읽기 실패는 기록하고 0 벡터
            print("[extract] 읽기 실패 %s: %s" % (self.rows[i]["audio_path"], e), flush=True)
            wav = np.zeros(SR // 2, np.float32)
        if self.kind == "eat":
            import torchaudio
            w = torch.from_numpy(wav - wav.mean())
            v = torchaudio.compliance.kaldi.fbank(w.unsqueeze(0), htk_compat=True, sample_frequency=SR, use_energy=False,
                                                  window_type="hanning", num_mel_bins=128, dither=0.0, frame_shift=10)
            return i, v
        wav = (wav - wav.mean()) / np.sqrt(wav.var() + 1e-7)                # 네 음성 백본 모두 사전학습 때 정규화 입력
        return i, torch.from_numpy(wav.astype(np.float32))


def _collate(kind):
    def f(items):
        import torch
        idx = [i for i, _ in items]
        lens = torch.tensor([len(v) for _, v in items])
        n = int(lens.max())
        if kind == "eat":
            n = int(math.ceil(n / 16.0) * 16)
        x = torch.zeros((len(items), n) + tuple(items[0][1].shape[1:]))
        for k, (_, v) in enumerate(items):
            x[k, :len(v)] = v
        if kind == "eat":
            x = ((x + 4.268) / (4.569 * 2)).unsqueeze(1)
        return idx, x, lens
    return f


def _batches(rows, max_batch_sec, max_items, max_sec):
    out, cur, longest = [], [], 0.0
    for i, r in enumerate(rows):                                            # 행은 이미 길이순
        d = min(float(r["duration_sec"]), max_sec)
        if cur and (max(longest, d) * (len(cur) + 1) > max_batch_sec or len(cur) >= max_items):
            out.append(cur)
            cur, longest = [], 0.0
        cur.append(i)
        longest = max(longest, d)
    if cur:
        out.append(cur)
    return out


class _Encoder(object):
    def __init__(self, name, device):
        import torch
        self.kind, loc, self.layers, self.width = BACKBONES[name]
        self.device = device
        if self.kind == "fairseq":
            from shared.fairseq_guard import ensure_fairseq
            ensure_fairseq()
            from fairseq import checkpoint_utils
            models, _ = checkpoint_utils.load_model_ensemble([str(getattr(P, loc))])
            m = models[0]
            m.remove_pretraining_modules()
            m.encoder.layerdrop = 0.0
        else:
            from huggingface_hub import snapshot_download
            hub = P.MODEL_ROOT / "hf" / "hub"
            try:
                path = snapshot_download(loc, cache_dir=str(hub), local_files_only=True)
            except Exception:                                               # refs/main 없이 스냅숏만 받아 둔 경우
                path = str(sorted((hub / ("models--" + loc.replace("/", "--")) / "snapshots").iterdir())[-1])
            if self.kind == "hf_w2v2":
                from transformers import Wav2Vec2Model
                m = Wav2Vec2Model.from_pretrained(path)
            elif self.kind == "hf_wavlm":
                from transformers import WavLMModel
                m = WavLMModel.from_pretrained(path)
            else:
                from transformers import AutoModel
                from transformers.dynamic_module_utils import get_cached_module_file
                for s in sorted(_pl.Path(path).glob("*.py")):                   # 중첩 import 모듈까지 캐시
                    get_cached_module_file(path, s.name, local_files_only=True)
                m = AutoModel.from_pretrained(path, trust_remote_code=True)
            if hasattr(m.config, "layerdrop"):
                m.config.layerdrop = 0.0
        self.m = m.eval().to(device)
        for p in self.m.parameters():
            p.requires_grad = False
        self.n_out = len(self.layers) * (2 if self.kind == "eat" else 1)
        self.params = sum(p.numel() for p in self.m.parameters())

    def __call__(self, x, lens):
        import torch
        x, lens = x.to(self.device), lens.to(self.device)

        def pool(h, valid):
            v = valid.to(h.dtype).unsqueeze(-1)
            return (h.float() * v).sum(1) / v.sum(1).clamp_min(1.0)

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            if self.kind == "eat":
                m = self.m.model
                h = m.local_encoder(x)
                if m.fixed_positional_encoder is not None:
                    h = h + m.fixed_positional_encoder(h, None)[:, :h.shape[1], :]
                h = torch.cat((m.extra_tokens.expand(len(h), -1, -1), h), 1)
                h = m.pos_drop(m.pre_norm(h))
                tp = x.shape[2] // 16
                vt = torch.arange(tp, device=self.device)[None, :] < ((lens + 15) // 16)[:, None]
                valid = vt.repeat_interleave(8, dim=1)
                pad = torch.cat((torch.zeros((len(h), 1), dtype=torch.bool, device=self.device), ~valid), 1)
                means, clss = [], []
                for li, blk in enumerate(m.blocks, 1):
                    h, _ = blk(h, padding_mask=pad)
                    if li in self.layers:
                        means.append(pool(h[:, 1:], valid))
                        clss.append(h[:, 0].float())
                res = means + clss
            else:
                valid = torch.arange(x.shape[1], device=self.device)[None, :] < lens[:, None]
                if self.kind == "fairseq":
                    o = self.m.extract_features(source=x, padding_mask=~valid, mask=False)
                    pm = o.get("padding_mask")
                    hs = [o["layer_results"][li - 1][0].transpose(0, 1) for li in self.layers]
                    fv = torch.ones(hs[0].shape[:2], dtype=torch.bool, device=self.device) if pm is None else ~pm
                else:
                    am = valid.long() if self.kind == "hf_wavlm" or self.m.config.feat_extract_norm == "layer" else None
                    o = self.m(input_values=x, attention_mask=am, output_hidden_states=True, return_dict=True)
                    hs = [o.hidden_states[li] for li in self.layers]
                    fv = self.m._get_feature_vector_attention_mask(hs[0].shape[1], valid.long()).bool()
                res = [pool(h, fv) for h in hs]
        r = torch.stack(res, 1)
        return torch.nan_to_num(r).clamp(-50, 50).cpu().numpy().astype(np.float16)


def cmd_extract(a):
    import torch
    from torch.utils.data import DataLoader
    dev = torch.device("cuda")
    enc = _Encoder(a.backbone, dev)
    print("[extract] %s 파라미터 %.0fM · 저장 %d개(층%s)" % (
        a.backbone, enc.params / 1e6, enc.n_out, " × 평균/CLS" if enc.kind == "eat" else ""), flush=True)
    meta = {"backbone": a.backbone, "params": enc.params, "layers": enc.layers, "max_sec": a.max_sec,
            "outputs": (["L%d_mean" % l for l in enc.layers] + ["L%d_cls" % l for l in enc.layers])
            if enc.kind == "eat" else ["L%d" % l for l in enc.layers]}
    for split in a.splits.split(","):
        rows = list(csv.DictReader(open(str(out_dir() / ("manifest_%s.csv" % split)), encoding="utf-8")))
        if a.limit:
            rows = rows[:a.limit]
        fn = out_dir() / ("feats_%s_%s%s.npy" % (a.backbone, split, "_limit" if a.limit else ""))
        if fn.exists() and not a.limit and not a.force:
            print("[extract] 있음, 건너뜀:", fn, flush=True)
            continue
        feats = np.lib.format.open_memmap(str(fn) + ".part", mode="w+", dtype=np.float16, shape=(len(rows), enc.n_out, enc.width))
        bs = _batches(rows, a.batch_sec, a.batch_items, a.max_sec)
        dl = DataLoader(_Clips(rows, enc.kind, a.max_sec), batch_sampler=bs, num_workers=a.workers,
                        collate_fn=_collate(enc.kind), prefetch_factor=4)
        t0, done = time.time(), 0
        for k, (idx, x, lens) in enumerate(dl):
            feats[idx] = enc(x, lens)
            done += len(idx)
            if k % 200 == 0:
                el = time.time() - t0
                print("[extract] %s %s %d/%d (%.0fs, 남은 %.0f분)" % (
                    a.backbone, split, done, len(rows), el, el / max(1, done) * (len(rows) - done) / 60), flush=True)
        feats.flush()
        del feats
        os.replace(str(fn) + ".part", str(fn))
        meta[split] = {"rows": len(rows), "seconds": round(time.time() - t0, 1)}
        print("[extract] %s %s 끝 %.1f분" % (a.backbone, split, (time.time() - t0) / 60), flush=True)
    meta["peak_gib"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
    old = out_dir() / ("feats_%s_meta.json" % a.backbone)
    if old.exists() and not a.limit:
        prev = json.loads(old.read_text())
        prev.update(meta)
        meta = prev
    old.write_text(json.dumps(meta, ensure_ascii=False, indent=1))


# ───────────────────────────── probe ─────────────────────────────
def _metrics(logits, y, ncls):
    top = np.argsort(-logits, 1)[:, :5]
    h1 = top[:, 0] == y
    h5 = (top == y[:, None]).any(1)
    present = [c for c in range(ncls) if (y == c).any()]
    rec = [h1[y == c].mean() for c in present]
    rec5 = [h5[y == c].mean() for c in present]
    f1 = []
    for c in present:
        tp = float(((top[:, 0] == c) & (y == c)).sum())
        p = tp / max(1, (top[:, 0] == c).sum())
        r = tp / max(1, (y == c).sum())
        f1.append(0 if p + r == 0 else 2 * p * r / (p + r))
    return {"n": int(len(y)), "top1": round(float(h1.mean()), 4), "top5": round(float(h5.mean()), 4),
            "macro_recall": round(float(np.mean(rec)), 4), "macro_top5": round(float(np.mean(rec5)), 4),
            "macro_f1": round(float(np.mean(f1)), 4), "classes": len(present)}


def _train_head(xtr, ytr, xva, yva, ncls, a, dev):
    import torch
    import torch.nn.functional as F
    xtr, xva = torch.from_numpy(xtr).to(dev), torch.from_numpy(xva).to(dev)
    ytr_t = torch.from_numpy(ytr).long().to(dev)
    cnt = np.bincount(ytr, minlength=ncls).astype(np.float32)
    w = torch.from_numpy(np.where(cnt > 0, (cnt.sum() / np.maximum(cnt, 1)) ** a.balance, 0.0)).float().to(dev)
    w = w / w[cnt > 0].mean()
    torch.manual_seed(a.seed)
    head = torch.nn.Linear(xtr.shape[1], ncls).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=a.wd)
    best, best_state, bad = -1.0, None, 0
    n = len(xtr)
    for ep in range(a.epochs):
        perm = torch.randperm(n, device=dev)
        for s in range(0, n, a.bs):
            b = perm[s:s + a.bs]
            loss = F.cross_entropy(head(xtr[b]), ytr_t[b], weight=w)
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            m = _metrics(head(xva).cpu().numpy(), yva, ncls)
        score = m["top1"] + m["macro_recall"]                              # 다수 클래스·소수 클래스를 같이 본다
        if score > best:
            best, best_state, bad = score, {k: v.clone() for k, v in head.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= a.patience:
                break
    head.load_state_dict(best_state)
    return head, best


def cmd_probe(a):
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = out_dir()
    meta = json.loads((d / ("feats_%s_meta.json" % a.backbone)).read_text())
    rows = {s: list(csv.DictReader(open(str(d / ("manifest_%s.csv" % s)), encoding="utf-8"))) for s in ("train", "valid", "test")}
    labels = sorted({r["label"] for r in rows["train"]})
    c2i = {c: i for i, c in enumerate(labels)}
    ys = {s: np.array([c2i.get(r["label"], -1) for r in rows[s]]) for s in rows}
    keep = {s: ys[s] >= 0 for s in rows}                                    # 학습에 없는 국가는 평가에서 뺀다
    F_ = {s: np.load(str(d / ("feats_%s_%s.npy" % (a.backbone, s))), mmap_mode="r") for s in rows}
    res = {"backbone": a.backbone, "labels": labels, "layers": {}, "args": vars(a)}
    logits, heads = {}, {}
    for li, name in enumerate(meta["outputs"]):
        X = {s: np.asarray(F_[s][:, li, :], dtype=np.float32)[keep[s]] for s in rows}
        mu, sd = X["train"].mean(0), X["train"].std(0) + 1e-5
        X = {s: (X[s] - mu) / sd for s in X}
        head, _ = _train_head(X["train"], ys["train"][keep["train"]], X["valid"], ys["valid"][keep["valid"]], len(labels), a, dev)
        with torch.no_grad():
            out = {s: _metrics(head(torch.from_numpy(X[s]).to(dev)).cpu().numpy(), ys[s][keep[s]], len(labels))
                   for s in ("valid", "test")}
            logits[name] = head(torch.from_numpy(X["test"]).to(dev)).cpu().numpy().astype(np.float16)
        heads[name] = {"index": li, "weight": head.weight.detach().cpu().numpy(), "bias": head.bias.detach().cpu().numpy(),
                       "mu": mu, "sd": sd}
        res["layers"][name] = out
        print("[probe] %s %-8s valid top1 %.4f mrec %.4f | test top1 %.4f top5 %.4f mrec %.4f mF1 %.4f" % (
            a.backbone, name, out["valid"]["top1"], out["valid"]["macro_recall"], out["test"]["top1"],
            out["test"]["top5"], out["test"]["macro_recall"], out["test"]["macro_f1"]), flush=True)
    best = max(res["layers"], key=lambda k: res["layers"][k]["valid"]["top1"] + res["layers"][k]["valid"]["macro_recall"])
    res["best_layer"] = best
    res["best"] = res["layers"][best]
    np.save(str(d / ("testlogits_%s_best.npy" % a.backbone)), logits[best])     # 클래스별 분석용(행 = manifest_test 중 학습 국가)
    h = heads[best]                                                         # 잡음·분리 시험에서 다시 쓴다
    np.savez(str(d / ("head_%s_best.npz" % a.backbone)), layer=best, index=h["index"], weight=h["weight"], bias=h["bias"],
             mu=h["mu"], sd=h["sd"], labels=np.array(labels))
    (d / ("probe_%s.json" % a.backbone)).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print("[probe] %s 최적층(valid 기준) %s → test %s" % (a.backbone, best, res["best"]["test"]), flush=True)


class _DeployedClips(object):
    """배포 주 모델 입력 그대로: 원 파형(정규화 없음) 앞 12초 → 무음 정규화(서비스 KD_SILENCE_NORM=1)."""
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        import torch
        from country_classifier.wav_augment import silence_normalize
        wav = _load(self.rows[i]["audio_path"], 12.0)
        wav = silence_normalize(wav.astype(np.float32), 1.5e-3)
        return i, torch.from_numpy(np.asarray(wav, np.float32))


def _deployed_eval(names, a):
    """배포 주 모델 run_zeroth(ep0.pt, EMA · 300M ≤17층 KD)로 noise_sep 조건 채점. 확인 모델 게이트는 뺐다."""
    import torch
    from torch.utils.data import DataLoader
    from country_classifier.kd_student import build_student, forward_pooled
    d = out_dir()
    dev = torch.device("cuda")
    enc, head, mu, sd, labels, layer = build_student(P.XLSR_300M, P.NAT_OUTPUTS / "webmic_kd" / "run_zeroth" / "ep0.pt", dev)
    enc.eval()
    c2i = {c: i for i, c in enumerate(labels)}
    res_path = d / "noise_sep" / "results_run_zeroth.json"
    res = json.loads(res_path.read_text()) if res_path.exists() else {}
    for name in names:
        rows = list(csv.DictReader(open(str(d / "noise_sep" / ("%s.csv" % name)), encoding="utf-8")))
        rows = sorted(rows, key=lambda r: float(r["duration_sec"]))
        logits = np.zeros((len(rows), len(labels)), np.float32)
        dl = DataLoader(_DeployedClips(rows), batch_sampler=_batches(rows, 300.0, 64, 12.0), num_workers=a.workers,
                        collate_fn=_collate("wave"))
        for idx, x, lens in dl:
            x, lens = x.to(dev), lens.to(dev)
            pm = torch.arange(x.shape[1], device=dev)[None, :] >= lens[:, None]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                pooled = forward_pooled(enc, x, pm, layer)
                logits[idx] = head((pooled.float() - mu) / sd).float().cpu().numpy()
        y = np.array([c2i[r["label"]] for r in rows])
        np.savez(str(d / "noise_sep" / ("pred_run_zeroth_%s.npz" % name)), logits=logits.astype(np.float16), y=y,
                 wav=np.array([_pl.Path(r["audio_path"]).name for r in rows]), speaker=np.array([r["speaker"] for r in rows]))
        res[name] = _metrics(logits, y, len(labels))
        print("[evalset] run_zeroth %-16s top1 %.4f top5 %.4f macro_recall %.4f" % (
            name, res[name]["top1"], res[name]["top5"], res[name]["macro_recall"]), flush=True)
        res_path.write_text(json.dumps(res, ensure_ascii=False, indent=1))


def cmd_evalset(a):
    """noise_sep/<조건>.csv 들을 저장된 최적층 헤드로 채점(noise_separation_prep.py 가 만든 음성)."""
    import torch
    from torch.utils.data import DataLoader
    d = out_dir()
    names = a.sets.split(",") if a.sets else sorted(json.loads((d / "noise_sep" / "conditions.json").read_text())["conditions"])
    if not a.sets:
        names = names + [n + "_sep" for n in names]
    if a.backbone == "run_zeroth":
        return _deployed_eval(names, a)
    h = np.load(str(d / ("head_%s_best.npz" % a.backbone)))
    labels = [str(x) for x in h["labels"]]
    c2i = {c: i for i, c in enumerate(labels)}
    enc = _Encoder(a.backbone, torch.device("cuda"))
    W, b = torch.from_numpy(h["weight"]).float().cuda(), torch.from_numpy(h["bias"]).float().cuda()
    mu, sd = torch.from_numpy(h["mu"]).float().cuda(), torch.from_numpy(h["sd"]).float().cuda()
    li = int(h["index"])
    res_path = d / "noise_sep" / ("results_%s.json" % a.backbone)
    res = json.loads(res_path.read_text()) if res_path.exists() else {}
    for name in names:
        rows = list(csv.DictReader(open(str(d / "noise_sep" / ("%s.csv" % name)), encoding="utf-8")))
        order = sorted(range(len(rows)), key=lambda i: float(rows[i]["duration_sec"]))
        rows_s = [rows[i] for i in order]
        logits = np.zeros((len(rows_s), len(labels)), np.float32)
        dl = DataLoader(_Clips(rows_s, enc.kind, 12.0), batch_sampler=_batches(rows_s, a.batch_sec, 128, 12.0),
                        num_workers=a.workers, collate_fn=_collate(enc.kind))
        for idx, x, lens in dl:
            f = torch.from_numpy(enc(x, lens)[:, li, :].astype(np.float32)).cuda()
            logits[idx] = (((f - mu) / sd) @ W.T + b).cpu().numpy()
        y = np.array([c2i[r["label"]] for r in rows_s])
        res[name] = _metrics(logits, y, len(labels))
        print("[evalset] %s %-16s top1 %.4f top5 %.4f macro_recall %.4f" % (
            a.backbone, name, res[name]["top1"], res[name]["top5"], res[name]["macro_recall"]), flush=True)
        res_path.write_text(json.dumps(res, ensure_ascii=False, indent=1))


class _TrainClips(_Clips):
    """학습용: 앞이 아니라 무작위 위치에서 --crop 초를 자른다."""
    def __init__(self, rows, crop, seed):
        _Clips.__init__(self, rows, "wave", 1e9)
        self.crop, self.rnd = crop, random.Random(seed)

    def __getitem__(self, i):
        import torch
        wav = _load(self.rows[i]["audio_path"], 60.0)
        n = int(self.crop * SR)
        if len(wav) > n:
            s = self.rnd.randrange(0, len(wav) - n)
            wav = wav[s:s + n]
        wav = (wav - wav.mean()) / np.sqrt(wav.var() + 1e-7)
        return i, torch.from_numpy(wav.astype(np.float32))


class _FTModel(object):
    """백본 앞 --depth 층만 남기고 --unfreeze_from 층부터 학습, 마지막 층 평균풀링 → 선형 헤드."""
    def __init__(self, name, depth, unfreeze_from, ncls, dev):
        import torch
        enc = _Encoder(name, dev)
        self.kind, self.m, self.depth, self.dev = enc.kind, enc.m, depth, dev
        if self.kind == "fairseq":
            del self.m.encoder.layers[depth:]
            layers = self.m.encoder.layers
        else:
            self.m.encoder.layers = torch.nn.ModuleList(list(self.m.encoder.layers)[:depth])
            layers = self.m.encoder.layers
        for p in self.m.parameters():
            p.requires_grad = False
        self.train_params = []
        for li in range(unfreeze_from - 1, depth):                          # 층 번호는 1부터
            for p in layers[li].parameters():
                p.requires_grad = True
                self.train_params.append(p)
        self.head = torch.nn.Linear(enc.width, ncls).to(dev)
        self.mu = torch.zeros(enc.width, device=dev)
        self.sd = torch.ones(enc.width, device=dev)

    def __call__(self, x, lens):
        return self.head((self.pool(x, lens) - self.mu) / self.sd)

    def pool(self, x, lens):
        import torch
        valid = torch.arange(x.shape[1], device=self.dev)[None, :] < lens[:, None]
        if self.kind == "fairseq":
            o = self.m.extract_features(source=x, padding_mask=~valid, mask=False, layer=self.depth - 1)
            h = o["layer_results"][self.depth - 1][0].transpose(0, 1)
            pm = o.get("padding_mask")
            fv = torch.ones(h.shape[:2], dtype=torch.bool, device=self.dev) if pm is None else ~pm
        else:
            o = self.m(input_values=x, attention_mask=valid.long(), output_hidden_states=True, return_dict=True)
            h = o.hidden_states[self.depth]
            fv = self.m._get_feature_vector_attention_mask(h.shape[1], valid.long()).bool()
        v = fv.to(h.dtype).unsqueeze(-1)
        pooled = (h * v).sum(1) / v.sum(1).clamp_min(1.0)
        return torch.nan_to_num(pooled.float()).clamp(-50, 50)

    def train(self, flag):
        # 두 백본 모두 드롭아웃·시간 마스킹 없이 학습한다(HF 는 train 모드면 SpecAugment 를 걸고 fairseq 는 mask=False 라
        # train 모드를 켜면 조건이 달라진다). 학습 여부는 requires_grad 로만 가른다.
        self.m.train(False)


def cmd_finetune(a):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    d = out_dir()
    dev = torch.device("cuda")
    torch.manual_seed(a.seed)
    rows = {s: list(csv.DictReader(open(str(d / ("manifest_%s.csv" % s)), encoding="utf-8"))) for s in ("train", "valid", "test")}
    labels = sorted({r["label"] for r in rows["train"]})
    c2i = {c: i for i, c in enumerate(labels)}
    for s in ("valid", "test"):
        rows[s] = [r for r in rows[s] if r["label"] in c2i]
    M = _FTModel(a.backbone, a.depth, a.unfreeze_from, len(labels), dev)
    cnt = np.bincount([c2i[r["label"]] for r in rows["train"]], minlength=len(labels)).astype(np.float32)
    w = torch.from_numpy((cnt.sum() / np.maximum(cnt, 1)) ** 0.5).float().to(dev)
    w = w / w.mean()
    opt = torch.optim.AdamW([{"params": M.train_params, "lr": a.lr}, {"params": M.head.parameters(), "lr": a.head_lr}], weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    tag = "ft_%s_d%d_u%d_e%d" % (a.backbone, a.depth, a.unfreeze_from, a.epochs)
    log = {"args": vars(a), "evals": []}

    def evaluate(split):
        M.train(False)
        rs = sorted(rows[split], key=lambda r: float(r["duration_sec"]))
        lg = np.zeros((len(rs), len(labels)), np.float32)
        dl = DataLoader(_Clips(rs, "wave", 12.0), batch_sampler=_batches(rs, 600.0, 128, 12.0), num_workers=a.workers,
                        collate_fn=_collate("wave"))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for idx, x, lens in dl:
                lg[idx] = M(x.to(dev), lens.to(dev)).float().cpu().numpy()
        return _metrics(lg, np.array([c2i[r["label"]] for r in rs]), len(labels)), lg

    tr = sorted(rows["train"], key=lambda r: min(float(r["duration_sec"]), a.crop))
    # 선형 프로브로 시작(LP-FT): 동결 상태 풀링 특징 일부로 mu/sd(고정)와 헤드 초기값을 잡는다 — 배포 모델 학습과 같은 방식.
    # 이게 없으면 WavLM 처럼 크기가 큰 차원이 있는 백본은 헤드 학습이 느려 비교가 불공정해진다.
    def pooled_of(rs, sec):
        out = np.zeros((len(rs), M.head.in_features), np.float32)
        dl = DataLoader(_Clips(rs, "wave", sec), batch_sampler=_batches(rs, 600.0, 128, sec),
                        num_workers=a.workers, collate_fn=_collate("wave"))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for idx, x, lens in dl:
                out[idx] = M.pool(x.to(dev), lens.to(dev)).float().cpu().numpy()
        return out

    lp_rows = sorted(random.Random(a.seed).sample(tr, min(a.lp_rows, len(tr))), key=lambda r: float(r["duration_sec"]))
    va_rows = sorted(rows["valid"], key=lambda r: float(r["duration_sec"]))
    feats, vfeats = pooled_of(lp_rows, a.crop), pooled_of(va_rows, 12.0)
    mu, sd = feats.mean(0), feats.std(0) + 1e-5
    M.mu, M.sd = torch.from_numpy(mu).to(dev), torch.from_numpy(sd).to(dev)
    ylp = np.array([c2i[r["label"]] for r in lp_rows])
    yva = np.array([c2i[r["label"]] for r in va_rows])
    lp_args = argparse.Namespace(balance=0.5, seed=a.seed, lr=1e-3, wd=1e-4, epochs=60, bs=512, patience=8)
    lp_head, _ = _train_head((feats - mu) / sd, ylp, (vfeats - mu) / sd, yva, len(labels), lp_args, dev)
    M.head.load_state_dict(lp_head.state_dict())
    log["lp"] = {"rows": len(lp_rows)}
    m0, _ = evaluate("valid")
    log["evals"].append({"step": 0, "valid": m0})
    print("[ft] %s 선형 프로브 시작점 valid top1 %.4f mrec %.4f" % (tag, m0["top1"], m0["macro_recall"]), flush=True)
    batches = _batches(tr, a.batch_sec, 64, a.crop)
    total = len(batches) * a.epochs
    step, t0 = 0, time.time()
    best = m0["top1"] + m0["macro_recall"]
    best_state = {"head": {k: v.detach().cpu().clone() for k, v in M.head.state_dict().items()},
                  "enc": {id(p): p.detach().cpu().clone() for p in M.train_params}}
    for ep in range(a.epochs):
        rnd = random.Random(a.seed + ep)
        order = list(batches)
        rnd.shuffle(order)
        dl = DataLoader(_TrainClips(tr, a.crop, a.seed + ep), batch_sampler=order, num_workers=a.workers, collate_fn=_collate("wave"))
        M.train(True)
        for idx, x, lens in dl:
            lr_scale = min(1.0, (step + 1) / 200.0) * max(0.1, 0.5 * (1 + math.cos(math.pi * step / total)))
            for g, base in zip(opt.param_groups, (a.lr, a.head_lr)):
                g["lr"] = base * lr_scale
            y = torch.tensor([c2i[tr[i]["label"]] for i in idx], device=dev)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = F.cross_entropy(M(x.to(dev), lens.to(dev)).float(), y, weight=w)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(M.train_params + list(M.head.parameters()), 5.0)
            scaler.step(opt)
            scaler.update()
            step += 1
            if step % 100 == 0:
                print("[ft] %s step %d/%d loss %.3f (%.0f분)" % (tag, step, total, float(loss), (time.time() - t0) / 60), flush=True)
            if step % a.eval_every == 0 or step == total:
                m, _ = evaluate("valid")
                score = m["top1"] + m["macro_recall"]
                log["evals"].append({"step": step, "valid": m})
                print("[ft] %s step %d valid top1 %.4f mrec %.4f" % (tag, step, m["top1"], m["macro_recall"]), flush=True)
                if score > best:
                    best = score
                    best_state = {"head": {k: v.detach().cpu().clone() for k, v in M.head.state_dict().items()},
                                  "enc": {id(p): p.detach().cpu().clone() for p in M.train_params}}
                M.train(True)
    M.head.load_state_dict(best_state["head"])                              # valid 최고 시점으로 되돌려 test 1회
    with torch.no_grad():
        for p in M.train_params:
            p.copy_(best_state["enc"][id(p)].to(dev))
    log["test"], lg = evaluate("test")
    np.save(str(d / ("testlogits_%s.npy" % tag)), lg.astype(np.float16))
    log["minutes"] = round((time.time() - t0) / 60, 1)
    (d / ("%s.json" % tag)).write_text(json.dumps(log, ensure_ascii=False, indent=1))
    print("[ft] %s test %s" % (tag, log["test"]), flush=True)


def cmd_probe_wsum(a):
    """층 가중합 헤드(SUPERB 식): 층마다 표준화 → softmax(학습 가중치) 가중합 → (선택) 은닉층 → 선형. 캐시된 층별 평균만 쓴다."""
    import torch
    import torch.nn.functional as F
    dev = torch.device("cuda")
    d = out_dir()
    rows = {s: list(csv.DictReader(open(str(d / ("manifest_%s.csv" % s)), encoding="utf-8"))) for s in ("train", "valid", "test")}
    labels = sorted({r["label"] for r in rows["train"]})
    c2i = {c: i for i, c in enumerate(labels)}
    meta = json.loads((d / ("feats_%s_meta.json" % a.backbone)).read_text())
    keep_out = [i for i, n in enumerate(meta["outputs"]) if not n.endswith("_cls")]
    X, Y = {}, {}
    for s in rows:
        m = np.array([r["label"] in c2i for r in rows[s]])
        f = np.load(str(d / ("feats_%s_%s.npy" % (a.backbone, s))), mmap_mode="r")
        X[s] = torch.from_numpy(np.ascontiguousarray(f[:, keep_out, :][m])).to(dev)          # fp16 (N, L, D)
        Y[s] = torch.tensor([c2i[r["label"]] for r, k in zip(rows[s], m) if k], device=dev)
    L, D = X["train"].shape[1:]
    mu = X["train"].float().mean(0)
    sd = torch.sqrt(((X["train"].float() - mu) ** 2).mean(0)) + 1e-5 if X["train"].shape[0] < 50000 else None
    if sd is None:                                                          # 메모리 아끼려고 나눠서
        acc = torch.zeros_like(mu)
        for i in range(0, X["train"].shape[0], 8192):
            acc += ((X["train"][i:i + 8192].float() - mu) ** 2).sum(0)
        sd = torch.sqrt(acc / X["train"].shape[0]) + 1e-5
    cnt = torch.bincount(Y["train"], minlength=len(labels)).float()
    w = (cnt.sum() / cnt.clamp_min(1)) ** 0.5
    w = w / w[cnt > 0].mean()
    res = {"backbone": a.backbone, "variants": {}}
    class MoE(torch.nn.Module):
        """전문가 K개(각각 은닉 H MLP) + 게이트(입력마다 전문가 가중치, softmax). 쏠림 방지 보조손실(전문가 사용량 분산)."""
        def __init__(self, k, h):
            super().__init__()
            self.gate = torch.nn.Linear(D, k)
            self.experts = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(D, h), torch.nn.GELU(), torch.nn.Dropout(0.2),
                                                                    torch.nn.Linear(h, len(labels))) for _ in range(k)])
            self.aux = torch.zeros((), device=dev)
            self.last_gate = None

        def forward(self, x):
            g = torch.softmax(self.gate(x), -1)                             # (B, K)
            out = torch.stack([e(x) for e in self.experts], 1)              # (B, K, C)
            imp = g.mean(0)
            self.aux = (imp.var() / (imp.mean() ** 2 + 1e-9))               # 중요도 변동계수² (Shazeer 2017)
            self.last_gate = g.detach()
            return (g.unsqueeze(-1) * out).sum(1)

    for spec in a.hidden.split(","):
        torch.manual_seed(a.seed)
        lw = torch.zeros(L, device=dev, requires_grad=True)
        if spec.startswith("moe"):
            k, h = (int(v) for v in spec[3:].split("x"))
            body = MoE(k, h).to(dev)
        else:
            hidden = int(spec)
            body = (torch.nn.Sequential(torch.nn.Linear(D, hidden), torch.nn.GELU(), torch.nn.Dropout(0.2), torch.nn.Linear(hidden, len(labels)))
                    if hidden else torch.nn.Linear(D, len(labels))).to(dev)
        opt = torch.optim.AdamW([{"params": [lw], "lr": 1e-2}, {"params": body.parameters(), "lr": 1e-3}], weight_decay=1e-4)

        def fwd(x):
            z = (x.float() - mu) / sd
            return body((torch.softmax(lw, 0)[None, :, None] * z).sum(1))

        def run(s):
            with torch.no_grad():
                body.eval()
                out = torch.cat([fwd(X[s][i:i + 4096]) for i in range(0, X[s].shape[0], 4096)]).cpu().numpy()
                body.train()
            return out
        best, best_state, bad = -1, None, 0
        n = X["train"].shape[0]
        for ep in range(80):
            perm = torch.randperm(n, device=dev)
            for i in range(0, n, 512):
                b = perm[i:i + 512]
                loss = F.cross_entropy(fwd(X["train"][b]), Y["train"][b], weight=w)
                if isinstance(body, MoE):
                    loss = loss + 0.01 * body.aux
                opt.zero_grad()
                loss.backward()
                opt.step()
            m = _metrics(run("valid"), Y["valid"].cpu().numpy(), len(labels))
            sc = m["top1"] + m["macro_recall"]
            if sc > best:
                best, bad = sc, 0
                best_state = ({k: v.clone() for k, v in body.state_dict().items()}, lw.detach().clone())
            else:
                bad += 1
                if bad >= 10:
                    break
        body.load_state_dict(best_state[0])
        with torch.no_grad():
            lw.copy_(best_state[1])
        lg = run("test")
        t = _metrics(lg, Y["test"].cpu().numpy(), len(labels))
        lws = torch.softmax(lw, 0).detach().cpu().numpy()
        top = [meta["outputs"][keep_out[i]] for i in np.argsort(-lws)[:5]]
        name = ("wsum_" + spec) if spec.startswith("moe") else ("wsum_linear" if spec == "0" else "wsum_mlp" + spec)
        res["variants"][name] = {"test": t, "valid_best": best, "top_layers": top,
                                 "layer_weights": [round(float(v), 4) for v in lws],
                                 "params": int(sum(p.numel() for p in body.parameters()))}
        if isinstance(body, MoE):                                           # 전문가가 국가별로 나뉘었는지
            with torch.no_grad():
                body.eval()
                gs = torch.cat([torch.softmax(body.gate(((X["test"][i:i + 4096].float() - mu) / sd * torch.softmax(lw, 0)[None, :, None]).sum(1)), -1)
                                for i in range(0, X["test"].shape[0], 4096)]).cpu().numpy()
            yt = Y["test"].cpu().numpy()
            res["variants"][name]["expert_share"] = [round(float(v), 3) for v in gs.mean(0)]
            res["variants"][name]["top_expert_by_class"] = {labels[c]: int(gs[yt == c].mean(0).argmax()) for c in np.unique(yt)}
            res["variants"][name]["gate_entropy"] = round(float((-(gs * np.log(gs + 1e-9)).sum(1)).mean()), 3)
        np.save(str(d / ("testlogits_%s_%s.npy" % (a.backbone, name))), lg.astype(np.float16))
        print("[wsum] %s %-12s test top1 %.4f top5 %.4f mrec %.4f mF1 %.4f · 가중 큰 층 %s" % (
            a.backbone, name, t["top1"], t["top5"], t["macro_recall"], t["macro_f1"], top), flush=True)
    (d / ("probe_wsum_%s.json" % a.backbone)).write_text(json.dumps(res, ensure_ascii=False, indent=1))


def cmd_report(a):
    d = out_dir()
    lines = ["| 백본 | 파라미터 | 최적층(valid) | test top-1 | test top-5 | macro recall | macro top-5 | macro F1 |",
             "|---|---|---|---|---|---|---|---|"]
    for b in BACKBONES:
        f = d / ("probe_%s.json" % b)
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        meta = json.loads((d / ("feats_%s_meta.json" % b)).read_text())
        t = r["best"]["test"]
        lines.append("| %s | %.0fM | %s | %.4f | %.4f | %.4f | %.4f | %.4f |" % (
            b, meta["params"] / 1e6, r["best_layer"], t["top1"], t["top5"], t["macro_recall"], t["macro_top5"], t["macro_f1"]))
    txt = "\n".join(lines)
    (d / "report.md").write_text(txt + "\n", encoding="utf-8")
    print(txt)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("prep")
    p.add_argument("--cap", type=int, default=3000, help="train 클래스당 최대 행")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_sec", type=float, default=12.0, help="시간 집계용(추출과 같게)")
    e = sub.add_parser("extract")
    e.add_argument("--backbone", required=True, choices=sorted(BACKBONES))
    e.add_argument("--splits", default="valid,test,train")
    e.add_argument("--max_sec", type=float, default=12.0, help="앞 N초만(서빙과 같다)")
    e.add_argument("--batch_sec", type=float, default=600.0, help="배치 총 길이(초, 가장 긴 것 × 개수)")
    e.add_argument("--batch_items", type=int, default=128)
    e.add_argument("--workers", type=int, default=8)
    e.add_argument("--limit", type=int, default=0, help="앞 N행만(점검용)")
    e.add_argument("--force", action="store_true")
    q = sub.add_parser("probe")
    q.add_argument("--backbone", required=True, choices=sorted(BACKBONES))
    q.add_argument("--epochs", type=int, default=60)
    q.add_argument("--patience", type=int, default=8)
    q.add_argument("--bs", type=int, default=512)
    q.add_argument("--lr", type=float, default=1e-3)
    q.add_argument("--wd", type=float, default=1e-4)
    q.add_argument("--balance", type=float, default=0.5, help="클래스 가중 (1/빈도)^balance")
    q.add_argument("--seed", type=int, default=0)
    v = sub.add_parser("evalset")
    v.add_argument("--backbone", required=True, choices=sorted(BACKBONES) + ["run_zeroth"], help="run_zeroth = 배포 주 모델")
    v.add_argument("--sets", default="", help="noise_sep 안 조건 이름들(쉼표). 비우면 전부 + _sep")
    v.add_argument("--batch_sec", type=float, default=600.0)
    v.add_argument("--workers", type=int, default=8)
    t = sub.add_parser("finetune", help="같은 조건 부분 미세조정 비교")
    t.add_argument("--backbone", required=True, choices=sorted(BACKBONES))
    t.add_argument("--depth", type=int, default=18, help="앞 N층만 남긴다(배포 모델 = 18)")
    t.add_argument("--unfreeze_from", type=int, default=13, help="이 층부터 학습(배포 모델 = 13)")
    t.add_argument("--epochs", type=int, default=1)
    t.add_argument("--crop", type=float, default=8.0)
    t.add_argument("--batch_sec", type=float, default=320.0)
    t.add_argument("--lr", type=float, default=2e-5)
    t.add_argument("--head_lr", type=float, default=1e-3)
    t.add_argument("--eval_every", type=int, default=500)
    t.add_argument("--workers", type=int, default=8)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--lp_rows", type=int, default=20000, help="시작 선형 프로브·mu/sd 에 쓸 train 행 수")
    w_ = sub.add_parser("probe_wsum", help="층 가중합 헤드(동결 캐시 특징)")
    w_.add_argument("--backbone", required=True, choices=sorted(BACKBONES))
    w_.add_argument("--hidden", default="0,512,moe4x512,moe8x256", help="0 = 선형, N = 은닉 N MLP, moeKxH = 전문가 K개(은닉 H)")
    w_.add_argument("--seed", type=int, default=0)
    sub.add_parser("report")
    a = ap.parse_args()
    if a.cmd is None:
        ap.print_help()
        return
    {"prep": cmd_prep, "extract": cmd_extract, "probe": cmd_probe, "evalset": cmd_evalset, "finetune": cmd_finetune, "probe_wsum": cmd_probe_wsum, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
