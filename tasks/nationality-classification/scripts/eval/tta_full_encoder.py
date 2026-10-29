# -*- coding: utf-8 -*-
"""TTA — 같은 녹음을 바꿔 여러 번 넣는다. 변형마다 test 로짓을 남긴다 (2026-10-26).

finetune_full_encoder.py --eval_only 와 같은 경로(앞 12초 · bf16 · 길이 묶음 배치)로 돌리고, 불러온 파형에만 변형을 건다.
  변형 (결과 21 과 같다): orig 원본 · sp09 속도 0.9 · sp11 속도 1.1 (resample_poly, 음높이도 같이 바뀐다)
                          · noise30 백색 잡음 30 dB (발화마다 고정 시드) · head85 앞 85% · tail85 뒤 85%
  변형마다 <out>/<변형>.npz (ok · L · labels) 를 남기고, 이미 있으면 건너뛴다. 평균은 ensemble_logits_report.py 로 낸다.

    $PY scripts/eval/tta_full_encoder.py --ckpt <best.pt> --test <test.csv> --out <폴더> [--variants orig sp09 ...]
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src"))),
                 str(_r / "tasks" / "nationality-classification" / "experiments")]

import argparse, csv, importlib.util, json, os, time, zlib
import numpy as np

VARIANTS = ("orig", "sp09", "sp11", "noise30", "head85", "tail85")
LEN_FACTOR = {"orig": 1.0, "sp09": 1 / 0.9, "sp11": 1 / 1.1, "noise30": 1.0, "head85": 0.85, "tail85": 0.85}


def transform(name, w, key):
    """w: float32 1차원 파형(16 kHz). key: 잡음 시드용 문자열(오디오 경로)."""
    if name == "orig":
        return w
    if name in ("sp09", "sp11"):
        from scipy.signal import resample_poly
        f = 0.9 if name == "sp09" else 1.1
        return resample_poly(w, 100, int(round(100 * f))).astype(np.float32)
    if name == "noise30":
        rng = np.random.default_rng(zlib.crc32(key.encode("utf-8")))
        p = float(np.mean(w.astype(np.float64) ** 2)) + 1e-12
        return (w + rng.normal(0, np.sqrt(p / 10 ** 3), len(w))).astype(np.float32)
    n = len(w)
    if name == "head85":
        return w[: int(n * 0.85)]
    if name == "tail85":
        return w[int(n * 0.15):]
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    ap.add_argument("--eval_batch_sec", type=float, default=240.0)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader, Dataset
    from country_classifier.dataset import AudioCountryDataset, collate_audio_batch
    from country_classifier.kd_student import LengthBucketBatchSampler, forward_pooled, write_loader_csv
    spec = importlib.util.spec_from_file_location(
        "finetune_full_encoder", str(_r / "tasks" / "nationality-classification" / "scripts" / "train" / "finetune_full_encoder.py"))
    ff = importlib.util.module_from_spec(spec); spec.loader.exec_module(ff)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = dict(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda")
    ck = torch.load(a.ckpt, map_location="cpu")
    xlsr_ck = ck.get("xlsr") or os.path.join(str(ff.MODEL_ROOT), ff.BACKBONES[ck.get("backbone", "300m")][0])
    if ck.get("xh_cfg"):
        raise SystemExit("운율 실험 체크포인트(xh_cfg)는 지원하지 않는다")
    enc = ff.load_backbone(xlsr_ck, ck["layer"], dev); enc.load_state_dict(ck["encoder"]); enc.eval()
    labels = list(ck["labels"])
    head = torch.nn.Linear(ck["head"]["weight"].shape[1], len(labels)).to(dev); head.load_state_dict(ck["head"]); head.eval()
    mu = torch.as_tensor(ck["mu"]).to(dev); sd = torch.as_tensor(ck["sd"]).to(dev)

    with open(a.test, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    os.makedirs(a.out, exist_ok=True)
    loader_csv = write_loader_csv(rows, os.path.join(a.out, "_test_loader.csv"))
    base = AudioCountryDataset(loader_csv, 16000, min_duration_sec=0.25, max_duration_sec=12.0)

    class Wrapped(Dataset):
        def __init__(self, name):
            self.name = name

        def __len__(self):
            return len(base)

        def __getitem__(self, i):
            it = base[i]
            if it.get("_skip"):
                return it
            w = it["waveform"].numpy().astype(np.float32)
            w2 = transform(self.name, w, it["audio_path"])
            it["waveform"] = torch.from_numpy(np.ascontiguousarray(w2 if len(w2) >= 4000 else w))
            return it

    dur = np.array([min(float(r["duration_sec"]), 12.0) for r in rows])
    for name in a.variants:
        dst = os.path.join(a.out, f"{name}.npz")
        if os.path.exists(dst):
            print(f"[tta] {name} 있음 — 건너뜀", flush=True); continue
        t0 = time.time()
        k = LEN_FACTOR[name]
        bs = LengthBucketBatchSampler(dur * k, 12.0 * k + 0.01, a.eval_batch_sec, 128, shuffle=False)
        ld = DataLoader(Wrapped(name), batch_sampler=bs, num_workers=a.workers, collate_fn=collate_audio_batch, pin_memory=True)
        L = np.zeros((len(rows), len(labels)), np.float32); seen = np.zeros(len(rows), bool)
        with torch.no_grad():
            for bt in ld:
                if bt is None:
                    continue
                idx = np.array([int(s) for s in bt["speaker_ids"]])
                with torch.autocast(**amp):
                    pooled = forward_pooled(enc, bt["waveforms"].to(dev), bt["padding_mask"].to(dev), ck["layer"])
                L[idx] = head((pooled - mu) / sd).float().cpu().numpy(); seen[idx] = True
        np.savez(dst + ".tmp.npz", ok=seen, L=L, labels=np.array(labels))
        os.replace(dst + ".tmp.npz", dst)
        print(f"[tta] {name} 끝 · {int(seen.sum()):,}/{len(rows):,}행 · {time.time() - t0:.0f}초", flush=True)
    json.dump({"ckpt": a.ckpt, "test": a.test, "variants": a.variants}, open(os.path.join(a.out, "meta.json"), "w"), ensure_ascii=False)
    print("TTA_DONE", flush=True)


if __name__ == "__main__":
    main()
