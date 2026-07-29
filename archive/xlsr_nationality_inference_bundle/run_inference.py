#!/usr/bin/env python3
"""Run XLS-R nationality/language-area inference from wav files.

This script is inference-only. It loads XLS-R once, loads the portable
linear/MLP classifier checkpoints, then runs:

wav -> mono/16 kHz -> XLS-R -> mean pooling -> classifier ensemble -> top-k.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio
from transformers import AutoFeatureExtractor, AutoModel


DEFAULT_MODEL_NAME = "facebook/wav2vec2-xls-r-2b"
DEFAULT_WEIGHTS = {
    "linear": 0.80,
    "mlp30": 0.05,
    "mlp60": 0.15,
}


class CachedMLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float):
        super().__init__()
        if hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.net = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.net(x)


def parse_crop_seconds(value: str | None) -> float | None:
    if value is None:
        return None
    value = str(value).strip().lower()
    if value in {"none", "full", "all", ""}:
        return None
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("--crop_seconds must be positive or 'none'")
    return parsed


def parse_args():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="XLS-R nationality inference and latency benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--audio", nargs="*", default=[], help="One or more wav paths.")
    parser.add_argument("--audio_list", type=Path, help="Text file with one wav path per line.")
    parser.add_argument(
        "--manifest_csv",
        nargs="*",
        type=Path,
        default=[],
        help="One or more CSV files with audio_path,duration_sec,label_name columns.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Pick samples near target durations from --manifest_csv and run latency benchmark.",
    )
    parser.add_argument("--target_seconds", nargs="+", type=float, default=[5, 10, 15, 30, 45, 60, 120])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--checkpoint_dir", type=Path, default=root / "checkpoints")
    parser.add_argument("--output_dir", type=Path, default=root / "outputs")
    parser.add_argument("--crop_seconds", type=parse_crop_seconds, default=None)
    parser.add_argument(
        "--chunk_seconds",
        type=parse_crop_seconds,
        default=None,
        help="If set, split the loaded audio into chunks and average logits.",
    )
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fp32", action="store_true", help="Use float32 instead of fp16 on CUDA.")
    parser.add_argument("--local_files_only", action="store_true", help="Do not download the HF model.")
    parser.add_argument(
        "--print_labels",
        action="store_true",
        help="Print classifier label_id -> label mapping and exit without loading XLS-R.",
    )
    return parser.parse_args()


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def now():
    sync()
    return time.perf_counter()


def load_head(name: str, path: Path, device: str):
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint for {name}: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt["classifier"]
    if "net.0.weight" in state:
        input_dim = state["net.0.weight"].shape[1]
    else:
        input_dim = state["net.weight"].shape[1]
    labels = ckpt["labels"]
    head = CachedMLPClassifier(
        input_dim=input_dim,
        hidden_dim=int(ckpt["hidden_dim"]),
        num_classes=len(labels),
        dropout=float(ckpt["dropout"]),
    )
    head.load_state_dict(state)
    head.eval().to(device)
    ckpt["x_mean"] = ckpt["x_mean"].to(device)
    ckpt["x_std"] = ckpt["x_std"].to(device)
    return ckpt, head, labels


def load_labels(checkpoint_dir: Path):
    path = checkpoint_dir / "linear_best_portable.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint for labels: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt["labels"]


def load_heads(checkpoint_dir: Path, device: str):
    specs = [
        ("linear", checkpoint_dir / "linear_best_portable.pt", DEFAULT_WEIGHTS["linear"]),
        ("mlp30", checkpoint_dir / "mlp30_best_portable.pt", DEFAULT_WEIGHTS["mlp30"]),
        ("mlp60", checkpoint_dir / "mlp60_best_portable.pt", DEFAULT_WEIGHTS["mlp60"]),
    ]
    heads = []
    labels = None
    for name, path, weight in specs:
        ckpt, head, labels = load_head(name, path, device)
        heads.append((name, weight, ckpt, head))
    return heads, labels


def load_audio(path: Path, sample_rate: int, crop_seconds: float | None):
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    original_duration = len(audio) / float(sr)
    if crop_seconds is not None:
        audio = audio[: int(crop_seconds * sr)]
    if sr != sample_rate:
        wav = torch.from_numpy(audio).unsqueeze(0)
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
        audio = wav.squeeze(0).numpy()
        sr = sample_rate
    return np.asarray(audio, dtype=np.float32), sr, original_duration


def audio_chunks(audio: np.ndarray, sample_rate: int, chunk_seconds: float | None):
    if chunk_seconds is None:
        return [audio]
    chunk_samples = int(chunk_seconds * sample_rate)
    if chunk_samples <= 0:
        return [audio]
    chunks = []
    for start in range(0, len(audio), chunk_samples):
        chunk = audio[start : start + chunk_samples]
        if len(chunk) > 0:
            chunks.append(chunk)
    return chunks or [audio]


@torch.inference_mode()
def logits_for_audio(audio, sample_rate, extractor, encoder, heads, labels, device, dtype, chunk_seconds):
    chunk_logits = []
    feature_extractor_sec = 0.0
    xlsr_forward_pool_sec = 0.0
    classifier_sec = 0.0

    for chunk in audio_chunks(audio, sample_rate, chunk_seconds):
        t1 = now()
        inputs = extractor(chunk, sampling_rate=sample_rate, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        if dtype == torch.float16:
            inputs = {k: (v.half() if v.dtype.is_floating_point else v) for k, v in inputs.items()}
        t2 = now()
        feature_extractor_sec += t2 - t1

        out = encoder(**inputs)
        feat = out.last_hidden_state.mean(dim=1).float()
        t3 = now()
        xlsr_forward_pool_sec += t3 - t2

        ensemble_logits = None
        for name, weight, ckpt, head in heads:
            x = (feat - ckpt["x_mean"]) / ckpt["x_std"]
            logits = head(x)
            ensemble_logits = logits * weight if ensemble_logits is None else ensemble_logits + logits * weight
        t4 = now()
        classifier_sec += t4 - t3
        chunk_logits.append(ensemble_logits)

    logits = torch.stack(chunk_logits, dim=0).mean(dim=0)
    timings = {
        "feature_extractor_sec": feature_extractor_sec,
        "xlsr_forward_pool_sec": xlsr_forward_pool_sec,
        "classifier_ensemble_sec": classifier_sec,
    }
    return logits, timings


def predict_audio(path, extractor, encoder, heads, labels, args):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    t0 = now()
    audio, sr, original_duration = load_audio(path, args.sample_rate, args.crop_seconds)
    t1 = now()

    logits, timings = logits_for_audio(
        audio=audio,
        sample_rate=sr,
        extractor=extractor,
        encoder=encoder,
        heads=heads,
        labels=labels,
        device=args.device,
        dtype=torch.float32 if args.fp32 or args.device == "cpu" else torch.float16,
        chunk_seconds=args.chunk_seconds,
    )
    t4 = now()

    probs = torch.softmax(logits, dim=-1)
    top = probs.topk(min(args.top_k, probs.size(1)), dim=-1)
    top_labels = [labels[i] for i in top.indices[0].tolist()]
    top_probs = [float(v) for v in top.values[0].tolist()]
    timings = {
        "audio_load_resample_sec": t1 - t0,
        **timings,
        "request_total_sec": t4 - t0,
    }
    return {
        "audio_path": str(path),
        "original_duration_sec": original_duration,
        "used_duration_sec": len(audio) / float(sr),
        "sample_rate": sr,
        "crop_seconds": args.crop_seconds,
        "chunk_seconds": args.chunk_seconds,
        "top1": top_labels[0],
        "topk": [{"label": label, "prob": prob} for label, prob in zip(top_labels, top_probs)],
        "timings": timings,
    }


def read_audio_paths(args):
    paths = list(args.audio)
    if args.audio_list:
        for line in args.audio_list.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                paths.append(line)
    return paths


def load_manifest(path: Path):
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["_duration_float"] = float(row["duration_sec"])
        row["_manifest_csv"] = str(path)
    return rows


def pick_benchmark_items(manifest_csvs, target_seconds):
    if not manifest_csvs:
        raise ValueError("--manifest_csv is required for --benchmark when --audio is empty.")
    rows = []
    for manifest_csv in manifest_csvs:
        rows.extend(r for r in load_manifest(manifest_csv) if Path(r["audio_path"]).exists())
    if not rows:
        raise ValueError(f"No usable wav paths found in {manifest_csvs}")
    items = []
    used = set()
    for target in target_seconds:
        candidates = [r for r in rows if r["audio_path"] not in used]
        if not candidates:
            break
        row = min(candidates, key=lambda r: abs(r["_duration_float"] - target))
        used.add(row["audio_path"])
        items.append(
            {
                "target_sec": target,
                "audio_path": row["audio_path"],
                "gold": row.get("label_name"),
                "speaker_id": row.get("speaker_id"),
                "manifest_duration_sec": row["_duration_float"],
                "manifest_csv": row["_manifest_csv"],
            }
        )
    return items


def print_prediction(result):
    print("=" * 100)
    print("audio:", result["audio_path"])
    print(f"duration: original={result['original_duration_sec']:.3f}s used={result['used_duration_sec']:.3f}s")
    print("top1:", result["top1"])
    print("top-k:")
    for rank, item in enumerate(result["topk"], start=1):
        print(f"  {rank}. {item['label']}: {item['prob']:.4f}")
    print("timings:")
    for key, value in result["timings"].items():
        print(f"  {key}: {value:.4f}s")
    print("10s SLA:", "OK" if result["timings"]["request_total_sec"] <= 10.0 else "SLOW")


def run_benchmark(items, extractor, encoder, heads, labels, args):
    if not items:
        raise ValueError("No benchmark items.")
    _ = predict_audio(items[0]["audio_path"], extractor, encoder, heads, labels, args)

    rows = []
    for item in items:
        for run_idx in range(1, args.repeats + 1):
            result = predict_audio(item["audio_path"], extractor, encoder, heads, labels, args)
            row = {
                "target_sec": item.get("target_sec"),
                "manifest_duration_sec": item.get("manifest_duration_sec"),
                "manifest_csv": item.get("manifest_csv"),
                "gold": item.get("gold"),
                "speaker_id": item.get("speaker_id"),
                "audio_path": item["audio_path"],
                "run_idx": run_idx,
                "original_duration_sec": result["original_duration_sec"],
                "used_duration_sec": result["used_duration_sec"],
                "top1": result["top1"],
                "topk_labels": " | ".join(x["label"] for x in result["topk"]),
                "crop_seconds": result["crop_seconds"],
                "chunk_seconds": result["chunk_seconds"],
            }
            row.update(result["timings"])
            rows.append(row)
    runs = pd.DataFrame(rows)
    summary = (
        runs.groupby(["target_sec", "audio_path", "manifest_csv", "gold", "speaker_id"], dropna=False)
        .agg(
            original_duration_sec=("original_duration_sec", "first"),
            used_duration_sec=("used_duration_sec", "first"),
            repeats=("run_idx", "count"),
            total_mean_sec=("request_total_sec", "mean"),
            total_min_sec=("request_total_sec", "min"),
            total_max_sec=("request_total_sec", "max"),
            audio_load_mean_sec=("audio_load_resample_sec", "mean"),
            feature_extractor_mean_sec=("feature_extractor_sec", "mean"),
            xlsr_forward_pool_mean_sec=("xlsr_forward_pool_sec", "mean"),
            classifier_mean_sec=("classifier_ensemble_sec", "mean"),
            top1=("top1", "first"),
            topk_labels=("topk_labels", "first"),
        )
        .reset_index()
        .sort_values("used_duration_sec")
    )
    return runs, summary


def write_outputs(args, payload=None, runs=None, summary=None):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if payload is not None:
        (args.output_dir / "latest_inference_result.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if runs is not None:
        runs.to_csv(args.output_dir / "latency_benchmark_runs.csv", index=False, encoding="utf-8-sig")
    if summary is not None:
        summary.to_csv(args.output_dir / "latency_benchmark_summary.csv", index=False, encoding="utf-8-sig")
        md = [
            "# Latency Benchmark Summary",
            "",
            f"- device: `{args.device}`",
            f"- model: `{args.model_name}`",
            f"- crop_seconds: `{args.crop_seconds}`",
            f"- chunk_seconds: `{args.chunk_seconds}`",
            "",
            "| target | used sec | gold | top1 | repeats | total mean | xlsr mean | classifier mean |",
            "|---:|---:|---|---|---:|---:|---:|---:|",
        ]
        for _, row in summary.iterrows():
            md.append(
                f"| {row['target_sec']} | {row['used_duration_sec']:.3f} | {row['gold']} | "
                f"{row['top1']} | {int(row['repeats'])} | {row['total_mean_sec']:.4f} | "
                f"{row['xlsr_forward_pool_mean_sec']:.4f} | {row['classifier_mean_sec']:.4f} |"
            )
        (args.output_dir / "latency_benchmark_summary.md").write_text("\n".join(md), encoding="utf-8")


def main():
    args = parse_args()
    if args.print_labels:
        labels = load_labels(args.checkpoint_dir)
        print("num_labels:", len(labels))
        print("label_id,label")
        for idx, label in enumerate(labels):
            print(f"{idx},{label}")
        return

    dtype = torch.float32 if args.fp32 or args.device == "cpu" else torch.float16

    t0 = now()
    extractor = AutoFeatureExtractor.from_pretrained(args.model_name, local_files_only=args.local_files_only)
    encoder = AutoModel.from_pretrained(
        args.model_name,
        local_files_only=args.local_files_only,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    encoder.eval().to(args.device)
    heads, labels = load_heads(args.checkpoint_dir, args.device)
    t1 = now()

    print("device:", args.device)
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
        print(f"gpu_allocated_gb: {torch.cuda.memory_allocated() / 1024**3:.2f}")
    print("model:", args.model_name)
    print("num_labels:", len(labels))
    print(f"model_and_heads_load_sec: {t1 - t0:.4f}")
    print("crop_seconds:", args.crop_seconds)
    print("chunk_seconds:", args.chunk_seconds)

    audio_paths = read_audio_paths(args)
    payload = {
        "model_name": args.model_name,
        "device": args.device,
        "crop_seconds": args.crop_seconds,
        "chunk_seconds": args.chunk_seconds,
        "model_and_heads_load_sec": t1 - t0,
        "utterance_results": [],
    }

    if audio_paths and not args.benchmark:
        _ = predict_audio(audio_paths[0], extractor, encoder, heads, labels, args)
        for path in audio_paths:
            result = predict_audio(path, extractor, encoder, heads, labels, args)
            payload["utterance_results"].append(result)
            print_prediction(result)
        write_outputs(args, payload=payload)
        print("wrote:", args.output_dir / "latest_inference_result.json")
        return

    if args.benchmark:
        if audio_paths:
            items = [{"audio_path": p, "target_sec": None, "gold": None, "speaker_id": None} for p in audio_paths]
        else:
            items = pick_benchmark_items(args.manifest_csv, args.target_seconds)
        runs, summary = run_benchmark(items, extractor, encoder, heads, labels, args)
        print(summary.round(4).to_string(index=False))
        write_outputs(args, payload=payload, runs=runs, summary=summary)
        print("wrote:", args.output_dir)
        return

    raise SystemExit("Provide --audio/--audio_list for inference or --benchmark with --manifest_csv.")


if __name__ == "__main__":
    main()
