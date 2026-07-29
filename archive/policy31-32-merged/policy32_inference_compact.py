#!/usr/bin/env python3
"""Run Policy32 inference with the optimizer-free XLS-R checkpoint."""

import os
import argparse
import json
import random
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn


PROJECT_ROOT = Path(os.path.expandvars("$HOME"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from country_classifier.audio import load_wav_mono
from country_classifier.model import build_model


def sync_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def select_test_audio(manifest_path, group, seed):
    manifest = pd.read_csv(manifest_path)
    if group == "foreign":
        manifest = manifest[manifest["label_id"] != 31]
    elif group == "korean":
        manifest = manifest[manifest["label_id"] == 31]

    random_state = seed
    if random_state is None:
        random_state = random.SystemRandom().randint(0, 2**32 - 1)
    row = manifest.sample(n=1, random_state=random_state).iloc[0]
    return Path(row["audio_path"]), row.to_dict(), random_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path)
    parser.add_argument(
        "--group",
        choices=("all", "foreign", "korean"),
        default="all",
        help="random test subset used when --audio is omitted",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-sec", type=float, default=16.0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--xlsr-checkpoint",
        type=Path,
        default=Path(os.path.expandvars("$SPEECH_AI_DATA/models/xlsr2_2B_1000k_inference.pt")),
    )
    parser.add_argument(
        "--head-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "taeho/policy32_othergpu_split_16s_probe/linear_best.pt",
    )
    parser.add_argument(
        "--test-manifest",
        type=Path,
        default=PROJECT_ROOT / "taeho/policy32_othergpu_split_seed42/test.csv",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    head_payload = torch.load(args.head_checkpoint, map_location="cpu")
    labels = list(head_payload["labels"])

    classifier = nn.Linear(1920, len(labels))
    classifier.load_state_dict(head_payload["head"])
    classifier = classifier.to(device).eval()
    mean = torch.as_tensor(head_payload["mu"], dtype=torch.float32, device=device)
    std = torch.as_tensor(head_payload["sd"], dtype=torch.float32, device=device)

    load_started = time.perf_counter()
    encoder = build_model(
        backend="fairseq_xlsr",
        num_classes=len(labels),
        checkpoint_path=args.xlsr_checkpoint,
        fairseq_root=PROJECT_ROOT / "fairseq",
        freeze_encoder=True,
    ).to(device).eval()
    sync_cuda(device)
    model_load_sec = time.perf_counter() - load_started

    metadata = None
    random_state = None
    if args.audio:
        audio_path = args.audio
    else:
        audio_path, metadata, random_state = select_test_audio(
            args.test_manifest,
            args.group,
            args.seed,
        )

    total_started = time.perf_counter()
    waveform, _ = load_wav_mono(audio_path, target_sample_rate=16000)
    original_sec = waveform.numel() / 16000
    waveform = waveform[: int(args.max_sec * 16000)].contiguous()
    used_sec = waveform.numel() / 16000
    waveforms = waveform.unsqueeze(0).to(device)
    padding_mask = torch.zeros_like(waveforms, dtype=torch.bool)

    sync_cuda(device)
    encoder_started = time.perf_counter()
    with torch.inference_mode():
        if device.type == "cuda":
            with torch.autocast(device_type="cuda"):
                features, feature_mask = encoder._extract_features(waveforms, padding_mask)
                pooled = encoder._pool(features, feature_mask)
        else:
            features, feature_mask = encoder._extract_features(waveforms, padding_mask)
            pooled = encoder._pool(features, feature_mask)
    sync_cuda(device)
    encoder_sec = time.perf_counter() - encoder_started

    classifier_started = time.perf_counter()
    with torch.inference_mode():
        logits = classifier((pooled.float() - mean) / std)
        probabilities = torch.softmax(logits, dim=-1)[0]
        scores, indices = probabilities.topk(min(args.top_k, len(labels)))
    sync_cuda(device)
    classifier_sec = time.perf_counter() - classifier_started
    inference_sec = time.perf_counter() - total_started

    predictions = [
        {
            "rank": rank,
            "label_id": int(index),
            "label": labels[int(index)],
            "probability": float(score),
        }
        for rank, (score, index) in enumerate(
            zip(scores.cpu(), indices.cpu()),
            start=1,
        )
    ]
    result = {
        "audio_path": str(audio_path),
        "true_label": metadata.get("label_name") if metadata else None,
        "random_state": random_state,
        "device": str(device),
        "model_load_sec": model_load_sec,
        "original_duration_sec": original_sec,
        "used_duration_sec": used_sec,
        "encoder_sec": encoder_sec,
        "classifier_sec": classifier_sec,
        "inference_total_sec": inference_sec,
        "realtime_factor": inference_sec / max(used_sec, 1e-9),
        "predictions": predictions,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
