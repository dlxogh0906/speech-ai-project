"""CSV manifest Dataset and collate utilities for audio classification."""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists(): _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]


import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from .audio import AudioLoadError, load_wav_mono


def read_manifest(path):
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["label_id"] = int(row["label_id"])
        if row.get("duration_sec"):
            row["duration_sec"] = float(row["duration_sec"])
    return rows


class AudioCountryDataset(Dataset):
    def __init__(
        self,
        manifest_path,
        sample_rate=16000,
        min_duration_sec=0.25,
        max_duration_sec=15.0,
        skip_errors=True,
    ):
        self.manifest_path = Path(manifest_path)
        self.rows = read_manifest(self.manifest_path)
        self.sample_rate = sample_rate
        self.min_samples = int(min_duration_sec * sample_rate)
        self.max_samples = int(max_duration_sec * sample_rate) if max_duration_sec else 0
        self.skip_errors = skip_errors

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        try:
            waveform, sample_rate = load_wav_mono(row["audio_path"], self.sample_rate)
            if sample_rate != self.sample_rate:
                raise AudioLoadError(f"unexpected sample rate: {sample_rate}")
            if waveform.numel() < self.min_samples:
                raise AudioLoadError(f"too short: {waveform.numel() / self.sample_rate:.3f}s")
            if self.max_samples and waveform.numel() > self.max_samples:
                waveform = waveform[: self.max_samples]
            return {
                "waveform": waveform,
                "label": row["label_id"],
                "audio_path": row["audio_path"],
                "json_path": row["json_path"],
                "speaker_id": row["speaker_id"],
                "label_name": row["label_name"],
            }
        except Exception as exc:
            if not self.skip_errors:
                raise
            return {"_skip": True, "error": str(exc), "audio_path": row.get("audio_path", "")}


def collate_audio_batch(items):
    skipped = [item for item in items if item is None or item.get("_skip")]
    items = [item for item in items if item is not None and not item.get("_skip")]
    if not items:
        return None

    lengths = torch.tensor([item["waveform"].numel() for item in items], dtype=torch.long)
    max_len = int(lengths.max().item())
    waveforms = torch.zeros(len(items), max_len, dtype=torch.float32)
    padding_mask = torch.ones(len(items), max_len, dtype=torch.bool)
    for idx, item in enumerate(items):
        length = item["waveform"].numel()
        waveforms[idx, :length] = item["waveform"]
        padding_mask[idx, :length] = False

    return {
        "waveforms": waveforms,
        "padding_mask": padding_mask,
        "labels": torch.tensor([item["label"] for item in items], dtype=torch.long),
        "lengths": lengths,
        "audio_paths": [item["audio_path"] for item in items],
        "json_paths": [item["json_path"] for item in items],
        "speaker_ids": [item["speaker_id"] for item in items],
        "label_names": [item["label_name"] for item in items],
        "skipped": skipped,
    }


def build_dataloader(
    manifest_path,
    sample_rate,
    batch_size,
    shuffle,
    seed,
    min_duration_sec=0.25,
    max_duration_sec=15.0,
    num_workers=0,
):
    dataset = AudioCountryDataset(
        manifest_path=manifest_path,
        sample_rate=sample_rate,
        min_duration_sec=min_duration_sec,
        max_duration_sec=max_duration_sec,
        skip_errors=True,
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_audio_batch,
        generator=generator if shuffle else None,
        pin_memory=torch.cuda.is_available(),
    )
