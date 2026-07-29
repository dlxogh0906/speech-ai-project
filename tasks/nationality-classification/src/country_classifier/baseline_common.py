"""Shared helpers for the XLS-R country-classification baseline scripts.

This module is the bridge between the *fairseq-style* manifests under
``$NATIONALITY_DATA`` (``{split}.tsv`` / ``.lang`` / ``.spk`` / ``dict.lang.txt`` /
``labels.txt``) and the *CSV-manifest* ``AudioCountryDataset`` that already lives in
``$HOME/country_classifier``.  We do not modify the existing package; instead we
materialise a small CSV manifest that ``AudioCountryDataset`` already knows how to read.

It also provides:
  * label-map construction (label2id / id2label) from labels.txt or dict.lang.txt
  * checkpoint save/restore that wraps ``model.checkpoint_state()``
  * synthetic-WAV generation, used when the real AIHub audio is not present on this
    machine (the manifests reference an absolute root such as ``/home/<other-user>/...`` that
    may not exist locally).
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists(): _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import DATA_ROOT


import csv
import json
import os
import sys
import wave
from pathlib import Path

import numpy as np

# Make the local `country_classifier` package importable regardless of CWD.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_SAMPLE_RATE = 16000
CSV_FIELDS = ["audio_path", "json_path", "speaker_id", "label_name", "label_id", "duration_sec"]

# The manifests were authored on another machine; their TSV root points at
# $PROJECT_ROOT/131.인공지능 학습을 위한 ... (spaces). On THIS server the same
# corpus lives here (underscores), with per-split subdirs 1.Training / 2.Validation:
DEFAULT_AUDIO_ROOT = (
    f"{DATA_ROOT}/aihub/foreign_korean_speech/"
    "131.인공지능_학습을_위한_외국인_한국어_발화_음성_데이터/01.데이터_new_20220719"
)


# --------------------------------------------------------------------------------------
# fairseq manifest parsing
# --------------------------------------------------------------------------------------
def resolve_real_root(orig_root, audio_root):
    """Remap a manifest TSV root onto the local audio base.

    The manifest root ends with the split dir (e.g. ".../01.데이터_new_20220719/2.Validation").
    We keep that trailing token and join it onto `audio_root`.
    """
    orig_root = Path(orig_root)
    if not audio_root:
        return orig_root
    tail = orig_root.name  # "1.Training" or "2.Validation"
    return Path(audio_root) / tail


def build_audio_index(audio_root, exts=(".wav",)):
    """Map {basename -> absolute path} over the extracted corpus.

    The AIHub zips extract into per-language folder layouts that do NOT always match
    the manifests' literal rel-paths (e.g. Training Vietnamese is doubly nested as
    ``TS_1.베트남어/TS_1. 베트남어/...``, and some folders carry encoding-mangled names).
    Utterance filenames (e.g. ``VN16RC139_VN0019_20210806.wav``) are globally unique,
    so resolving by basename is robust to all those quirks. Scanning ~1M files is ~3s.
    """
    idx = {}
    if not audio_root or not Path(audio_root).exists():
        return idx
    exts = tuple(e.lower() for e in exts)
    for dirpath, _, filenames in os.walk(audio_root):
        for f in filenames:
            if f.lower().endswith(exts):
                idx[f] = os.path.join(dirpath, f)
    return idx


def manifest_paths(manifest_dir, split):
    manifest_dir = Path(manifest_dir)
    return {
        "tsv": manifest_dir / f"{split}.tsv",
        "lang": manifest_dir / f"{split}.lang",
        "spk": manifest_dir / f"{split}.spk",
        "dict": manifest_dir / "dict.lang.txt",
        "labels": manifest_dir / "labels.txt",
    }


def read_fairseq_split(manifest_dir, split, audio_root=None, audio_index=None):
    """Return (root, entries) where entries is a list of dicts.

    Each entry: {audio_path, rel_path, num_frames, label_name, speaker_id}.
    `audio_root` (if given) remaps the manifest TSV root onto the local corpus.
    `audio_index` (if given, see build_audio_index) resolves each audio_path by
    basename, which is robust to the corpus's irregular folder layouts.
    Raises FileNotFoundError / ValueError with clear messages on malformed input.
    """
    paths = manifest_paths(manifest_dir, split)
    if not paths["tsv"].exists():
        raise FileNotFoundError(f"missing TSV: {paths['tsv']}")
    if not paths["lang"].exists():
        raise FileNotFoundError(f"missing label file: {paths['lang']}")

    tsv_lines = paths["tsv"].read_text(encoding="utf-8").splitlines()
    if not tsv_lines:
        raise ValueError(f"empty TSV: {paths['tsv']}")
    orig_root = Path(tsv_lines[0].strip())
    root = resolve_real_root(orig_root, audio_root)

    rel_entries = []
    for ln in tsv_lines[1:]:
        if not ln.strip():
            continue
        parts = ln.rsplit("\t", 1)
        rel = parts[0]
        frames = int(parts[1]) if len(parts) == 2 and parts[1].strip().isdigit() else 0
        rel_entries.append((rel, frames))

    langs = [l.strip() for l in paths["lang"].read_text(encoding="utf-8").splitlines() if l.strip() != ""]
    if len(langs) != len(rel_entries):
        raise ValueError(
            f"{split}: tsv rows ({len(rel_entries)}) != lang rows ({len(langs)}) "
            f"in {manifest_dir}"
        )

    if paths["spk"].exists():
        spks = [s.strip() for s in paths["spk"].read_text(encoding="utf-8").splitlines() if s.strip() != ""]
        if len(spks) != len(rel_entries):
            raise ValueError(
                f"{split}: tsv rows ({len(rel_entries)}) != spk rows ({len(spks)})"
            )
    else:
        spks = ["" for _ in rel_entries]

    entries = []
    for (rel, frames), lang, spk in zip(rel_entries, langs, spks):
        if audio_index is not None:
            audio_path = audio_index.get(Path(rel).name, str(root / rel))
        else:
            audio_path = str(root / rel)
        entries.append(
            {
                "rel_path": rel,
                "audio_path": audio_path,
                "num_frames": frames,
                "label_name": lang,
                "speaker_id": spk,
            }
        )
    return root, entries


# --------------------------------------------------------------------------------------
# label map
# --------------------------------------------------------------------------------------
def load_labels(manifest_dir):
    """Ordered label list. Prefer labels.txt; fall back to dict.lang.txt."""
    paths = manifest_paths(manifest_dir, "train")
    if paths["labels"].exists():
        labels = [l.strip() for l in paths["labels"].read_text(encoding="utf-8").splitlines() if l.strip()]
    elif paths["dict"].exists():
        labels = []
        for ln in paths["dict"].read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            labels.append(ln.split()[0])
    else:
        raise FileNotFoundError(f"no labels.txt or dict.lang.txt in {manifest_dir}")
    if not labels:
        raise ValueError(f"empty label set in {manifest_dir}")
    return labels


def build_label_maps(manifest_dir):
    labels = load_labels(manifest_dir)
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for idx, label in enumerate(labels)}
    return labels, label2id, id2label


def load_alias_map(alias_config):
    """Optional fine-grained label alias map (configs/*.json -> {'aliases': {...}})."""
    if not alias_config:
        return {}
    data = json.loads(Path(alias_config).read_text(encoding="utf-8"))
    return data.get("aliases", data) if isinstance(data, dict) else {}


# --------------------------------------------------------------------------------------
# CSV materialisation for AudioCountryDataset
# --------------------------------------------------------------------------------------
def write_csv_manifest(entries, label2id, out_csv, alias_map=None, limit=None,
                       sample_rate=DEFAULT_SAMPLE_RATE):
    """Write a CSV that AudioCountryDataset can read. Returns (n_written, n_skipped_label)."""
    alias_map = alias_map or {}
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    n_skipped_label = 0
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for entry in entries:
            if limit is not None and n_written >= limit:
                break
            label_name = alias_map.get(entry["label_name"], entry["label_name"])
            if label_name not in label2id:
                n_skipped_label += 1
                continue
            frames = entry.get("num_frames", 0)
            duration = (frames / float(sample_rate)) if frames else ""
            writer.writerow(
                {
                    "audio_path": entry["audio_path"],
                    "json_path": entry.get("json_path", ""),
                    "speaker_id": entry.get("speaker_id", ""),
                    "label_name": label_name,
                    "label_id": label2id[label_name],
                    "duration_sec": duration,
                }
            )
            n_written += 1
    return n_written, n_skipped_label


def prepare_split_csv(manifest_dir, split, out_csv, label2id, alias_map=None, limit=None,
                      sample_rate=DEFAULT_SAMPLE_RATE, audio_root=None, audio_index=None):
    _, entries = read_fairseq_split(manifest_dir, split, audio_root=audio_root,
                                    audio_index=audio_index)
    return write_csv_manifest(entries, label2id, out_csv, alias_map=alias_map,
                              limit=limit, sample_rate=sample_rate)


# --------------------------------------------------------------------------------------
# synthetic audio (used when real AIHub wavs are not present locally)
# --------------------------------------------------------------------------------------
def write_wav(path, audio, sample_rate=DEFAULT_SAMPLE_RATE):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


def generate_synthetic_csv(out_dir, labels, label2id, n_per_split,
                           sample_rate=DEFAULT_SAMPLE_RATE, seconds=4.0, seed=0):
    """Create tiny random-noise WAVs + a CSV manifest. Returns the CSV path.

    Each class gets a slightly different tone so a model *can* in principle learn it;
    this exercises the full data->model->loss->metric path without the real corpus.
    """
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    wav_dir = out_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    n_classes = len(labels)
    for i in range(n_per_split):
        label = labels[i % n_classes]
        dur = seconds * (0.6 + 0.8 * rng.random())  # vary length to exercise padding
        t = np.linspace(0, dur, int(dur * sample_rate), endpoint=False)
        freq = 110.0 * (1 + (label2id[label] % n_classes))
        tone = 0.1 * np.sin(2 * np.pi * freq * t)
        noise = 0.01 * rng.standard_normal(t.shape[0])
        audio = (tone + noise).astype("float32")
        wav_path = wav_dir / f"synth_{i:05d}_{label}.wav"
        write_wav(wav_path, audio, sample_rate)
        rows.append(
            {
                "rel_path": wav_path.name,
                "audio_path": str(wav_path),
                "num_frames": audio.shape[0],
                "label_name": label,
                "speaker_id": f"spk{i % max(1, n_per_split // 3)}",
            }
        )
    csv_path = out_dir / "synthetic_manifest.csv"
    write_csv_manifest(rows, label2id, csv_path, sample_rate=sample_rate)
    return csv_path


def audio_exists_fraction(csv_path, sample=20):
    """Return (n_present, n_checked) by probing the first `sample` audio_paths."""
    csv_path = Path(csv_path)
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    checked = rows[:sample]
    present = sum(1 for r in checked if Path(r["audio_path"]).exists())
    return present, len(checked)


# --------------------------------------------------------------------------------------
# checkpoint save / restore
# --------------------------------------------------------------------------------------
def save_checkpoint(path, model, label2id, id2label, config, extra=None):
    import torch

    payload = {
        "model_state": model.checkpoint_state(),
        "label2id": label2id,
        "id2label": {str(k): v for k, v in id2label.items()},
        "num_classes": len(label2id),
        "config": config,
    }
    if extra:
        payload.update(extra)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def restore_model(ckpt_path, fairseq_root=None, xlsr_ckpt=None, device="cpu"):
    """Rebuild a model from a checkpoint saved by save_checkpoint and load its weights."""
    import torch

    from country_classifier.model import build_model

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = payload["model_state"]
    backend = state.get("backend", "tiny")
    num_classes = payload["num_classes"]

    if backend == "fairseq_xlsr":
        checkpoint_path = xlsr_ckpt or state.get("encoder_checkpoint")
        model = build_model(
            backend="fairseq_xlsr",
            num_classes=num_classes,
            checkpoint_path=checkpoint_path,
            fairseq_root=fairseq_root,
            freeze_encoder=state.get("freeze_encoder", True),
        )
        model.load_classifier(state["classifier"])
    else:
        model = build_model(backend="tiny", num_classes=num_classes)
        model.load_state_dict(state["model"])

    model.to(device)
    model.eval()
    labels = [payload["id2label"][str(i)] for i in range(num_classes)]
    return model, payload, labels


def str2bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}
