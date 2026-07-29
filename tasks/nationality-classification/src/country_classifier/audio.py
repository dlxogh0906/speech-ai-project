"""Small WAV loader used to keep the pipeline independent of torchaudio."""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]


import math
import wave
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly


class AudioLoadError(RuntimeError):
    pass


def read_wav_info(path):
    path = Path(path)
    try:
        with wave.open(str(path), "rb") as wav:
            frames = wav.getnframes()
            sample_rate = wav.getframerate()
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
    except (wave.Error, OSError) as exc:
        raise AudioLoadError(f"{path}: {exc}") from exc
    if sample_rate <= 0 or frames <= 0 or channels <= 0 or sample_width <= 0:
        raise AudioLoadError(f"{path}: invalid WAV header")
    return {
        "num_frames": frames,
        "sample_rate": sample_rate,
        "channels": channels,
        "sample_width": sample_width,
        "duration_sec": frames / float(sample_rate),
    }


def load_wav_mono(path, target_sample_rate=16000):
    path = Path(path)
    try:
        with wave.open(str(path), "rb") as wav:
            sample_rate = wav.getframerate()
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            frames = wav.getnframes()
            raw = wav.readframes(frames)
    except (wave.Error, OSError) as exc:
        raise AudioLoadError(f"{path}: {exc}") from exc

    if frames <= 0:
        raise AudioLoadError(f"{path}: empty audio")
    audio = _pcm_to_float(raw, sample_width)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    audio = audio.astype(np.float32, copy=False)

    if sample_rate != target_sample_rate:
        audio = resample_audio(audio, sample_rate, target_sample_rate)
        sample_rate = target_sample_rate

    return torch.from_numpy(np.ascontiguousarray(audio)), sample_rate


def _pcm_to_float(raw, sample_width):
    if sample_width == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        return (data - 128.0) / 128.0
    if sample_width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32)
        return data / 32768.0
    if sample_width == 3:
        bytes_ = np.frombuffer(raw, dtype=np.uint8)
        if bytes_.size % 3 != 0:
            raise AudioLoadError("malformed 24-bit PCM data")
        triples = bytes_.reshape(-1, 3).astype(np.int32)
        data = triples[:, 0] | (triples[:, 1] << 8) | (triples[:, 2] << 16)
        data = np.where(data & 0x800000, data | ~0xFFFFFF, data)
        return data.astype(np.float32) / 8388608.0
    if sample_width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32)
        return data / 2147483648.0
    raise AudioLoadError(f"unsupported sample width: {sample_width}")


def resample_audio(audio, orig_sample_rate, target_sample_rate):
    if orig_sample_rate <= 0 or target_sample_rate <= 0:
        raise AudioLoadError("sample rates must be positive")
    gcd = math.gcd(int(orig_sample_rate), int(target_sample_rate))
    up = int(target_sample_rate) // gcd
    down = int(orig_sample_rate) // gcd
    return resample_poly(audio, up, down).astype(np.float32)
