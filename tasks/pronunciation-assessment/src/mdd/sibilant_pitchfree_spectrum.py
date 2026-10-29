"""Pitch-free low-band observations from original runtime PCM, not TAG bounds.

40ms windows on the global5ms grid; FFT zero padding gives15.625Hz bins,
not independent15.625Hz resolution. No pitch, formant or phonation label.
"""
import hashlib
import json
import numpy as np

SCHEMA = 'originalPCM_low4k_logpower40ms_hop5ms_fft1024_fp16_v1'
BANDS = (0, 100, 500, 1000, 1500, 2200, 3000, 4001)
WINDOW, HOP, NFFT, BINS = 640, 80, 1024, 257
FREQUENCIES = np.arange(BINS)*16000/NFFT


def observations(source, feature, reference):
    """Return stored-precision spectrum/energy, relative clock and global index.

    Exact constant/nonfinite windows are missing, with no amplitude threshold.
    Per-window DC removal and clip DC-free RMS remove gain/offset information.
    F0 absence is not a failure criterion. No sample outside runtime ROI used.
    """
    left, right = int(feature['left']), int(feature['right'])
    if left < 0 or right > len(source) or right <= left:
        raise ValueError('runtime ROI outside original PCM')
    clip = np.asarray(source[left:right], np.float64)
    first = (left+HOP-1)//HOP
    starts = np.arange(first*HOP, right-WINDOW+1, HOP, dtype=np.int64)
    x = np.full((len(starts), BINS), np.nan, np.float16)
    energy = np.full(len(starts), np.nan, np.float16)
    relative = np.clip(((starts+(WINDOW-1)/2)/16000-reference)/.31, -2, 2).astype(np.float32)
    if not np.isfinite(clip).all() or np.ptp(clip) == 0:
        return x, energy, relative, first
    rms = np.sqrt(np.mean((clip-clip.mean())**2))
    win = np.hanning(WINDOW)
    for j, start in enumerate(starts):
        frame = np.asarray(source[start:start+WINDOW], np.float64)
        if not np.isfinite(frame).all() or np.ptp(frame) == 0:
            continue
        frame = (frame-frame.mean())/rms
        power = abs(np.fft.rfft(frame*win, NFFT))[:BINS]**2/np.sum(win**2)
        x[j] = (10*np.log10(np.maximum(power, 1e-12))).astype(np.float16)
        energy[j] = np.float16(20*np.log10(max(np.sqrt(np.mean(frame**2)), 1e-6)))
    return x, energy, relative, first


def coarse_bands(values):
    """Power-average observed fine bins; retain missing frames and dimensions."""
    x = np.asarray(values, np.float32)
    if x.shape[-1] != BINS:
        raise ValueError('257 low-band bins required')
    result = np.full_like(x, np.nan)
    valid = np.isfinite(x).all(-1)
    for low, high in zip(BANDS[:-1], BANDS[1:]):
        ix = np.flatnonzero((FREQUENCIES >= low) & (FREQUENCIES < high))
        power = np.mean(10**(x[..., ix]/10), axis=-1)
        mean = 10*np.log10(np.maximum(power, 1e-12))
        result[..., ix] = np.where(valid, mean, np.nan)[..., None]
    return result


def normalization_digest(normalization):
    return hashlib.sha256(json.dumps(normalization, sort_keys=True, allow_nan=False,
                                    separators=(',', ':')).encode()).hexdigest()


def normalized_input(values, energy, relative, normalization, mode):
    if (normalization.get('schema') != SCHEMA or normalization.get('partition') != 'train'
            or mode not in ('fine', 'coarse_bands')):
        raise ValueError('typed TRAIN scaler and spectral mode required')
    x = np.asarray(values, np.float32)
    if mode == 'coarse_bands':
        x = coarse_bands(x)
    valid = np.isfinite(x).all(-1) & np.isfinite(energy)
    mean, scale = [np.asarray(normalization[k], np.float32) for k in ('mean', 'scale')]
    z = np.where(valid[..., None], np.clip((x-mean)/scale, -8, 8), 0)
    e = np.where(valid, np.clip((energy-normalization['energy_mean'])/
                              normalization['energy_scale'], -8, 8), 0)
    output = np.concatenate([z, e[..., None], valid[..., None], relative[..., None]], -1)
    return output.astype(np.float32), valid
