"""Retain runtime CTC anchoring but observe original, continuous audio.

The VAD sample map is preprocessing provenance, not a phonetic boundary.
No removed samples are synthesized or taken from an unrelated syllable.
"""
import numpy as np


def cached_anchor(left, right, length, relative, vad_samples):
    frames = (1 + (vad_samples - 400) // 160) // 2
    if frames < 1 or length < 3 or right <= left:
        raise ValueError('unsupported pinned feature clock')
    dt = vad_samples / (16000. * frames)
    center0 = left + .5 * (right-left) / length
    frame = int(round((center0/16000. - float(relative[0])*.31)/dt - .5))
    if not 0 <= frame < frames:
        raise ValueError('cached relative position does not encode a valid anchor')
    expected = (max(0, int((frame-15)*dt*16000)), min(vad_samples, int((frame+16)*dt*16000)))
    if expected != (left, right):
        raise ValueError('cached bounds do not reproduce on pinned CTC clock')
    centers = left + (np.arange(length)+.5) * (right-left) / length
    rel = np.clip((centers/16000. - (frame+.5)*dt)/.31, -2, 2).astype(np.float32)
    np.testing.assert_array_equal(rel, relative[:length])
    return frame, dt


def source_bounds(sample_map, frame, dt, source_length):
    indices = np.asarray(sample_map)
    if indices.ndim != 1 or not len(indices) or np.any(np.diff(indices) <= 0):
        raise ValueError('strictly increasing retained source sample map required')
    center = min(len(indices)-1, max(0, int((frame+.5)*dt*16000)))
    shift = int(indices[center])-center
    left = max(0, int((frame-15)*dt*16000)+shift)
    right = min(source_length, int((frame+16)*dt*16000)+shift)
    if not 0 <= left < right <= source_length:
        raise ValueError('empty source observation')
    return left, right, shift


def source_relative(left, right, length, shift, frame, dt):
    # Subtract the integer coordinate shift before fractional arithmetic. When
    # the original and retained clips coincide, this reproduces cached bits.
    centers = (left-shift) + (np.arange(length)+.5) * (right-left) / length
    return np.clip((centers/16000. - (frame+.5)*dt)/.31, -2, 2).astype(np.float32)
