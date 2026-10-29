"""Existing acoustic primitives on their native 2ms/5ms observation grids.

10ms partition balances retain 2ms samples. Other spectra retain 25ms windows
and periodicity retains 40ms windows; nearest 5ms observations are explicitly
associated with 2ms centers. This does not create new temporal resolution in
those longer-window channels or establish frication/resonance/voicing onsets.
"""
import numpy as np
from mdd.sibilant_acoustics import partition_balance_tracks
from mdd.sibilant_transition import acoustic_tracks, ACOUSTIC_NAMES

SCHEMA = 'originalPCM_balance10ms_hop2ms_spectrum25ms_hop5ms_ACF40ms_v1'
NAMES = tuple('upper_lower_db_'+str(k) for k in (2000, 3000, 4000)) + ACOUSTIC_NAMES


def observations(source, feature, reference):
    left, right = int(feature['left']), int(feature['right'])
    start = (left+31)//32*32
    wave = source[start:right]
    balance = partition_balance_tracks(wave, 16000)
    times, spectral = acoustic_tracks(wave, 16000)
    target = balance['times']
    x = np.full((len(target), len(NAMES)), np.nan, np.float32)
    x[:, :3] = np.stack([balance[name] for name in NAMES[:3]], -1)
    ix = np.searchsorted(times, target)
    a, b = np.maximum(ix-1, 0), np.minimum(ix, len(times)-1)
    near = np.where(abs(times[a]-target) <= abs(times[b]-target), a, b)
    supported = (target >= times[0]) & (target <= times[-1])
    x[supported, 3:] = spectral[near[supported]]
    relative = np.clip(((start+target*16000)/16000-reference)/.31, -2, 2).astype(np.float32)
    return x, relative, start//32


def coarsen_20ms(values, first_frame):
    """Average observed 2ms values within global20ms bins, keep missing missing.

    The comparison retains original per-channel validity and all fine-grid
    coordinates. It removes within-bin variation without filling failed
    observations, appending another syllable, or changing model capacity.
    """
    x = np.asarray(values)
    if x.ndim != 2:
        raise ValueError('frame/channel observations required')
    output = np.full_like(x, np.nan)
    groups = (first_frame+np.arange(len(x)))//10
    for group in np.unique(groups):
        ix = np.flatnonzero(groups == group)
        z = x[ix]
        valid = np.isfinite(z)
        mean = np.where(valid, z, 0).sum(0)/np.maximum(valid.sum(0), 1)
        output[ix] = np.where(valid, mean[None], np.nan)
    return output


def normalization_digest(normalization):
    import hashlib
    h = hashlib.sha256()
    for key in ('mean', 'scale'):
        a = np.asarray(normalization[key], np.float32)
        h.update(key.encode())
        h.update(a.tobytes())
    return h.hexdigest()


def normalized_input(values, normalization):
    valid = np.isfinite(values)
    mean, scale = [np.asarray(normalization[k], np.float32) for k in ('mean', 'scale')]
    z = np.where(valid, np.clip((values-mean)/scale, -8, 8), 0).astype(np.float32)
    return np.concatenate([z, valid.astype(np.float32)], -1)
