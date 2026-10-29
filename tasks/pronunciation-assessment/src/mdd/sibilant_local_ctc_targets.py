"""TRAIN output-only CTC supervision on unchanged runtime acoustic windows."""
import numpy as np
from mdd.sibilant_frame_clock import stacked_window_centers


def target_region(feature, tag, actual, vocab, fraction=.5, edge=.01):
    tag = [(a, b, ''.join(c for c in token if not c.isspace())) for a, b, token in tag]
    flat = [c for _, _, token in tag if token not in ('', '.', '-', 'spn', '<unk>') for c in token]
    if flat != list(actual):
        return None, 'TAG_sequence_mismatch'
    if any(t in ('.', '-', 'spn', '<unk>') or any(c not in vocab for c in t) for _, _, t in tag):
        return None, 'unresolved_TAG_symbol'
    if (any(not np.isfinite([a, b]).all() or not a < b for a, b, _ in tag)
            or any(a[1] > b[0] + 1e-6 for a, b in zip(tag, tag[1:]))):
        return None, 'invalid_TAG_time_order'
    left, right, length = feature['left'], feature['right'], feature['input_frames']
    lo, hi = left / 16000., right / 16000.
    centers = stacked_window_centers(left, right, length) / 16000.
    selected = []
    for i, (a, b, token) in enumerate(tag):
        overlap = max(0., min(b, hi) - max(a, lo)) / (b - a)
        # A short internal phone need not own a 20ms output-frame center.
        # CTC supervises the known sequence, not a fabricated per-phone grid.
        # A partial grouped interval does not reveal which member is observed.
        # Keep all members only when the whole shared interval is in the clip.
        required = 1. - 1e-6 if len(token) > 1 else fraction
        if token and overlap >= required:
            selected.append(i)
    if not selected:
        return None, 'no_supported_phone'
    first, last = selected[0], selected[-1]
    if any(tag[i][2] and i not in selected for i in range(first, last + 1)):
        return None, 'unsupported_internal_phone'
    a, b = tag[first][0], tag[last][1]
    # Adjacent empty annotation intervals carry no output label. Include them
    # in the sequence-loss region; never reinterpret them as physical silence.
    if first and not tag[first - 1][2] and abs(tag[first - 1][1] - a) < 1e-6:
        a = tag[first - 1][0]
    if last + 1 < len(tag) and not tag[last + 1][2] and abs(tag[last + 1][0] - b) < 1e-6:
        b = tag[last + 1][1]
    supported = np.flatnonzero((centers >= max(lo, a) + edge) & (centers < min(hi, b) - edge))
    if not len(supported):
        return None, 'no_supported_output_frame'
    tokens = [c for i in selected for c in tag[i][2]]
    minimum = len(tokens) + sum(a == b for a, b in zip(tokens, tokens[1:]))
    if len(supported) < minimum:
        return None, 'CTC_path_impossible'
    return {'roi': [int(supported[0]), int(supported[-1]) + 1], 'tokens': tokens,
            'ids': [vocab[c] for c in tokens], 'TAG_indices': selected,
            'source_loss_interval': [max(lo, a), min(hi, b)]}, 'supported'
