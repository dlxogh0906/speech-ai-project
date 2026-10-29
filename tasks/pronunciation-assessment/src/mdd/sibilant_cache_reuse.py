"""Runtime input reuse invariants, with no label/model decisions."""
import numpy as np


def pair_reference_equal(before, after, q):
    if (len(before) != len(after) or not 0 <= q < len(before)
            or before[q] not in ("ㅅ", "ㅆ") or after[q] not in ("ㅅ", "ㅆ")):
        return False
    return all(a == b or (i == q and {a, b} == {"ㅅ", "ㅆ"})
               for i, (a, b) in enumerate(zip(before, after)))


def local_relative(start, end, length, anchor_frame, frame_seconds, scale):
    if length < 3 or end <= start or frame_seconds <= 0 or scale <= 0:
        raise ValueError("valid observed clip and original runtime clock required")
    # Preserve the original operation order, including a clipped left edge.
    centers = start + (np.arange(length) + .5) * (end - start) / length
    return np.clip((centers / 16000. - (anchor_frame + .5) * frame_seconds) / scale, -2, 2).astype(np.float32)
