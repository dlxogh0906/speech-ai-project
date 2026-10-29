"""Analysis-window centers for the pinned 16kHz SeamlessM4T fbank extractor.

400-sample windows, 160-sample hop, center=False, pairs of adjacent80-mel
frames. Only pairs whose second member is observed enter the encoder mask.
Centers are analysis-window support midpoints, not phonetic boundaries.
"""
import numpy as np


def stacked_window_centers(left, right, length):
    left, right, length = int(left), int(right), int(length)
    if left < 0 or right <= left or length < 1:
        raise ValueError("positive observed clip support required")
    raw_frames = max(0, 1 + (right - left - 400) // 160)
    if length != raw_frames // 2:
        raise ValueError("observed mask differs from pinned fbank/stack clock")
    return left + 279.5 + np.arange(length, dtype=np.float64) * 320
