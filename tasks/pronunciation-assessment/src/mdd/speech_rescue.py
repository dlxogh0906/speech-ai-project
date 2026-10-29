"""Optional offline rescue of long-padding rejects using independent speech evidence.

The original accepted path is unchanged. This gate never crops or denoises the
waveform used for pronunciation assessment. A speech detector cannot establish
foreground-speaker identity or correct pronunciation.
"""
import numpy as np

from mdd.noise_preprocess import SileroEdges
from mdd.speech_span_quality import assess_span


class SpeechRescue:
    def __init__(self, model_path, threshold=.8):
        self.detector = SileroEdges(model_path, threshold=threshold, padding_ms=150)

    def assess(self, wav, original_code, min_seconds, min_ratio=.30):
        x = np.asarray(wav, dtype=np.float32)
        if x.ndim != 1 or not np.isfinite(x).all():
            raise ValueError("finite mono 16kHz PCM required")
        if original_code is None:
            return {"accepted": True, "rescued": False, "reason": "original_accept"}
        if original_code != "low_speech" or len(x) < 4800:
            return {"accepted": False, "rescued": False, "reason": "ineligible_rejection"}
        span = assess_span(x, 16000, min_seconds, min_ratio)
        if not span["accepted"]:
            return {"accepted": False, "rescued": False, "reason": "insufficient_active_span", "span": span}
        # Discard the detector's cropped waveform. It is not assessment input.
        _, speech = self.detector(x)
        accepted = not speech["no_speech"] and speech["speech_sec"] >= min_seconds
        return {"accepted": bool(accepted), "rescued": bool(accepted),
                "reason": "speech_supported_span" if accepted else "insufficient_detected_speech",
                "span": span, "speech": speech}
