"""Optional CPU preprocessors for isolated pronunciation experiments.

No production defaults are changed. Callers supply the ONNX model or native
RNNoise library. Model sessions are reused, but utterance state is reset.
"""

import ctypes
import time

import numpy as np


class SileroEdges:
    """Detect speech and trim its outer boundaries; keep all internal pauses."""

    def __init__(self, model_path, threshold=0.5, padding_ms=150):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_path), sess_options=options,
                                           providers=["CPUExecutionProvider"])
        self.threshold = threshold
        self.padding = int(padding_ms * 16)

    def __call__(self, wav):
        started = time.perf_counter()
        n = len(wav)
        state = np.zeros((2, 1, 128), dtype=np.float32)
        context = np.zeros((1, 64), dtype=np.float32)
        probabilities = []
        for start in range(0, n, 512):
            frame = np.zeros((1, 512), dtype=np.float32)
            part = wav[start:start + 512]
            frame[0, :len(part)] = part
            probability, state = self.session.run(None, {
                "input": np.concatenate((context, frame), axis=1),
                "state": state, "sr": np.array(16000, dtype=np.int64)})
            probabilities.append(float(probability[0, 0]))
            context = frame[:, -64:]

        spans = []
        speech_start = None
        tentative_end = None
        for i, probability in enumerate(probabilities):
            start = i * 512
            if probability >= self.threshold:
                if speech_start is None:
                    speech_start = start
                tentative_end = None
            elif probability < max(0.01, self.threshold - 0.15) and speech_start is not None:
                if tentative_end is None:
                    tentative_end = start
                if start + 512 - tentative_end >= 2400:  # 150 ms silence
                    if tentative_end - speech_start >= 4000:  # 250 ms speech
                        spans.append((speech_start, tentative_end))
                    speech_start = tentative_end = None
        if speech_start is not None:
            end = min(n, tentative_end if tentative_end is not None else n)
            if end - speech_start >= 4000:
                spans.append((speech_start, end))

        speech_samples = sum(end - start for start, end in spans)
        if spans:
            lo = max(0, spans[0][0] - self.padding)
            hi = min(n, spans[-1][1] + self.padding)
            result = wav[lo:hi].copy()
        else:
            # Keep input for unconditional research scoring. A production gate
            # would abstain instead; no-speech is reported separately below.
            lo, hi, result = 0, n, wav.copy()
        info = {"speech_sec": speech_samples / 16000.0,
                "speech_ratio": speech_samples / max(1, n),
                "segments": len(spans), "no_speech": not bool(spans),
                "left_cut_sec": lo / 16000.0, "right_cut_sec": (n - hi) / 16000.0,
                "preprocess_ms": (time.perf_counter() - started) * 1000}
        return result.astype(np.float32), info


class RNNoise16k:
    """16k -> 48k -> RNNoise -> 16k, with fixed library-delay compensation."""

    def __init__(self, library_path, delay_frames=2):
        self.library = ctypes.CDLL(str(library_path))
        self.library.rnnoise_create.argtypes = [ctypes.c_void_p]
        self.library.rnnoise_create.restype = ctypes.c_void_p
        self.library.rnnoise_destroy.argtypes = [ctypes.c_void_p]
        frame_query = getattr(self.library, "rnnoise_get_frame_size", None)
        if frame_query is not None:
            frame_query.restype = ctypes.c_int
        self.library.rnnoise_process_frame.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float)]
        self.library.rnnoise_process_frame.restype = ctypes.c_float
        self.frame_size = frame_query() if frame_query is not None else 480
        if self.frame_size != 480:
            raise ValueError("Expected the RNNoise 48k/480-sample interface")
        self.delay_samples = self.frame_size * delay_frames

    def __call__(self, wav):
        from scipy.signal import resample_poly

        started = time.perf_counter()
        upsampled = resample_poly(wav, 3, 1).astype(np.float32)
        wanted = len(upsampled)
        total = ((wanted + self.delay_samples + self.frame_size - 1)
                 // self.frame_size) * self.frame_size
        source = np.zeros(total, dtype=np.float32)
        source[:wanted] = upsampled * 32768.0
        output = np.empty(total, dtype=np.float32)
        state = self.library.rnnoise_create(None)
        if not state:
            raise RuntimeError("RNNoise state creation failed")
        probabilities = []
        try:
            for start in range(0, total, self.frame_size):
                a = source[start:start + self.frame_size]
                b = output[start:start + self.frame_size]
                probabilities.append(float(self.library.rnnoise_process_frame(
                    state, b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))))
        finally:
            self.library.rnnoise_destroy(state)
        output = output[self.delay_samples:self.delay_samples + wanted] / 32768.0
        result = resample_poly(output, 1, 3).astype(np.float32)[:len(wav)]
        if len(result) != len(wav) or not np.isfinite(result).all():
            raise RuntimeError("RNNoise output length/finite check failed")
        result = np.clip(result, -1.0, 1.0)
        info = {"preprocess_ms": (time.perf_counter() - started) * 1000,
                "delay_compensation_ms": self.delay_samples / 48.0,
                "rnnoise_speech_probability": float(np.mean(probabilities))}
        return result, info
