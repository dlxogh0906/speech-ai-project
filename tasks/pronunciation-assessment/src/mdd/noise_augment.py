"""Length-preserving waveform augmentation for actual-phone CTC targets.

Noise is mixed in memory; neither source audio nor labels are overwritten.
SNR uses whole-utterance RMS, matching the previous noise experiments.
"""

import random
import wave
from collections import OrderedDict

import numpy as np

SR = 16000


def read_pcm(path):
    with wave.open(str(path), "rb") as stream:
        if (stream.getframerate(), stream.getnchannels(), stream.getsampwidth()) != (SR, 1, 2):
            raise ValueError("Expected mono PCM16 at 16 kHz")
        return np.frombuffer(stream.readframes(stream.getnframes()), np.int16).astype(np.float32) / 32768.0


def mix_at_snr(speech, noise, snr):
    if speech.shape != noise.shape:
        raise ValueError("Noise/speech shape mismatch")
    gain = np.sqrt((np.mean(speech ** 2) + 1e-10) /
                   ((np.mean(noise ** 2) + 1e-10) * 10 ** (snr / 10.0)))
    result = speech + gain * noise
    return (result / max(1.0, float(np.max(np.abs(result))) / 0.99)).astype(np.float32)


def rumble(seed, n):
    from scipy.signal import lfilter
    rng = np.random.RandomState(seed % (2 ** 32))
    a = np.exp(-2 * np.pi * 300 / SR)
    filtered = lfilter([1 - a], [1, -a], rng.randn(n + 4000))[4000:]
    return (filtered / (filtered.std() + 1e-9) + 0.1 * rng.randn(n)).astype(np.float32)


def reverberate(speech, seed, rt60):
    """Synthetic decaying impulse response with an undelayed direct path.

    This is a reverberation stress proxy, not a measured booth response.
    Keep duration/phone alignment; crop only the reverberation tail.
    """
    from scipy.signal import fftconvolve
    rng = np.random.RandomState(seed % (2 ** 32))
    n = int(SR * rt60)
    response = rng.randn(n) * np.exp(-np.log(1000) * np.arange(n) / n)
    response[:int(0.008 * SR)] = 0
    response *= 0.25 / (np.linalg.norm(response) + 1e-9)
    response[0] = 1.0
    result = fftconvolve(speech, response)[:len(speech)]
    result /= max(1.0, float(np.max(np.abs(result))) / 0.99)
    return result.astype(np.float32)


class NoiseAugmenter:
    def __init__(self, pool, probability=0.5, cache_size=64):
        if not pool or not 0 <= probability <= 1:
            raise ValueError("Invalid noise pool/probability")
        self.pool = list(pool)
        self.probability = probability
        self.cache_size = cache_size
        self.cache = OrderedDict()

    def voice(self, path):
        if path not in self.cache:
            self.cache[path] = read_pcm(path)
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        self.cache.move_to_end(path)
        return self.cache[path]

    def babble(self, rng, n):
        noise = np.zeros(n, np.float32)
        for _ in range(6):
            voice = self.voice(rng.choice(self.pool))
            pieces = [voice]
            while sum(map(len, pieces)) < n:
                pieces.append(self.voice(rng.choice(self.pool)))
            voice = np.concatenate(pieces) if len(pieces) > 1 else voice
            start = rng.randrange(len(voice) - n + 1)
            noise += voice[start:start + n] * rng.uniform(0.6, 1.0)
        return noise

    def __call__(self, speech, seed):
        rng = random.Random(seed)
        if rng.random() >= self.probability:
            return speech, "clean"
        kind = rng.random()
        if kind < 0.4:
            result = mix_at_snr(speech, self.babble(rng, len(speech)), rng.uniform(5, 20))
            condition = "babble"
        elif kind < 0.7:
            result = mix_at_snr(speech, rumble(seed + 17, len(speech)), rng.uniform(0, 15))
            condition = "rumble"
        else:
            result = reverberate(speech, seed + 29, rng.uniform(0.15, 0.5))
            condition = "reverb"
            if kind < 0.9:
                result = mix_at_snr(result, self.babble(rng, len(speech)), rng.uniform(5, 20))
                condition = "reverb_babble"
        if result.shape != speech.shape or not np.isfinite(result).all():
            raise ValueError("Invalid augmented waveform")
        return result, condition
