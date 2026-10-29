# -*- coding: utf-8 -*-
"""오디오 읽기·쓰기. 다른 task 의 로더와 합치지 않는다(요구사항이 다르다 — docs/DECISIONS.md #1).

wav/flac 은 soundfile 로, 그 밖(webm·m4a·mp3 — 브라우저·폰 녹음)은 ffmpeg 로 읽어 16 kHz 모노 float32 로 맞춘다.
"""
import io
import subprocess
import wave

import numpy as np

SR = 16000


def load_audio(path, sr=SR):
    path = str(path)
    try:
        import soundfile as sf
        x, fs = sf.read(path, dtype="float32", always_2d=True)
        x = x.mean(axis=1)
    except Exception:
        from shared.paths import FFMPEG
        cmd = [FFMPEG, "-v", "error", "-nostdin", "-i", path, "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if r.returncode != 0:
            raise RuntimeError("오디오를 읽지 못했습니다 (%s): %s" % (path, r.stderr.decode("utf-8", "replace")[-300:]))
        return np.frombuffer(r.stdout, dtype=np.float32).copy()
    if fs != sr:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(int(fs), sr)
        x = resample_poly(x, sr // g, int(fs) // g)
    return np.ascontiguousarray(x, dtype=np.float32)


def decode_bytes(raw, sr=SR):
    """업로드 바이트(브라우저 webm·mp4, 폰 m4a, wav …) → 16 kHz 모노 float32.

    ffmpeg 은 mp4 를 파이프로 못 읽을 때가 있어 임시 파일을 거친다 — 닫는 즉시 지워진다(녹음을 남기지 않는다)."""
    try:
        import soundfile as sf
        x, fs = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
        x = x.mean(axis=1)
        if fs != sr:
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(int(fs), sr)
            x = resample_poly(x, sr // g, int(fs) // g)
        return np.ascontiguousarray(x, dtype=np.float32)
    except Exception:
        pass
    import tempfile
    from shared.paths import FFMPEG
    with tempfile.NamedTemporaryFile(suffix=".bin") as f:
        f.write(raw)
        f.flush()
        r = subprocess.run([FFMPEG, "-v", "error", "-nostdin", "-i", f.name, "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0 or not r.stdout:
        raise ValueError(r.stderr.decode("utf-8", "replace").strip()[-300:] or "빈 오디오")
    return np.frombuffer(r.stdout, dtype=np.float32).copy()


def wav_bytes(x, sr=SR):
    """float → 16-bit PCM WAV 바이트(보고서에 넣는 용도)."""
    pcm = (np.clip(np.asarray(x, dtype=np.float64), -1.0, 1.0) * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def save_wav(path, x, sr=SR):
    with open(path, "wb") as f:
        f.write(wav_bytes(x, sr))
