# -*- coding: utf-8 -*-
"""녹음 받기 — 바이트 → 16 kHz 파형, 말소리 확인, 앞뒤 무음 맞추기. 요청 파라미터 읽기(SpeechSuper 방식).

앞뒤 무음은 **0.5초만 남기고** 자른다. 점수기를 학습한 speechocean762 녹음이 앞뒤로 0.5~0.6초씩
무음을 갖고 있어서(중앙값 0.58·0.64초), 앱 녹음의 긴 무음이 문장 특징(앞뒤 무음·말한 시간)을 학습 분포
밖으로 밀지 않게 한다. 문장 **중간**의 쉼은 유창성 판단에 쓰이므로 건드리지 않는다.
"""

import io
import json
import shutil
import subprocess
import wave

import numpy as np

SR = 16000
FRAME_SEC = 0.03
REL_THRESH = 0.06          # 프레임 RMS ≥ 최대 RMS × 이 값이면 말소리
ABS_FLOOR = 8e-4
EDGE_KEEP_SEC = 0.5


def decode_audio(raw, ffmpeg="ffmpeg"):
    """녹음 바이트 → 16k mono float32. 16k mono 16bit WAV 는 그대로, 나머지는 ffmpeg(표준입력)으로."""
    if raw[:4] == b"RIFF":
        try:
            with wave.open(io.BytesIO(raw), "rb") as w:
                if w.getframerate() == SR and w.getnchannels() == 1 and w.getsampwidth() == 2:
                    data = w.readframes(w.getnframes())
                    if data:                                 # 헤더 길이가 0 으로 남은 WAV 는 ffmpeg 로 넘긴다
                        return np.frombuffer(data, np.int16).astype(np.float32) / 32768.0
        except Exception:
            pass
    exe = shutil.which(ffmpeg) or ffmpeg
    p = subprocess.run([exe, "-nostdin", "-hide_banner", "-loglevel", "error", "-threads", "1", "-i", "pipe:0",
                        "-f", "f32le", "-ac", "1", "-ar", str(SR), "pipe:1"],
                       input=raw, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0 or not p.stdout:
        raise ValueError(p.stderr.decode("utf-8", "replace").strip()[-300:] or "빈 오디오")
    return np.nan_to_num(np.frombuffer(p.stdout, np.float32))


def voiced_mask(wav):
    f = int(SR * FRAME_SEC)
    n = len(wav) // f
    if n == 0:
        return np.zeros(0, bool)
    r = np.sqrt((wav[: n * f].reshape(n, f).astype(np.float64) ** 2).mean(1))
    return r >= max(float(r.max()) * REL_THRESH, ABS_FLOOR)


def voiced_info(wav):
    m = voiced_mask(wav)
    return {"voiced_ratio": round(float(m.mean()), 3) if m.size else 0.0,
            "voiced_sec": round(float(m.sum()) * FRAME_SEC, 2)}


def trim_edges_offset(wav, keep=EDGE_KEEP_SEC):
    """앞뒤 무음을 keep 초만 남기고 자른다(원래 여유가 짧으면 무음을 채운다). 말소리가 없으면 그대로.

    → (잘린 파형, 잘린 파형 0번 표본의 **원래 녹음 기준 시각(초)**). 무음을 채웠으면 음수일 수 있다 —
    응답의 span 을 원래 녹음 기준으로 돌려줄 때 쓴다.
    """
    m = voiced_mask(wav)
    if not m.any():
        return wav, 0.0
    idx = np.flatnonzero(m)
    f = int(SR * FRAME_SEC)
    lo = max(0, idx[0] * f - int(keep * SR))
    hi = min(len(wav), (idx[-1] + 1) * f + int(keep * SR))
    out = wav[lo:hi]
    pad_lo = int(keep * SR) - (idx[0] * f - lo)
    pad_hi = int(keep * SR) - (hi - (idx[-1] + 1) * f)
    if pad_lo > 0 or pad_hi > 0:
        out = np.concatenate([np.zeros(max(pad_lo, 0), np.float32), out, np.zeros(max(pad_hi, 0), np.float32)])
    return out, (lo - max(pad_lo, 0)) / float(SR)


def trim_edges(wav, keep=EDGE_KEEP_SEC):
    """`trim_edges_offset` 의 파형만."""
    return trim_edges_offset(wav, keep)[0]


def parse_request_params(form):
    """SpeechSuper 요청에서 refText·tokenId·userId 를 꺼낸다(한국어 API 와 같은 규칙).

    multipart 안의 connect/start JSON(필드 이름 무관 — 값이 JSON 이고 start/request 를 품고 있으면) 또는
    간편 방식 필드(refText·tokenId·userId).
    """
    out = {"refText": "", "tokenId": "", "userId": "guest", "coreType": ""}
    d = lambda x: x if isinstance(x, dict) else {}           # 모양이 틀린 JSON 이 500 을 내지 않게
    txt = lambda x: x if isinstance(x, str) else ("" if x is None else str(x))
    for key, val in form.items():
        if not isinstance(val, str) or "{" not in val:
            continue
        try:
            j = json.loads(val)
        except Exception:
            continue
        if not isinstance(j, dict):
            continue
        sp = d(d(j.get("start")).get("param"))
        req = d(sp.get("request")) or d(d(j.get("param")).get("request")) or d(j.get("request"))
        app_ = d(sp.get("app"))
        if req or app_:
            out["refText"] = txt(req.get("refText")) or out["refText"]
            out["tokenId"] = txt(req.get("tokenId")) or out["tokenId"]
            out["coreType"] = txt(req.get("coreType")) or out["coreType"]
            out["userId"] = txt(app_.get("userId")) or out["userId"]
            break
    for k in ("refText", "tokenId", "userId"):
        v = form.get(k)
        if isinstance(v, str) and v and not out[k]:
            out[k] = v
    return out


def save_take(out_dir, wav, ref_text, extra=None):
    """녹음을 16k WAV + 메타로 남긴다(SpeechSuper 비교·한국인 재보정용). items.jsonl 에 한 줄을 더해
    `scripts/eval/compare_speechsuper_en.py --items` 로 바로 쓸 수 있게 한다. → stem"""
    import os
    import time
    os.makedirs(out_dir, exist_ok=True)
    stem = time.strftime("%Y%m%d-%H%M%S") + "-%03d" % (int(time.time() * 1000) % 1000)
    path = os.path.join(out_dir, stem + ".wav")
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(wav, -1, 1) * 32767).astype(np.int16).tobytes())
    rec = {"id": stem, "wav": path, "text": ref_text, "sec": round(len(wav) / SR, 2)}
    rec.update(extra or {})
    with open(os.path.join(out_dir, "items.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return stem
