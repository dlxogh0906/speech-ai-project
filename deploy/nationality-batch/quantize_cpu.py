"""Prototype: int8 dynamic quantization of XLS-R 2B on CPU, measure speed + accuracy.
Answers: 'how many seconds per clip on CPU?' and 'does quantization keep accuracy?'
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
if "--help" in _sys.argv or "-h" in _sys.argv:   # 인자 파서가 없는 스크립트다. --help 로 본체가 실행되면 안 된다.
    print(__doc__ or "설명이 없다 — 소스를 열어볼 것"); raise SystemExit(0)
from shared.fairseq_guard import ensure_fairseq
ensure_fairseq()
from shared.paths import DATA_ROOT, FAIRSEQ_ROOT, MODEL_ROOT, NAT_OUTPUTS

import sys, time, subprocess
import numpy as np
import torch, torch.nn as nn
from country_classifier.model import build_model

CKPT = f"{MODEL_ROOT}/xlsr2_2B_1000k.pt"
HEAD_PT = f"{NAT_OUTPUTS}/0702_country41_sent_best/head.pt"
LAYER, SR = 31, 16000
torch.set_num_threads(torch.get_num_threads())
print(f"CPU threads: {torch.get_num_threads()}", flush=True)


def decode(path):
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    p = subprocess.run([ff, "-nostdin", "-i", path, "-f", "f32le", "-ac", "1", "-ar", str(SR), "pipe:1"],
                       capture_output=True)
    return np.frombuffer(p.stdout, dtype=np.float32).copy()


print("[load] XLS-R 2B on CPU ...", flush=True); t0 = time.time()
model = build_model(backend="fairseq_xlsr", num_classes=61, checkpoint_path=CKPT,
                    fairseq_root=str(FAIRSEQ_ROOT), freeze_encoder=True).eval()
ck = torch.load(HEAD_PT, map_location="cpu", weights_only=False)
head = nn.Linear(1920, len(ck["labels"])); head.load_state_dict(ck["head"]); head.eval()
mu = torch.as_tensor(ck["mu"]).float(); sd = torch.as_tensor(ck["sd"]).float(); labels = ck["labels"]
print(f"[load] {time.time()-t0:.0f}s", flush=True)


@torch.no_grad()
def predict(enc, wave, max_sec=12):
    x = torch.from_numpy(np.ascontiguousarray(wave[: max_sec * SR])).unsqueeze(0)
    pm = torch.zeros_like(x, dtype=torch.bool)
    res = enc.encoder.extract_features(source=x, padding_mask=pm, mask=False)
    feat = res["layer_results"][LAYER][0].transpose(0, 1).mean(1).float().clamp(-50, 50)
    probs = torch.softmax(head((feat - mu) / sd), -1)[0]
    i = int(probs.argmax())
    return labels[i], float(probs[i]), probs


def bench(enc, wave, k=5):
    _ = predict(enc, wave)  # warmup
    t = time.time()
    for _ in range(k):
        predict(enc, wave)
    return (time.time() - t) / k


# real clips
base = f"{DATA_ROOT}/aihub/foreign_korean_speech/131.인공지능_학습을_위한_외국인_한국어_발화_음성_데이터/01.데이터_new_20220719/1.Training/원천데이터"
clips = {"Vietnam": f"{base}/TS_1. 베트남어/1. 한국일반/VN10QC226_VN0012_20210802.wav",
         "English": f"{base}/TS_2. 영어/1. 한국일반/EN10QC226_EN0144_20211026.wav",
         "China":  f"{base}_230113_add/TS_4. 중국어_add/1. 한국일반/CN10QC226_CN0003_20210730.wav"}
import os
waves = {k: decode(v) for k, v in clips.items() if os.path.exists(v)}
one = list(waves.values())[0]

print("\n[fp32] benchmarking CPU ...", flush=True)
t_fp32 = bench(model, one, k=5)
pred_fp32 = {k: predict(model, w)[:2] for k, w in waves.items()}

print("[int8] applying dynamic quantization (Linear -> int8) ...", flush=True)
tq = time.time()
model_q = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
print(f"       quantized in {time.time()-tq:.0f}s", flush=True)
t_int8 = bench(model_q, one, k=5)
pred_int8 = {k: predict(model_q, w)[:2] for k, w in waves.items()}

print("\n================ RESULT (CPU, per clip) ================")
print(f"fp32  : {t_fp32:5.2f} s/clip")
print(f"int8  : {t_int8:5.2f} s/clip   ({t_fp32/t_int8:.2f}x faster)")
print("\n정확도 비교 (fp32 -> int8):")
print(f"{'clip':9}{'fp32 top1':16}{'int8 top1':16}{'same':5}")
for k in waves:
    a, pa = pred_fp32[k]; b, pb = pred_int8[k]
    print(f"{k:9}{a+f' ({pa:.2f})':16}{b+f' ({pb:.2f})':16}{'O' if a==b else 'X'}")
