# -*- coding: utf-8 -*-
"""일부러 틀리게 읽은 한국어(한국 → 한쿡 등) TTS 시험셋 — `eval_tense_ft.py --deliberate` 가 읽는 꼴(manifest.json)로 만든다.

목소리 = Zeroth 한국어 **test** 화자 5명(CC BY 4.0, 학습 목소리와 겹치지 않음). 낱말 15개 × (낱말 단독 · "저는 ○○를
좋아해요.") × (맞게 · 일부러 틀리게) = 목소리당 60개. 엔진마다 따로 만든다 — 학습에 쓴 엔진과 시험 엔진이 같으면
효과가 부풀려진다(2026-10-11 리뷰). 2026-10-10 첫 시험셋(`pair_rescue_2026-10-10/tts_deliberate`)은 Chatterbox 판이다.
  --engine chatterbox  tts 환경(/data/speech-ai/envs/tts) · 목소리 참조 = 화자의 첫 녹음
  --engine qwen        qwentts 환경(/data/speech-ai/envs/qwentts) · ICL(참조 녹음 + Zeroth 전사)
외부 API 는 쓰지 않는다.

    HF_HOME=/data/speech-ai/hf_cache HF_HUB_OFFLINE=1 /data/speech-ai/envs/qwentts/bin/python .../make_tts_deliberate.py \\
        --engine qwen --out $MDD_OUTPUTS/analysis/tts_deliberate_qwen
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import DATA_ROOT

import argparse, glob, json, os, re, wave

VOICES = ["104", "105", "112", "121", "132"]
ITEMS = [("한국", "한쿡"), ("가방", "카방"), ("공부", "콩부"), ("다리", "타리"), ("바다", "파다"), ("자동차", "차동차"),
         ("사과", "싸과"), ("고기", "꼬기"), ("두부", "뚜부"), ("바지", "빠지"), ("자두", "짜두"),
         ("커피", "거피"), ("토끼", "도끼"), ("포도", "보도"), ("친구", "진구")]


def particle(w):
    return "을" if (ord(w[-1]) - 0xAC00) % 28 else "를"


def expected(t):
    return max(1, len(re.findall(r"[가-힣]", t))) * 0.3


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", choices=("chatterbox", "qwen"), required=True)
    ap.add_argument("--zeroth", default=os.path.join(str(DATA_ROOT), "zeroth", "wav16k", "test_data_01", "003"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import numpy as np
    import torch
    import torchaudio

    os.makedirs(a.out, exist_ok=True)
    if a.engine == "chatterbox":
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        m = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
    else:
        from qwen_tts import Qwen3TTSModel
        home = os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
        snap = sorted(glob.glob(os.path.join(home, "hub", "models--Qwen--Qwen3-TTS-12Hz-1.7B-Base", "snapshots", "*")))
        m = Qwen3TTSModel.from_pretrained(snap[-1] if snap else "Qwen/Qwen3-TTS-12Hz-1.7B-Base", device_map="cuda:0",
                                          dtype=torch.bfloat16)

    def save(p, x, sr):
        x = torchaudio.functional.resample(torch.as_tensor(np.asarray(x, dtype=np.float32)).reshape(1, -1), sr, 16000)
        with wave.open(p, "wb") as f:
            f.setnchannels(1); f.setsampwidth(2); f.setframerate(16000)
            f.writeframes((x[0].clamp(-1, 1).numpy() * 32767).astype(np.int16).tobytes())

    rows, n = [], 0
    for v in VOICES:
        d = os.path.join(a.zeroth, v)
        ref = sorted(f for f in os.listdir(d) if f.endswith(".wav"))[0]
        if a.engine == "chatterbox":
            m.prepare_conditionals(os.path.join(d, ref), exaggeration=0.5)
        else:
            trans = [f for f in os.listdir(d) if f.endswith(".trans.txt")][0]
            tr = dict(l.strip().split(" ", 1) for l in open(os.path.join(d, trans), encoding="utf-8") if " " in l.strip())
            prompt = m.create_voice_clone_prompt(ref_audio=os.path.join(d, ref), ref_text=tr[ref[:-4]], x_vector_only_mode=False)
            plist = prompt if isinstance(prompt, list) else [prompt]
        for k, (good, bad) in enumerate(ITEMS):
            for form in ("word", "sent"):
                ref_text = good if form == "word" else "저는 %s%s 좋아해요." % (good, particle(good))
                for kind, w in (("correct", good), ("wrong", bad)):
                    spoken = w if form == "word" else "저는 %s%s 좋아해요." % (w, particle(good))
                    p = os.path.join(a.out, "%s_%02d_%s_%s.wav" % (v, k, form, kind))
                    if not os.path.exists(p):
                        e = expected(spoken)
                        for s in range(3):
                            torch.manual_seed(1000 * s + k)
                            if a.engine == "chatterbox":
                                x = m.generate(spoken, language_id="ko", exaggeration=0.5, cfg_weight=0.5)[0].cpu().numpy()
                                sr = m.sr
                            else:
                                ws, sr = m.generate_voice_clone(text=spoken, language="Korean", voice_clone_prompt=plist,
                                                                temperature=0.7, subtalker_temperature=0.7)
                                x = ws[0]
                            if 0.3 * e <= len(x) / sr <= 2.5 * e + 2:
                                break
                        save(p, x, sr)
                        n += 1
                    rows.append({"wav": p, "voice": v, "item": k, "form": form, "kind": kind, "ref_text": ref_text,
                                 "spoken": spoken, "good": good, "bad": bad, "engine": a.engine})
        print(v, n, flush=True)
    json.dump(rows, open(os.path.join(a.out, "manifest.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    print("done", len(rows), flush=True)


if __name__ == "__main__":
    main()
