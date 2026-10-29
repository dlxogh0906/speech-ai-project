# -*- coding: utf-8 -*-
"""세기 쌍 짝 문장(`build_synth_tense_pairs.py` 의 texts_qwen.jsonl)을 Qwen3-TTS-12Hz-1.7B-Base(Apache-2.0, 로컬 GPU)로
읽혀 16 kHz WAV 로 저장한다. 외부 API 는 쓰지 않는다. qwentts 환경(파이썬 3.11, qwen-tts)에서 돈다 — 학습 환경($PY)이 아니다.

- 목소리 = 문장마다 적힌 Zeroth 참조 녹음. 목소리 절반은 ICL(참조 녹음 + 전사 — 말투까지 따라 한다), 절반은
  x-vector(음색만) — 한 가지 조건에 쏠리지 않게.
- 짝(…c/…w)은 한 배치에 같이 넣는다(같은 목소리 · 같은 seed 로 시작).
- 길이가 비정상(예상 = 한글 음절 × 0.3초, 0.3배 미만 · 2.5배+2초 초과)이면 seed 를 바꿔 하나씩 최대 3번 다시 만든다.
- 이미 있는 파일은 건너뛴다(이어 돌리기). GPU 1장에 조각(--shard i/n) 여러 개를 같이 돌려도 된다(조각당 약 8 GB).

    HF_HOME=/data/speech-ai/hf_cache HF_HUB_OFFLINE=1 /data/speech-ai/envs/qwentts/bin/python .../synth_tense_tts_qwen.py \\
        --texts $MDD_OUTPUTS/synth_tense_v2/texts_qwen.jsonl --out $MDD_OUTPUTS/synth_tense_v2/wav --shard 0/2
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import argparse, collections, glob, json, os, re, time, wave

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"


def expected(t):
    return max(1, len(re.findall(r"[가-힣]", t))) * 0.3


def local_snapshot(name):
    """HF 캐시의 스냅샷 경로 — 오프라인(HF_HUB_OFFLINE=1)에서 저장소 이름으로 부르면 qwen_tts 가 API 를 불러 죽는다."""
    home = os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
    snaps = sorted(glob.glob(os.path.join(home, "hub", "models--" + name.replace("/", "--"), "snapshots", "*")))
    return snaps[-1] if snaps else name


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--texts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--batch", type=int, default=8, help="한 번에 읽힐 문장 수(짝 단위로 채운다)")
    ap.add_argument("--temperature", type=float, default=0.7)
    a = ap.parse_args()
    import numpy as np
    import torch
    import torchaudio
    from qwen_tts import Qwen3TTSModel

    si, sn = (int(x) for x in a.shard.split("/"))
    rows = [json.loads(l) for l in open(a.texts, encoding="utf-8")]
    voices = sorted(set(r["voice"] for r in rows))
    mine = set(voices[si::sn])                                   # 목소리 단위로 조각을 나눈다(참조 조건을 한 번만 만들게)
    rows = [r for r in rows if r["voice"] in mine]
    os.makedirs(a.out, exist_ok=True)
    has = lambda r: os.path.exists(os.path.join(a.out, r["sid"] + ".wav"))
    by_voice = collections.OrderedDict()
    for r in sorted(rows, key=lambda r: (r["voice"], r["sid"])):
        by_voice.setdefault(r["voice"], []).append(r)
    todo_n = sum(1 for r in rows if not has(r))
    print("[qwen] 조각 %d/%d · 목소리 %d · 문장 %d · 남은 것 %d" % (si, sn, len(by_voice), len(rows), todo_n), flush=True)
    m = Qwen3TTSModel.from_pretrained(local_snapshot(MODEL), device_map="cuda:0", dtype=torch.bfloat16)
    t0, done, bad = time.time(), 0, 0

    def save(r, w, sr):
        x = torch.from_numpy(np.asarray(w, dtype=np.float32)).unsqueeze(0)
        x = torchaudio.functional.resample(x, sr, 16000).clamp(-1, 1)
        tmp = os.path.join(a.out, r["sid"] + ".tmp.wav")
        with wave.open(tmp, "wb") as f:
            f.setnchannels(1); f.setsampwidth(2); f.setframerate(16000)
            f.writeframes((x[0].numpy() * 32767).astype(np.int16).tobytes())
        os.replace(tmp, os.path.join(a.out, r["sid"] + ".wav"))

    for vi, (voice, vrows) in enumerate(by_voice.items()):
        todo = [r for r in vrows if not has(r)]
        if not todo:
            continue
        xvec = (voices.index(voice) % 2) == 1                    # 목소리 절반은 x-vector
        prompt = m.create_voice_clone_prompt(ref_audio=voice, ref_text=None if xvec else vrows[0]["voice_text"],
                                             x_vector_only_mode=xvec)
        plist = prompt if isinstance(prompt, list) else [prompt]
        for b in range(0, len(todo), a.batch):
            chunk = todo[b:b + a.batch]
            torch.manual_seed(int(re.sub(r"\D", "", chunk[0]["sid"])))
            wavs, sr = m.generate_voice_clone(text=[r["text_spoken"] for r in chunk], language=["Korean"] * len(chunk),
                                              voice_clone_prompt=plist * len(chunk), temperature=a.temperature,
                                              subtalker_temperature=a.temperature)
            for r, w in zip(chunk, wavs):
                e, k = expected(r["text_spoken"]), 0
                while not (0.3 * e <= len(w) / sr <= 2.5 * e + 2) and k < 3:
                    k += 1
                    torch.manual_seed(int(re.sub(r"\D", "", r["sid"])) + 1000 * k)
                    w = m.generate_voice_clone(text=r["text_spoken"], language="Korean", voice_clone_prompt=plist,
                                               temperature=a.temperature, subtalker_temperature=a.temperature)[0][0]
                if not (0.3 * e <= len(w) / sr <= 2.5 * e + 2):
                    bad += 1
                save(r, w, sr)
                done += 1
        if vi % 5 == 0:
            print("  목소리 %d/%d · %d/%d · %.0fs · 길이 비정상 %d" % (vi + 1, len(by_voice), done, todo_n, time.time() - t0, bad),
                  flush=True)
    print("[완료] 조각 %d/%d · %d문장 · %.0fs · 길이 비정상(3번 다 실패) %d" % (si, sn, done, time.time() - t0, bad), flush=True)


if __name__ == "__main__":
    main()
