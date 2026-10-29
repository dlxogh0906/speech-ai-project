# -*- coding: utf-8 -*-
"""세기 쌍 오류 문장(`build_synth_tense_texts.py`)을 Chatterbox Multilingual(MIT, 로컬 GPU)로 읽혀 16 kHz WAV 로 저장한다.
외부 API 는 쓰지 않는다. kokoro/tts 환경(파이썬 3.11, chatterbox-tts 0.1.7)에서 돈다 — 학습 환경($PY)이 아니다.

- 목소리는 문장마다 적힌 참조 녹음(Zeroth train 화자). 같은 목소리끼리 모아 참조 조건을 한 번만 만든다.
- 길이가 비정상(예상 = 한글 음절 × 0.3초, 0.3배 미만 · 2.5배+2초 초과)이면 seed 를 바꿔 최대 3번 다시 만든다.
- 이미 있는 파일은 건너뛴다(이어 돌리기). GPU 1장에 조각(--shard i/n) 여러 개를 같이 돌려도 된다.
- 2차 짝 목록(`build_synth_tense_pairs.py` 의 texts_chatterbox.jsonl)도 그대로 받는다 — 짝(…c/…w)은 같은 seed 로 읽는다.

    HF_HOME=/data/speech-ai/hf_cache HF_HUB_OFFLINE=1 /data/speech-ai/envs/tts/bin/python .../synth_tense_tts.py \\
        --texts $MDD_OUTPUTS/synth_tense_v1/texts.jsonl --out $MDD_OUTPUTS/synth_tense_v1/wav --shard 0/6
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import argparse, json, os, re, time


def expected(t):
    return max(1, len(re.findall(r"[가-힣]", t))) * 0.3


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--texts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--cfg", type=float, default=0.5)
    ap.add_argument("--exag", type=float, default=0.5)
    a = ap.parse_args()
    import torch, torchaudio
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    si, sn = (int(x) for x in a.shard.split("/"))
    rows = [json.loads(l) for l in open(a.texts, encoding="utf-8")][si::sn]
    rows.sort(key=lambda r: (r["voice"], r["sid"]))
    os.makedirs(a.out, exist_ok=True)
    # CUDA device-side assert 는 프로세스를 죽이고 같은 문장에서 다시 난다 — 시도 표시만 있고 결과가 없는 문장은 건너뛴다
    has = lambda r, ext: os.path.exists(os.path.join(a.out, r["sid"] + ext))
    crashed = [r["sid"] for r in rows if has(r, ".try") and not has(r, ".wav")]
    todo = [r for r in rows if not has(r, ".wav") and not has(r, ".try")]
    print("[tts] 조각 %d/%d · %d문장 · 남은 것 %d · 이전에 죽은 문장 건너뜀 %d" % (si, sn, len(rows), len(todo), len(crashed)), flush=True)
    m = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
    cur, t0, bad = None, time.time(), 0
    for n, r in enumerate(todo):
        if r["voice"] != cur:
            m.prepare_conditionals(r["voice"], exaggeration=a.exag)
            cur = r["voice"]
        text, e = r["text_spoken"], expected(r["text_spoken"])
        open(os.path.join(a.out, r["sid"] + ".try"), "w").close()
        for k in range(3):
            torch.manual_seed(int(re.sub(r"\D", "", r["sid"])) * 7 + k)      # 2차 짝(…c/…w)은 같은 seed
            wav = m.generate(text, language_id="ko", exaggeration=a.exag, cfg_weight=a.cfg)
            wav = torchaudio.functional.resample(wav, m.sr, 16000)
            d = wav.shape[-1] / 16000.0
            if 0.3 * e <= d <= 2.5 * e + 2:
                break
        else:
            bad += 1
        tmp = os.path.join(a.out, r["sid"] + ".tmp.wav")
        torchaudio.save(tmp, wav, 16000, encoding="PCM_S", bits_per_sample=16)
        os.replace(tmp, os.path.join(a.out, r["sid"] + ".wav"))
        if n % 200 == 0:
            print("  %d/%d · %.0fs · 길이 비정상 %d" % (n + 1, len(todo), time.time() - t0, bad), flush=True)
    print("[완료] 조각 %d/%d · %.0fs · 길이 비정상(3번 다 실패) %d" % (si, sn, time.time() - t0, bad), flush=True)


if __name__ == "__main__":
    main()
