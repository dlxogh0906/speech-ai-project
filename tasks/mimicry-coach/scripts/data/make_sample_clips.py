# -*- coding: utf-8 -*-
"""데모·귀 확인용 견본 대사를 Chatterbox Multilingual(로컬 GPU, MIT)로 만든다 — 원작 음성이 아니다.

명대사 텍스트만 빌리고 목소리는 Zeroth 학습 화자(공개 낭독)를 참조로 쓴다. 배우 목소리 흉내가 아니다.
대사마다 셋을 만든다:
  ref     — 목소리 1, 과장(exaggeration) 0.7 : 원본 역할
  take_b  — 목소리 2, 과장 0.4            : 다른 사람이 따라 한 것 역할(목소리·억양 둘 다 다름)
  take_a  — 목소리 1, 과장 0.7, 다른 seed : 같은 사람이 다시 한 것 역할(억양만 조금 다름)
tts 환경(파이썬 3.11, chatterbox-tts)에서 돈다 — 학습 환경($PY)이 아니다.

    HF_HOME=/data/speech-ai/hf_cache HF_HUB_OFFLINE=1 /data/speech-ai/envs/tts/bin/python \\
        tasks/mimicry-coach/scripts/data/make_sample_clips.py --voices <voices.json> --out <출력 폴더>
voices.json = {"low": 낮은 목소리 wav, "high": 높은 목소리 wav} (기준 음높이를 소리에서 재서 고른다 — 메타데이터 안 씀)
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import argparse, json, os, re, time

LINES = [
    ("hawaii", "니가 가라 하와이", "친구 (2001)", "low"),
    ("mai", "마이 묵었다 아이가", "친구 (2001)", "low"),
    ("sidabari", "내가 니 시다바리가", "친구 (2001)", "low"),
    ("idae", "나 이대 나온 여자야", "타짜 (2006)", "high"),
    ("bab", "밥은 먹고 다니냐", "살인의 추억 (2003)", "low"),
    ("neona", "너나 잘하세요", "친절한 금자씨 (2005)", "high"),
    ("eoi", "어이가 없네", "베테랑 (2015)", "low"),
    ("mwosi", "뭣이 중헌디", "곡성 (2016)", "high"),
    ("seojang", "느그 서장 남천동 살제", "범죄와의 전쟁 (2012)", "low"),
    ("salaitne", "살아있네", "범죄와의 전쟁 (2012)", "low"),
]
# (이름, 목소리 키 — "same" 이면 ref 와 같은 목소리, 과장, seed 더하기)
TAKES = [("ref", "same", 0.7, 0), ("take_a", "same", 0.7, 101), ("take_b", "other", 0.4, 202)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voices", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cfg", type=float, default=0.3, help="Chatterbox 권장: 과장할 땐 cfg 를 낮춘다")
    a = ap.parse_args()
    import torch, torchaudio
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    voices = json.load(open(a.voices, encoding="utf-8"))
    os.makedirs(a.out, exist_ok=True)
    m = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
    clips, t0 = [], time.time()
    for n, (cid, line, title, vkey) in enumerate(LINES):
        other = "high" if vkey == "low" else "low"
        files = {}
        for name, which, exag, seed in TAKES:
            vk = vkey if which == "same" else other
            m.prepare_conditionals(voices[vk], exaggeration=exag)
            expect = len(re.findall(r"[가-힣]", line)) * 0.3
            for k in range(3):
                torch.manual_seed(1000 * n + seed + k)
                wav = m.generate(line, language_id="ko", exaggeration=exag, cfg_weight=a.cfg)
                wav = torchaudio.functional.resample(wav, m.sr, 16000)
                if 0.4 * expect <= wav.shape[-1] / 16000.0 <= 2.5 * expect + 1.5:
                    break
            fn = "%s_%s.wav" % (cid, name)
            torchaudio.save(os.path.join(a.out, fn), wav, 16000, encoding="PCM_S", bits_per_sample=16)
            files[name] = fn
        clips.append(dict(id=cid, line=line, title=title + " — TTS 견본(원작 음성 아님)", files=files,
                          voice=vkey, other_voice=other))
        print("[%d/%d] %s %.0fs" % (n + 1, len(LINES), line, time.time() - t0), flush=True)
    json.dump(dict(clips=clips, engine="chatterbox-multilingual", cfg=a.cfg), open(os.path.join(a.out, "clips.json"), "w",
                                                                                   encoding="utf-8"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
