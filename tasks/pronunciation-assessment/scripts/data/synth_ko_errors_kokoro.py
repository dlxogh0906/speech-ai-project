# -*- coding: utf-8 -*-
"""한국인 대표 발음 오류를 일부러 넣어 TTS 로 읽힌 학습 데이터 — 정답 = **실제로 읽힌 소리**.

Kokoro-82M(Apache-2.0, 음소 문자열 입력)로 읽힌다. 규칙은 `mdd_en.l1_ko_errors`(ð→d · z→s · f→p · v→b ·
r↔l · ɪ↔iː · 끝소리 모음 삽입 …). 발화마다 오류 밀도(0 · 0.5 · 1 · 1.5배)·목소리(영어 28개)·말 속도를 바꾼다.
0배(표준대로 읽음)도 섞어 "틀리지 않은 소리는 그대로" 도 배우게 한다.

⚠️ 고정 파이썬 환경($PY 3.8)이 아니라 **Kokoro 전용 환경**에서 돈다:
    /data/speech-ai/envs/kokoro (tts 환경 위에 --system-site-packages 로 만든 venv · kokoro 0.9.4)

    HF_HOME=... /data/speech-ai/envs/kokoro/bin/python \\
        tasks/pronunciation-assessment/scripts/data/synth_ko_errors_kokoro.py --shard 0 --nshard 3

산출: $SPEECH_AI_DATA/english/synth_ko_v1/wav/<shard>/<id>.wav (16 kHz) · manifest_<shard>.jsonl
      한 줄 = {id, wav, sec, phones(읽힌 소리), canon(표준), errs, voice, speed, rate, sid, split, synthetic}
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import DATA_ROOT, MDD_OUTPUTS

import argparse, hashlib, json, os, random, time

EN_VOICES = ("af_alloy af_aoede af_bella af_heart af_jessica af_kore af_nicole af_nova af_river af_sarah af_sky "
             "am_adam am_echo am_eric am_fenrir am_liam am_michael am_onyx am_puck am_santa "
             "bf_alice bf_emma bf_isabella bf_lily bm_daniel bm_fable bm_george bm_lewis").split()
RATES = [(0.0, 0.20), (0.5, 0.25), (1.0, 0.35), (1.5, 0.20)]


def pick(rng, table):
    x, acc = rng.random(), 0.0
    for v, p in table:
        acc += p
        if x < acc:
            return v
    return table[-1][0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--texts", default=os.path.join(str(MDD_OUTPUTS), "english", "synth_ko", "texts.jsonl"))
    ap.add_argument("--out", default=os.path.join(str(DATA_ROOT), "english", "synth_ko_v1"))
    ap.add_argument("--per-text", dest="per_text", type=int, default=3)
    ap.add_argument("--dev-frac", dest="dev_frac", type=float, default=0.03)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="시험용 — 문장 N개만")
    a = ap.parse_args()
    import torch, torchaudio
    import soundfile as sf
    from huggingface_hub import hf_hub_download
    from kokoro import KModel
    from mdd_en.l1_ko_errors import apply_errors, strip, to_kokoro

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = KModel(repo_id="hexgrad/Kokoro-82M").to(dev).eval()
    packs = dict((v, torch.load(hf_hub_download("hexgrad/Kokoro-82M", "voices/%s.pt" % v), weights_only=True))
                 for v in EN_VOICES)
    rows = [json.loads(l) for l in open(a.texts, encoding="utf-8") if l.strip()]
    rows = [r for k, r in enumerate(rows) if k % a.nshard == a.shard]
    if a.limit:
        rows = rows[: a.limit]
    wdir = os.path.join(a.out, "wav", str(a.shard))
    os.makedirs(wdir, exist_ok=True)
    mp = os.path.join(a.out, "manifest_%d.jsonl" % a.shard)
    done = set()
    if os.path.exists(mp):                                  # 이어 만들기
        done = set(json.loads(l)["id"] for l in open(mp, encoding="utf-8") if l.strip())
    f = open(mp, "a", encoding="utf-8")
    t0, n = time.time(), 0
    for r in rows:
        split = "dev" if int(hashlib.md5(r["sid"].encode()).hexdigest(), 16) % 1000 < a.dev_frac * 1000 else "train"
        for k in range(a.per_text):
            uid = "%s_%d" % (r["sid"], k)
            if uid in done:
                continue
            rng = random.Random("%d-%s-%d" % (a.seed, r["sid"], k))
            rate = pick(rng, RATES)
            words, errs = apply_errors(r["words"], rng, rate)
            ph = to_kokoro(words, final="." if r["text"].rstrip()[-1:] in ".!?" else "")
            if len(ph) > 500:
                continue
            voice = rng.choice(EN_VOICES)
            speed = round(rng.uniform(0.85, 1.15), 3)
            with torch.no_grad():
                out = model(ph, packs[voice][len(ph) - 1].to(dev), speed)
            wav = out.audio if hasattr(out, "audio") else out
            wav = torchaudio.functional.resample(wav.float().cpu(), 24000, 16000).numpy()
            path = os.path.join(wdir, uid + ".wav")
            sf.write(path, wav, 16000, subtype="PCM_16")
            f.write(json.dumps({"id": uid, "wav": path, "sec": round(len(wav) / 16000.0, 3),
                                "phones": " ".join(strip(p) for w in words for p in w["ph"]),
                                "canon": " ".join(strip(p) for w in r["words"] for p in w["ph"]),
                                "errs": [[e[2], e[3], e[4]] for e in errs], "voice": voice, "speed": speed,
                                "rate": rate, "sid": r["sid"], "text": r["text"], "split": split,
                                "synthetic": True, "kokoro": ph}, ensure_ascii=False) + "\n")
            n += 1
            if n % 500 == 0:
                f.flush()
                print("[shard %d] %d발화 · %.0f초" % (a.shard, n, time.time() - t0), flush=True)
    f.close()
    print("[done] shard %d · %d발화 · %.0f초 → %s" % (a.shard, n, time.time() - t0, mp), flush=True)


if __name__ == "__main__":
    main()
