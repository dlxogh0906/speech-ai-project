# -*- coding: utf-8 -*-
"""세기 쌍 합성 오류 음성 + 실제 학습 데이터 → 이어 학습용 매니페스트(`train_w2vbert_ctc.py --manifest`).

합성(`synth_tense_tts.py` 결과)
  - 파일이 있고 길이가 정상인 것만(한글 음절 × 0.3초 기준 0.3배~2.5배+2초).
  - --check-model 을 주면 그 모델로 받아쓴 자모열과 정답(actual)의 차이율이 --max-cer 를 넘는 것을 뺀다 — TTS 가 문장을
    빼먹거나 뭉갠 녹음을 거른다(표적 한 자리 차이는 이 문턱보다 훨씬 작다).
  - 문장 단위로 --dev-frac 만큼을 synth_dev 로 떼어 학습에 넣지 않는다(합성 쪽 학습 확인용).
  - 학습에는 --replicate 번 반복해 넣는다.
실제: manifest_full_tag_replay10 train 에서 무작위 --n-real 행(원어민 반복 행 포함, 비율 그대로). 경로·라벨만 옮긴다.
valid: replay10 valid 그대로(학습 중 UER 확인용).

    $PY .../build_synth_tense_manifest.py --check-model $SPEECH_AI_DATA/checkpoints/w2vbert_stage2/step_38000
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import MDD_OUTPUTS

import argparse, collections, json, os, random, re, shutil, wave


def expected(t):
    return max(1, len(re.findall(r"[가-힣]", t))) * 0.3


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--texts", default=os.path.join(str(MDD_OUTPUTS), "synth_tense_v1", "texts.jsonl"))
    ap.add_argument("--wav-dir", dest="wav_dir", default=os.path.join(str(MDD_OUTPUTS), "synth_tense_v1", "wav"))
    ap.add_argument("--real", default=os.path.join(str(MDD_OUTPUTS), "manifest_full_tag_replay10"))
    ap.add_argument("--n-real", dest="n_real", type=int, default=150000)
    ap.add_argument("--replicate", type=int, default=3)
    ap.add_argument("--dev-frac", dest="dev_frac", type=float, default=0.05)
    ap.add_argument("--check-model", dest="check_model", default=None)
    ap.add_argument("--max-cer", dest="max_cer", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(str(MDD_OUTPUTS), "manifest_synth_tense_v1"))
    a = ap.parse_args()

    vocab = set(l.split()[0] for l in open(os.path.join(a.real, "dict.ltr.txt"), encoding="utf-8") if l.strip())
    rows, drop = [], collections.Counter()
    for l in open(a.texts, encoding="utf-8"):
        r = json.loads(l)
        p = os.path.join(a.wav_dir, r["sid"] + ".wav")
        if not os.path.exists(p):
            drop["no_wav"] += 1; continue
        with wave.open(p, "rb") as w:
            fr = w.getnframes()
        e = expected(r["text_spoken"])
        if not (0.3 * e <= fr / 16000.0 <= 2.5 * e + 2):
            drop["length"] += 1; continue
        if any(t not in vocab for t in r["actual"] + r["canon"]):
            drop["vocab"] += 1; continue
        r["path"], r["frames"] = p, fr
        rows.append(r)
    if a.check_model:
        import edlib
        import numpy as np
        from mdd.gop_feats_v2 import HFDecoder, forward_lprobs
        from mdd.vad import trim
        dec = HFDecoder(a.check_model)
        keep = []
        for n, r in enumerate(rows):
            with wave.open(r["path"], "rb") as w:
                x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0
            x, _ = trim(x, 16000)
            lp, Ls = forward_lprobs(dec, [x])
            hyp, prev = [], None
            for v in lp[0, :Ls[0]].argmax(-1).tolist():
                if v != prev and v != dec.blank:
                    t = dec.d[v]
                    if len(t) == 1 and 0x3131 <= ord(t) <= 0x3163:
                        hyp.append(t)
                prev = v
            ref = "".join(r["actual"])
            cer = edlib.align("".join(hyp), ref)["editDistance"] / max(1, len(ref)) if hyp else 1.0
            r["check_cer"] = round(cer, 4)
            if cer > a.max_cer:
                drop["check_cer"] += 1
            else:
                keep.append(r)
            if n % 1000 == 0:
                print("  확인 %d/%d" % (n + 1, len(rows)), flush=True)
        rows = keep
    rng = random.Random(a.seed)
    rng.shuffle(rows)
    n_dev = int(len(rows) * a.dev_frac)
    dev, syn = rows[:n_dev], rows[n_dev:]

    with open(os.path.join(a.real, "train.tsv"), encoding="utf-8") as f:
        root = f.readline().rstrip("\n")
        real_rows = [l.rstrip("\n") for l in f]
    real_ltr = open(os.path.join(a.real, "train.ltr"), encoding="utf-8").read().splitlines()
    assert root == "/" and len(real_rows) == len(real_ltr), (root, len(real_rows), len(real_ltr))
    pick = sorted(random.Random(a.seed + 1).sample(range(len(real_rows)), min(a.n_real, len(real_rows))))

    os.makedirs(a.out, exist_ok=True)
    rel = lambda p: os.path.abspath(p).lstrip("/")
    with open(os.path.join(a.out, "train.tsv"), "w", encoding="utf-8") as ft, \
            open(os.path.join(a.out, "train.ltr"), "w", encoding="utf-8") as fl, \
            open(os.path.join(a.out, "train.kind"), "w", encoding="utf-8") as fk:
        ft.write("/\n")
        for i in pick:
            ft.write(real_rows[i] + "\n"); fl.write(real_ltr[i] + "\n"); fk.write("real\n")
        for _ in range(a.replicate):
            for r in syn:
                ft.write("%s\t%d\n" % (rel(r["path"]), r["frames"])); fl.write(" ".join(r["actual"]) + "\n")
                fk.write("synth_%s\n" % ("ok" if r["pair"] is None else r["pair"]))
    with open(os.path.join(a.out, "synth_dev.tsv"), "w", encoding="utf-8") as ft, \
            open(os.path.join(a.out, "synth_dev.ltr"), "w", encoding="utf-8") as fl, \
            open(os.path.join(a.out, "synth_dev.canon"), "w", encoding="utf-8") as fc:
        ft.write("/\n")
        for r in dev:
            ft.write("%s\t%d\n" % (rel(r["path"]), r["frames"])); fl.write(" ".join(r["actual"]) + "\n")
            fc.write(" ".join(r["canon"]) + "\n")
    for ext in ("tsv", "ltr"):
        shutil.copy(os.path.join(a.real, "valid." + ext), os.path.join(a.out, "valid." + ext))
    shutil.copy(os.path.join(a.real, "dict.ltr.txt"), os.path.join(a.out, "dict.ltr.txt"))
    stats = {"real_rows": len(pick), "synth_sentences_train": len(syn), "synth_rows_train": len(syn) * a.replicate,
             "synth_dev": len(dev), "dropped": dict(drop), "replicate": a.replicate,
             "synth_pairs_train": dict(collections.Counter(r["pair"] or "맞게" for r in syn)),
             "synth_hours_train": round(sum(r["frames"] for r in syn) / 16000 / 3600, 2),
             "synth_voices_train": len(set(r["voice"] for r in syn))}
    json.dump(stats, open(os.path.join(a.out, "stats.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(stats, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
