# -*- coding: utf-8 -*-
"""견본 대사(make_sample_clips.py 산출물) 품질 검사 + 대사 밖 소리 자르기.

TTS 는 대사 뒤에 엉뚱한 소리를 덧붙이거나 글자를 빼먹을 때가 있다. 그래서
  1. 대사를 정렬해 말소리 구간을 잡고, 앞 0.15초 · 뒤 0.3초만 남기고 자른다. 원본은 *_raw.wav 로 남긴다.
  2. 자른 소리를 정렬 모델로 받아 적어(greedy CTC) 대사 자모와의 자모 오류율(JER)을 잰다 — 0.35 넘으면 불량
     (남은 군소리는 삽입으로, 빠진 글자는 삭제로 잡힌다).
clips.json 에 파일별 qc(jer, 자른 길이, ok)를 적는다. 데모(deploy/mimicry-demo)는 ref 가 ok 인 대사만 쓴다.

    PYTHONPATH=/data/speech-ai/tools/hf_py38:/data/speech-ai/tools/mimic_py38 HF_HUB_OFFLINE=1 \\
        $PY tasks/mimicry-coach/scripts/data/qc_sample_clips.py --dir /data/speech-ai/out/mimic/sample_clips_v1
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import argparse, json, os, shutil

JER_MAX = 0.35
LEAD, TAIL = 0.15, 0.3


def edit_distance(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
    return d[len(b)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    from mimicry.analyze import analyze
    from mimicry.align import Aligner
    from mimicry.audio import load_audio, save_wav
    from mimicry.text import align_tokens, parse_line

    meta_p = os.path.join(a.dir, "clips.json")
    meta = json.load(open(meta_p, encoding="utf-8"))
    al = Aligner()
    inv = {v: k for k, v in al.vocab.items()}
    for c in meta["clips"]:
        words, syls, _ = parse_line(c["line"])
        want = [t for s in align_tokens(words, syls)[0] for t in s]
        c["qc"] = {}
        for name, fn in c["files"].items():
            p = os.path.join(a.dir, fn)
            raw = p[:-4] + "_raw.wav"
            if not os.path.exists(raw):
                shutil.copyfile(p, raw)                    # 처음 한 번만 원본 보관
            x = load_audio(raw)
            A = analyze(x, c["line"], al, keep_tracks=False)
            s0, s1 = max(0.0, A["speech"][0] - LEAD), min(len(x) / 16000.0, A["speech"][1] + TAIL)
            y = x[int(s0 * 16000): int(s1 * 16000)]
            save_wav(p, y)
            ids = al.log_probs(y).argmax(-1).tolist()
            hyp = [inv[t] for k, t in enumerate(ids) if t != al.blank and (k == 0 or t != ids[k - 1]) and t in inv]
            jer = edit_distance(want, hyp) / max(1, len(want))
            c["qc"][name] = dict(jer=round(jer, 3), raw_sec=round(len(x) / 16000.0, 2), kept=(round(s0, 2), round(s1, 2)),
                                 ok=jer <= JER_MAX, hyp="".join(hyp))
            print("%-9s %-7s JER %.2f %s  %.1fs → %.2f~%.2f" % (c["id"], name, jer, "ok " if jer <= JER_MAX else "불량",
                                                            len(x) / 16000.0, s0, s1), flush=True)
    json.dump(meta, open(meta_p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
