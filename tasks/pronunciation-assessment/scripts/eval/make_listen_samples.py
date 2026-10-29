# -*- coding: utf-8 -*-
"""청취 페이지 샘플(`samples.json`) 생성 — 표준발음 · 사람 전사(TAG) · 모델 둘의 출력을 모은다.

예측은 `eval_actual_ctc.py score` 가 남긴 `<접두어>.preds.jsonl` 을 그대로 읽는다(재디코드 없음).
`--stems-from` 으로 기존 samples.json 의 발화를 그대로 이어받으면 모델만 바뀐 페이지를 만들 수 있다.

    $PY .../make_listen_samples.py --manifest $MDD_OUTPUTS/manifest_foreign_clean_v2 --split test_unseen \\
        --new <FINALv2 out 접두어> --old <OLD out 접두어> --dir $MDD_OUTPUTS/experiments/final_ctc/listen \\
        --stems-from $MDD_OUTPUTS/experiments/final_ctc/listen/samples.json

⚠️ AIHub 오디오를 복사한다 — 외부에 공개하지 말 것(로컬·사내 확인용).
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]

import argparse, importlib.util, json, os, random, shutil

from mdd.jamo_to_kog2p import jamo_to_phones


def dist(ref, hyp):
    import edlib
    return edlib.align("".join(ref), "".join(hyp))["editDistance"]


def load_preds(prefix, n):
    P = {}
    for l in open(prefix + ".preds.jsonl", encoding="utf-8"):
        o = json.loads(l)
        P[o["i"]] = [t for t in o["pred"] if len(t) == 1 and 0x3131 <= ord(t) <= 0x3163]
    assert len(P) == n, (len(P), n)
    return [P[i] for i in range(n)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True); ap.add_argument("--split", required=True)
    ap.add_argument("--append", action="store_true", help="기존 samples.json 에 이어붙인다(여러 split 을 한 페이지에)")
    ap.add_argument("--model", action="append", default=[], dest="models",
                    help="'이름=out 접두어' — 여러 번 주면 그 순서로 줄이 생긴다")
    ap.add_argument("--new", default=None, help="(옛 방식) 새 모델 out 접두어")
    ap.add_argument("--old", default=None, help="(옛 방식) 대조 모델 out 접두어")
    ap.add_argument("--dir", required=True, help="samples.json 과 wav/ 를 쓸 폴더")
    ap.add_argument("--stems-from", dest="stems_from", default=None, help="기존 samples.json — 같은 발화를 재사용")
    ap.add_argument("--n", type=int, default=24); ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    spec = importlib.util.spec_from_file_location("ev", str(_pl.Path(__file__).with_name("eval_actual_ctc.py")))
    ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
    wavs, ltr, canon, meta = ev.read_split(a.manifest, a.split)
    specs = [v.split("=", 1) for v in a.models] or \
            [("모델", a.new), ("기존 모델", a.old)]
    if any(p is None for _, p in specs):
        ap.error("--model 또는 --new/--old 가 필요하다")
    PRED = [(name, load_preds(pre, len(wavs))) for name, pre in specs]

    if a.stems_from:
        want = [s["stem"] for s in json.load(open(a.stems_from, encoding="utf-8"))]
        by = {os.path.basename(w[0])[:-4]: i for i, w in enumerate(wavs)}
        idx = [by[s] for s in want if s in by]
    else:                                   # 오발음/정상 절반씩, 짧은 것 위선
        mis = [i for i in range(len(wavs)) if ltr[i] != canon[i]]
        ok = [i for i in range(len(wavs)) if ltr[i] == canon[i]]
        rnd = random.Random(a.seed)
        idx = rnd.sample(mis, min(a.n // 2, len(mis))) + rnd.sample(ok, min(a.n - a.n // 2, len(ok)))

    os.makedirs(os.path.join(a.dir, "wav"), exist_ok=True)
    out = []
    for i in idx:
        stem = os.path.basename(wavs[i][0])[:-4]
        dst = os.path.join(a.dir, "wav", stem + ".wav")
        if not os.path.exists(dst):
            shutil.copyfile(wavs[i][0], dst)
        m = meta[i]
        out.append({"split": a.split, "group": "오발음" if ltr[i] != canon[i] else "정상",
                    "stem": stem, "prompt": m.get("prompt") or m.get("norm_prompt") or "",
                    "lang": m.get("language") or "?", "task": m.get("task") or "?",
                    "sec": round(wavs[i][1] / 16000, 3),
                    # 자모·음소열은 리스트 그대로 둔다 — make_listen_page.mark() 가 토큰 단위로 읽는다
                    "canon": canon[i], "tag": ltr[i],
                    "canon_ph": jamo_to_phones(canon[i]), "tag_ph": jamo_to_phones(ltr[i]),
                    "preds": [{"name": name, "jamo": P[i], "ph": jamo_to_phones(P[i]),
                               "d": dist(ltr[i], P[i])} for name, P in PRED]})
    p = os.path.join(a.dir, "samples.json")
    if a.append and os.path.exists(p):
        prev = json.load(open(p, encoding="utf-8"))
        have = {s["stem"] for s in out}
        out = [s for s in prev if s["stem"] not in have] + out
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[done] {p} · 샘플 {len(out)}개 (오발음 {sum(s['group'] == '오발음' for s in out)})", flush=True)


if __name__ == "__main__":
    main()
