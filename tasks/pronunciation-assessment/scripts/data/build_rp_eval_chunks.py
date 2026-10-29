# -*- coding: utf-8 -*-
"""처음 보는 글을 읽은 학습자 시험셋 — 문단 읽기(RP) 녹음을 쉼에서 잘라 조각마다 기준 문장·표준음·실제 발음을 붙인다.

왜(2026-10-21): 학습자 지표는 모두 학습 문항 224개(같은 문장을 수천 번 들은 모델)로 쟀다. 앱 사용자는 대개 처음 보는
문장을 읽는다. 운영 모델(섞은 모델 · 100% 모델)은 문단 읽기 RP002~RP004 를 학습에 한 번도 안 썼다(E2 만 학습 화자의
RP002·RP003 조각을 썼고 운영에 안 넣었다). 화자도 `manifest_unseen_prompt_rp`(학습 화자 제외) 그대로 쓴다.

  - 자르기: TAG phones 티어의 쉼(틈 ≥ --min-gap)에서, 조각 길이 --min-chunk ~ --max-chunk 초(`build_rp_train_chunks.chunks_of`).
  - 조각마다: 실제 발음 = TAG phones(시각이 조각 안) · 표준음 = LAB phonemic.phones(가운데 시각이 조각 안) ·
    기준 문장 = TAG words(가운데 시각이 조각 안, 원문 철자). spn·<unk> 가 든 조각은 버린다.
  - **녹음은 새로 쓰지 않는다**(/data 여유가 적다) — `test.seg` 의 시각으로 평가 때 원본에서 잘라 쓴다.

출력(<out>/): test.tsv(원본 경로 · 조각 표본 수) · test.ltr(실제) · test.canon(표준음) · test.text(기준 문장) ·
              test.seg(시작 끝 초) · test.meta.jsonl · stats.json

    $PY .../build_rp_eval_chunks.py --out $MDD_OUTPUTS/manifest_unseen_rp_chunks
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import MDD_OUTPUTS

import argparse, collections, glob, importlib.util, json, os, re

SR = 16000


def _load(name):
    spec = importlib.util.spec_from_file_location(name, str(_pl.Path(__file__).with_name(name + ".py")))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_tier(path, name):
    raw = open(path, "rb").read()
    txt = None
    for enc in ("utf-8", "utf-16"):
        try:
            txt = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if txt is None:
        return None
    m = re.search(r'name = "%s".*?(?=name = "|\Z)' % name, txt, re.S)
    if not m:
        return None
    return [(float(a), float(b), t.strip()) for a, b, t in re.findall(r'xmin = ([\d.]+)\s*xmax = ([\d.]+)\s*text = "([^"]*)"', m.group(0))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=os.path.join(str(MDD_OUTPUTS), "manifest_unseen_prompt_rp"),
                    help="녹음 목록(학습 화자를 뺀 RP 녹음) — test.meta.jsonl 의 stem")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-gap", dest="min_gap", type=float, default=0.15)
    ap.add_argument("--min-chunk", dest="min_chunk", type=float, default=4.0)
    ap.add_argument("--max-chunk", dest="max_chunk", type=float, default=15.0)
    ap.add_argument("--min-jamo", dest="min_jamo", type=int, default=8)
    a = ap.parse_args()
    rpb = _load("build_unseen_prompt_eval")
    rtc = _load("build_rp_train_chunks")
    stems = [json.loads(l)["stem"] for l in open(os.path.join(a.src, "test.meta.jsonl"), encoding="utf-8")]
    want = set(stems)
    tags, labs = {}, {}
    for kind, ext, d in (("tag", ".TextGrid", tags), ("lab", ".json", labs)):
        for root in glob.glob(os.path.join(rpb.AIHUB, "0[0-9][0-9]*", "01-1.정식개방데이터", "*", "02.라벨링데이터", "*_pronunciation_" + kind)):
            for p in glob.glob(os.path.join(root, "*" + ext)):
                b = os.path.basename(p)[: -len(ext)]
                if b in want:
                    d[b] = p
    print("[녹음] %d · TAG %d · LAB %d" % (len(stems), len(tags), len(labs)), flush=True)
    rows, drop = [], collections.Counter()
    for stem in stems:
        if stem not in tags or stem not in labs:
            drop["라벨 없음"] += 1; continue
        wp = rpb.wavpath(stem)
        if not os.path.exists(wp):
            drop["wav 없음"] += 1; continue
        phones, words = read_tier(tags[stem], "phones"), read_tier(tags[stem], "words")
        try:
            lab = json.load(open(labs[stem], encoding="utf-8"))["RecordingMetadata"]["phonemic"]["phones"]
        except Exception:
            drop["LAB 파싱"] += 1; continue
        if not phones or not words:
            drop["TAG 파싱"] += 1; continue
        ch = rtc.chunks_of(phones, a.min_gap, a.min_chunk, a.max_chunk)
        if ch is None:
            drop["쉼 없이 김"] += 1; continue
        n_frames = rpb.frames_of(wp)
        for k, (t0, t1, ph) in enumerate(ch):
            labs_ = "".join(t for _, _, t in ph)
            if "spn" in labs_ or "unk" in labs_:
                drop["spn 조각"] += 1; continue
            actual = [c for c in labs_ if rtc.is_jamo(c)]
            canon = [c for x in lab if t0 <= (x["start"] + x["end"]) / 2 < t1 for c in str(x.get("phone", "")).strip() if rtc.is_jamo(c)]
            text = " ".join(w for s, e, w in words if w and t0 <= (s + e) / 2 < t1)
            if len(canon) < a.min_jamo or len(actual) < a.min_jamo or not re.search(r"[가-힣]", text):
                drop["짧은 조각"] += 1; continue
            if int(t1 * SR) > n_frames:
                t1 = n_frames / SR
            parts = stem.split("-")
            rows.append({"wav": wp, "n": int((t1 - t0) * SR), "ltr": actual, "canon": canon, "text": text, "t0": round(t0, 3),
                         "t1": round(t1, 3), "meta": {"stem": stem, "k": k, "item": rpb.ITEM.search(stem).group(1), "spk": parts[0],
                                                      "nationality": parts[3] if len(parts) > 3 else None,
                                                      "t0": round(t0, 3), "t1": round(t1, 3), "text": text}})
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "test.tsv"), "w", encoding="utf-8") as f:
        f.write("/\n")
        for r in rows:
            f.write("%s\t%d\n" % (r["wav"].lstrip("/"), r["n"]))
    for ext, fn in (("ltr", lambda r: " ".join(r["ltr"])), ("canon", lambda r: " ".join(r["canon"])), ("text", lambda r: r["text"]),
                    ("seg", lambda r: "%.3f %.3f %d" % (r["t0"], r["t1"], r["meta"]["k"]))):
        with open(os.path.join(a.out, "test.%s" % ext), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(fn(r) + "\n")
    with open(os.path.join(a.out, "test.meta.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r["meta"], ensure_ascii=False) + "\n")
    st = {"src": a.src, "recordings": len(set(r["meta"]["stem"] for r in rows)), "chunks": len(rows),
          "speakers": len(set(r["meta"]["spk"] for r in rows)), "hours": round(sum(r["n"] for r in rows) / SR / 3600, 2),
          "by_item": dict(collections.Counter(r["meta"]["item"] for r in rows)), "dropped": dict(drop),
          "note": "운영 모델이 학습에 안 쓴 문단(RP002~004) · 학습 화자 제외. 녹음은 원본에서 test.seg 시각으로 잘라 쓴다"}
    json.dump(st, open(os.path.join(a.out, "stats.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(st, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
