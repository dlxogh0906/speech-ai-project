# -*- coding: utf-8 -*-
"""AIHub 037(한국인의 영어) 발화를 **서버와 같은 경로**로 채점해 문장 점수기 학습용 특징표를 만든다.

입력: `build_037_ctc_manifest.py` 가 만든 $MDD_OUTPUTS/english/aihub037_ctc/{train,dev}.jsonl (PRN, 낭독)
      또는 --list 로 준 목록(예: `build_037_spk_eval.py` 의 SPK 말하기 평가셋, --carry rating,task)
      쓰는 칸은 id · wav · text · spk(화자 단위 분할용) · sec · 사람 점수(--carry) 뿐이다. 나이·수준 칸은 읽지 않는다.
화자마다 --per-spk 개를 고른다(화자 단위 평가가 고르게 되도록).
경로: 녹음 →(앞뒤 무음 맞추기 · 말소리 검사 = deploy/pronunciation-en/app.py 와 같은 값)→ EnglishAssessor.assess
출력 한 줄 = {id, spk, text_key(정규화한 문장의 해시), art, pros, sec, feats{mdd_en.sent_score.FEAT_COLS}, api{…}}
  api = 지금 응답(speechocean762 기준) 4칸 + integrity — 재보정 전후 비교용
GPU 1장에 조각 여러 개(--shard i/n)를 같이 돌려도 된다. 이미 쓴 id 는 건너뛴다(이어 돌리기).

    $PY .../en_extract_037_feats.py --model synth-d --shard 0/2
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import FFMPEG, MDD_OUTPUTS

import argparse, collections, hashlib, json, os, random, re, time

MIN_VOICED_SEC, MIN_VOICED_RATIO, MAX_SEC = 0.3, 0.10, 30.0     # deploy/pronunciation-en/app.py 와 같은 값
KEEP = ("id", "wav", "text", "spk", "sec")


def text_key(t):
    return hashlib.sha1(re.sub(r"\s+", " ", re.sub(r"[^a-z0-9' ]", " ", t.lower())).strip().encode("utf-8")).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="synth-d", help="mdd_en.models 의 이름")
    ap.add_argument("--scorer-dir", dest="scorer_dir", default=None, help="기본: $MDD_OUTPUTS/english/scorer/<model>")
    ap.add_argument("--manifest", default=os.path.join(str(MDD_OUTPUTS), "english", "aihub037_ctc"))
    ap.add_argument("--list", default=None, help="manifest 대신 이 jsonl 목록을 쓴다")
    ap.add_argument("--carry", default="art,pros", help="출력에 그대로 옮길 사람 점수 칸")
    ap.add_argument("--per-spk", dest="per_spk", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", default="0/1", help="i/n — 고른 발화의 i 번째 조각")
    ap.add_argument("--limit", type=int, default=0, help="점검용 — 조각 앞의 N개만")
    ap.add_argument("--out", default=None, help="기본: $MDD_OUTPUTS/english/score037/<model>")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    keep = KEEP + tuple(c for c in a.carry.split(",") if c)
    rows = []
    for p in ([a.list] if a.list else [os.path.join(a.manifest, sp + ".jsonl") for sp in ("train", "dev")]):
        with open(p, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                rows.append(dict((k, d[k]) for k in keep))
    by = collections.defaultdict(list)
    for r in rows:
        by[r["spk"]].append(r)
    rng = random.Random(a.seed)
    pick = []
    for s in sorted(by):
        rs = sorted(by[s], key=lambda r: r["id"])
        rng.shuffle(rs)
        pick += rs[: a.per_spk]
    pick.sort(key=lambda r: r["id"])
    si, sn = (int(x) for x in a.shard.split("/"))
    pick = pick[si::sn]
    if a.limit:
        pick = pick[: a.limit]

    out_dir = a.out or os.path.join(str(MDD_OUTPUTS), "english", "score037", a.model)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "feats.%d-of-%d.jsonl" % (si, sn))
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            done = set(json.loads(l)["id"] for l in f if l.strip())
    todo = [r for r in pick if r["id"] not in done]
    print("[037] 화자 %d명 · 고른 발화 %d · 이 조각 %d · 이미 함 %d" % (len(by), sum(min(len(v), a.per_spk) for v in by.values()),
                                                         len(pick), len(done)), flush=True)

    from mdd_en.audio import SR, decode_audio, trim_edges_offset, voiced_info
    from mdd_en.engine import EnglishAssessor, NoWords, TextTooLong
    from mdd_en.gop import AlignError
    from mdd_en.models import resolve
    from mdd_en.response import build_response
    from mdd_en.sent_score import sentence_features

    hf, kind, vocab = resolve(a.model)
    sd = a.scorer_dir or os.path.join(str(MDD_OUTPUTS), "english", "scorer", a.model)
    eng = EnglishAssessor(hf, kind, sd, vocab_dir=vocab, device=a.device)
    skip = collections.Counter()
    t0 = time.time()
    with open(path, "a", encoding="utf-8") as fo:
        for n, r in enumerate(todo):
            with open(r["wav"], "rb") as f:
                wav = decode_audio(f.read(), FFMPEG)[: int(MAX_SEC * SR)]
            tw, off = trim_edges_offset(wav)
            vi = voiced_info(tw)
            if vi["voiced_sec"] < MIN_VOICED_SEC or vi["voiced_ratio"] < MIN_VOICED_RATIO:
                skip["low_speech"] += 1
                continue
            try:
                raw = eng.assess(tw, r["text"])
            except NoWords:
                skip["no_words"] += 1
                continue
            except TextTooLong:
                skip["too_long"] += 1
                continue
            except AlignError:
                skip["align"] += 1
                continue
            raw.pop("scores_ko", None)                     # api = 재보정 전(speechocean762 기준) 응답
            raw.pop("human_ko", None)
            res = build_response(raw, r["text"], flag_rule=eng.flag_rule, offset_sec=off)["result"]
            rec = {"id": r["id"], "spk": r["spk"], "text_key": text_key(r["text"]), "sec": r["sec"],
                   "feats": sentence_features(raw),
                   "api": dict((k, res[k]) for k in ("overall", "pronunciation", "fluency", "rhythm", "integrity"))}
            rec.update((k, r[k]) for k in keep if k not in KEEP)
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if n % 500 == 0:
                fo.flush()
                print("  %d/%d · %.0fs · 건너뜀 %s" % (n + 1, len(todo), time.time() - t0, dict(skip)), flush=True)
    print("[완료] %s · %.0fs · 건너뜀 %s" % (path, time.time() - t0, dict(skip)), flush=True)


if __name__ == "__main__":
    main()
