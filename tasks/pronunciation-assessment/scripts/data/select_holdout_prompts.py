# -*- coding: utf-8 -*-
"""Stage 2 **문항 hold-out** 선택 — 화자·문항이 모두 처음인 평가를 만들기 위한 첫 단계.

기존 test 는 화자만 분리돼 있고 문항 225개가 전부 학습에 있다(`scripts/eval/split_leakage_check.py`).
여기서 문항 일부를 골라 학습에서 완전히 빼고, 그 일부를 valid(모델 선택용)와 test(최종 1회)로 나눈다.

선택 규칙(결정적 · 시드 고정):
  1. 후보는 단어(RW)·문장(RS) 문항. 문단(RP)은 길이·과제가 달라 제외.
  2. 과제 비율을 전체와 맞춘다 — 문항 수 기준 RW:RS = 150:75 = 2:1.
  3. 과제 안에서 **오발음 비율 3분위**로 층화해 난이도를 대표하게 한다.
  4. 발화 수가 과제 안 하위 25% 인 문항은 뺀다(평가 표본이 너무 작아짐).
  5. **단어 문항이 다른 문항(문장) 안에 부분 문자열로 들어 있으면 뺀다** — 그 단어를 학습 문장에서
     이미 들었다면 "안 본 문항"이 아니다. 문장 문항도 다른 문항에 통째로 들어 있으면 뺀다.
  6. 층마다 1/3 을 valid, 나머지를 test 로.

    $PY .../select_holdout_prompts.py --n-rw 16 --n-rs 8 --out $MDD_OUTPUTS/holdout_prompts_v1.json
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.paths import DATA_ROOT, MDD_OUTPUTS

import argparse, collections, gzip, json, os, random


def inventory(cache, wav_index, min_sec, max_sec, exclude_sr):
    """문항별 발화 수·오발음 비율·평균 길이 — 학습 매니페스트와 같은 품질 필터를 적용한다."""
    idx = {}
    with open(wav_index, encoding="utf-8") as f:
        next(f)
        for line in f:
            st, sr, ch, fr = line.rstrip("\n").split("\t")[:4]
            idx[st] = (int(sr) if sr else None, int(fr))
    inv = collections.defaultdict(lambda: {"n": 0, "mis": 0, "sec": 0.0, "task": None, "prompt": None, "norm": None,
                                           "speakers": set()})
    with gzip.open(cache, "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if "fail" in d or not d.get("ltr") or not d.get("canon") or not d.get("norm_prompt") or d["stem"] not in idx:
                continue
            sr, fr = idx[d["stem"]]
            sec = fr / 16000.0
            if not (min_sec <= sec <= max_sec) or sr in exclude_sr:
                continue
            it = inv[d["item"]]
            it["n"] += 1; it["mis"] += int(d["ltr"] != d["canon"]); it["sec"] += sec
            it["task"] = d["task"]; it["prompt"] = it["prompt"] or d["prompt"]; it["norm"] = it["norm"] or d["norm_prompt"]
            it["speakers"].add(d["spk"])
    out = {}
    for k, v in inv.items():
        out[k] = {"task": v["task"], "n": v["n"], "mis_rate": round(v["mis"] / float(v["n"]), 4),
                  "mean_sec": round(v["sec"] / v["n"], 2), "norm": v["norm"], "prompt": v["prompt"],
                  "speakers": len(v["speakers"])}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=os.path.join(str(MDD_OUTPUTS), "cache", "foreign_tag_lab_parsed.jsonl.gz"))
    ap.add_argument("--wav-index", dest="wav_index", default=os.path.join(str(DATA_ROOT), "mdd_gold_wav16k", "_index.tsv"))
    ap.add_argument("--n-rw", dest="n_rw", type=int, default=16)
    ap.add_argument("--n-rs", dest="n_rs", type=int, default=8)
    ap.add_argument("--valid-frac", dest="valid_frac", type=float, default=1.0 / 3)
    ap.add_argument("--min-sec", dest="min_sec", type=float, default=0.5)
    ap.add_argument("--max-sec", dest="max_sec", type=float, default=15.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit("이미 있다(덮어쓰기 금지): %s" % a.out)

    inv = inventory(a.cache, a.wav_index, a.min_sec, a.max_sec, {8000})
    norms = dict((k, v["norm"]) for k, v in inv.items())
    rng = random.Random(a.seed)
    chosen = {"valid": [], "test": []}
    report = {"inventory": {"items": len(inv), "by_task": dict(collections.Counter(v["task"] for v in inv.values()))},
              "excluded": collections.defaultdict(list), "strata": {}}
    for task, n_pick in (("RW", a.n_rw), ("RS", a.n_rs)):
        items = sorted(k for k, v in inv.items() if v["task"] == task)
        ns = sorted(inv[k]["n"] for k in items)
        n_floor = ns[len(ns) // 4]
        elig = []
        for k in items:
            nm = norms[k]
            inside = sorted(o for o, onm in norms.items() if o != k and onm and nm and nm in onm)
            if inside:
                report["excluded"]["substring_of_other_prompt"].append({"item": k, "text": inv[k]["prompt"], "inside": inside[:5]})
                continue
            if inv[k]["n"] < n_floor:
                report["excluded"]["low_support"].append(k)
                continue
            elig.append(k)
        elig.sort(key=lambda k: inv[k]["mis_rate"])
        tert = [elig[i * len(elig) // 3:(i + 1) * len(elig) // 3] for i in range(3)]
        per = [n_pick // 3 + (1 if i < n_pick % 3 else 0) for i in range(3)]
        for ti, (bucket, m) in enumerate(zip(tert, per)):
            pick = sorted(rng.sample(bucket, min(m, len(bucket))))
            n_val = max(1, int(round(len(pick) * a.valid_frac)))
            rng.shuffle(pick)
            chosen["valid"] += sorted(pick[:n_val]); chosen["test"] += sorted(pick[n_val:])
            report["strata"]["%s_tertile%d" % (task, ti + 1)] = {
                "eligible": len(bucket), "mis_rate_range": [inv[bucket[0]]["mis_rate"], inv[bucket[-1]]["mis_rate"]] if bucket else None,
                "picked": sorted(pick), "valid": sorted(pick[:n_val]), "test": sorted(pick[n_val:])}
    held = set(chosen["valid"]) | set(chosen["test"])
    train_items = sorted(k for k in inv if k not in held)

    def agg(keys):
        n = sum(inv[k]["n"] for k in keys)
        return {"items": len(keys), "utterances": n,
                "by_task": dict(collections.Counter(inv[k]["task"] for k in keys)),
                "mis_rate_utt_weighted": round(sum(inv[k]["mis"] if False else inv[k]["mis_rate"] * inv[k]["n"] for k in keys) / max(1, n), 4),
                "mean_sec_utt_weighted": round(sum(inv[k]["mean_sec"] * inv[k]["n"] for k in keys) / max(1, n), 2)}
    out = {"version": "holdout_prompts_v1", "seed": a.seed, "rule": __doc__.split("선택 규칙")[1].split("$PY")[0].strip(),
           "valid_items": sorted(chosen["valid"]), "test_items": sorted(chosen["test"]),
           "summary": {"train": agg(train_items), "valid": agg(chosen["valid"]), "test": agg(chosen["test"])},
           "items": dict((k, dict(inv[k], split=("valid" if k in chosen["valid"] else "test"))) for k in sorted(held)),
           "report": {"inventory": report["inventory"], "strata": report["strata"],
                      "excluded": dict(report["excluded"])}}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
    for s in ("train", "valid", "test"):
        print("[%-5s] %s" % (s, out["summary"][s]))
    print("[valid] %s" % ", ".join("%s(%s)" % (k, inv[k]["prompt"]) for k in out["valid_items"]))
    print("[test ] %s" % ", ".join("%s(%s)" % (k, inv[k]["prompt"]) for k in out["test_items"]))
    print("[excl ] 부분문자열 %d · 표본부족 %d" % (len(report["excluded"]["substring_of_other_prompt"]), len(report["excluded"]["low_support"])))
    print("[saved] %s" % a.out)


if __name__ == "__main__":
    main()
