"""근거 기반 통계 추출 (학습/추출 재실행 없음 — npz 화자배열 + 매니페스트만 읽음).
baseline(505+한국어)과 merged(505+한국어+041) 두 split의 클래스별 train/valid/test
화자·발화 수를 계산해 JSON으로 저장."""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
if "--help" in _sys.argv or "-h" in _sys.argv:   # 인자 파서가 없는 스크립트다. --help 로 본체가 실행되면 안 된다.
    print(__doc__ or "설명이 없다 — 소스를 열어볼 것"); raise SystemExit(0)
from shared.paths import DATA_ROOT, NAT_DATA, NAT_TASK

import csv, json, sys
from collections import Counter, defaultdict
import numpy as np
from country_classifier.make_country40_probe import SPLIT, LANG2C
from country_classifier.make_speaker_split import assign_speakers

ASIA = f"{DATA_ROOT}/taeho_layers_full"
KOR = f"{DATA_ROOT}/taeho_layers_korean/korean.npz"
EURO = f"{DATA_ROOT}/taeho_layers_euro/train.npz"
EURO_MANIFEST = f"{NAT_DATA}/euro_country40_manifest.csv"
OUT = f"{NAT_TASK}/outputs/report_stats.json"

# 505 speaker -> country40
rows = list(csv.DictReader(open(f"{NAT_DATA}/fine_labels.csv", encoding="utf-8")))
spk_mt = {}; spk_cc = defaultdict(Counter)
for r in rows:
    spk_mt[r["speaker_id"]] = r["mother_tongue"]; spk_cc[r["speaker_id"]][(r["country"] or "").strip()] += 1
spk_c = {s: c.most_common(1)[0][0] for s, c in spk_cc.items()}
def c40(s):
    mt = spk_mt.get(s); return SPLIT[mt].get(spk_c.get(s)) if mt in SPLIT else LANG2C.get(mt)

# per-utterance (speaker, label, origin) — EXACT order as merge_country41_euro.py
utt_505, utt_kor, utt_041 = [], [], []
for sp in ["train", "valid", "test"]:
    d = np.load(f"{ASIA}/{sp}.npz", allow_pickle=True)
    for s in d["speaker"]:
        l = c40(str(s))
        if l: utt_505.append((str(s), l, "505"))
kd = np.load(KOR, allow_pickle=True)
for s in kd["speaker"]:
    utt_kor.append(("KOR:" + str(s), "한국어", "kor"))
e2l = {r["speaker_id"]: r["label_name"] for r in csv.DictReader(open(EURO_MANIFEST, encoding="utf-8"))}
de = np.load(EURO, allow_pickle=True)
n_041_npz = len(de["speaker"])
for s in de["speaker"]:
    l = e2l.get(str(s))
    if l: utt_041.append((str(s), l, "041"))

def split_of(universe):
    items = [{"speaker_id": s, "label_name": l, "num_frames": 0} for s, l, _ in universe]
    return assign_speakers(items, [8, 1, 1], seed=42)

def comp(universe, assign, origin_filter=None):
    """per-class: {split: (utt_count, set(speakers))} optionally filtered by origin."""
    out = defaultdict(lambda: {sp: [0, set()] for sp in ["train", "valid", "test"]})
    for s, l, o in universe:
        if origin_filter and o != origin_filter: continue
        sp = assign[s]
        out[l][sp][0] += 1; out[l][sp][1].add(s)
    # to plain
    res = {}
    for l, d in out.items():
        res[l] = {sp: {"utt": d[sp][0], "spk": len(d[sp][1])} for sp in d}
        res[l]["total"] = {"utt": sum(d[sp][0] for sp in ["train","valid","test"]),
                           "spk": len(set().union(*[d[sp][1] for sp in ["train","valid","test"]]))}
    return res

base_univ = utt_505 + utt_kor
merged_univ = utt_505 + utt_kor + utt_041
base_assign = split_of(base_univ)
merged_assign = split_of(merged_univ)

stats = {
    "comp_505_baseline_split": comp(utt_505, base_assign),          # 505 40개국 (baseline split)
    "comp_041_merged_split": comp(utt_041, merged_assign),          # 041 (merged split)
    "comp_505_in_merged": comp(utt_505, merged_assign),             # 505 (merged split) — 비교용
    "kor_baseline": comp(utt_kor, base_assign).get("한국어"),
    "kor_merged": comp(utt_kor, merged_assign).get("한국어"),
    "raw_041_npz_utts": n_041_npz,
    "raw_041_manifest_utts": len(utt_041),
}
# dataset-level totals
def totals(comp_dict, assign, universe):
    tot_utt = {"train":0,"valid":0,"test":0}; tot_spk = {"train":set(),"valid":set(),"test":set()}
    for s,l,o in universe:
        sp = assign[s]; tot_utt[sp]+=1; tot_spk[sp].add(s)
    return {"utt": tot_utt, "spk": {k:len(v) for k,v in tot_spk.items()},
            "classes": len(comp_dict), "total_utt": sum(tot_utt.values()),
            "total_spk": len(set().union(*tot_spk.values()))}
stats["totals_505"] = totals(stats["comp_505_baseline_split"], base_assign, utt_505)
stats["totals_041"] = totals(stats["comp_041_merged_split"], merged_assign, utt_041)
stats["totals_merged"] = totals(comp(merged_univ, merged_assign), merged_assign, merged_univ)
stats["totals_kor"] = totals({"한국어":1}, merged_assign, utt_kor)

json.dump(stats, open(OUT, "w"), ensure_ascii=False, indent=1)

# 사람이 읽을 요약 출력
print("=== 데이터셋 totals ===")
for k in ["totals_505","totals_kor","totals_041","totals_merged"]:
    t=stats[k]; print(f"  {k:14} classes={t['classes']:>3} spk={t['total_spk']:>5,} utt={t['total_utt']:>9,}  "
                      f"tr/va/te utt={t['utt']['train']:,}/{t['utt']['valid']:,}/{t['utt']['test']:,}  "
                      f"spk={t['spk']['train']}/{t['spk']['valid']}/{t['spk']['test']}")
print(f"\n  041 원본npz={stats['raw_041_npz_utts']:,} / 41클래스매핑사용={stats['raw_041_manifest_utts']:,} / drop={stats['raw_041_npz_utts']-stats['raw_041_manifest_utts']:,}")
print(f"  041 추가된 클래스 수={len(stats['comp_041_merged_split'])}")
print(f"\n저장: {OUT}")
