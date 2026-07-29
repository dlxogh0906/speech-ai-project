# -*- coding: utf-8 -*-
"""041 문장(≥4단어) 구성 + 문장-병합 split 통계 (merge_country41_euro --euro_sent_only 와 동일 순서/split 재현).
npz 화자배열 + manifest + sent_mask 만 읽음 (학습/추출 재실행 없음)."""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists(): _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
if "--help" in _sys.argv or "-h" in _sys.argv:   # 인자 파서가 없는 스크립트다. --help 로 본체가 실행되면 안 된다.
    print(__doc__ or "설명이 없다 — 소스를 열어볼 것"); raise SystemExit(0)
from shared.paths import DATA_ROOT, NAT_DATA

import csv, json, os, sys
from collections import defaultdict, Counter, OrderedDict
import numpy as np
from country_classifier.make_country40_probe import SPLIT, LANG2C
from country_classifier.make_speaker_split import assign_speakers
OUT="/tmp/claude-7778/-home-tta/86db1df8-e52d-4e72-89e0-2554abdda961/scratchpad/sent_stats.json"

rows=list(csv.DictReader(open(f"{NAT_DATA}/fine_labels.csv",encoding="utf-8")))
spk_mt={}; spk_cc=defaultdict(Counter)
for r in rows:
    spk_mt[r["speaker_id"]]=r["mother_tongue"]; spk_cc[r["speaker_id"]][(r["country"]or"").strip()]+=1
spk_c={s:c.most_common(1)[0][0] for s,c in spk_cc.items()}
def c40(s):
    mt=spk_mt.get(s); return SPLIT[mt].get(spk_c.get(s)) if mt in SPLIT else LANG2C.get(mt)

# 041 문장 마스크 (npz 순서 정렬됨)
sent=np.load(f"{DATA_ROOT}/taeho_layers_euro/sent_mask_ge4.npy")
e2l={r["speaker_id"]:r["label_name"] for r in csv.DictReader(open(f"{NAT_DATA}/euro_country40_manifest.csv",encoding="utf-8"))}

# merge_country41_euro --euro_sent_only 와 동일 순서로 (speaker, label, origin)
items=[]
for sp in ["train","valid","test"]:
    d=np.load(f"{DATA_ROOT}/taeho_layers_full/{sp}.npz",allow_pickle=True)
    for s in d["speaker"]:
        l=c40(str(s))
        if l: items.append((str(s),l,"505"))
kd=np.load(f"{DATA_ROOT}/taeho_layers_korean/korean.npz",allow_pickle=True)
for s in kd["speaker"]: items.append(("KOR:"+str(s),"한국어","kor"))
de=np.load(f"{DATA_ROOT}/taeho_layers_euro/train.npz",allow_pickle=True)
for i,s in enumerate(de["speaker"]):
    if not sent[i]: continue
    l=e2l.get(str(s))
    if l: items.append((str(s),l,"041s"))

assign=assign_speakers([{"speaker_id":s,"label_name":l,"num_frames":0} for s,l,_ in items],[8,1,1],seed=42)

# per (origin) totals + per-class 041s composition
def blank(): return {sp:{"u":0,"s":set()} for sp in ["train","valid","test"]}
tot=defaultdict(blank)             # origin -> split totals
c041=defaultdict(blank)            # 041s class -> split
merged_tot=blank()
for s,l,o in items:
    sp=assign[s]
    tot[o][sp]["u"]+=1; tot[o][sp]["s"].add(s)
    merged_tot[sp]["u"]+=1; merged_tot[sp]["s"].add(s)
    if o=="041s": c041[l][sp]["u"]+=1; c041[l][sp]["s"].add(s)

def pack(b):
    r={sp:{"utt":b[sp]["u"],"spk":len(b[sp]["s"])} for sp in ["train","valid","test"]}
    allu=sum(b[sp]["u"] for sp in b); alls=len(set().union(*[b[sp]["s"] for sp in b]))
    r["total"]={"utt":allu,"spk":alls}; return r

stats={
 "totals_041s":pack(tot["041s"]),
 "totals_505_in_msent":pack(tot["505"]),
 "totals_kor_in_msent":pack(tot["kor"]),
 "totals_merged_sent":pack(merged_tot),
 "comp_041s":{l:pack(b) for l,b in c041.items()},
}
json.dump(stats,open(OUT,"w"),ensure_ascii=False,indent=1)
t=stats["totals_041s"]; m=stats["totals_merged_sent"]
print(f"041 문장: {t['total']['spk']}화자 {t['total']['utt']:,}발화  tr/va/te utt={t['train']['utt']:,}/{t['valid']['utt']:,}/{t['test']['utt']:,} spk={t['train']['spk']}/{t['valid']['spk']}/{t['test']['spk']}")
print(f"병합(문장): {m['total']['spk']}화자 {m['total']['utt']:,}발화  tr/va/te utt={m['train']['utt']:,}/{m['valid']['utt']:,}/{m['test']['utt']:,} spk={m['train']['spk']}/{m['valid']['spk']}/{m['test']['spk']}")
print(f"저장: {OUT}")
