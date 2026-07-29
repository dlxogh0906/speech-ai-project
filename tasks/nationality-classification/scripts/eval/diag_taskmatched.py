# -*- coding: utf-8 -*-
"""task-matched cross-domain: 041을 '문장(≥4단어)'만으로 걸러 505와 같은 과제로 비교.
캐시 feature + sent_mask_ge4.npy 사용 (재추출 없음). 격차가 내용/길이냐 녹음/화자냐 판정."""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
if "--help" in _sys.argv or "-h" in _sys.argv:   # 인자 파서가 없는 스크립트다. --help 로 본체가 실행되면 안 된다.
    print(__doc__ or "설명이 없다 — 소스를 열어볼 것"); raise SystemExit(0)
from shared.paths import DATA_ROOT, NAT_DATA

import csv, sys
from collections import Counter, defaultdict
import numpy as np
from country_classifier.make_country40_probe import SPLIT, LANG2C
import torch, torch.nn as nn, torch.nn.functional as F
dev="cuda"

rows=list(csv.DictReader(open(f"{NAT_DATA}/fine_labels.csv",encoding="utf-8")))
spk_mt={}; spk_cc=defaultdict(Counter)
for r in rows:
    spk_mt[r["speaker_id"]]=r["mother_tongue"]; spk_cc[r["speaker_id"]][(r["country"]or"").strip()]+=1
spk_c={s:c.most_common(1)[0][0] for s,c in spk_cc.items()}
def c40(s):
    mt=spk_mt.get(s); return SPLIT[mt].get(spk_c.get(s)) if mt in SPLIT else LANG2C.get(mt)

print("[load]",flush=True)
# 505
X5=[]; s5=[]; l5=[]
for sp in ["train","valid","test"]:
    d=np.load(f"{DATA_ROOT}/taeho_layers_full/{sp}.npz",allow_pickle=True); Xa=np.clip(np.nan_to_num(d["X"].astype(np.float32)),-50,50)[:,0,:]
    for i,s in enumerate(d["speaker"]):
        c=c40(str(s))
        if c: X5.append(Xa[i]); s5.append(str(s)); l5.append(c)
X5=np.stack(X5); s5=np.array(s5); l5=np.array(l5,dtype=object)
# 041 + 문장마스크
de=np.load(f"{DATA_ROOT}/taeho_layers_euro/train.npz",allow_pickle=True); Xe=np.clip(np.nan_to_num(de["X"].astype(np.float32)),-50,50)[:,0,:]
sent=np.load(f"{DATA_ROOT}/taeho_layers_euro/sent_mask_ge4.npy")
e2l={r["speaker_id"]:r["label_name"] for r in csv.DictReader(open(f"{NAT_DATA}/euro_country40_manifest.csv",encoding="utf-8"))}
se=np.array([str(s) for s in de["speaker"]]); le=np.array([e2l.get(s) for s in se],dtype=object)
euro20=sorted(set(le.tolist()))
# 505도 20클래스로 제한
keep5=np.array([c in euro20 for c in l5]); X5,s5,l5=X5[keep5],s5[keep5],l5[keep5]
print(f"  505={len(X5):,} / 041 mixed={len(Xe):,} / 041 sent={int(sent.sum()):,}",flush=True)

rng=np.random.default_rng(0)
def split_idx(spk_arr, idx):
    sp_=sorted(set(spk_arr[idx])); rng.shuffle(sp_); cut=int(len(sp_)*0.8); trs=set(sp_[:cut])
    tr=idx[np.array([spk_arr[i] in trs for i in idx])]; te=idx[np.array([spk_arr[i] not in trs for i in idx])]
    return tr,te
def fit_eval(Xtr,ytr,Xte,yte,C):
    mu=Xtr.mean(0,keepdims=True); sd=Xtr.std(0,keepdims=True)+1e-6; Xtr=(Xtr-mu)/sd; Xte=(Xte-mu)/sd
    torch.manual_seed(0); head=nn.Linear(Xtr.shape[1],C).to(dev); opt=torch.optim.AdamW(head.parameters(),lr=1e-3,weight_decay=1e-4)
    At=torch.from_numpy(Xtr).to(dev); Yt=torch.from_numpy(ytr).to(dev); n=len(Yt)
    for ep in range(40):
        head.train(); perm=torch.randperm(n,device=dev)
        for i in range(0,n,4096):
            j=perm[i:i+4096]; F.cross_entropy(head(At[j]),Yt[j]).backward(); opt.step(); opt.zero_grad()
    head.eval()
    with torch.no_grad(): tl=head(torch.from_numpy(Xte).to(dev)).cpu().numpy()
    t5=np.argpartition(-tl,4,axis=1)[:,:5]; h5=np.array([y in r for y,r in zip(yte,t5)]); h1=(tl.argmax(1)==yte)
    return h1.mean(),h5.mean(),len(yte)

c2i={c:i for i,c in enumerate(euro20)}; C=len(euro20)
# 데이터 풀
i5=np.arange(len(X5)); ie_mix=np.arange(len(Xe)); ie_sen=np.where(sent)[0]
tr5,te5=split_idx(s5,i5)
trm,tem=split_idx(se,ie_mix)
trs,tes=split_idx(se,ie_sen)
def Y(l,idx): return np.array([c2i[l[i]] for i in idx],np.int64)

print("\n"+"="*64+"\n  task-matched cross-domain (20 클래스, top1/top5)\n"+"="*64)
configs=[
 ("505 → 505",          X5[tr5],Y(l5,tr5),  X5[te5],Y(l5,te5)),
 ("505 → 041(mixed)",   X5[tr5],Y(l5,tr5),  Xe[tem],Y(le,tem)),
 ("505 → 041(sent)",    X5[tr5],Y(l5,tr5),  Xe[tes],Y(le,tes)),
 ("041(mix) → 041(mix)",Xe[trm],Y(le,trm),  Xe[tem],Y(le,tem)),
 ("041(sent)→041(sent)",Xe[trs],Y(le,trs),  Xe[tes],Y(le,tes)),
 ("041(mix) → 505",     Xe[trm],Y(le,trm),  X5[te5],Y(l5,te5)),
 ("041(sent)→ 505",     Xe[trs],Y(le,trs),  X5[te5],Y(l5,te5)),
]
for name,Xtr,ytr,Xte,yte in configs:
    a,b,n=fit_eval(Xtr,ytr,Xte,yte,C)
    print(f"  {name:<22} top1={a:.4f} top5={b:.4f}  (te {n:,})")
print("\n[done]",flush=True)
