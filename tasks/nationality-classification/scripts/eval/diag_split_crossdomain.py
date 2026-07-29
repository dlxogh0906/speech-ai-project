# -*- coding: utf-8 -*-
"""진단 #1(동결 split: baseline vs merged를 동일 test로 + seed 노이즈플로어) + #4(cross-domain).
캐시된 layer-31 feature만 사용 (재추출/인코더 재학습 없음)."""

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

import csv, sys, json
from collections import Counter, defaultdict
import numpy as np
from country_classifier.make_country40_probe import SPLIT, LANG2C
from country_classifier.make_speaker_split import assign_speakers
import torch, torch.nn as nn, torch.nn.functional as F

ASIA=f"{DATA_ROOT}/taeho_layers_full"; KOR=f"{DATA_ROOT}/taeho_layers_korean/korean.npz"
EURO=f"{DATA_ROOT}/taeho_layers_euro/train.npz"; EMAN=f"{NAT_DATA}/euro_country40_manifest.csv"
dev="cuda"

rows=list(csv.DictReader(open(f"{NAT_DATA}/fine_labels.csv",encoding="utf-8")))
spk_mt={}; spk_cc=defaultdict(Counter)
for r in rows:
    spk_mt[r["speaker_id"]]=r["mother_tongue"]; spk_cc[r["speaker_id"]][(r["country"]or"").strip()]+=1
spk_c={s:c.most_common(1)[0][0] for s,c in spk_cc.items()}
def c40(s):
    mt=spk_mt.get(s); return SPLIT[mt].get(spk_c.get(s)) if mt in SPLIT else LANG2C.get(mt)

print("[load] features ...", flush=True)
X=[]; sps=[]; lab=[]; dom=[]
for sp in ["train","valid","test"]:
    d=np.load(f"{ASIA}/{sp}.npz",allow_pickle=True); Xa=np.clip(np.nan_to_num(d["X"].astype(np.float32)),-50,50)[:,0,:]
    for i,s in enumerate(d["speaker"]):
        l=c40(str(s))
        if l: X.append(Xa[i]); sps.append(str(s)); lab.append(l); dom.append("505")
kd=np.load(KOR,allow_pickle=True); kX=np.clip(np.nan_to_num(kd["X"].astype(np.float32)),-50,50)[:,0,:]
for i,s in enumerate(kd["speaker"]): X.append(kX[i]); sps.append("KOR:"+str(s)); lab.append("한국어"); dom.append("kor")
e2l={r["speaker_id"]:r["label_name"] for r in csv.DictReader(open(EMAN,encoding="utf-8"))}
de=np.load(EURO,allow_pickle=True); eX=np.clip(np.nan_to_num(de["X"].astype(np.float32)),-50,50)[:,0,:]
for i,s in enumerate(de["speaker"]):
    l=e2l.get(str(s))
    if l: X.append(eX[i]); sps.append(str(s)); lab.append(l); dom.append("041")
X=np.stack(X); sps=np.array(sps); lab=np.array(lab,dtype=object); dom=np.array(dom)
print(f"[load] {len(X):,} utts", flush=True)

def fit_eval(tr, te, classes):
    c2i={c:i for i,c in enumerate(classes)}
    keep_tr=np.array([lab[i] in c2i for i in tr]); tr=tr[keep_tr]
    keep_te=np.array([lab[i] in c2i for i in te]); te=te[keep_te]
    ytr=np.array([c2i[lab[i]] for i in tr],np.int64); yte=np.array([c2i[lab[i]] for i in te],np.int64)
    Xtr=X[tr]; Xte=X[te]; mu=Xtr.mean(0,keepdims=True); sd=Xtr.std(0,keepdims=True)+1e-6
    Xtr=(Xtr-mu)/sd; Xte=(Xte-mu)/sd
    torch.manual_seed(0); head=nn.Linear(X.shape[1],len(classes)).to(dev)
    opt=torch.optim.AdamW(head.parameters(),lr=1e-3,weight_decay=1e-4)
    At=torch.from_numpy(Xtr).to(dev); Yt=torch.from_numpy(ytr).to(dev); n=len(Yt)
    for ep in range(40):
        head.train(); perm=torch.randperm(n,device=dev)
        for i in range(0,n,4096):
            j=perm[i:i+4096]; F.cross_entropy(head(At[j]),Yt[j]).backward(); opt.step(); opt.zero_grad()
    head.eval()
    with torch.no_grad(): tl=head(torch.from_numpy(Xte).to(dev)).cpu().numpy()
    t5=np.argpartition(-tl,4,axis=1)[:,:5]; hit5=np.array([y in r for y,r in zip(yte,t5)]); hit1=(tl.argmax(1)==yte)
    return yte, hit1, hit5, te

classes41=sorted(set(lab.tolist()))
items=[{"speaker_id":s,"label_name":l,"num_frames":0} for s,l in zip(sps,lab)]

# ===== (A) 동결 split: baseline(041 train 제외) vs merged, 동일 test =====
print("\n"+"="*70+"\n(A) 동결 split — baseline(no-041 train) vs merged, *동일 test*\n"+"="*70, flush=True)
assign=assign_speakers(items,[8,1,1],seed=42); split=np.array([assign[s] for s in sps])
tr_all=np.where(split=="train")[0]; te=np.where(split=="test")[0]
tr_no041=tr_all[dom[tr_all]!="041"]
yb,hb1,hb5,_=fit_eval(tr_no041,te,classes41)   # baseline head
ym,hm1,hm5,_=fit_eval(tr_all,te,classes41)      # merged head  (te identical)
dte=dom[te]
def micro(mask,h): return h[mask].mean() if mask.sum() else float('nan')
print(f"  [동일 test, 505분만] baseline top5={micro(dte=='505',hb5):.4f}  merged top5={micro(dte=='505',hm5):.4f}  Δ={micro(dte=='505',hm5)-micro(dte=='505',hb5):+.4f}")
print(f"  [동일 test, 한국어]   baseline top5={micro(dte=='kor',hb5):.4f}  merged top5={micro(dte=='kor',hm5):.4f}")
print("  클래스별 (505-test, baseline→merged top5, 동일 화자):")
c2i={c:i for i,c in enumerate(classes41)}
for c in ["우즈베키스탄","영국","미국","러시아","독일","스페인","프랑스","이탈리아","카자흐스탄"]:
    m=(dte=='505')&(yb==c2i[c])
    if m.sum(): print(f"    {c:<10} n={int(m.sum()):>5}  {hb5[m].mean():.3f} → {hm5[m].mean():.3f}  Δ={hm5[m].mean()-hb5[m].mean():+.3f}")
    else: print(f"    {c:<10} 505-test 없음")

# ===== (B) seed 노이즈 플로어 =====
print("\n"+"="*70+"\n(B) seed 노이즈 플로어 — merged, seed 3개, 클래스별 top5 std\n"+"="*70, flush=True)
seeds=[42,7,123]; perclass=defaultdict(list)
for sd in seeds:
    asg=assign_speakers(items,[8,1,1],seed=sd); spl=np.array([asg[s] for s in sps])
    tr=np.where(spl=="train")[0]; tee=np.where(spl=="test")[0]
    y,h1,h5,_=fit_eval(tr,tee,classes41)
    for c in set(y.tolist()):
        m=y==c; perclass[classes41[c]].append(h5[m].mean())
print(f"  seeds={seeds}")
print(f"  {'클래스':<12}{'top5(seed별)':<28}{'std':>7}")
for c in sorted(perclass, key=lambda c:-np.std(perclass[c]) if len(perclass[c])>1 else 0):
    v=perclass[c]
    if len(v)>=2:
        print(f"  {c:<12}{str([round(x,2) for x in v]):<28}{np.std(v):>7.3f}")
allstd=[np.std(v) for v in perclass.values() if len(v)>=3]
print(f"\n  >> 노이즈 플로어: 3-seed 모두 등장 클래스 top5 std 중앙값={np.median(allstd):.3f}, 최대={np.max(allstd):.3f}")

# ===== (C) cross-domain (20 공통 클래스) =====
print("\n"+"="*70+"\n(C) cross-domain — 505↔041 (20 공통 클래스, 한국어 제외)\n"+"="*70, flush=True)
euro20=sorted(set(lab[dom=='041'].tolist()))
i505=np.where(dom=='505')[0]; i041=np.where(dom=='041')[0]
i505=i505[np.array([lab[i] in euro20 for i in i505])];
rng=np.random.default_rng(0)
def half(idx):
    sp_=sorted(set(sps[idx])); rng.shuffle(sp_); cut=int(len(sp_)*0.8); trs=set(sp_[:cut])
    return idx[np.array([sps[i] in trs for i in idx])], idx[np.array([sps[i] not in trs for i in idx])]
tr505,te505=half(i505); tr041,te041=half(i041)
for name,tr,te in [("505→505",tr505,te505),("505→041",tr505,te041),("041→041",tr041,te041),("041→505",tr041,te505)]:
    y,h1,h5,_=fit_eval(tr,te,euro20)
    print(f"  {name}: top1={h1.mean():.4f} top5={h5.mean():.4f} (te n={len(y):,})")
print("\n[done]", flush=True)
