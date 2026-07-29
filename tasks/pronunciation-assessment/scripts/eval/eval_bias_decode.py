# -*- coding: utf-8 -*-
"""디코딩 편향 조정 실험: CTC 내부 LM(라벨 사전확률) 차감이 오류 검출을 살리는가.

가설: 교정형 FN에서 정답이 2등에 78% 대기 → argmax 전에 각 클래스 logit에서
      α·log P(class) 를 빼면(=사전확률 페널티) 2등이 1등으로 뒤집힌다.
판정: F1 상승 → 경계 문제(재학습 불필요 가능성) / 정밀도 붕괴 → 표현 문제(재학습 필요)

α 스윕을 test 부분표본에서 수행, 각 α의 weighted F1 / 오류클래스 P·R·F1 보고.
사전확률 = 학습 라벨 빈도(dict.ltr.txt의 카운트) 정규화.
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists(): _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.fairseq_guard import ensure_fairseq
ensure_fairseq()
from shared.paths import DATA_ROOT, MDD_OUTPUTS

import sys, os, json, argparse, time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from mdd import mdd_eval
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=f"{DATA_ROOT}/mdd_ckpt_tag_all_native/checkpoint_best.pt")
ap.add_argument("--manifest", default=f"{MDD_OUTPUTS}/manifest_tag_all")
ap.add_argument("--n", type=int, default=20000, help="부분표본 크기")
ap.add_argument("--alphas", default="0,0.1,0.2,0.3,0.5,0.7,1.0")
ap.add_argument("--out", default=f"{MDD_OUTPUTS}/eval_bias_sweep.json")
A = ap.parse_args()
mdd_eval.CKPT = A.ckpt
from mdd.mdd_eval import Decoder, load_wav
from mdd.mdd_reproduce_official import align, fixup

M = A.manifest
rel = [l.split("\t") for l in open(f"{M}/test.tsv").read().splitlines()[1:]]
canon = [l.split() for l in open(f"{M}/test.canon", encoding="utf-8").read().splitlines()]
manual = [l.split() for l in open(f"{M}/test.ltr", encoding="utf-8").read().splitlines()]
import random
idx = list(range(len(rel))); random.seed(0); random.shuffle(idx); idx = idx[:A.n]

dec = Decoder()
d = dec.d
# 사전확률: 학습 라벨 빈도 (dict.ltr.txt: "심볼 카운트")
prior = np.zeros(len(d))
for line in open(f"{M}/dict.ltr.txt", encoding="utf-8"):
    sym, cnt = line.split()
    prior[d.index(sym)] = float(cnt)
prior = prior / prior.sum()
log_prior = torch.tensor(np.log(np.clip(prior, 1e-12, None)), dtype=torch.float32)
# blank·특수토큰은 페널티 없음 (0)
special = set(range(d.nspecial)) | {dec.blank}
for s in special:
    log_prior[s] = 0.0

alphas = [float(x) for x in A.alphas.split(",")]

@torch.inference_mode()
def decode_lprobs(wavs):
    lens = [len(w) for w in wavs]; mx = max(lens)
    src = torch.zeros(len(wavs), mx); pm = torch.zeros(len(wavs), mx, dtype=torch.bool)
    for i, w in enumerate(wavs):
        t = torch.from_numpy(w)
        if dec.normalize:
            t = F.layer_norm(t, t.shape)
        src[i, :len(w)] = t; pm[i, len(w):] = True
    net = dec.model(source=src.half().cuda(), padding_mask=pm.cuda())
    lprobs = dec.model.get_normalized_probs(net, log_probs=True).transpose(0, 1).float().cpu()
    opm = net.get("padding_mask")
    Ls = []
    for b in range(len(wavs)):
        Ls.append(int((~opm[b]).sum()) if opm is not None else lprobs.shape[1])
    return lprobs, Ls

def greedy(lp, L, alpha):
    x = lp[:L] - alpha * log_prior.unsqueeze(0)
    toks = x.argmax(-1).unique_consecutive()
    return [d[i] for i in toks.tolist() if i != dec.blank and i >= d.nspecial or (i != dec.blank and d[i] not in ('<s>','</s>','<pad>','<unk>'))]

# 배치 디코드 (프레임 lprobs 1회 계산, α별 argmax 재사용)
res = {a: {"X": [], "Y": []} for a in alphas}
B = 24
t0 = time.time()
for bi in range(0, len(idx), B):
    batch = idx[bi:bi+B]
    wavs = [load_wav("/" + rel[i][0]) for i in batch]
    lprobs, Ls = decode_lprobs(wavs)
    for j, i in enumerate(batch):
        C, Mn = canon[i], manual[i]
        if not C or not Mn: continue
        for a in alphas:
            P = greedy(lprobs[j], Ls[j], a)
            n1, n2, n3 = fixup(*align("".join(C), "".join(Mn), "".join(P) or "-"))
            L = min(len(n1), len(n2), len(n3))
            res[a]["X"] += [int(n1[k] != n2[k]) for k in range(L)]
            res[a]["Y"] += [int(n1[k] != n3[k]) for k in range(L)]
    if bi % (B*40) == 0:
        print(f"  {bi:,}/{len(idx):,} ({bi/max(1e-9,time.time()-t0):.0f}/s)", flush=True)

out = {}
print(f"\n=== 디코딩 편향 스윕 (n={A.n:,}, ckpt={os.path.basename(os.path.dirname(A.ckpt))}) ===")
print(f"{'α':>5} | {'weighted F1':>11} | {'오류 P':>7} | {'오류 R':>7} | {'오류 F1':>7}")
for a in alphas:
    X = np.array(res[a]["X"]); Y = np.array(res[a]["Y"])
    wf1 = f1_score(X, Y, average="weighted")
    tp = int(((X==1)&(Y==1)).sum()); fp = int(((X==0)&(Y==1)).sum()); fn = int(((X==1)&(Y==0)).sum())
    p = tp/max(1,tp+fp); r = tp/max(1,tp+fn); f = 2*p*r/max(1e-9,p+r)
    out[str(a)] = {"weighted_f1": round(wf1,4), "err_p": round(p,4), "err_r": round(r,4), "err_f1": round(f,4)}
    print(f"{a:>5} | {wf1:>11.4f} | {p:>7.4f} | {r:>7.4f} | {f:>7.4f}")
json.dump(out, open(A.out,"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"[saved] {A.out}")
