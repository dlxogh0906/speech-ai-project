# -*- coding: utf-8 -*-
"""공식 MDD 모델로 GOP를 뽑으면 우리 모델보다 나은가.

공식 MDD도 자모(45) CTC라 우리와 **같은 표준발음 열**에 강제정렬할 수 있다.
같은 오디오·같은 canonical·같은 정답(TAG 전문가 전사)으로 두 인식기를 비교한다.
공식은 언어권별 4개이므로 발화의 원 코퍼스에 맞춰 라우팅한다(우리는 1개로 전부 처리).

두 정의를 모두 낸다:
  gop_maxnorm  현행 (1등 대비)   ← 0에 붕괴하는 정의. 두 모델 모두 같은 결함을 갖는지 확인
  margin2      2등 격차          ← 채택 후보
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists():
    if _r.parent == _r: raise RuntimeError("저장소 루트(pyproject.toml)를 찾지 못했습니다")
    _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.fairseq_guard import ensure_fairseq
ensure_fairseq()
from shared.paths import DATA_ROOT, MDD_OUTPUTS

import sys, os, json, time, argparse, itertools, collections
import numpy as np
import torch
import torch.nn.functional as F
import edlib
from sklearn.metrics import roc_auc_score, average_precision_score

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", default=f"{MDD_OUTPUTS}/manifest_tag_all")
ap.add_argument("--ours", default=f"{DATA_ROOT}/mdd_ckpt_tag_all_lar/checkpoint_best.pt")
ap.add_argument("--official_dir", default=f"{DATA_ROOT}/mdd_official_ckpt")
ap.add_argument("--n", type=int, default=8000)
ap.add_argument("--out", default=f"{MDD_OUTPUTS}/gop_official_vs_ours.json")
A = ap.parse_args()

from mdd import gop_feats_v2 as GF
from mdd import mdd_eval; mdd_eval.CKPT = A.ours
from mdd.mdd_eval import Decoder, load_wav, batched
from mdd.mdd_eval_official import load_official

AA = f"{DATA_ROOT}/aihub/asian_korean_speech/043.교육용_아시아어(중·일어_제외)_사용자의_한국어_음성_데이터/01-1.정식개방데이터"
EE = f"{DATA_ROOT}/aihub/european_korean_speech/041.교육용_유럽어_모국어_사용자의_한국어_음성_데이터/01-1.정식개방데이터/Training/02.라벨링데이터/extracted"
CORPUS = [("eng", f"{DATA_ROOT}/aihub/040_labels/extracted/lab"),
          ("eu", f"{EE}/TL_pronunciation_lab"),
          ("sjr", f"{DATA_ROOT}/aihub/042_labels/extracted/lab"),
          ("asia", f"{AA}/Training/02.라벨링데이터/TL_pronunciation_lab"),
          ("asia", f"{AA}/Validation/02.라벨링데이터/VL_pronunciation_lab")]


def canon_labels(canon, other):
    if not other:
        return np.ones(len(canon), dtype=np.int8)
    r = edlib.align(other, canon, mode="NW", task="path")
    lab = np.zeros(len(canon), dtype=np.int8); ti = 0; n = 0
    for ch in r["cigar"]:
        if ch.isdigit(): n = n * 10 + int(ch); continue
        if ch == "=": ti += n
        elif ch in "XD": lab[ti:ti + n] = 1; ti += n
        n = 0
    return lab


@torch.inference_mode()
def lprobs(model, wavs, normalize):
    lens = [len(w) for w in wavs]; mx = max(lens)
    src = torch.zeros(len(wavs), mx); pm = torch.zeros(len(wavs), mx, dtype=torch.bool)
    for i, w in enumerate(wavs):
        t = torch.from_numpy(w)
        if normalize: t = F.layer_norm(t, t.shape)
        src[i, :len(w)] = t; pm[i, len(w):] = True
    net = model(source=src.half().cuda(), padding_mask=pm.cuda())
    lp = model.get_normalized_probs(net, log_probs=True).transpose(0, 1).float().cpu()
    opm = net.get("padding_mask")
    Ls = [int((~opm[b]).sum()) if opm is not None else lp.shape[1] for b in range(len(wavs))]
    return lp, Ls


def variants(l, y, blank):
    """강제정렬 후 음소별 두 정의. 정렬 실패 시 None."""
    segs = GF.viterbi_align(l, y, blank=blank)
    if segs is None: return None
    g = l.argmax(-1); srt = np.sort(l, axis=-1); second = srt[:, -2]; top = srt[:, -1]
    mx, m2 = [], []
    for si, fs in enumerate(segs):
        if not fs:
            mx.append(-15.0); m2.append(-15.0); continue
        fa = np.asarray(fs); py = l[fa, y[si]]
        mx.append(float((py - top[fa]).mean()))
        ref = np.where(g[fa] == y[si], second[fa], top[fa])
        m2.append(float((py - ref).mean()))
    return mx, m2


tsv = open(f"{A.manifest}/test.tsv").read().splitlines()
canon = [l.split() for l in open(f"{A.manifest}/test.canon", encoding="utf-8").read().splitlines()]
manual = [l.split() for l in open(f"{A.manifest}/test.ltr", encoding="utf-8").read().splitlines()]
langmap = {}
for lg, dr in CORPUS:
    if os.path.isdir(dr):
        for fn in os.listdir(dr):
            if fn.endswith(".json"): langmap.setdefault(fn[:-5], lg)

items = []
for i, line in enumerate(tsv[1:]):
    rel, fr = line.split("\t"); stem = os.path.basename(rel)[:-4]
    if canon[i] and manual[i] and stem in langmap:
        items.append((int(fr), i, "/" + rel, stem, langmap[stem]))
import random; random.seed(0); random.shuffle(items); items = items[: A.n]
bylang = collections.Counter(x[4] for x in items)
print(f"[표본] {len(items):,}발화 · 언어권 {dict(bylang)}\n")

# 정답(TAG)은 모델과 무관하므로 한 번만
GT = {}
for _, ci, _, stem, _ in items:
    GT[ci] = canon_labels("".join(canon[ci]), "".join(manual[ci]))

runs = {}

# ── 우리 모델 (1개로 전부) ──
dec = Decoder(); d = dec.d
acc = {"maxnorm": [], "margin2": [], "y": []}
t0 = time.time(); done = 0
for idxs in batched(items):
    wavs = [load_wav(items[j][2]) for j in idxs]
    lp, Ls = GF.forward_lprobs(dec, wavs)
    for k, j in enumerate(idxs):
        ci = items[j][1]
        y = [d.index(t) for t in canon[ci]]
        if d.unk() in y: continue
        v = variants(lp[k, :Ls[k]].numpy(), y, dec.blank)
        if v is None or len(GT[ci]) != len(y): continue
        acc["maxnorm"] += v[0]; acc["margin2"] += v[1]; acc["y"] += GT[ci].tolist()
    done += len(idxs)
    if done % 4000 < len(idxs):
        print(f"  [우리] {done:,}/{len(items):,} ({done/max(1e-9,time.time()-t0):.0f}/s)", flush=True)
runs["ours"] = acc
del dec
torch.cuda.empty_cache()

# ── 공식 모델 (언어권별 4개 라우팅) ──
acc = {"maxnorm": [], "margin2": [], "y": []}
for lg in sorted(bylang):
    sub = [x for x in items if x[4] == lg]
    model, dic, norm = load_official(f"{A.official_dir}/checkpoint_mdd_{lg}.pt")
    t0 = time.time(); done = 0; oov = 0
    for idxs in batched(sub):
        wavs = [load_wav(sub[j][2]) for j in idxs]
        lp, Ls = lprobs(model, wavs, norm)
        for k, j in enumerate(idxs):
            ci = sub[j][1]
            y = [dic.index(t) for t in canon[ci]]
            if dic.unk() in y:
                oov += 1; continue
            v = variants(lp[k, :Ls[k]].numpy(), y, 0)
            if v is None or len(GT[ci]) != len(y): continue
            acc["maxnorm"] += v[0]; acc["margin2"] += v[1]; acc["y"] += GT[ci].tolist()
        done += len(idxs)
    print(f"  [공식 {lg}] {len(sub):,}발화 · 사전밖 {oov}", flush=True)
    del model
    torch.cuda.empty_cache()
runs["official"] = acc

print(f"\n{'모델':10}{'정의':12}{'ROC-AUC':>10}{'PR-AUC':>9}{'최빈값 쏠림':>12}{'음소 위치':>11}")
res = {}
for name, label in (("ours", "우리(1개)"), ("official", "공식(4개)")):
    a = runs[name]; Y = np.asarray(a["y"], dtype=int)
    res[name] = {"n_pos": int(len(Y)), "err_rate": round(float(Y.mean()), 4)}
    for key in ("maxnorm", "margin2"):
        x = np.asarray(a[key], dtype=np.float32)
        auc = roc_auc_score(Y, -x); apr = average_precision_score(Y, -x)
        r2 = np.round(x, 2); _, c = np.unique(r2, return_counts=True)
        z = float(c.max() / len(r2))
        res[name][key] = {"auc": round(float(auc), 4), "pr_auc": round(float(apr), 4),
                          "mode_frac": round(z, 4)}
        print(f"{label:10}{key:12}{auc:>10.4f}{apr:>9.4f}{z:>12.1%}{len(Y):>11,}")
    print()
json.dump(res, open(A.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"[saved] {A.out}")
