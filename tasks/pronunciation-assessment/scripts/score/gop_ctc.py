# -*- coding: utf-8 -*-
"""GOP(Goodness of Pronunciation) — 학습 없이 기존 MDD CTC 모델의 확신도로 발음 점수 산출.

변형 3종을 같은 발화에서 계산해 사람 점수(EvaluationMetadata)와의 PCC로 비교:
  A. gop_post : canonical을 CTC Viterbi 강제정렬 → 음소별 [정답 로그확률 − 최고 로그확률] 평균 (고전 GOP)
  B. nll_norm : canonical의 CTC loss / 길이 (정렬 자신도) 부호 반전
  C. neg_per  : 그리디 디코드와 canonical의 음소 오류율 부호 반전
정답: lab JSON의 PronunProfEval / ComprehendEval / FluencyEval (0=미평가 제외)
표본: TAG test(화자분리)에서 무작위 N — 점수 팩 목표 PCC ≥ 0.60 과 비교.
"""

# ── 저장소 부트스트랩 (설치 불필요) ──
import sys as _sys, pathlib as _pl
_r = _pl.Path(__file__).resolve()
while not (_r / "pyproject.toml").exists(): _r = _r.parent
_sys.path[:0] = [str(_r), *(str(p) for p in sorted(_r.glob("tasks/*/src")))]
from shared.fairseq_guard import ensure_fairseq
ensure_fairseq()
from shared.paths import DATA_ROOT, MDD_OUTPUTS

import sys, os, json, glob, random, time, argparse
import numpy as np
import torch
import torch.nn.functional as F
import editdistance

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=20000)
ap.add_argument("--ckpt", default=f"{DATA_ROOT}/mdd_ckpt_tag_all_lar/checkpoint_best.pt")
ap.add_argument("--manifest", default=f"{MDD_OUTPUTS}/manifest_tag_all")
ap.add_argument("--split", default="test", choices=["train", "valid", "test"])
ap.add_argument("--out", default=f"{MDD_OUTPUTS}/eval_gop.json")
ap.add_argument("--cache", default=f"{MDD_OUTPUTS}/gop_scores.jsonl")
A = ap.parse_args()

AA = f"{DATA_ROOT}/aihub/asian_korean_speech/043.교육용_아시아어(중·일어_제외)_사용자의_한국어_음성_데이터/01-1.정식개방데이터"
EE = f"{DATA_ROOT}/aihub/european_korean_speech/041.교육용_유럽어_모국어_사용자의_한국어_음성_데이터/01-1.정식개방데이터/Training/02.라벨링데이터/extracted"
LAB_DIRS = [f"{EE}/TL_pronunciation_lab",
            f"{AA}/Training/02.라벨링데이터/TL_pronunciation_lab",
            f"{AA}/Validation/02.라벨링데이터/VL_pronunciation_lab",
            f"{DATA_ROOT}/aihub/040_labels/extracted/lab",
            f"{DATA_ROOT}/aihub/042_labels/extracted/lab"]

from mdd import mdd_eval; mdd_eval.CKPT = A.ckpt
from mdd.mdd_eval import Decoder, load_wav, batched


@torch.inference_mode()
def forward_lprobs(dec, wavs):
    """디코더와 동일 forward, 전체 프레임 로그확률 (B,T,V) 와 유효길이 반환."""
    lens = [len(w) for w in wavs]; mx = max(lens)
    src = torch.zeros(len(wavs), mx); pm = torch.zeros(len(wavs), mx, dtype=torch.bool)
    for i, w in enumerate(wavs):
        t = torch.from_numpy(w)
        if dec.normalize:
            t = F.layer_norm(t, t.shape)
        src[i, :len(w)] = t; pm[i, len(w):] = True
    net = dec.model(source=src.half().cuda(), padding_mask=pm.cuda())
    lp = dec.model.get_normalized_probs(net, log_probs=True).transpose(0, 1).float().cpu()
    opm = net.get("padding_mask")
    Ls = [(int((~opm[b]).sum()) if opm is not None else lp.shape[1]) for b in range(len(wavs))]
    return lp, Ls


def viterbi_align(lp, y, blank=0):
    """CTC Viterbi 강제정렬. lp:(T,V) numpy, y: 라벨 idx 목록.
    반환: 프레임별 상태(2L+1, 짝수=blank) → 토큰별 프레임 구간. 정렬 불가면 None."""
    T = lp.shape[0]; L = len(y)
    S = 2 * L + 1
    ext = np.full(S, blank); ext[1::2] = y
    NEG = -1e30
    dp = np.full((T, S), NEG); bp = np.zeros((T, S), np.int32)
    dp[0, 0] = lp[0, blank]
    if S > 1: dp[0, 1] = lp[0, y[0]]
    for t in range(1, T):
        prev = dp[t - 1]
        stay = prev
        d1 = np.concatenate(([NEG], prev[:-1]))
        d2 = np.concatenate(([NEG, NEG], prev[:-2]))
        skip_ok = np.zeros(S, bool)
        skip_ok[2:] = (ext[2:] != blank) & (ext[2:] != ext[:-2])
        d2 = np.where(skip_ok, d2, NEG)
        cand = np.stack([stay, d1, d2])
        best = cand.argmax(0)
        dp[t] = cand[best, np.arange(S)] + lp[t, ext]
        bp[t] = best
    last = S - 1 if S == 1 else (S - 1 if dp[T - 1, S - 1] >= dp[T - 1, S - 2] else S - 2)
    if dp[T - 1, last] <= NEG / 2:
        return None
    path = np.zeros(T, np.int32); s = last
    for t in range(T - 1, -1, -1):
        path[t] = s
        s -= bp[t, s]
    segs = [[] for _ in range(L)]
    for t in range(T):
        if path[t] % 2 == 1:
            segs[path[t] // 2].append(t)
    return segs


def main():
    labidx = {}
    for d in LAB_DIRS:
        for f in os.listdir(d):
            if f.endswith(".json"):
                labidx.setdefault(f[:-5], os.path.join(d, f))
    print(f"[lab] {len(labidx):,} json", flush=True)

    tsv = open(f"{A.manifest}/{A.split}.tsv").read().splitlines()
    canon = [l.split() for l in open(f"{A.manifest}/{A.split}.canon", encoding="utf-8").read().splitlines()]
    items = []
    for i, line in enumerate(tsv[1:]):
        rel, fr = line.split("\t")
        stem = os.path.basename(rel)[:-4]
        if stem in labidx and canon[i]:
            items.append((int(fr), i, "/" + rel, stem))
    random.seed(0); random.shuffle(items)
    items = items[: A.n]
    print(f"[표본] {A.split}에서 라벨 매칭 {len(items):,}", flush=True)

    dec = Decoder()
    d = dec.d
    tok2idx = {}
    scores = []
    t0 = time.time(); done = 0
    fout = open(A.cache, "w", encoding="utf-8")
    for idxs in batched(items):
        wavs = [load_wav(items[j][2]) for j in idxs]
        lp, Ls = forward_lprobs(dec, wavs)
        for k, j in enumerate(idxs):
            fr, ci, wp, stem = items[j]
            y_toks = canon[ci]
            y = []
            ok = True
            for t_ in y_toks:
                if t_ not in tok2idx:
                    tok2idx[t_] = d.index(t_)
                yi = tok2idx[t_]
                if yi == d.unk():
                    ok = False; break
                y.append(yi)
            if not ok:
                continue
            l = lp[k, :Ls[k]].numpy()
            # C. greedy PER + 유창성 특징 (쉼 = blank 런, 말속도)
            g = l.argmax(-1)
            gtoks = [int(v) for v, _ in __import__('itertools').groupby(g) if v != dec.blank]
            per = editdistance.eval(gtoks, y) / max(1, len(y))
            dur = fr / 16000.0
            frame_sec = dur / max(1, l.shape[0])
            blank_frac = float((g == dec.blank).mean())
            mbr = cur = 0
            for v in g:
                cur = cur + 1 if v == dec.blank else 0
                mbr = max(mbr, cur)
            max_blank_sec = mbr * frame_sec
            rate = len(y) / max(0.1, dur)
            # B. CTC nll / len
            tl = torch.from_numpy(l).unsqueeze(1)
            nll = float(F.ctc_loss(tl, torch.tensor([y]), torch.tensor([l.shape[0]]),
                                   torch.tensor([len(y)]), blank=dec.blank, reduction="sum",
                                   zero_infinity=True)) / len(y)
            # A. 강제정렬 GOP
            segs = viterbi_align(l, y, blank=dec.blank)
            if segs is None:
                gop = -20.0
            else:
                per_tok = []
                for si, fset in enumerate(segs):
                    if not fset:
                        per_tok.append(-10.0); continue
                    fs = np.asarray(fset)
                    per_tok.append(float((l[fs, y[si]] - l[fs].max(-1)).mean()))
                gop = float(np.mean(per_tok))
            try:
                lab = json.load(open(labidx[stem], encoding="utf-8"))
                ev = lab.get("EvaluationMetadata", {})
            except Exception:
                continue
            rec = {"stem": stem, "gop": round(gop, 4), "neg_nll": round(-nll, 4),
                   "neg_per": round(-per, 4), "dur": round(dur, 2), "rate": round(rate, 3),
                   "blank_frac": round(blank_frac, 4), "max_blank": round(max_blank_sec, 3),
                   "pron": ev.get("PronunProfEval", 0), "comp": ev.get("ComprehendEval", 0),
                   "flu": ev.get("FluencyEval", 0)}
            scores.append(rec)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        done += len(idxs)
        if done % 4000 < len(idxs):
            print(f"  {done:,}/{len(items):,} ({done/max(1e-9,time.time()-t0):.0f}/s)", flush=True)
    fout.close()

    def pcc(xs, ys):
        if len(xs) < 10: return None
        return round(float(np.corrcoef(xs, ys)[0, 1]), 4)

    res = {"n": len(scores), "ckpt": A.ckpt, "pcc": {}}
    for dim in ("pron", "comp", "flu"):
        sub = [s for s in scores if s[dim] and s[dim] > 0]
        res["pcc"][dim] = {"n": len(sub)}
        for v in ("gop", "neg_nll", "neg_per"):
            res["pcc"][dim][v] = pcc([s[v] for s in sub], [s[dim] for s in sub])
    json.dump(res, open(A.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print(f"\n========== GOP vs 사람 점수 PCC (test 화자분리, n={len(scores):,}) ==========")
    print(f"{'차원':8}{'n':>8}{'GOP(정렬)':>12}{'−NLL':>10}{'−PER':>10}")
    for dim, name in (("pron", "발음숙달"), ("comp", "이해가능"), ("flu", "유창성")):
        r = res["pcc"][dim]
        print(f"{name:8}{r['n']:>8,}{str(r['gop']):>12}{str(r['neg_nll']):>10}{str(r['neg_per']):>10}")
    print(f"(점수 팩 목표 PCC ≥ 0.60 · 팩 저자 보고 0.7228)")
    print(f"[saved] {A.out}")


if __name__ == "__main__":
    main()
