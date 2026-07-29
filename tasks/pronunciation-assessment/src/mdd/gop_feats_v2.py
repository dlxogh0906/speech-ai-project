# -*- coding: utf-8 -*-
"""점수 3종용 확장 특징 추출 (v2) — 동결 MDD 모델의 부산물만 사용, 추가 학습 없음.

v1(gop_ctc.py)은 발화당 특징 6개(gop/nll/per/rate/blank_frac/max_blank)만 남겼다.
발음숙달 PCC가 0.561에서 막힌 원인이 "오류율 하나로 뭉뚱그린 것"이라 판단해,
사람 채점자가 실제로 보는 축을 분해한다:

  정확도 축  오류 유형별 비율(경음화·격음화·비음화·유음화·모음·탈락·첨가·초성·종성)
             + 오류 집중도(한 군데 몰림 vs 전체 분산)
  GOP 축     음소별 GOP의 분위수(p10/p25/중앙/최소) + 나쁜 음소 비율
             ← 평균만 보면 "몇 개가 심하게 나쁜" 경우를 놓친다
  유창성 축  쉼 개수·총 쉼 길이·평균 쉼 + 조음속도(쉼 제외 속도)

또한 **음소별 GOP 원값과 인식 자모열을 함께 저장**한다. 이후 특징을 더 만들고 싶을 때
GPU 재실행 없이 이 캐시만으로 CPU에서 파생할 수 있다.
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
from shared.paths import DATA_ROOT, MDD_CKPT, MDD_OUTPUTS

import sys, os, json, time, argparse, itertools
import numpy as np
import torch
import torch.nn.functional as F
import editdistance

# 인자 파싱은 main() 안에서 한다. 예전에는 모듈 최상단에서 parse_args() 를 불렀는데,
# 이 모듈을 import 하는 6개 스크립트가 자기 argv 를 먹히지 않게 하려고 매번
#     _argv = sys.argv[:]; sys.argv = [_argv[0]]; import gop_feats_v2; sys.argv = _argv
# 같은 우회를 해야 했다. import 부작용을 없애서 그 우회를 전부 제거했다.

from mdd import mdd_eval
# ⚠️ 여기서 mdd_eval.CKPT 를 건드리면 안 된다. 이 모듈을 import 하는 스크립트가
#    이미 자기 --ckpt 를 설정해 둔 경우 조용히 덮어써서 '다른 모델을 재고 맞다고
#    출력하는' 최악의 실패가 난다. 체크포인트는 main() 에서만 설정한다.


def _parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "valid", "test"])
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--ckpt", default=str(MDD_CKPT))
    ap.add_argument("--manifest", default=f"{MDD_OUTPUTS}/manifest_tag_all")
    ap.add_argument("--out", default=None)
    return ap.parse_args(argv)

from mdd.mdd_eval import Decoder, load_wav, batched
from mdd import pipeline as MD                      # _nw / ctype / VOWELS / JAMO

AA = f"{DATA_ROOT}/aihub/asian_korean_speech/043.교육용_아시아어(중·일어_제외)_사용자의_한국어_음성_데이터/01-1.정식개방데이터"
EE = f"{DATA_ROOT}/aihub/european_korean_speech/041.교육용_유럽어_모국어_사용자의_한국어_음성_데이터/01-1.정식개방데이터/Training/02.라벨링데이터/extracted"
LAB_DIRS = [f"{EE}/TL_pronunciation_lab",
            f"{AA}/Training/02.라벨링데이터/TL_pronunciation_lab",
            f"{AA}/Validation/02.라벨링데이터/VL_pronunciation_lab",
            f"{DATA_ROOT}/aihub/040_labels/extracted/lab",
            f"{DATA_ROOT}/aihub/042_labels/extracted/lab"]

ERR_TYPES = ["경음화", "격음화", "비음화", "유음화", "모음오류", "음운탈락", "음운첨가", "초성", "종성"]
# 강제정렬된 음소 구간 사이의 빈틈이 이 값 이상이면 '쉼'으로 본다.
# 0.15s는 CTC 정렬의 정상적인 음소 간격(프레임 20ms × 수 프레임)까지 쉼으로 세어버려 과대계상됐다 → 0.25s.
PAUSE_SEC = 0.25


@torch.inference_mode()
def forward_lprobs(dec, wavs):
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
    """CTC 강제정렬 → 라벨 토큰별 프레임 구간. 정렬 불가면 None."""
    T, L = lp.shape[0], len(y); S = 2 * L + 1
    ext = np.full(S, blank); ext[1::2] = y
    NEG = -1e30
    dp = np.full((T, S), NEG); bp = np.zeros((T, S), np.int32)
    dp[0, 0] = lp[0, blank]
    if S > 1: dp[0, 1] = lp[0, y[0]]
    for t in range(1, T):
        prev = dp[t - 1]
        d1 = np.concatenate(([NEG], prev[:-1]))
        d2 = np.concatenate(([NEG, NEG], prev[:-2]))
        ok = np.zeros(S, bool); ok[2:] = (ext[2:] != blank) & (ext[2:] != ext[:-2])
        cand = np.stack([prev, d1, np.where(ok, d2, NEG)])
        best = cand.argmax(0)
        dp[t] = cand[best, np.arange(S)] + lp[t, ext]; bp[t] = best
    last = S - 1 if S == 1 else (S - 1 if dp[T - 1, S - 1] >= dp[T - 1, S - 2] else S - 2)
    if dp[T - 1, last] <= NEG / 2:
        return None
    path = np.zeros(T, np.int32); s = last
    for t in range(T - 1, -1, -1):
        path[t] = s; s -= bp[t, s]
    segs = [[] for _ in range(L)]
    for t in range(T):
        if path[t] % 2 == 1:
            segs[path[t] // 2].append(t)
    return segs


def align_feats(canon, pred):
    """표준음 vs 인식열 정렬 → 오류 유형별 비율·집중도."""
    ops = MD._nw(canon, pred)
    n = len(canon)
    cnt = {k: 0 for k in ERR_TYPES}
    n_sub = n_del = n_ins = 0
    onset_e = onset_n = coda_e = coda_n = vow_e = vow_n = 0
    errpos = []
    for op in ops:
        k, tpos = op["kind"], op["tpos"]
        if op["kind"] != "ins" and tpos < n:
            ch = canon[tpos]
            if ch in MD.VOWELS:
                vow_n += 1
            elif ch == "ㅇ" or not (tpos + 1 < n and canon[tpos + 1] in MD.VOWELS):
                coda_n += 1
            else:
                onset_n += 1
        if k == "match":
            continue
        if k == "sub": n_sub += 1
        elif k == "del": n_del += 1
        else: n_ins += 1
        if k != "ins" and tpos < n:
            errpos.append(tpos)
            ch = canon[tpos]
            if ch in MD.VOWELS: vow_e += 1
            elif ch == "ㅇ" or not (tpos + 1 < n and canon[tpos + 1] in MD.VOWELS): coda_e += 1
            else: onset_e += 1
        is_coda = (op["kind"] != "ins" and tpos < n and canon[tpos] not in MD.VOWELS
                   and (canon[tpos] == "ㅇ" or not (tpos + 1 < n and canon[tpos + 1] in MD.VOWELS)))
        t = MD.ctype(op.get("target") or "", op.get("actual") or "", k, is_coda)
        if t in cnt: cnt[t] += 1
    # 집중도: 오류가 몇 덩어리로 뭉쳐 있나 (덩어리 수 / 오류 수). 1에 가까울수록 흩어짐
    errpos.sort()
    runs = 1 + sum(1 for a, b in zip(errpos, errpos[1:]) if b - a > 1) if errpos else 0
    f = {f"et_{k}": cnt[k] / max(1, n) for k in ERR_TYPES}
    f.update({
        "r_sub": n_sub / max(1, n), "r_del": n_del / max(1, n), "r_ins": n_ins / max(1, n),
        "r_onset": onset_e / max(1, onset_n), "r_coda": coda_e / max(1, coda_n),
        "r_vowel": vow_e / max(1, vow_n),
        "err_clump": runs / max(1, len(errpos)),
        "err_maxrun": max((len(list(g)) for _, g in itertools.groupby(
            range(n), key=lambda i: i in set(errpos))if _), default=0) / max(1, n),
    })
    return f


def main(argv=None):
    A = _parse_args(argv)
    OUT = A.out or f"{MDD_OUTPUTS}/gop_feats_v2_{A.split}.jsonl"
    mdd_eval.CKPT = A.ckpt
    labidx = {}
    for d in LAB_DIRS:
        if not os.path.isdir(d): continue
        for fn in os.listdir(d):
            if fn.endswith(".json"): labidx.setdefault(fn[:-5], os.path.join(d, fn))
    print(f"[lab] {len(labidx):,}", flush=True)

    tsv = open(f"{A.manifest}/{A.split}.tsv").read().splitlines()
    canon = [l.split() for l in open(f"{A.manifest}/{A.split}.canon", encoding="utf-8").read().splitlines()]
    items = []
    for i, line in enumerate(tsv[1:]):
        rel, fr = line.split("\t"); stem = os.path.basename(rel)[:-4]
        if stem in labidx and canon[i]:
            items.append((int(fr), i, "/" + rel, stem))
    import random; random.seed(0); random.shuffle(items); items = items[: A.n]
    print(f"[표본] {A.split} {len(items):,}", flush=True)

    dec = Decoder(); d = dec.d; tok2idx = {}
    fout = open(OUT, "w", encoding="utf-8")
    t0 = time.time(); done = 0; skipped = 0
    for idxs in batched(items):
        wavs = [load_wav(items[j][2]) for j in idxs]
        lp, Ls = forward_lprobs(dec, wavs)
        for k, j in enumerate(idxs):
            fr, ci, wp, stem = items[j]
            y_toks = canon[ci]; y = []
            for t_ in y_toks:
                if t_ not in tok2idx: tok2idx[t_] = d.index(t_)
                y.append(tok2idx[t_])
            if d.unk() in y:
                skipped += 1; continue
            l = lp[k, :Ls[k]].numpy()
            dur = fr / 16000.0; frame_sec = dur / max(1, l.shape[0])

            g = l.argmax(-1)
            pred_idx = [int(v) for v, _ in itertools.groupby(g) if v != dec.blank]
            pred = [d[v] for v in pred_idx]
            pred = [p for p in pred if p in MD.JAMO]
            per = editdistance.eval(pred_idx, y) / max(1, len(y))

            # ── 음소별 강제정렬 (GOP + 쉼 계산의 공통 기반) ──
            segs = viterbi_align(l, y, blank=dec.blank)
            if segs is None:
                gops = [-10.0] * len(y)
            else:
                gops = []
                for si, fs in enumerate(segs):
                    if not fs: gops.append(-10.0); continue
                    fa = np.asarray(fs)
                    gops.append(float((l[fa, y[si]] - l[fa].max(-1)).mean()))
            ga = np.asarray(gops)

            # ── 유창성 ──
            # 주의: CTC blank는 침묵이 아니라 음소 사이의 정상 출력이다.
            # blank run을 그대로 쉼으로 세면 조음속도가 수십 음소/초로 터진다.
            # → 강제정렬된 음소 구간 사이의 '빈틈'만 쉼으로 본다.
            blanks = (g == dec.blank)
            spans = [(fs[0] * frame_sec, (fs[-1] + 1) * frame_sec) for fs in (segs or []) if fs]
            if len(spans) >= 2:
                lead_sil = spans[0][0]
                trail_sil = max(0.0, dur - spans[-1][1])
                speech_span = max(0.1, spans[-1][1] - spans[0][0])
                gaps = [b0 - e0 for (_, e0), (b0, _) in zip(spans, spans[1:])]
                pauses = [gp for gp in gaps if gp >= PAUSE_SEC]
            else:
                lead_sil = trail_sil = 0.0
                speech_span = max(0.1, dur); pauses = []
            pause_tot = float(sum(pauses))
            speak = max(0.1, speech_span - pause_tot)      # 실제 조음 시간

            tl = torch.from_numpy(l).unsqueeze(1)
            nll = float(F.ctc_loss(tl, torch.tensor([y]), torch.tensor([l.shape[0]]),
                                   torch.tensor([len(y)]), blank=dec.blank,
                                   reduction="sum", zero_infinity=True)) / max(1, len(y))

            feat = {
                "gop": float(ga.mean()), "gop_p10": float(np.percentile(ga, 10)),
                "gop_p25": float(np.percentile(ga, 25)), "gop_med": float(np.median(ga)),
                "gop_min": float(ga.min()), "gop_std": float(ga.std()),
                "gop_bad": float((ga < -2.0).mean()),
                "neg_nll": -nll, "neg_per": -per,
                "dur": dur, "n_canon": len(y),
                "rate": len(y) / max(0.1, dur),              # 전체 길이 기준 속도
                "speech_rate": len(y) / speech_span,          # 발화 구간 기준 속도
                "artic_rate": len(y) / speak,                 # 쉼 제외 조음 속도
                "blank_frac": float(blanks.mean()),
                "lead_sil": lead_sil, "trail_sil": trail_sil, "speech_span": speech_span,
                "n_pause": len(pauses), "pause_tot": pause_tot,
                "pause_mean": float(np.mean(pauses)) if pauses else 0.0,
                "max_pause": float(max(pauses)) if pauses else 0.0,
                "pause_rate": len(pauses) / speech_span,
                "pause_frac": pause_tot / speech_span,
            }
            feat.update(align_feats(y_toks, pred))

            try:
                ev = json.load(open(labidx[stem], encoding="utf-8")).get("EvaluationMetadata", {})
            except Exception:
                skipped += 1; continue
            rec = {"stem": stem, **{k: round(v, 5) for k, v in feat.items()},
                   "pron": ev.get("PronunProfEval", 0), "comp": ev.get("ComprehendEval", 0),
                   "flu": ev.get("FluencyEval", 0),
                   "pred": "".join(pred), "gops": [round(x, 2) for x in gops]}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        done += len(idxs)
        if done % 8000 < len(idxs):
            print(f"  {done:,}/{len(items):,} ({done/max(1e-9,time.time()-t0):.0f}/s)", flush=True)
    fout.close()
    print(f"[done] {OUT} · skipped {skipped} · {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
