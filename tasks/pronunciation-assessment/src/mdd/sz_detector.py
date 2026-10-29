# -*- coding: utf-8 -*-
"""ㅅ/ㅆ · ㅈ/ㅉ/ㅊ 첫소리 세기 오류 판별기 — 받아쓰기(CTC)가 놓치는 오류를 소리 단서로 다시 본다(2026-10-22).

배경: 10월 22일 특정 평가 표본의 오류 보존 부족을 조사하며 만든 후보 판별기다.
뒤 모음 초기의 주기·배음 특징을 사용하지만, 추정값 반환이 실제 발성이나 음질 측정의 타당성을 보장하지 않는다.
10월 24일 후속 조사에서는 기존 모델도 모음 초기 변화에 반응했고, 짧은 분석 창의 결측·이웃 음소 침범과
문맥 의존성이 확인됐다. 초기 관측만으로 "모델이 이 단서를 듣지 못한다"고 해석하지 않는다.
현재 근거와 한계: docs/ㅅㅆ_구분_개선_탐색_2026-10-23.md 5절.

자리마다 특징(실행 중에 얻을 수 있는 것만 — NPU 모델은 로그확률만 낸다):
  - CTC: 그 자리 구간에서 계열 자모(ㅅㅆ / ㅈㅉㅊ)의 최대 로그확률을 표준음 대비 차로
  - 소리: 자음 구간 길이 · 쉿 소리 주파수 중심 · 쉿 소리 세기(모음 대비) · 모음 시작 F0(발화 기준 반음) · H1-H2 · F0 못 잼 표시
  - 맥락: 앞이 쉼/문장 처음 · 모음 · 받침
구간은 강제정렬(`gop_feats_v2.viterbi_align`) 프레임으로 정한다(학습도 같은 방식).

판별기 파일(json): 표준음마다 {classes, mean, scale, coef, intercept, threshold} — 다항 로지스틱 회귀.
`annotate(sites, ...)` 가 sites[i]["sz"] = {"alt": 자모, "p": 확률} 을 붙이고, `speechsuper.build` 가 SZ_RULE 이 켜져 있으면
p ≥ threshold 인 자리를 지적한다.
"""
import json
import math

import numpy as np

SR = 16000
SERIES = {"ㅅ": "ㅅㅆ", "ㅆ": "ㅅㅆ", "ㅈ": "ㅈㅉㅊ", "ㅉ": "ㅈㅉㅊ", "ㅊ": "ㅈㅉㅊ"}
VOWELS = set("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
FEATS = ["d1", "d2", "dur", "cent", "frel", "f0st", "h1h2", "f0miss", "ctx_start", "ctx_coda"]
GFEATS = ["g1", "g2"]      # 정렬 없는 비교 점수(2026-10-22, arXiv 2506.02080 식) — 그 자리만 바꾼 문장 전체 CTC 우도 − 표준음 문장 우도
ACOUSTIC = {"dur", "cent", "frel", "f0st", "h1h2", "f0miss"}       # 녹음 파형에서 재는 것(계산이 무겁다)


def yin(x, fmin=70.0, fmax=450.0, th=0.15):
    n = len(x)
    tmax, tmin = int(SR / fmin), int(SR / fmax)
    if n < tmax * 2:
        return None
    # A flat waveform has zero difference at every lag. Without this guard,
    # CMND becomes zero everywhere and the first lag masquerades as a pitch.
    # This is an exact invalid-input check, not a quiet-voice energy threshold.
    if not np.isfinite(x).all() or np.all(x == x[0]):
        return None
    x = x - x.mean()
    w = n - tmax
    d = np.array([np.sum((x[:w] - x[t:t + w]) ** 2) for t in range(tmax + 1)])
    cm = np.ones_like(d)
    cs = np.cumsum(d[1:])
    cm[1:] = d[1:] * np.arange(1, tmax + 1) / np.maximum(cs, 1e-12)
    for t in range(max(tmin, 1), tmax):
        if cm[t] < th and cm[t] <= cm[t + 1]:
            a, b, c = cm[t - 1], cm[t], cm[t + 1]
            den = a - 2 * b + c
            tt = t + (0.5 * (a - c) / den if den else 0)
            return SR / tt
    return None


def h1h2(x, f0):
    if f0 is None or len(x) < 256:
        return None
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)), 8192))
    fr = np.fft.rfftfreq(8192, 1 / SR)

    def amp(f):
        sel = (fr > f * 0.85) & (fr < f * 1.15)
        return 20 * np.log10(spec[sel].max() + 1e-9) if sel.any() else None
    a1, a2 = amp(f0), amp(2 * f0)
    return None if a1 is None or a2 is None else a1 - a2


def rms_db(x):
    return float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9)) if len(x) else None


def centroid(x):
    if len(x) < 128:
        return None
    seg = x * np.hanning(len(x))
    p = np.abs(np.fft.rfft(seg, 1024)) ** 2
    f = np.fft.rfftfreq(1024, 1 / SR)
    sel = (f >= 1000) & (f <= 7900)
    return float((f[sel] * p[sel]).sum() / max(p[sel].sum(), 1e-20)) / 1000


def target_sites(canon):
    """판별 대상 — ㅅㅆㅈㅉㅊ 첫소리(뒤가 모음)."""
    return [q for q, c in enumerate(canon) if c in SERIES and q + 1 < len(canon) and canon[q + 1] in VOWELS]


def ll_ratios(l, canon, to_id, blank, qs):
    """자리 q 마다 계열의 다른 자모 m 으로 바꾼 표준음 전체의 CTC 로그우도 − 표준음 로그우도 → {q: {m: 값}}.
    강제정렬 경계에 기대지 않는다(경계가 틀려도 덜 흔들린다). 한 번의 묶음 CTC 계산으로 다 잰다."""
    import torch
    import torch.nn.functional as F
    base = [to_id(t) for t in canon]
    seqs, keys = [base], []
    for q in qs:
        for m in SERIES[canon[q]]:
            if m != canon[q]:
                s2 = list(base); s2[q] = to_id(m); seqs.append(s2); keys.append((q, m))
    if not keys:
        return {}
    lp = torch.as_tensor(l).float()
    T, V = int(lp.shape[0]), int(lp.shape[1])
    N = len(seqs)
    tg = torch.tensor([x for s2 in seqs for x in s2], dtype=torch.long)
    lens = torch.tensor([len(s2) for s2 in seqs], dtype=torch.long)
    with torch.no_grad():
        nll = F.ctc_loss(lp.unsqueeze(1).expand(T, N, V).contiguous(), tg, torch.full((N,), T, dtype=torch.long), lens,
                         blank=blank, reduction="none", zero_infinity=True)
    ll = (-nll).tolist()
    out = {}
    for (q, m), v in zip(keys, ll[1:]):
        out.setdefault(q, {})[m] = float(v - ll[0])
    return out


def utt_f0_ref(wav, segs, canon, frame_sec):
    """발화 기준 음높이 — 모음 토큰 가운데 40 ms 의 F0 중앙값."""
    f = []
    for q, tok in enumerate(canon):
        if tok in VOWELS and q < len(segs) and segs[q]:
            m = (segs[q][0] + segs[q][-1] + 1) / 2 * frame_sec
            v = yin(wav[int((m - 0.02) * SR): int((m + 0.02) * SR)])
            if v:
                f.append(v)
    return float(np.median(f)) if len(f) >= 2 else None


def site_features(wav, l, segs, canon, q, to_id, frame_sec, f0_ref, acoustic=True, g=None):
    """표준음 q 번째(ㅅㅆㅈㅉㅊ 첫소리, 뒤가 모음) 자리의 특징 dict. 정렬이 없으면 None.
    acoustic=False 면 파형 특징은 계산하지 않고 0 으로 둔다(판별기가 안 쓸 때)."""
    c = canon[q]
    if c not in SERIES or q + 1 >= len(canon) or canon[q + 1] not in VOWELS or q >= len(segs) or not segs[q] or not segs[q + 1]:
        return None
    T = l.shape[0]
    t0, t1 = segs[q][0], segs[q][-1]
    a, b = max(0, t0 - 1), min(T, t1 + 3)
    lp = {m: float(l[a:b, to_id(m)].max()) for m in SERIES[c]}
    others = [m for m in SERIES[c] if m != c]
    feats = {"d1": lp[others[0]] - lp[c], "d2": (lp[others[1]] - lp[c]) if len(others) > 1 else 0.0}
    g = g or {}
    feats["g1"] = g.get(others[0], 0.0)
    feats["g2"] = g.get(others[1], 0.0) if len(others) > 1 else 0.0
    prev_end = segs[q - 1][-1] + 1 if (q > 0 and q - 1 < len(segs) and segs[q - 1]) else max(0, t0 - 4)
    cs = prev_end * frame_sec
    vs = segs[q + 1][0] * frame_sec
    nxt = segs[q + 2][0] if (q + 2 < len(segs) and segs[q + 2]) else segs[q + 1][-1] + 3
    ve = max(vs + 0.04, nxt * frame_sec)
    dur = min(0.3, max(0.0, vs - cs))
    prev = canon[q - 1] if q > 0 else None
    gap = (segs[q][0] - (segs[q - 1][-1] if (q > 0 and segs[q - 1]) else -99)) * frame_sec
    feats["ctx_start"] = 1.0 if (prev is None or gap > 0.3) else 0.0
    feats["ctx_coda"] = 1.0 if (prev is not None and prev not in VOWELS and feats["ctx_start"] == 0.0) else 0.0
    if not acoustic:
        feats.update({k: 0.0 for k in ACOUSTIC})
        return feats
    seg = wav[int((cs + 0.2 * dur) * SR): int((vs - 0.2 * dur) * SR)]
    vow = wav[int(vs * SR): int(ve * SR)]
    von = wav[int(vs * SR): int((vs + 0.04) * SR)]
    f0 = yin(von)
    feats["dur"] = dur * 1000
    cen = centroid(seg)
    feats["cent"] = cen if cen is not None else 4.0
    fd, vd = rms_db(seg), rms_db(vow)
    feats["frel"] = (fd - vd) if (fd is not None and vd is not None) else -15.0
    feats["f0st"] = 12 * math.log2(f0 / f0_ref) if (f0 and f0_ref) else 0.0
    hh = h1h2(von, f0)
    feats["h1h2"] = hh if hh is not None else 0.0
    feats["f0miss"] = 0.0 if (f0 and f0_ref and hh is not None) else 1.0
    return feats


class Detector:
    def __init__(self, path):
        self.m = json.load(open(path, encoding="utf-8"))
        self.acoustic = any(k in ACOUSTIC for d in self.m.values() for k in d.get("feats", FEATS))
        self.needs_g = any(k in GFEATS for d in self.m.values() for k in d.get("feats", FEATS))

    def predict(self, c, feats):
        """→ (가장 그럴듯한 다른 자모, 그 확률). 판별기가 없는 표준음이면 None."""
        d = self.m.get(c)
        if not d:
            return None
        x = (np.array([feats[k] for k in d.get("feats", FEATS)]) - np.array(d["mean"])) / np.array(d["scale"])
        z = np.array(d["coef"]) @ x + np.array(d["intercept"])
        if len(d["classes"]) == 2 and z.ndim == 1 and len(z) == 1:      # 이항: 두 번째 클래스 확률
            p1 = 1 / (1 + math.exp(-z[0]))
            probs = {d["classes"][0]: 1 - p1, d["classes"][1]: p1}
        else:
            e = np.exp(z - z.max())
            probs = dict(zip(d["classes"], e / e.sum()))
        alt = max((k for k in probs if k != c), key=lambda k: probs[k])
        return alt, float(probs[alt])

    def threshold(self, c):
        d = self.m.get(c)
        return d.get("threshold") if d else None


def annotate(sites, wav, l, segs, canon, to_id, frame_sec, det, blank=0):
    """sites(표준음 순서) 에 판별 결과를 붙인다. wav 는 로그확률을 낸 바로 그(무음 다듬은) 녹음."""
    if det is None or segs is None:
        return sites
    f0_ref = utt_f0_ref(wav, segs, canon, frame_sec) if det.acoustic else None
    G = ll_ratios(l, canon, to_id, blank, target_sites(canon)) if det.needs_g else {}
    by_i = {s["i"]: s for s in sites if s.get("i") is not None}
    for q, c in enumerate(canon):
        if c not in SERIES or q not in by_i:
            continue
        f = site_features(wav, l, segs, canon, q, to_id, frame_sec, f0_ref, acoustic=det.acoustic, g=G.get(q))
        if f is None:
            continue
        r = det.predict(c, f)
        if r:
            by_i[q]["sz"] = {"alt": r[0], "p": round(r[1], 3), "th": det.threshold(c)}
    return sites
