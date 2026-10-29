# -*- coding: utf-8 -*-
"""녹음 하나 → 음절별 특징: 경계·길이·앞 쉼·음높이 수준/변화·세기·정렬 확신.

경계 잡는 법:
  1. CTC 정렬 토큰을 앵커로 쓴다. 음절 i 의 앵커 = 첫 토큰 시작 ~ 마지막 토큰 끝.
     첫 토큰(초성, 없으면 모음)은 음절 시작 근처에서 나오고, 앞 음절의 마지막 토큰(모음·종성)은
     그 모음 앞쪽에서 나온다 — 그래서 두 앵커의 가운데는 앞 음절 모음 한복판이다(쓰면 안 된다).
  2. 앞 음절 앵커 끝 ~ 이 음절 앵커 시작 사이에 PAUSE_MIN 이상 무음이 있으면 쉼 → 경계를 쉼 양끝에 둔다.
     없으면 이 음절 첫 토큰 바로 앞(DIP_BEFORE ~ DIP_AFTER)에서 모음 대역 에너지 골짜기(DIP_DB 이상 깊을 때),
     아니면 첫 토큰 시작 − LEAD_DEFAULT.
  3. 첫 음절 시작·마지막 음절 끝은 앵커를 감싸는 유성(무음 아님) 구간 끝까지 넓힌다
     — 대사 끝을 길게 끄는 성대모사에서 마지막 모음이 잘리지 않게.
음높이는 화자 기준(녹음 전체 유성 프레임 중앙값)에서 뺀 반음이다. 부분 재녹음이면 `baseline_hz` 로
전체 녹음의 기준을 넘긴다(조각 자체 중앙값으로 정규화하면 기준이 흔들린다).
음절 음높이(높이·끝 억양)는 이웃 유성 프레임 사이 JUMP_MAX 넘게 튀는 곳이 있으면 끊은 가장 긴 조각에서 잰다 — 긁는 소리·
발성 시작의 튐이 "끝이 내려갔어요(−7.1)"를 만들었다(results/08). 녹음 전체를 받아 적어 대사와 맞춰 본 JER 도 남긴다
(compare 의 대사 확인).
"""
import numpy as np

from . import prosody
from .align import FRAME, OFFSET
from .locate import line_check
from .text import align_tokens, parse_line

PAUSE_MIN = 0.15      # 이보다 긴 무음은 쉼 — 음절 경계를 쉼 양끝에 둔다
SIL_RUN_MIN = 0.03    # 이보다 짧은 무음 조각은 무시
DIP_DB = 3.0          # 에너지 골짜기로 경계를 옮기는 최소 깊이
DIP_BEFORE = 0.08     # 골짜기를 찾는 창: 첫 토큰 시작 − 80 ms ~ + 20 ms
DIP_AFTER = 0.02
LEAD_DEFAULT = 0.02   # 골짜기가 없을 때 경계 = 첫 토큰 시작 − 20 ms
EDGE = 0.01           # 음절 양끝에서 음높이 계산에 안 쓰는 폭
LEAD_MAX = 0.12       # 첫 앵커 앞으로 넓힐 수 있는 최대 폭 — 0.25 는 말 앞 숨소리까지 첫 음절에 넣었다
BOUND_TOL = 0.05      # crosscheck: 강제 정렬 경계와 DTW 투영 경계가 이 안에서 맞아야 그 음절을 믿는다
TAIL_MAX = 2.0        # 마지막 앵커 뒤로 넓힐 수 있는 최대 폭
INT_MIN_PITCH = 100.0 # 세기 창(3.2/100 = 32 ms) — 원본·녹음에 같은 값을 써야 비교가 된다
JUMP_MAX = 3.0        # 이웃 유성 프레임(10 ms) 사이 이보다 큰 음높이 차는 소리 결 탓 — 사람 억양은 10 ms 에 1반음도 못 움직인다
SMOOTH_MIN = 0.7      # 튀는 곳에서 끊은 가장 긴 조각이 유성 프레임의 이 비율은 돼야 그 음절 음높이를 잰다
PAUSE_CTC = 0.02      # 무음 문턱을 못 넘는 쉼(잡음·잔향·자동 음량이 채운 것): 정렬 모델이 대사 소리(공백 아닌 토큰)를 ±40 ms 안에서
PAUSE_QUIET_DB = 6.0  # 이 확률도 안 주고 말소리 세기 중앙값보다 이만큼 낮은 구간이 — 끝을 길게 끄는 모음은 세서 안 걸린다
PAUSE_CTC_MIN = 0.2   # 앵커 사이에서 이만큼 이어지면 쉼
PAUSE_FILL = 0.04     # 이보다 짧은 끊김(웅성거림 조각)은 이어 붙인다
NOISE_MIN = 0.15      # 말소리 밖 구간이 이만큼은 있어야 잡음 바닥을 잰다(없으면 신호대잡음비 모름)


def _interp(t, x, q):
    return float(np.interp(q, t, x))


def _nan_to_none(a):
    return [None if (v is None or not np.isfinite(v)) else round(float(v), 3) for v in a]


def smooth_run(vs, jump=JUMP_MAX):
    """유성 프레임 값들을 이웃 차가 jump 넘는 곳에서 끊어 가장 긴 조각."""
    if len(vs) < 2:
        return vs
    cut = np.flatnonzero(np.abs(np.diff(vs)) > jump) + 1
    return max(np.split(vs, cut), key=len)


def pitch_stats(t_p, st, s, e):
    """음절 [s, e] 의 음높이 → (높이, 끝 억양, 유성 프레임 수, 프레임 수). 양끝 EDGE 는 안 쓴다.
    높이 = 중앙값(4개 이상일 때), 끝 억양 = 끝 1/3 중앙값 − 앞 1/3 중앙값(6개 이상일 때). 튀는 곳이 없으면 유성 프레임
    전체에서, 있으면 끊은 가장 긴 조각에서 잰다 — 그 조각이 유성 프레임의 SMOOTH_MIN 이 안 되면 둘 다 못 잼(None)."""
    m = (t_p >= s + EDGE) & (t_p <= e - EDGE)
    mv = m & ~np.isnan(st)
    vs = st[mv]
    run = smooth_run(vs)
    if len(run) < SMOOTH_MIN * len(vs):
        run = run[:0]                                                # 어느 쪽이 진짜 음높이인지 모른다
    level = float(np.median(run)) if len(run) >= 4 else None
    delta = None
    if len(run) >= 6:
        k = len(run) // 3
        delta = float(np.median(run[-k:]) - np.median(run[:k]))     # 끝 1/3 − 앞 1/3
    return level, delta, int(mv.sum()), int(m.sum())


def baseline_st(t_p, st_abs, speech):
    """화자 기준 음높이(반음) = 말소리 구간 유성 프레임 중앙값. 5개 미만이면 0."""
    v = (~np.isnan(st_abs)) & (t_p >= speech[0]) & (t_p <= speech[1])
    return float(np.median(st_abs[v])) if v.sum() >= 5 else 0.0


def analyze(wav, text, aligner, sr=16000, baseline_hz=None, lp=None, keep_tracks=True):
    words, syls, word_of = parse_line(text)
    toks, tok_src = align_tokens(words, syls)
    if lp is None:
        lp = aligner.log_probs(wav)
    al = aligner.align(wav, toks, lp=lp)
    line_jer, line_heard = line_check(lp, aligner.token_ids(toks)[0], aligner.blank)
    n = len(syls)
    anchors = []
    for i in range(n):
        ts = [a for a in al if a["syl"] == i]
        anchors.append((ts[0]["start"], ts[-1]["end"], float(np.mean([a["prob"] for a in ts]))))

    t_p, f0, prng = prosody.pitch_track(wav, sr)
    st_abs = prosody.clean_octaves(prosody.semitones(f0))
    t_i, db = prosody.intensity_track(wav, sr, min_pitch=INT_MIN_PITCH)
    t_b, eb = prosody.band_energy(wav, sr)
    sil, thr = prosody.silence_mask(db)
    hop = prosody.HOP
    sil_runs = [(t_i[a] - hop / 2, t_i[b - 1] + hop / 2) for a, b in prosody.runs(sil) if (b - a) * hop >= SIL_RUN_MIN]
    voc_runs = [(t_i[a] - hop / 2, t_i[b - 1] + hop / 2) for a, b in prosody.runs(~sil) if (b - a) * hop >= SIL_RUN_MIN]
    speech_db = float(np.median(db[~sil])) if (~sil).any() else float(np.max(db))
    L = lp.numpy() if hasattr(lp, "numpy") else np.asarray(lp)
    nb = 1.0 - np.exp(L[:, aligner.blank])                          # 정렬 모델이 대사 소리라고 보는 확률
    act = np.max(np.stack([np.roll(nb, d) for d in (-2, -1, 0, 1, 2)]), axis=0)      # ±40 ms 안 가장 큰 값
    quiet = (np.interp(t_i, np.arange(len(act)) * FRAME + OFFSET, act) < PAUSE_CTC) & (db < speech_db - PAUSE_QUIET_DB)
    for a, b in prosody.runs(~quiet):                                # 짧은 끊김은 메운다
        if (b - a) * hop < PAUSE_FILL and a > 0 and b < len(quiet):
            quiet[a:b] = True
    ctc_runs = [(t_i[a] - hop / 2, t_i[b - 1] + hop / 2) for a, b in prosody.runs(quiet) if (b - a) * hop >= PAUSE_CTC_MIN]

    # 말 시작·끝
    first, last = anchors[0][0], anchors[-1][1]
    s0 = first
    for a, b in voc_runs:
        if b >= first - 0.05:
            s0 = max(first - LEAD_MAX, min(first, a))
            break
    s1 = last
    for a, b in reversed(voc_runs):
        if a <= last + 0.05:
            s1 = min(last + TAIL_MAX, max(last, b), len(wav) / sr)
            break

    starts, ends, pauses = [s0] + [None] * (n - 1), [None] * (n - 1) + [s1], [0.0] * n
    for i in range(1, n):
        a, b = anchors[i - 1][1], anchors[i][0]
        best = (0.0, None, None)
        for ra, rb in sil_runs:
            oa, ob = max(ra, a), min(rb, b)
            if ob - oa >= SIL_RUN_MIN and ob - oa > best[0]:
                best = (ob - oa, oa, ob)
        if best[0] < PAUSE_MIN:                                     # 무음 문턱을 못 넘은 쉼(잡음이 채운 것)
            for ra, rb in ctc_runs:
                oa, ob = max(ra, a), min(rb, b)
                if ob - oa >= PAUSE_CTC_MIN and ob - oa > best[0]:
                    best = (ob - oa, oa, ob)
        pauses[i] = best[0]
        if best[0] >= PAUSE_MIN:
            ends[i - 1], starts[i] = best[1], best[2]
            continue
        lo = max(a, b - DIP_BEFORE)
        hi = max(lo, b + DIP_AFTER)
        bd = max(lo, b - LEAD_DEFAULT)
        m = (t_b >= lo) & (t_b <= hi)
        if m.any():
            k = int(np.argmin(eb[m]))
            edge = max(_interp(t_b, eb, lo), _interp(t_b, eb, hi))
            if edge - eb[m][k] >= DIP_DB:
                bd = float(t_b[m][k])
        ends[i - 1] = starts[i] = bd

    out_sp = (t_i < s0 - 0.05) | (t_i > s1 + 0.1)                    # 말소리 밖 = 잡음 바닥
    snr = speech_db - float(np.median(db[out_sp])) if out_sp.sum() * hop >= NOISE_MIN else None

    if baseline_hz is None:
        base_st = baseline_st(t_p, st_abs, (s0, s1))
        baseline_hz = float(100.0 * 2 ** (base_st / 12.0))
    else:
        base_st = float(prosody.semitones(baseline_hz))
    st = st_abs - base_st

    out = []
    for i in range(n):
        s, e = starts[i], ends[i]
        level, delta, nv, nf = pitch_stats(t_p, st, s, e)
        mi = (t_i >= s) & (t_i <= e)
        energy = float(np.percentile(db[mi], 80)) if mi.any() else None
        out.append(dict(i=i, text=syls[i], word=word_of[i], start=round(s, 4), end=round(e, 4),
                        dur=round(e - s, 4), pause_before=round(pauses[i], 4), level=level, delta=delta,
                        energy=energy, voiced=round(float(nv) / max(1, nf), 3),
                        n_voiced=nv, prob=round(anchors[i][2], 4),
                        anchor=(round(anchors[i][0], 4), round(anchors[i][1], 4))))

    res = dict(text=text, words=words, duration=round(len(wav) / sr, 4), speech=(round(s0, 4), round(s1, 4)),
               token_source=tok_src, baseline_hz=round(baseline_hz, 2), pitch_range=prng,
               silence_db=round(thr, 2), snr_db=None if snr is None else round(snr, 1), syllables=out, tokens=al,
               line_jer=round(float(line_jer), 3), line_heard=round(float(line_heard), 3))
    if keep_tracks:
        res["tracks"] = dict(t_pitch=[round(float(v), 3) for v in t_p], st=_nan_to_none(st),
                             t_int=[round(float(v), 3) for v in t_i], db=[round(float(v), 2) for v in db])
    return res


def line_ids(text, aligner):
    """대사 → (토큰 id 들, 토큰마다 음절 번호, 음절 수) — 녹음에서 대사 자리 찾기(locate.locate_line)용."""
    words, syls, _ = parse_line(text)
    toks, _ = align_tokens(words, syls)
    ids, owner, _ = aligner.token_ids(toks)
    return ids, owner, len(syls)


def bounds(res):
    S = res["syllables"]
    return [s["start"] for s in S] + [S[-1]["end"]]


def keep_spans(res):
    """말소리 구간을 쉼(PAUSE_MIN 이상)에서 끊은 시간 구간들."""
    S = res["syllables"]
    spans, a = [], res["speech"][0]
    for i in range(1, len(S)):
        if S[i]["start"] - S[i - 1]["end"] >= PAUSE_MIN - 1e-6:
            spans.append((a, S[i - 1]["end"]))
            a = S[i]["start"]
    spans.append((a, res["speech"][1]))
    return spans


def crosscheck(wav_ref, ref, wav_take, take, sr=16000, tol=BOUND_TOL, feats=None, hop=None, off=None):
    """녹음 경계를 두 방법으로 잡아 맞춰 본다 — 녹음 강제 정렬(take) vs 원본 경계의 DTW 투영.

    둘은 다르게 틀린다: 정렬은 잡음·음색 왜곡에서 토큰을 한쪽에 몰아 넣거나 늘인 음절의 시작을 늦게 잡고,
    DTW 는 화자가 다르면 경로가 틀어질 수 있다. 음절의 시작·끝 경계가 모두 `tol` 안에서 맞는 음절만
    `bound_ok` — compare 는 나머지 음절을 지적하지 않는다(확신 없으면 지적하지 않는다).
    경계는 점이 아니라 구간으로 맞춘다: 쉼이 있으면 (앞 음절 끝, 이 음절 시작) 구간이고, 두 구간이 겹치면 오차 0.
    특징 DTW 는 양쪽 쉼을 빼고 말소리끼리만 맞춘다(dtw.project_keep).
    feats=(원본 특징, 녹음 특징)·hop·off 를 주면 그 특징(정렬 모델 은닉층)으로, 안 주면 MFCC 로 DTW 한다.
    take 를 고쳐 쓰고(음절마다 dtw·bound_ok, 전체 bound_agree) 일치율을 돌려준다."""
    from . import dtw
    R, S = ref["syllables"], take["syllables"]
    n = len(R)
    pts = [R[0]["start"]] + [v for i in range(1, n) for v in (R[i - 1]["end"], R[i]["start"])] + [R[-1]["end"]]
    if feats is not None:                    # 쉼을 뺀 말소리끼리 맞춘다 — 원본의 긴 쉼을 안 따라 해도 경계가 안 끌려간다
        iv = dtw.project_keep(feats[0], feats[1], pts, hop, off, keep_spans(ref), keep_spans(take), "cos")
    else:
        iv = [(p, p) for p in dtw.project(wav_ref, wav_take, pts, ref["speech"], take["speech"], sr=sr)]
    d_iv = [iv[0]] + [(min(iv[2 * i - 1][0], iv[2 * i][0]), max(iv[2 * i - 1][1], iv[2 * i][1])) for i in range(1, n)] + \
        [iv[-1]]
    c_iv = [(S[0]["start"], S[0]["start"])] + [(S[i - 1]["end"], S[i]["start"]) for i in range(1, n)] + \
        [(S[-1]["end"], S[-1]["end"])]

    def gap(a, b):
        return max(0.0, max(min(a), min(b)) - min(max(a), max(b)))
    err = [gap(a, b) for a, b in zip(c_iv, d_iv)]
    for i, s in enumerate(S):
        s["dtw"] = (round(float(d_iv[i][1]), 4), round(float(d_iv[i + 1][0]), 4))
        s["bound_ok"] = bool(err[i] <= tol and err[i + 1] <= tol)
    take["bound_agree"] = round(float(np.mean(np.array(err) <= tol)), 4)
    return take["bound_agree"]
