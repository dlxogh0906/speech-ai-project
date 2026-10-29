# -*- coding: utf-8 -*-
"""긴 오디오(장면 클립)·사용자 녹음에서 대사 한 줄이 어디 있는지 찾는다.

정렬 모델로 받아 적은(greedy CTC) 자모열에서, 대사 자모열과 편집 거리가 가장 작은 부분열을 찾는다
(근사 부분 문자열 매칭 — 대사는 전부, 받아 적은 쪽은 아무 데서 시작·끝나도 된다). 자모 오류율(JER)이
그 구간이 대사인지의 확신이다. 긴 오디오는 겹치는 조각으로 받아 적어 시간순으로 잇는다.
사용자 녹음엔 locate_line(대사 자리 · 군말 · 음절별 못 받아 적은 비율) → trim_bounds(군말 잘라 내기)를 쓴다.
"""
import numpy as np

from .align import FRAME, OFFSET


def greedy(lp, blank):
    """log 확률 [T, V] → [(토큰 id, 프레임)] (연속 중복·blank 제거)."""
    ids = np.asarray(lp.argmax(-1))
    out, prev = [], -1
    for t, k in enumerate(ids.tolist()):
        if k != prev and k != blank:
            out.append((k, t))
        prev = k
    return out


def transcribe(aligner, wav, sr=16000, chunk=30.0, overlap=4.0):
    """→ [(토큰 id, 시각 초)]. chunk 초씩 overlap 겹쳐 돌리고, 각 조각은 겹침 절반 안쪽만 쓴다."""
    n = len(wav)
    step = int((chunk - overlap) * sr)
    size = int(chunk * sr)
    out = []
    a = 0
    while True:
        seg = wav[a: a + size]
        lp = aligner.log_probs(seg)
        lo = 0.0 if a == 0 else overlap / 2.0
        hi = len(seg) / sr if a + size >= n else chunk - overlap / 2.0
        for k, f in greedy(lp, aligner.blank):
            t = f * FRAME + OFFSET
            if lo <= t < hi:
                out.append((k, a / sr + t))
        if a + size >= n:
            break
        a += step
    return out


def find(hyp, target, last=False):
    """hyp=[(id, 시각)] 안에서 target=[id] 와 가장 가까운 부분 → (시작 시각, 끝 시각, JER, (i0, i1)).

    Sellers 근사 부분 문자열 매칭: E[0][j]=0(어디서 시작해도 됨), 답은 min_j E[m][j](어디서 끝나도 됨).
    last: 같은 거리면 가장 늦게 끝나는 것 — 사용자가 대사를 두 번 말하면 마지막 것을 본다(장면에서 찾을 땐 False)."""
    H = [k for k, _ in hyp]
    m, n = len(target), len(H)
    if not m or not n:
        return None
    Ha = np.array(H)
    ar = np.arange(n + 1)
    E = np.zeros((m + 1, n + 1))
    E[1:, 0] = np.arange(1, m + 1)
    for i in range(1, m + 1):
        diag = E[i - 1, :-1] + (Ha != target[i - 1])                    # 맞춤/바꿈
        tmp = np.concatenate([[float(i)], np.minimum(diag, E[i - 1, 1:] + 1)])   # 대사 쪽 빠짐 +1
        E[i] = np.minimum.accumulate(tmp - ar) + ar                     # 받아 적은 쪽 끼어듦 +1 (누적 최소)
    row = E[m, 1:]
    ties = np.flatnonzero(row == row.min())
    # 같은 거리면 늦게 끝나는 쪽 — 끝 모음을 다른 모음으로 받아 적어도 포함하게. 단 가장 이른 끝에서 0.25초 안까지만
    # (쉬고 나온 다음 말 "에?" 까지 대사로 끌어오지 않게)
    t_first = hyp[int(ties[0])][1]
    j = int(ties[-1] if last else max(k for k in ties if hyp[int(k)][1] <= t_first + 0.25)) + 1
    best = float(E[m, j])
    i, jj = m, j
    while i > 0 and jj > 0:                                             # 되짚어 시작 위치
        if E[i, jj] == E[i - 1, jj - 1] + (H[jj - 1] != target[i - 1]):
            i, jj = i - 1, jj - 1
        elif E[i, jj] == E[i - 1, jj] + 1:
            i -= 1
        else:
            jj -= 1
    j0 = jj
    return hyp[j0][1], hyp[j - 1][1], best / m, (j0, j - 1)


def align_ops(want, got):
    """편집 거리 정렬 → want 토큰마다 'm'(맞음)·'s'(바뀜)·'d'(못 받아 적음). 되짚을 땐 맞춤/바꿈을 빠짐보다 먼저 고른다."""
    n, m = len(want), len(got)
    D = np.zeros((n + 1, m + 1), dtype=int)
    D[:, 0], D[0, :] = np.arange(n + 1), np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + (want[i - 1] != got[j - 1]))
    st, i, j = ["d"] * n, n, m
    while i > 0:
        if j > 0 and D[i, j] == D[i - 1, j - 1] + (want[i - 1] != got[j - 1]):
            st[i - 1] = "m" if want[i - 1] == got[j - 1] else "s"
            i, j = i - 1, j - 1
        elif D[i, j] == D[i - 1, j] + 1:
            i -= 1
        else:
            j -= 1
    return st


def locate_line(lp, target, owner, n_syl, blank):
    """녹음 받아 적기에서 대사 자리를 찾는다 — 앞뒤 군말은 세지 않는다.
    target·owner: 대사 토큰 id 와 그 토큰의 음절 번호(Aligner.token_ids). → dict(
      jer    대사와 가장 가까운 부분의 JER(compare 의 대사 확인에 쓴다) · heard 받아 적은 토큰 수 ÷ 대사 토큰 수
      span   (대사 첫 토큰 시각, 끝 토큰 시각) · before/after 그 앞·뒤에 받아 적힌 토큰 시각들(군말)
      absent 음절별로 받아 적기에서 못 받은 토큰 비율 — 빠진 어절 판정(compare.missing_syllables)에 쓴다)"""
    hyp = [(k, f * FRAME + OFFSET) for k, f in greedy(lp, blank)]
    heard = len(hyp) / float(max(1, len(target)))
    r = find(hyp, target, last=True)
    if r is None:
        return dict(jer=1.0, heard=heard, span=None, before=[], after=[], absent=[1.0] * n_syl)
    t0, t1, d, (j0, j1) = r
    st = align_ops(target, [k for k, _ in hyp[j0:j1 + 1]])
    absent = []
    for si in range(n_syl):
        q = [st[x] == "d" for x in range(len(target)) if owner[x] == si]
        absent.append(round(float(np.mean(q)), 3) if q else 0.0)
    return dict(jer=float(d), heard=heard, span=(t0, t1), before=[t for _, t in hyp[:j0]], after=[t for _, t in hyp[j1 + 1:]],
                absent=absent)


TRIM_LEAD = 0.1       # 대사 첫 토큰 앞으로 이만큼은 남긴다 — 첫 자음은 토큰보다 먼저 시작한다
TRIM_TAIL = 0.12      # 대사 끝 토큰 뒤로 이만큼은 남긴다 — 끝 모음이 이어진다
TRIM_DIP = 0.25       # 자르는 자리의 세기가 대사 구간 중앙값의 이 배(−12 dB) 아래여야 자른다 — 이어 말한 소리는 안 자른다


def trim_bounds(wav, loc, sr=16000):
    """대사 앞뒤에 다른 말(받아 적힌 토큰)이 있으면 그 사이 가장 조용한 곳에서 자른다 → (시작, 끝) 초.
    다른 말이 없거나, 사이에 조용한 틈이 없으면 그쪽은 자르지 않는다(끝을 길게 끄는 성대모사의 끝 모음을 지키려고)."""
    x = np.asarray(wav, dtype=float)
    dur = len(x) / float(sr)
    if loc.get("span") is None or not (loc["before"] or loc["after"]):
        return 0.0, dur
    hop, win = int(0.01 * sr), int(0.03 * sr)
    nf = max(1, 1 + (len(x) - win) // hop)
    env = np.sqrt(np.array([np.mean(x[k * hop:k * hop + win] ** 2) for k in range(nf)]) + 1e-12)
    tc = (np.arange(nf) * hop + win / 2.0) / sr
    t0, t1 = loc["span"]
    inside = env[(tc >= t0) & (tc <= t1)]
    floor = TRIM_DIP * float(np.median(inside)) if len(inside) else 0.0

    def quiet(lo, hi):
        m = (tc >= lo) & (tc <= hi)
        if hi - lo < 0.02 or not m.any():
            return None
        k = int(np.argmin(np.where(m, env, np.inf)))
        return float(tc[k]) if env[k] <= floor else None
    a = quiet(loc["before"][-1] + 0.03, t0 - TRIM_LEAD) if loc["before"] else None
    b = quiet(t1 + TRIM_TAIL, loc["after"][0] - TRIM_LEAD) if loc["after"] else None
    return (a if a is not None else 0.0), (b if b is not None else dur)


def line_check(lp, target, blank):
    """녹음 전체를 받아 적어 대사와 맞춰 본다 → (JER, 받아 적은 토큰 수 ÷ 대사 토큰 수).

    다른 대사·잡음만 든 녹음을 음절 지적 전에 거른다(compare.LINE_JER_MAX). 강제 정렬은 무엇이 들어오든 대사에 끼워 맞추므로
    이게 없으면 다른 대사를 녹음해도 음절 지적이 나온다(results/08: 42%)."""
    hyp = [k for k, _ in greedy(lp, blank)]
    return jer(target, hyp), len(hyp) / float(max(1, len(target)))


def jer(want, got):
    """자모 오류율 = 편집 거리 / 대사 길이."""
    d = list(range(len(got) + 1))
    for i, x in enumerate(want, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(got, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
    return d[len(got)] / max(1, len(want))
