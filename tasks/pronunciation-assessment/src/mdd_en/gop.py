# -*- coding: utf-8 -*-
"""표준 음소마다 "얼마나 그 소리로 들렸나"(GOP) 특징 — CTC 음소 사후확률에서 바로 계산한다(Kaldi 불필요).

두 갈래를 같이 쓴다.
  ① 강제정렬(torchaudio forced_align): 표준 음소열을 프레임에 맞춘 뒤 그 구간의
     표준 음소 로그확률(lpp) · 다른 음소와의 차(margin) · 구간 안 다른 음소의 최고값(peak_other)
  ② 정렬 없는 CTC-GOP(Cao et al. 2024, "alignment-free GOP"): 음소열 전체 우도를
     "i 번째를 다른 음소로 바꾼 열"·"i 번째를 뺀 열" 과 비교한다
       lpr_sub = log P(표준열) − max_q log P(i→q 로 바꾼 열)
       lpr_del = log P(표준열) − log P(i 를 뺀 열)
       p_af    = P(표준열) / (P(표준열) + Σ_q P(바꾼 열) + P(뺀 열))   ← 이 자리가 표준대로일 사후확률
     어디서 무엇으로 바뀌었을 것 같은지(best_sub)도 같이 나온다.

입력 post 는 `PhoneRecognizer.phone_post` 의 (T, blank + 음소) 로그확률 — 0 번이 blank.
"""

import numpy as np

FEATS = ("lpp", "margin", "peak_other", "lpr_sub", "lpr_del", "p_af", "dur")


class AlignError(ValueError):
    """녹음이 문장보다 너무 짧아 표준 음소열을 프레임에 맞출 수 없다(CTC 는 음소마다 프레임이 1개 이상 필요)."""


def frames_needed(ids):
    """CTC 로 ids 를 내는 데 필요한 최소 프레임 수(같은 음소가 이어지면 사이에 blank 가 하나 더 필요)."""
    return len(ids) + sum(1 for a, b in zip(ids, ids[1:]) if a == b)


def _ctc_loglik(post, targets, budget_bytes=1 << 30):
    """여러 음소열의 CTC 로그우도. post (T, K) torch 로그확률 · targets [[id …], …] → np.array.

    한 번에 묶는 개수는 메모리 예산(기본 1 GB)으로 정한다 — 입력 복사본(T×K)과 CTC 표(T×(2L+1))가 열마다 든다.
    GPU 를 다른 서비스와 같이 쓰므로 긴 문장에서 메모리를 한꺼번에 잡지 않게.
    """
    import torch
    import torch.nn.functional as F
    T, K = post.shape[0], post.shape[1]
    L = max([len(t) for t in targets] or [1])
    max_batch = int(max(1, min(4096, budget_bytes // (T * (2 * L + 1 + K) * 4))))
    out = []
    for s in range(0, len(targets), max_batch):
        chunk = targets[s: s + max_batch]
        B = len(chunk)
        lp = post[:, None, :].expand(T, B, post.shape[1]).contiguous()
        lens = torch.tensor([len(t) for t in chunk], dtype=torch.long)
        flat = torch.tensor([i for t in chunk for i in t] or [1], dtype=torch.long)
        with torch.no_grad():
            nll = F.ctc_loss(lp, flat.to(post.device), torch.full((B,), T, dtype=torch.long), lens,
                             blank=0, reduction="none", zero_infinity=False)
        out.append((-nll).double().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def gop_features(post, canon_ids, alt_ids):
    """→ (특징 (N, len(FEATS)), best_sub (N,) — 가장 그럴듯한 대체 음소 id, 정렬 구간 [(시작, 끝)]).

    post: (T, K) torch 로그확률(0 = blank) · canon_ids: 표준 음소 id 목록 · alt_ids: 대체 후보 음소 id 목록
    """
    import torch
    from torchaudio.functional import forced_align
    N = len(canon_ids)
    F_ = np.zeros((N, len(FEATS)), np.float32)
    best = np.full(N, -1, np.int64)
    spans = [(0, 0)] * N
    if N == 0:
        return F_, best, spans
    T = post.shape[0]
    p32 = post.float()
    # ① 강제정렬 — 프레임이 모자라면 정렬도, 정렬 없는 GOP 도 성립하지 않는다(모든 우도가 −inf).
    #    그 상태의 특징(전부 0)을 점수기에 넣으면 "완벽한 발음"처럼 읽혀 엉뚱한 점수가 나왔다 → 오류로 돌려보낸다.
    if T < frames_needed(canon_ids):
        raise AlignError("frames %d < needed %d" % (T, frames_needed(canon_ids)))
    starts, ends = [None] * N, [None] * N
    from torchaudio.functional import merge_tokens
    tgt = torch.tensor([canon_ids], dtype=torch.int32, device=post.device)
    ali, sc = forced_align(p32[None], tgt, blank=0)
    toks = merge_tokens(ali[0], sc[0].exp())
    if len(toks) == N:
        for i, tk in enumerate(toks):
            starts[i], ends[i] = tk.start, tk.end
    P = p32.cpu().numpy()
    other = P.copy()
    other[:, 0] = -np.inf
    for i in range(N):
        s, e = starts[i], ends[i]
        if s is None:
            continue
        spans[i] = (s, e)
        c = canon_ids[i]
        seg = P[s:e, c]
        oth = other[s:e].copy()
        oth[:, c] = -np.inf
        # 앞뒤 blank 까지 포함한 넓은 구간에서 다른 음소가 튀었는지(문장 양 끝은 ±10 프레임 = 0.1초)
        lo = ends[i - 1] if i > 0 and ends[i - 1] is not None else max(0, s - 10)
        hi = starts[i + 1] if i + 1 < N and starts[i + 1] is not None else min(T, e + 10)
        wide = other[lo:max(hi, e)].copy()
        wide[:, c] = -np.inf
        F_[i, 0] = seg.mean()
        F_[i, 1] = (seg - oth.max(1)).mean()
        F_[i, 2] = wide.max()
        F_[i, 6] = e - s
    # ② 정렬 없는 GOP — 표준열 · (자리마다) 대체열 · 삭제열의 우도
    seqs = [list(canon_ids)]
    index = []
    for i in range(N):
        alts = [q for q in alt_ids if q != canon_ids[i]]
        for q in alts:
            seqs.append(canon_ids[:i] + [q] + canon_ids[i + 1:])
        seqs.append(canon_ids[:i] + canon_ids[i + 1:] if N > 1 else [canon_ids[i]])
        index.append(alts)
    ll = _ctc_loglik(p32, seqs)
    ll = np.where(np.isfinite(ll), ll, -1e4)
    L0, k = ll[0], 1
    for i in range(N):
        n = len(index[i])
        sub, dele = ll[k: k + n], ll[k + n]
        k += n + 1
        if N == 1:
            dele = -1e4
        j = int(np.argmax(sub))
        best[i] = index[i][j]
        F_[i, 3] = L0 - sub[j]
        F_[i, 4] = L0 - dele
        allv = np.concatenate([[L0], sub, [dele]])
        m = allv.max()
        F_[i, 5] = float(np.exp(L0 - m) / np.exp(allv - m).sum())
    return F_, best, spans


def choose_variants(post, word_variants, to_ids):
    """단어마다 CMUdict 발음 후보가 여럿이면(the: DH AH / DH IY, to: T UW / T AH …) 소리에 가장 맞는 것을 고른다.

    word_variants: [[후보 음소열, …], …] (첫 번째가 기본) · to_ids: 음소열 → 모델 id 열
    다른 단어는 기본 발음으로 둔 채 한 단어씩 바꿔 CTC 우도를 비교한다(한 번에 묶어 계산).
    → 단어마다 고른 후보 번호
    """
    base = [to_ids(v[0]) for v in word_variants]
    seqs, owner = [], []
    for w, vs in enumerate(word_variants):
        if len(vs) < 2:
            continue
        for k, v in enumerate(vs):
            seqs.append([i for j, b in enumerate(base) for i in (to_ids(v) if j == w else b)])
            owner.append((w, k))
    pick = [0] * len(word_variants)
    if not seqs:
        return pick
    ll = _ctc_loglik(post.float(), seqs)
    best = {}
    for (w, k), v in zip(owner, ll):
        if w not in best or v > best[w][1]:
            best[w] = (k, v)
    for w, (k, _) in best.items():
        pick[w] = k
    return pick
