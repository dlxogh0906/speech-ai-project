# -*- coding: utf-8 -*-
"""표준발음 자리별 음향 증거 — 학습(`scripts/data/extract_phone_feats.py`)과 서빙(`mdd.phone_detector`)이 **같은 코드**를 쓴다.

특징 정의가 두 곳에서 조금이라도 어긋나면 에러 없이 판별 정확도만 떨어진다(학습·서빙 불일치). 그래서 여기 한 곳에만 둔다.

    forward_anchor   음향 기준 모델(원어민 Stage 1) forward → 로그사후확률 · 마지막 층 · 중간 층 은닉
    align_featurize  표준발음 강제정렬 → 자리별 [스칼라 7 · 스파이크/구간 평균 로그사후확률 · 구간 평균 은닉 2종 · 프레임 구간]

프레임은 20 ms(50 Hz). 구간(span)은 이웃 토큰 중심과의 중간점까지 넓힌 것이다 — CTC 스파이크는 1~2 프레임뿐이라
그대로는 은닉 벡터를 평균 낼 구간이 너무 좁다.
"""
import numpy as np

N_SCAL = 7
FRAME_SEC = 0.02


def groups_from_path(path, blank):
    """정렬 경로(프레임별 토큰 id) → 토큰별 [시작, 끝) 프레임. 같은 기호가 이어지면 한 토큰이다(사이에 blank 가 있으면 다른 토큰)."""
    out, prev = [], blank
    for t, s in enumerate(path):
        if s != blank and s != prev:
            out.append([t, t + 1])
        elif s != blank:
            out[-1][1] = t + 1
        prev = s
    return out


def forward_anchor(model, xs, normalize, mid_layer, torch, F):
    """xs: float32 파형 리스트(16 kHz). 반환 (lp_all[B,T,V] float32, h[B,T,C], hm[B,T,C]|None, 유효 프레임 수 리스트)."""
    q = next(model.parameters())
    dev = q.device
    w2v = model.w2v_encoder
    mx = max(len(x) for x in xs)
    src = torch.zeros(len(xs), mx); pm = torch.zeros(len(xs), mx, dtype=torch.bool)
    for j, x in enumerate(xs):
        t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        if normalize:
            t = F.layer_norm(t, t.shape)
        src[j, :len(x)] = t; pm[j, len(x):] = True
    res = w2v.w2v_model.extract_features(source=src.to(dev, q.dtype), padding_mask=pm.to(dev), mask=False)
    h = res["x"]                                                    # B,T,C (마지막 층)
    opm = res["padding_mask"]
    lp_all = F.log_softmax(w2v.proj(h).float(), -1)
    hm = res["layer_results"][mid_layer - 1][0].transpose(0, 1) if mid_layer else None      # T,B,C → B,T,C
    Ts = [h.shape[1] if opm is None else int((~opm[j]).sum()) for j in range(len(xs))]
    return lp_all, h, hm, Ts


def align_featurize(lp, hj, hmj, y, blank, torch, torchaudio):
    """lp[T,V] · hj[T,C] · hmj[T,C]|None · y: 표준발음 토큰 id 리스트. 정렬 불가면 None."""
    dev = lp.device
    T, V = lp.shape
    L = len(y)
    try:
        path, _ = torchaudio.functional.forced_align(lp[None], torch.tensor([y], device=dev), blank=blank)
        g = groups_from_path(path[0].tolist(), blank)
        if len(g) != L:
            return None
    except RuntimeError:
        return None
    C = hj.shape[1]
    cen = [(u + v - 1) / 2.0 for u, v in g]
    bnd = [max(0, g[0][0] - 3)] + [int(round((cen[t] + cen[t + 1]) / 2 + 0.5)) for t in range(L - 1)] + [min(T, g[-1][1] + 4)]
    cs_lp = torch.cat([lp.new_zeros(1, V), lp.cumsum(0)])
    hj = hj.float(); cs_h = torch.cat([hj.new_zeros(1, C), hj.cumsum(0)])
    gs = torch.tensor([u for u, _ in g], device=dev); ge = torch.tensor([v for _, v in g], device=dev)
    sa = torch.tensor(bnd[:-1], device=dev); sb = torch.tensor(bnd[1:], device=dev)
    sb = torch.maximum(sb, sa + 1).clamp(max=T); sa = torch.minimum(sa, sb - 1)
    m_spike = (cs_lp[ge] - cs_lp[gs]) / (ge - gs)[:, None]
    m_span = (cs_lp[sb] - cs_lp[sa]) / (sb - sa)[:, None]
    yt = torch.tensor(y, device=dev); ar = torch.arange(L, device=dev)
    canon_lp = m_spike[ar, yt]
    comp = m_spike.clone(); comp[ar, yt] = -1e4
    mx_canon = torch.stack([lp[int(u):int(v), k].max() for u, v, k in zip(sa.tolist(), sb.tolist(), y)])
    pnb = m_spike.exp(); pnb[:, blank] = 0; pnb = pnb / pnb.sum(-1, keepdim=True).clamp_min(1e-8)
    ent = -(pnb * pnb.clamp_min(1e-8).log()).sum(-1)
    scal = torch.stack([canon_lp, canon_lp - comp.max(-1).values, mx_canon, m_span[:, blank],
                        (ge - gs).float().log(), (sb - sa).float().log(), ent], -1)
    out = {"scal": scal, "lpp_spike": m_spike.clamp(-60, 0), "lpp_span": m_span.clamp(-60, 0),
           "emb_last": (cs_h[sb] - cs_h[sa]) / (sb - sa)[:, None], "emb_mid": None,
           "span": torch.stack([sa, sb], -1), "spike": torch.stack([gs, ge], -1)}
    if hmj is not None:
        hmj = hmj.float(); cs_m = torch.cat([hmj.new_zeros(1, C), hmj.cumsum(0)])
        out["emb_mid"] = (cs_m[sb] - cs_m[sa]) / (sb - sa)[:, None]
    return out
