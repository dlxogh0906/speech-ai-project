# -*- coding: utf-8 -*-
"""세기 쌍(평음·경음·격음) 되살리기 — 받아쓰기가 표준 자모로 덮은 자리를, 다른 세기로 냈을 근거가 충분하면 그 소리로 본다.

CTC 받아쓰기는 프레임마다 확률이 가장 높은 자모만 남긴다. 외국인이 ㅅ 자리를 ㅆ 로 냈어도 모델이 ㅅ 0.55 · ㅆ 0.40 을 내면
ㅅ 로 적혀 "맞게 읽음" 이 된다(results/17 §7 — ㅅ→ㅆ 은 열에 아홉을 표준으로 덮는다). 여기서는 표준발음 전체를 기준으로
    LLR(j, a) = log P_ctc(표준발음에서 j 자리만 a 로 바꾼 열 | 녹음) − log P_ctc(표준발음 | 녹음)
을 재고, 쌍(c→a)마다 정한 문턱 τ 를 넘으면 그 자리를 a 로 본다. 영어 엔진의 한국인 대표 오류 규칙(`mdd_en` 지적 규칙 v2)과
같은 방식이다. 문턱과 효과는 `scripts/eval/eval_pair_rescue.py` 가 잰다.
"""

import torch
import torch.nn.functional as F

FAMILIES = ("ㄱㄲㅋ", "ㄷㄸㅌ", "ㅂㅃㅍ", "ㅈㅉㅊ", "ㅅㅆ")
FAMILY_OF = dict((c, f) for f in FAMILIES for c in f)
PAIRS = [c + a for f in FAMILIES for c in f for a in f if a != c]       # "ㅅㅆ" = 표준 ㅅ → 실제 ㅆ
VOWELS = set("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
SONORANTS = set("ㄴㄹㅁㅇ")


def position(canon, j):
    """표준발음 자리 j 의 위치 — 세기 대립은 위치마다 단서가 다르다(어두: 음높이 · 공명음 뒤: 유성성 · 장애음 뒤: 중화).
    → "coda"(받침: 뒤에 모음이 없다) · "initial"(발화 첫 자리) · "intervocalic"(모음 뒤) · "post_sonorant"(ㄴㄹㅁㅇ 뒤)
      · "post_obstruent"(그 밖의 자음 뒤 — 경음화 맥락)
    """
    if j + 1 >= len(canon) or canon[j + 1] not in VOWELS:
        return "coda"
    if j == 0:
        return "initial"
    prev = canon[j - 1]
    if prev in VOWELS:
        return "intervocalic"
    return "post_sonorant" if prev in SONORANTS else "post_obstruent"


def candidates(canon, keep=None):
    """[(j, c, a)] — 표준 자모 c 가 세기 계열이면 같은 계열의 다른 자모 a 마다 하나. keep(j) 가 거짓인 자리는 뺀다."""
    out = []
    for j, c in enumerate(canon):
        if c in FAMILY_OF and (keep is None or keep(j)):
            out += [(j, c, a) for a in FAMILY_OF[c] if a != c]
    return out


@torch.no_grad()
def pair_llrs(lprobs, y, cands, to_id, blank=0):
    """lprobs: (T, V) 로그확률 · y: 표준발음 id 열 · cands: [(j, c, a)] → LLR 목록(정렬이 안 되는 열은 -inf).

    표준발음 자체가 정렬되지 않으면(녹음이 너무 짧다 등) None.
    """
    if not cands:
        return []
    lprobs = torch.as_tensor(lprobs)                  # 발음 API 는 numpy 로그확률을 들고 있다
    T, V = int(lprobs.shape[0]), int(lprobs.shape[1])
    tg = [list(y)] + [[to_id(a) if k == j else v for k, v in enumerate(y)] for j, _c, a in cands]
    N, L = len(tg), len(y)
    lp = lprobs.float().unsqueeze(1).expand(T, N, V).contiguous()
    nll = F.ctc_loss(lp, torch.tensor(tg, dtype=torch.long), torch.full((N,), T, dtype=torch.long),
                     torch.full((N,), L, dtype=torch.long), blank=blank, reduction="none", zero_infinity=False)
    ll = (-nll).tolist()
    if ll[0] == float("-inf") or ll[0] != ll[0]:
        return None
    return [(v - ll[0]) if v == v else float("-inf") for v in ll[1:]]


def rescue(proj_pred, cands, llrs, tau):
    """받아쓰기(표준 축 투영)가 표준 자모 그대로인 자리 중 LLR > τ[c+a] 인 것을 a 로 바꾼다.

    proj_pred: 표준발음 자리마다 받아쓴 자모(삭제면 None) · tau: {"ㅅㅆ": τ, …}(없는 쌍은 안 본다)
    → (새 투영, {j: (a, llr)}). 한 자리에서 여러 a 가 넘으면 문턱을 가장 크게 넘은 것.
    """
    out, best = list(proj_pred), {}
    for (j, c, a), v in zip(cands, llrs or []):
        t = tau.get(c + a)
        if t is None or proj_pred[j] != c or v is None or not (v > t):     # NaN 은 넘지 못한 것으로 본다
            continue
        if j not in best or v - t > best[j][2]:
            best[j] = (a, v, v - t)
    for j, (a, v, _m) in best.items():
        out[j] = a
    return out, dict((j, (a, v)) for j, (a, v, _m) in best.items())
