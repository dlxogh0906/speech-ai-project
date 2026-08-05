# -*- coding: utf-8 -*-
"""지적 후처리 — 억울한 지적을 줄이는 필터.

왜 필요한가
-----------
2026-07-31 실측: 현역 모델의 정상 자리 오탐 29,074건 중 **38.8%(11,293건)가 삭제형**이다.
모델이 그 자리에서 음소를 아예 내지 못한 경우인데, 지금 시스템은 이를 "발음이 틀렸다"로
표시한다. 그건 발음 오류가 아니라 **인식 실패**다.

삭제형을 지적에서 빼면 (레슨 = 30발화 기준):

    레슨당 억울한 지적    4.82 → 2.97회   (-1.85)
    억울한 지적 0회 레슨  16.5% → 23.3%
    레슨당 정확 포착      3.97 → 3.31개   (-0.66)

억울한 지적 1.85회를 줄이는 데 포착 0.66개를 낸다 — **2.8배 유리한 교환**이다.
재학습·GPU·모델 변경이 전혀 없다.

무엇을 고치는 것인가
--------------------
근본 원인(287문장 암기, `results/08_문장암기_일반화.md`)은 이걸로 안 고쳐진다.
이건 **모델 출력을 사용자에게 어떻게 해석해 보여줄지**의 문제다.
"못 알아들었다"와 "틀리게 발음했다"를 같은 것으로 취급하던 매핑을 바로잡는다.

사용
----
    from mdd.suppress import classify_ops, SuppressPolicy

    pol = SuppressPolicy(hide_deletions=True)
    shown, hidden = classify_ops(ops, pol)     # ops = pipeline._nw() 출력
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any


@dataclass
class SuppressPolicy:
    """어떤 지적을 사용자에게 보일지 정하는 정책.

    hide_deletions
        모델이 해당 자리에서 음소를 내지 못한 경우(삭제형)를 오류 지적에서 뺀다.
        실측상 오탐의 38.8%가 여기 해당하고, 진짜 오류 중 삭제로 잡힌 것은
        포착 전체의 16.6%다. 즉 잃는 것보다 얻는 것이 2.8배 크다.
    hide_insertions
        모델이 없는 음소를 덧붙인 경우. canonical 위치가 없어 사용자에게 보여줄
        자리가 모호하다. 기본으로 숨긴다.
    """
    hide_deletions: bool = True
    hide_insertions: bool = True


def classify_ops(ops: List[Dict[str, Any]],
                 policy: SuppressPolicy = SuppressPolicy()
                 ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """정렬 연산 목록을 (보여줄 지적, 숨길 지적)으로 가른다.

    ops 의 각 원소는 `kind` 를 갖는다: match / sub / del / ins.
      del  — 표준발음에 있는 음소를 모델이 내지 못함  → 인식 실패로 본다
      ins  — 표준발음에 없는 음소를 모델이 냄          → 표시 위치가 모호하다
      sub  — 다른 음소로 들음                          → 진짜 발음 오류 후보

    숨긴 항목은 버리지 않고 반환한다. 호출부가 "이 구간은 잘 안 들렸습니다" 같은
    별도 안내로 쓸 수 있어야 하기 때문이다.
    """
    shown, hidden = [], []
    for op in ops:
        kind = op.get("kind")
        if kind == "match":
            continue
        if kind == "del" and policy.hide_deletions:
            hidden.append({**op, "reason": "not_recognized"})
        elif kind == "ins" and policy.hide_insertions:
            hidden.append({**op, "reason": "no_canonical_position"})
        else:
            shown.append(op)
    return shown, hidden
