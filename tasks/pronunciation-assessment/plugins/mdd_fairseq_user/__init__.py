# -*- coding: utf-8 -*-
"""fairseq user-dir 플러그인 — `common.user_dir=<이 폴더>` 로 올린다. (src/ 밖에 둔 이유: fairseq 가 user_dir 의 **부모 폴더를 sys.path 맨 앞에** 넣는다 —
   src/mdd 안에 두면 그 안의 모듈들이 최상위 이름으로 노출돼 다른 import 를 가릴 수 있다.)

wildcard_ctc  라벨의 일부 자리를 "아무 자모나(*)" 로 남겨 두는 CTC.
    약지도 데이터(AIHub 505: 문장은 알지만 어느 자리를 틀리게 읽었는지는 모른다)에서, 판별기가 확신하는 자리만
    표준발음을 정답으로 주고 나머지는 `*` 로 둔다. 그러면 모델은 외국인 목소리·모바일 채널에는 적응하지만
    **오발음을 표준발음이라고 배우는 일(되돌림)은 생기지 않는다.**

    구현: 사전에 없는 기호 `*` 는 fairseq 가 <unk> 로 인코딩한다. 학습 때만 로그확률 끝에 가상 클래스 하나
    (= 비특수 기호 전체의 logsumexp, 즉 "blank 가 아닌 아무 자모")를 붙이고 <unk> 타깃을 그 클래스로 바꾼다.
    CTC 는 방출 확률이 정규화돼 있을 필요가 없어 표준 `F.ctc_loss` 를 그대로 쓴다. 검증은 원래 CTC 와 같다
    (검증 셋에는 `*` 를 넣지 않는다).
"""
import torch
from fairseq.criterions import register_criterion
from fairseq.criterions.ctc import CtcCriterion, CtcCriterionConfig


@register_criterion("wildcard_ctc", dataclass=CtcCriterionConfig)
class WildcardCtcCriterion(CtcCriterion):
    def forward(self, model, sample, reduce=True, **kwargs):
        d = self.task.target_dictionary
        unk = d.unk()
        tgt = sample["target"]
        if not model.training or not bool((tgt == unk).any()):
            return super().forward(model, sample, reduce=reduce, **kwargs)
        n_cls, nsp = len(d), d.nspecial
        orig = model.get_normalized_probs

        def with_wildcard(net_output, log_probs, sample=None):
            lp = orig(net_output, log_probs=log_probs)                       # (T, B, C) float32
            return torch.cat([lp, torch.logsumexp(lp[..., nsp:], dim=-1, keepdim=True)], dim=-1)

        model.get_normalized_probs = with_wildcard
        sample["target"] = tgt.masked_fill(tgt == unk, n_cls)
        try:
            return super().forward(model, sample, reduce=reduce, **kwargs)
        finally:
            del model.get_normalized_probs                                   # 인스턴스 속성을 지워 클래스 메서드로 되돌린다
            sample["target"] = tgt
