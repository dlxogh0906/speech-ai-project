"""Adapt shared C/V evidence to the known runtime following-vowel context.

No expected sibilant or TAG actual is a model input. A zero-initialized affine
conditioner preserves the original function initially. Learned context effects
are hypotheses, not fixed phonetic rules or actual-vowel transcription.
"""
import torch
from mdd.sibilant_cv_support import SupportReadout
from mdd.sibilant_c_trajectory import c_summary

VOWEL_TOKENS = tuple("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
VOWEL_IDS = {v: i + 1 for i, v in enumerate(VOWEL_TOKENS)}


class VowelConditionReadout(SupportReadout):
    def __init__(self, hidden_size):
        super().__init__(hidden_size, True)
        self.condition = torch.nn.Embedding(len(VOWEL_IDS) + 1, 128)
        torch.nn.init.zeros_(self.condition.weight)

    def load_control(self, state):
        missing, unexpected = self.load_state_dict(state, strict=False)
        if missing != ["condition.weight"] or unexpected:
            raise ValueError("unexpected control parameter layout")
        torch.nn.init.zeros_(self.condition.weight)

    def forward(self, hidden, mask, relative, vowel_ids):
        if mask.shape != hidden.shape[:2] or relative.shape != mask.shape or not mask.any(1).all():
            raise ValueError("observed acoustic mask and runtime position required")
        if vowel_ids.dtype != torch.long or vowel_ids.shape != (len(hidden),) or not ((vowel_ids >= 0) & (vowel_ids <= len(VOWEL_IDS))).all():
            raise ValueError("one valid runtime context ID per input required")
        phase = self.support(torch.cat([hidden.float(), relative[..., None]], -1))
        summary, _, _ = c_summary(hidden, mask, phase)
        scale, bias = self.condition(vowel_ids).chunk(2, -1)
        value = self.actual[0](summary) * (1 + scale) + bias
        return self.actual[2](self.actual[1](value)), phase
