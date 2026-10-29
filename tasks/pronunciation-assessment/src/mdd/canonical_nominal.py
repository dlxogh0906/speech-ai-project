"""Reviewed nominal-suffix correction candidate, not activated by serving.

NIKL's 2025-09-12 explanation explicitly gives 여덟도 [여덜도] and
여덟과 [여덜과]. It separately says the phrase 여덟 시 has no prescribed
standard pronunciation. Do not turn that statement into a mandatory ㅆ rule.
"""
import re

SOURCE = "https://www.korean.go.kr/front/onlineQna/onlineQnaView.do?mn_id=216&qna_seq=320607"

# Whole orthographic words only. Do not alter vowel-initial particles, other
# ㄼ stems, compound numerals, or lexical strings such as 여덟과학.
_SUFFIX = re.compile(r"(?<![0-9A-Za-z가-힣_])여덟(?P<suffix>도(?:요)?|과(?:는|도|만)?)(?![0-9A-Za-z가-힣_])")


def rewrite_nominal_suffixes(text):
    return _SUFFIX.sub(lambda m: "여덜" + m.group("suffix"), text)


class NominalCorrectionBuilder:
    """Wrap the existing canonical builder, preserving its other rules.

    Integration candidate: app.S['cb'] = NominalCorrectionBuilder(app.S['cb']).
    No service registers this wrapper by default. A runtime comparison must
    recompute alignment and scores against the resulting canonical sequence.
    """
    def __init__(self, base):
        self.base = base

    def fixed(self, text):
        return self.base.fixed(rewrite_nominal_suffixes(text))

    def variants(self, text, *args, **kwargs):
        return self.base.variants(rewrite_nominal_suffixes(text), *args, **kwargs)
