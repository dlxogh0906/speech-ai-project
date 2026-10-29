"""Inactive, source-supported time-phrase reference candidate.

NIKL does not prescribe a single pronunciation for the phrase 여덟 시.
TUFS teaching material explicitly presents [여덜씨]. This candidate adds that
reading to the existing automatic CTC reference chooser; it does not declare
every ㄼ+s sequence tense, nor provide evidence of actual-phone recognition.
"""
import re

SOURCES = (
    "https://www.korean.go.kr/front/onlineQna/onlineQnaView.do?mn_id=216&qna_seq=320607",
    "https://www.coelang.tufs.ac.jp/modules/ko/gmod/courses/c02/lesson07/step5/explanation/427.html",
)
KIND = "여덟 시 교육자료 발음"
# Positive time-unit forms only; 시간, 시작, 시청 and compound numerals excluded.
TIME_PHRASE = re.compile(r"(?<![0-9A-Za-z가-힣_])여덟\s+(?P<unit>시(?:까지|부터|에)?)(?![0-9A-Za-z가-힣_])")


class TimePhraseBuilder:
    """Wrap app.S['cb'] explicitly for offline evaluation; never auto-register."""
    def __init__(self, base):
        self.base = base

    def fixed(self, text):
        return self.base.fixed(text)

    def variants(self, text, *args, **kwargs):
        base, alts = self.base.variants(text, *args, **kwargs)
        out = list(alts)
        kinds = kwargs.get("kinds", args[0] if args else None)
        exclude = kwargs.get("exclude", args[1] if len(args) > 1 else ())
        if (kinds is not None and not any(KIND.startswith(k) for k in kinds)) or KIND in exclude:
            return base, out
        for match in TIME_PHRASE.finditer(text):
            q = match.start("unit")
            proposed = self.base.fixed(text[:q] + "씨" + text[q + 1:])
            if len(proposed) != len(base):
                continue
            changed = [i for i, (a, b) in enumerate(zip(base, proposed)) if a != b]
            # Refuse any unexpected G2P side effect, including boundary changes.
            if len(changed) != 1:
                continue
            i = changed[0]
            if (base[i] != "ㅅ" or proposed[i] != "ㅆ" or
                    base[max(0, i - 4):i] != list("ㅕㄷㅓㄹ") or base[i + 1:i + 2] != ["ㅣ"]):
                continue
            alt = (i, i + 1, ["ㅆ"], KIND)
            if not any((s, e, list(value)) == alt[:3] for s, e, value, _ in out):
                out.append(alt)
        return base, out
