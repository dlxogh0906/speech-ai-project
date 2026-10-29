"""Prefer the attested 여덜씨 reference when the decoder already heard ㅆ.

This is an inactive reference/acceptance adapter, not an acoustic classifier.
NIKL does not prescribe one standard for the phrase; TUFS explicitly teaches
여덜씨. Do not infer that every realization, deletion or third phone is allowed.
"""
from mdd.canonical_time import TimePhraseBuilder, KIND, TIME_PHRASE, SOURCES


class _FixedOnly:
    def __init__(self, builder):
        self.builder = builder

    def fixed(self, text):
        return self.builder.fixed(text)

    def variants(self, text, *args, **kwargs):
        return self.fixed(text), []


def observed_time_reference(builder, text, canonical, sites):
    selected = list(canonical)
    if not TIME_PHRASE.search(text):
        return selected, [], "not_time_phrase"
    base, alternatives = TimePhraseBuilder(_FixedOnly(builder)).variants(text)
    permitted = {start for start, end, value, kind in alternatives
                 if kind == KIND and end == start + 1 and value == ["ㅆ"]}
    if len(selected) != len(base) or len(sites) != len(selected):
        return selected, [], "reference_length_differs"
    if any(a != b and not (i in permitted and {a, b} == {"ㅅ", "ㅆ"})
           for i, (a, b) in enumerate(zip(base, selected))):
        return selected, [], "other_reference_change"
    accepted = []
    for q in sorted(permitted):
        site = sites[q]
        if site.get("i") != q or site.get("canon") != selected[q]:
            raise ValueError("canonical/site alignment changed")
        if site.get("pred") == "ㅆ":
            selected[q] = "ㅆ"
            accepted.append(q)
    return selected, accepted, "observed_attested_reading" if accepted else "attested_reading_not_observed"


class FixedReference:
    """A single-request reference for rerunning the original scoring path."""
    def __init__(self, text, canonical):
        self.text, self.canonical = text, list(canonical)

    def fixed(self, text):
        if text != self.text:
            raise ValueError("forced reference belongs to another request")
        return list(self.canonical)

    def variants(self, text, *args, **kwargs):
        return self.fixed(text), []
