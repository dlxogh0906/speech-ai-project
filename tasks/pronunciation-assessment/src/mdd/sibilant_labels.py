"""Keep full TAG outcomes at expected sibilants, including unresolved mappings."""
VOWELS = set("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
PAIR = {"ㅅ": "ㅆ", "ㅆ": "ㅅ"}


def label_sites(canonical, actual, mapping):
    output = []
    if len(mapping) != len(canonical):
        raise ValueError("mapping length mismatch")
    for q, expected in enumerate(canonical[:-1]):
        if expected not in PAIR or canonical[q + 1] not in VOWELS:
            continue
        options = mapping[q]
        if not options or any(a is not None and (not isinstance(a, int) or not 0 <= a < len(actual)) for a in options):
            raise ValueError("invalid actual index options")
        ambiguous = len(options) != 1
        a = next(iter(options)) if not ambiguous else None
        heard = actual[a] if a is not None else None
        kind = ("ambiguous" if ambiguous else "deletion" if a is None else
                "correct" if heard == expected else "pair_error" if heard == PAIR[expected] else "third_phone")
        output.append({"q": q, "a": a, "canon": expected, "actual": heard,
                       "alignment_ambiguous": ambiguous, "label_kind": kind,
                       "actual_onset": a is not None and a + 1 < len(actual) and actual[a + 1] in VOWELS,
                       "following": canonical[q + 1],
                       "position": "initial" if q == 0 else "post_vowel" if canonical[q - 1] in VOWELS else "post_consonant",
                       "alignment_options": [{"a": j, "actual": actual[j] if j is not None else None}
                                             for j in sorted(options, key=lambda x: -1 if x is None else x)]})
    return output
