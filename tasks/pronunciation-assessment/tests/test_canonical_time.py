import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks/pronunciation-assessment/src")]

from mdd.canonical_fix import CanonicalBuilder, choose
from mdd.canonical_time import TimePhraseBuilder, KIND
from mdd.word_judge import KoCanonical


class TimePhraseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = CanonicalBuilder(KoCanonical())
        cls.candidate = TimePhraseBuilder(cls.base)

    def test_exact_local_change_through_actual_g2p(self):
        for text in ("여덟 시", "오늘 여덟 시에 만나요.", "여덟 시까지", "여덟 시부터"):
            base, alts = self.candidate.variants(text)
            self.assertEqual(base, self.base.fixed(text))
            added = [a for a in alts if a[3] == KIND]
            self.assertEqual(len(added), 1)
            s, e, value, _ = added[0]
            self.assertEqual(base[s:e], ["ㅅ"])
            self.assertEqual(value, ["ㅆ"])
            self.assertEqual(self.candidate.fixed(text), self.base.fixed(text))

    def test_guard_words_and_lexical_contrasts(self):
        for text in ("여덟 시간", "여덟 시작", "여덟 시청", "여덟 시민", "열여덟 시", "여덟도",
                     "여덟과", "밟고", "사다 싸다", "다섯 시", "여덟시", "여덟 씨", "여덟 시외"):
            self.assertEqual(self.candidate.variants(text), self.base.variants(text))

    def test_kind_filters_and_multiple_occurrences(self):
        text = "여덟 시부터 여덟 시까지"
        self.assertEqual(len([a for a in self.candidate.variants(text)[1] if a[3] == KIND]), 2)
        self.assertEqual(self.candidate.variants(text, kinds=("의",)), self.base.variants(text, kinds=("의",)))
        self.assertEqual(self.candidate.variants(text, exclude=(KIND,)), self.base.variants(text, exclude=(KIND,)))

    def test_audio_evidence_and_existing_margin_still_required(self):
        import numpy as np
        base, alts = self.candidate.variants("여덟 시")
        vocab = list(dict.fromkeys(["_"] + base + ["ㅆ"]))
        index = vocab.index
        q = next(a[0] for a in alts if a[3] == KIND)
        for actual in ("ㅅ", "ㅆ"):
            seq = list(base)
            seq[q] = actual
            lp = np.full((2 * len(seq) + 1, len(vocab)), -12., dtype=np.float32)
            lp[::2, 0] = 0.
            for i, token in enumerate(seq):
                lp[2 * i + 1, index(token)] = 0.
            selected, _ = choose(lp, base, alts, index, margin=1.)
            self.assertEqual(selected[q], actual)


if __name__ == "__main__":
    unittest.main()
