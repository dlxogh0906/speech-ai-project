import copy
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from mdd import speechsuper as ss
from mdd.sibilant_warning_review import defer_warnings, review_queries


class WarningReviewTest(unittest.TestCase):
    def setUp(self):
        self.formatter = ss.SpeechSuperFormatter(jamo_to_phones=list,
            phone_context=lambda c, i, v: '초성', ctype=lambda *a: '음운대치')
        self.formatter._score = lambda key, margin: margin
        canon = list('ㅅㅏㄱㅏ')
        self.raw = {'canon': canon, 'sites': [dict(canon=c, pred=p, j=i, prob=.99)
                    for i, (c, p) in enumerate(zip(canon, 'ㅆㅏㄱㅏ'))],
                    'margins': [10, 10, 90, 90], 'scores': {}, 'feats': {}}
        self.vowels = {'ㅏ'}
        self.body = self.formatter.build('사가', canon, self.raw['sites'], self.raw['margins'], vowels=self.vowels)

    def test_deferral_preserves_tokens_scores_and_recomputes_aggregates(self):
        before = copy.deepcopy(self.body)
        result = defer_warnings(self.formatter, self.body, self.raw, {0: {}}, self.vowels)
        first = result['result']['words'][0]
        self.assertEqual(self.body, before)
        self.assertEqual(before['result']['words'][0]['readType'], ss.READ_BAD)
        self.assertEqual(first['readType'], ss.READ_NORMAL)
        self.assertEqual(first['score'], 10)
        self.assertEqual(result['result']['pronunciation'], 50)
        p = first['phonemes'][0]
        self.assertEqual(p['actual_jamo'], 'ㅆ')
        self.assertEqual(p['pronunciation'], 10)
        self.assertTrue(p['uncertain'])
        self.assertNotIn('errorType', p)
        self.assertEqual(result['result']['words'][1], before['result']['words'][1])

    def test_unlinked_warning_cannot_be_deferred(self):
        self.raw['sites'][0]['j'] = None
        with self.assertRaises(ValueError):
            defer_warnings(self.formatter, self.body, self.raw, {0: {}}, self.vowels)

    def test_decision_requires_support_expected_class_and_existing_warning(self):
        phones = [p for w in self.body['result']['words'] for p in w['phonemes']]
        args = (self.raw['canon'], phones, [{'q_runtime': 0}])
        self.assertEqual(set(review_queries(*args, np.array([[.8, .1, .05, .05]]), [True], {'ㅅ': .7})), {0})
        self.assertEqual(review_queries(*args, np.array([[.8, .1, .05, .05]]), [False], {'ㅅ': .7}), {})
        self.assertEqual(review_queries(*args, np.array([[.1, .8, .05, .05]]), [True], {'ㅅ': .7}), {})


if __name__ == '__main__':
    unittest.main()
