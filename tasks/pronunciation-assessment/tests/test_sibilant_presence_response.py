import copy
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from mdd import speechsuper as ss
from mdd.sibilant_presence_response import apply_decisions


class ResponseTest(unittest.TestCase):
    def setUp(self):
        self.fmt = ss.SpeechSuperFormatter(jamo_to_phones=list,
            phone_context=lambda c, i, v: '초성', ctype=lambda *a: '음운대치')
        self.fmt._score = lambda key, margin: margin
        canon = list('ㅅㅏㄱㅏ')
        self.raw = {'canon': canon, 'sites': [dict(canon=c, pred=p, j=i if p else None, prob=.99)
                    for i, (c, p) in enumerate(zip(canon, [None, 'ㅏ', 'ㄱ', 'ㅏ']))],
                    'margins': [10, 10, 90, 90], 'scores': {}, 'feats': {}}
        self.body = self.fmt.build('사가', canon, self.raw['sites'], self.raw['margins'], vowels={'ㅏ'})

    def test_unlinked_C_warning_can_be_deferred_without_changing_actual_or_score(self):
        before = copy.deepcopy(self.body)
        result = apply_decisions(self.fmt, self.body, self.raw, {0: {'kind': 'defer'}}, {'ㅏ'})
        self.assertEqual(self.body, before)
        word = result['result']['words'][0]
        self.assertEqual(word['readType'], ss.READ_NORMAL)
        phone = word['phonemes'][0]
        self.assertIsNone(phone['actual_jamo'])
        self.assertIsNone(phone['pronunciation'])
        self.assertTrue(phone['uncertain'])
        self.assertNotIn('errorType', phone)

    def test_unlinked_C_pair_call_preserves_numeric_None_and_updates_word(self):
        result = apply_decisions(self.fmt, self.body, self.raw,
            {0: {'kind': 'pair', 'actual': 'ㅆ', 'probability': .99}}, {'ㅏ'})
        phone = result['result']['words'][0]['phonemes'][0]
        self.assertEqual(phone['actual_jamo'], 'ㅆ')
        self.assertEqual(phone['actual'], 'ㅆ')
        self.assertEqual(phone['errorType'], '음운대치')
        self.assertIsNone(phone['pronunciation'])
        self.assertEqual(result['result']['words'][1], self.body['result']['words'][1])

    def test_unobserved_next_V_cannot_be_used(self):
        self.raw['sites'][1]['j'] = None
        with self.assertRaises(ValueError):
            apply_decisions(self.fmt, self.body, self.raw, {0: {'kind': 'defer'}}, {'ㅏ'})


if __name__ == '__main__':
    unittest.main()
