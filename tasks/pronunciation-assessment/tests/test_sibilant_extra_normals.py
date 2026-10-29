"""Check that normal-target inventory does not silently require clean speech."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT/'tasks/pronunciation-assessment/src'),
    str(ROOT/'tasks/pronunciation-assessment/scripts/data')]
import unittest
from inventory_sibilant_extra_normals import original_row


class ExtraNormalTest(unittest.TestCase):
    def test_other_phone_error_and_actual_indices_survive(self):
        # An inserted vowel before the target shifts its actual index. It must
        # neither turn this known normal sibilant into an error nor drop the row.
        meta = {'stem':'original', 'spk':'speaker', 'item':'item', 'prompt':'prompt',
            'norm_prompt':'prompt', 'ltr':list('ㄱㅏㅓㅅㅣ'), 'canon':list('ㄱㅏㅅㅣ')}
        row, state = original_row(meta, 16000, 5, Path('original.wav'), Path('manifest'),
                                  set('ㄱㅏㅓㅅㅣ'))
        self.assertEqual(state, 'eligible')
        self.assertFalse(row['utterance_all_correct'])
        self.assertEqual(row['sites'][0]['label_kind'], 'correct')
        self.assertEqual((row['sites'][0]['q'], row['sites'][0]['a']), (2, 3))
        self.assertEqual(row['actual'], meta['ltr'])
        dropped = dict(meta, dropped={'unknown_phone':1})
        self.assertEqual(original_row(dropped, 16000, 5, Path('original.wav'), Path('manifest'),
            set('ㄱㅏㅓㅅㅣ')), (None, 'annotation_missing_dropped_or_oov'))


if __name__ == '__main__':
    unittest.main()
