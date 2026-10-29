import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    if ROOT.parent == ROOT:
        raise RuntimeError('repository root not found')
    ROOT = ROOT.parent
sys.path.insert(0,str(ROOT/'tasks/pronunciation-assessment/scripts/data'))
import unittest
from inventory_sibilant_native_control import select_contexts


class NativeControlTest(unittest.TestCase):
    def test_identity_selection_does_not_follow_class_labels(self):
        rows = [{'id':str(i),'speaker':'s'+str(i//2),'phone':'s0' if i%2 else 'ss',
                 'label_consonant_interval':[0.,.1],'label_vowel_interval':[.1,.3],
                 'previous_phone':'aa','following_vowel':'ii','at_labeled_word_start':False,
                 'at_labeled_utterance_start':False} for i in range(20)]
        _,original,states,repeats=select_contexts(rows,20261025)
        self.assertEqual(repeats,10)
        self.assertEqual(len(original[0]['ids']),10)
        for row in rows:
            row['phone']='ss' if row['phone']=='s0' else 's0'
        _,changed,_,_=select_contexts(rows,20261025)
        self.assertEqual(original[0]['ids'],changed[0]['ids'])
        self.assertEqual(original[0]['halves'],changed[0]['halves'])
        rows[0]['previous_phone']='<private>'
        rows[1]['label_vowel_interval']=[-.1,.3]
        _,_,states,_=select_contexts(rows,20261025)
        self.assertEqual(states[0]['state'],'previous_phone_unknown_or_marked')
        self.assertEqual(states[1]['state'],'invalid_target_annotation')


if __name__ == '__main__':
    unittest.main()
