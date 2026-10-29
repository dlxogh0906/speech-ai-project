import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    if ROOT.parent == ROOT:
        raise RuntimeError('repository root not found')
    ROOT = ROOT.parent
sys.path.insert(0,str(ROOT/'tasks/pronunciation-assessment/scripts/eval'))
import unittest
import numpy as np
from audit_sibilant_distance_nuisance import paired_scores


class NuisanceTest(unittest.TestCase):
    def test_phonetic_vs_code_with_other_attribute_matched(self):
        y = np.array([0,0,1,1]);code = np.array(['F','M','F','M'])
        phone = (y[:,None] != y[None,:]).astype(float)
        attribute = (code[:,None] != code[None,:]).astype(float)
        score, count, total = paired_scores(phone,y,y,code,code)
        np.testing.assert_array_equal(score,[1.,.5])
        self.assertEqual((count,total),(8,8))
        score,_,_ = paired_scores(attribute,y,y,code,code)
        np.testing.assert_array_equal(score,[.5,1.])

    def test_missing_joint_groups_or_unknown_codes_not_dropped(self):
        y = np.array([0,0,1,1]);code = np.array(['F','F','F','F']);d=np.zeros((4,4))
        score,count,total = paired_scores(d,y,y,code,code)
        np.testing.assert_array_equal(score,[.5,.5]);self.assertEqual((count,total),(0,8))
        code[:] = 'X'
        score,count,total = paired_scores(d,y,y,code,code)
        np.testing.assert_array_equal(score,[.5,.5]);self.assertEqual((count,total),(0,8))


if __name__ == '__main__':
    unittest.main()
