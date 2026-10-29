import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import unittest
import torch
from mdd.sibilant_cv_support import SupportReadout
from mdd.sibilant_vowel_condition import VowelConditionReadout, VOWEL_IDS


class VowelConditionTest(unittest.TestCase):
    def test_zero_condition_matches_control_and_only_observed_contexts_learn(self):
        torch.manual_seed(53)
        old = SupportReadout(6, True)
        head = VowelConditionReadout(6)
        head.load_control(old.state_dict())
        h, mask, relative = torch.randn(2, 8, 6), torch.ones(2, 8, dtype=torch.bool), torch.randn(2, 8)
        ids = torch.tensor([VOWEL_IDS['ㅣ'], VOWEL_IDS['ㅜ']])
        result = head(h, mask, relative, ids)
        baseline = old(h, mask, relative)
        self.assertTrue(torch.equal(result[0], baseline[0]))
        self.assertTrue(torch.equal(result[1], baseline[1]))
        result[0].square().sum().backward()
        active = head.condition.weight.grad.abs().sum(-1) > 0
        self.assertEqual(active.nonzero().flatten().tolist(), sorted(ids.tolist()))

    def test_invalid_context_is_not_silently_used(self):
        head = VowelConditionReadout(6)
        h, mask, relative = torch.zeros(1, 8, 6), torch.ones(1, 8, dtype=torch.bool), torch.zeros(1, 8)
        for ids in [torch.tensor([99]), torch.tensor([1.]), torch.tensor([1, 2])]:
            with self.assertRaises(ValueError):
                head(h, mask, relative, ids)


if __name__ == '__main__':
    unittest.main()
