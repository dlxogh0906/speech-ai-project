import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mdd.sibilant_context_batches import make_plan


class ContextBatchTest(unittest.TestCase):
    def sample(self):
        y = np.array([0, 1, 1, 0, 0, 2, 3])
        specs = [("A", "ㅅ", "correct"), ("A", "ㅅ", "pair_error"),
                 ("B", "ㅆ", "correct"), ("B", "ㅆ", "pair_error"), ("C", "ㅅ", "correct")]
        labels = [dict(feature_index=i, item=item, q=0, expected=expected, following="ㅣ", label_kind=kind,
                       actual="ㅅ" if y[i] == 0 else "ㅆ", alignment_ambiguous=False)
                  for i, (item, expected, kind) in enumerate(specs)]
        features = [dict(canonical=x[1]) for x in specs] + [{}, {}]
        return y, labels, features

    def test_every_pair_has_same_context_and_both_actuals(self):
        y, labels, features = self.sample()
        plan, contexts, _ = make_plan(y, labels, features, 3, 8, "paired_context")
        for row, keys in zip(plan, contexts):
            self.assertEqual(np.bincount(y[row], minlength=4).tolist(), [4, 4, 4, 4])
            for j, key in enumerate(keys):
                a, b = labels[row[j]], labels[row[j + 4]]
                self.assertEqual((a["item"], a["q"], a["expected"], a["following"]), key)
                self.assertEqual((b["item"], b["q"], b["expected"], b["following"]), key)
                self.assertEqual({a["label_kind"], b["label_kind"]}, {"correct", "pair_error"})

    def test_eligible_control_excludes_unpaired_context_without_changing_classes(self):
        y, labels, features = self.sample()
        plan, _, _ = make_plan(y, labels, features, 3, 30, "eligible_actual")
        self.assertNotIn(4, plan)
        self.assertTrue(all(np.bincount(y[row], minlength=4).tolist() == [4] * 4 for row in plan))
        labels[0]["alignment_ambiguous"] = True
        with self.assertRaises(ValueError):
            make_plan(y, labels, features, 3, 1, "paired_context")


if __name__ == "__main__":
    unittest.main()
