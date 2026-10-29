import sys
import unittest
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'tasks/pronunciation-assessment/scripts/data'),
               str(ROOT / 'tasks/pronunciation-assessment/scripts/eval'),
               str(ROOT / 'tasks/pronunciation-assessment/src')]
from prepare_sibilant_local_coverage import matched_plan, unique_labels


class LocalCoverageTest(unittest.TestCase):
    def test_keeps_actual_class_error_kind_and_original_unsupported_draw(self):
        original = np.array([[0, 1, 2]])
        y = np.array([0, 0, 1, 0, 0, 1])
        targets = [{'target': {}} for _ in y]
        targets[2]['target'] = None
        labels = {i: {'label_kind': 'correct' if i in (0, 3) else 'pair_error'} for i in range(6)}
        plan = matched_plan(original, y[:3], targets[:3], labels, y, targets, labels, 13)
        self.assertEqual(int(plan[0, 2]), 2)
        for old, new in zip(original.ravel()[:2], plan.ravel()[:2]):
            self.assertEqual(y[old], y[new])
            self.assertEqual(labels[old]['label_kind'], labels[new]['label_kind'])
        self.assertTrue(np.array_equal(plan, matched_plan(original, y[:3], targets[:3], labels, y, targets, labels, 13)))

    def test_unknown_control_classes_cannot_become_supervision(self):
        with self.assertRaises(ValueError):
            matched_plan(np.array([[0]]), np.array([-100]), [{'target': None}], {},
                         np.array([-100]), [{'target': None}], {}, 1)

    def test_multiple_GT_slots_cannot_silently_overwrite_a_feature(self):
        with self.assertRaises(ValueError):
            unique_labels([{'feature_index': 0, 'q': 2}, {'feature_index': 0, 'q': 3}])


if __name__ == '__main__':
    unittest.main()
