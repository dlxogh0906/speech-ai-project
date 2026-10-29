"""Reuse raw-CV preparation while preserving the original supervision eligibility."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT / 'pyproject.toml').exists():
    if ROOT.parent == ROOT:
        raise RuntimeError('repository root not found')
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / 'tasks/pronunciation-assessment/src')]
import argparse
import json
import numpy as np
import prepare_sibilant_raw_cv_targets as preparer


def retained_policy(original_y, target_function):
    """Never promote an old unknown or silently change a known actual class."""
    def target(label, c, v, left, right, centers, adjacent, edge=.01):
        i = label['feature_index']
        if original_y[i] < 0:
            return int(original_y[i]), np.full(len(centers), -100, np.int64), 'original_unknown_preserved'
        actual, phase, state = target_function(label, c, v, left, right, centers, adjacent, edge=edge)
        if actual != original_y[i]:
            raise ValueError('known original actual class/support changed: '+str(i))
        return actual, phase, state
    return target


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    p = json.loads(Path(args.protocol).read_text())
    original_y = np.load(p['original_targets'])['y']
    original = preparer.training_target
    preparer.training_target = retained_policy(original_y, original)
    try:
        # The established entry point owns full input hash/partition checks,
        # independent source geometry, frame clocks, equality and outputs.
        preparer.main()
    finally:
        preparer.training_target = original


if __name__ == '__main__':
    main()
