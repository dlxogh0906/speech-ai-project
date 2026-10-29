"""Reuse the fixed context-paired trainer, changing only vowel readout pooling.

Parameter layout, sampling, class/auxiliary losses and encoder update layers
remain identical. A typed artifact prevents silent loading with learned E/L
pooling in a later evaluator.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    if ROOT.parent == ROOT:
        raise RuntimeError("repository root not found")
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tasks/pronunciation-assessment/src")]
import argparse
import json
import shutil
import torch
import train_sibilant_context_pairs as trainer
from mdd.sibilant_joint_vowel import JointVowelReadout


class ProgressReadout(JointVowelReadout):
    def __init__(self, hidden_size, learned):
        if not learned:
            raise ValueError('this entrypoint reuses only the learned-support fit')
        super().__init__(hidden_size, 'vowel_progress')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--root', required=True)
    ap.add_argument('--stage', choices=('fit',), required=True)
    a = ap.parse_args()
    trainer.SupportReadout = ProgressReadout
    trainer.main()
    out = Path(a.root) / 'heads'
    artifact = torch.load(out / 'learned_support.pt', map_location='cpu')
    artifact.update(readout='joint_vowel', pooling='vowel_progress', role_supervision='split')
    torch.save(artifact, out / 'learned_support.pt')
    report = json.loads((out / 'fit.json').read_text())
    report['readout'] = 'vowel_progress'
    trainer.write(out / 'fit.json', report)
    trainer.write(out / 'fit.finished.json', {'sha256': trainer.trainer.service.sha(out / 'fit.json')})
    shutil.copy2(__file__, out / Path(__file__).name)


if __name__ == '__main__':
    main()
