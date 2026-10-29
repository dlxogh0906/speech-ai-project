"""Reuse the matched trainer with only per-context sample replacement changed."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():ROOT=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'tasks/pronunciation-assessment/src')]
import json
import numpy as np
import train_sibilant_context_pairs as trainer
from mdd.sibilant_context_batches import make_plan as original_plan
from mdd.sibilant_coverage_batches import make_plan as coverage_plan


def make_plan(y,labels,features,seed,steps,mode):
    base,contexts,_=original_plan(y,labels,features,seed,steps,mode)
    expected=np.load(REFERENCE_PLAN)
    np.testing.assert_array_equal(base,expected)
    proposed,newcontexts,report=coverage_plan(y,labels,features,seed,steps,mode)
    if newcontexts!=contexts:raise AssertionError('context schedule changed')
    return proposed,newcontexts,report


if __name__=='__main__':
    # Let the established entrypoint own argument parsing, hashing and output.
    if '--help' in sys.argv or '-h' in sys.argv:
        trainer.main()
    at=sys.argv.index('--protocol')
    protocol=json.loads(Path(sys.argv[at+1]).read_text())
    REFERENCE_PLAN=protocol['reference_1500_plan']
    trainer.make_plan=make_plan
    trainer.main()
