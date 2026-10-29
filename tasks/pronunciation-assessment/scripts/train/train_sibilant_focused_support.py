"""Run the frozen matched C/V trainer with squared support pooling only."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    if ROOT.parent == ROOT:
        raise RuntimeError("repository root not found")
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tasks/pronunciation-assessment/src")]
import train_sibilant_cv_support as trainer
from mdd.sibilant_focused_support import SquaredSupportReadout


def main():
    trainer.SupportReadout = SquaredSupportReadout
    trainer.main()


if __name__ == "__main__":
    main()
