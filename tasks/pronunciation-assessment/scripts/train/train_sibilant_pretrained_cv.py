"""Run the matched C/V trainer on a base encoder without a CTC output head."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    if ROOT.parent == ROOT:
        raise RuntimeError("repository root not found")
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tasks/pronunciation-assessment/src")]
import train_sibilant_cv_support as trainer
from mdd.sibilant_pretrained_encoder import load_encoder


def main():
    trainer.load_encoder = load_encoder
    trainer.main()


if __name__ == "__main__":
    main()
