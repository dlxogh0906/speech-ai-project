"""Check pretrained C/V requests using the exact base-encoder loader."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    if ROOT.parent == ROOT:
        raise RuntimeError("repository root not found")
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tasks/pronunciation-assessment/src")]
import audit_sibilant_cv_batching as auditor
from mdd.sibilant_pretrained_encoder import load_encoder


def main():
    auditor.load_encoder = load_encoder
    auditor.main()


if __name__ == "__main__":
    main()
