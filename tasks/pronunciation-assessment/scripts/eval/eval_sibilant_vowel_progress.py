"""Single-request calibration/API replay of explicitly typed vowel pooling.

Keep the existing formatter/comparators unchanged. Runtime input is acoustic x,
observed length and automatic relative position only. Predicted role support is
not a measurement of physical aspiration, resonance or glottal onset.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    if ROOT.parent == ROOT:
        raise RuntimeError("repository root not found")
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tasks/pronunciation-assessment/src"),
               str(ROOT / "tasks/pronunciation-assessment/scripts/train")]
import argparse
import json
import shutil
import time
import numpy as np
import torch
import eval_sibilant_context_rank as evaluator
from train_sibilant_local_encoder import load_encoder, hidden
from mdd.sibilant_cv_support import reliable_support
from mdd.sibilant_joint_vowel import JointVowelReadout, strict_or_joint_support

SUPPORT_POLICY = None


def model(protocol, artifact):
    if artifact.get('readout') != 'joint_vowel' or artifact.get('pooling') != protocol['pooling']:
        raise ValueError('explicit matching vowel-pooling artifact required')
    encoder = load_encoder(protocol, False)
    encoder.encoder.layers[-2:].load_state_dict(artifact['last2'])
    head = JointVowelReadout(encoder.config.hidden_size, artifact['pooling']).to(next(encoder.parameters()).device)
    head.load_state_dict(artifact['head'])
    return encoder, head


def score_single(encoder, head, inputs):
    if set(inputs) != {'x', 'lengths', 'relative'} or SUPPORT_POLICY not in ('strict', 'joint'):
        raise ValueError('observed-only inputs and a frozen support policy required')
    output, available = [], []
    device, started = next(encoder.parameters()).device, time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    support = reliable_support if SUPPORT_POLICY == 'strict' else strict_or_joint_support
    with torch.no_grad():
        for i in range(len(inputs['x'])):
            x = torch.as_tensor(inputs['x'][i:i + 1], device=device)
            lengths = torch.as_tensor(inputs['lengths'][i:i + 1], device=device)
            mask = torch.arange(x.shape[1], device=device)[None] < lengths[:, None]
            relative = torch.as_tensor(inputs['relative'][i:i + 1], device=device)
            logits, roles = head(hidden(encoder, x, mask), mask, relative)
            output.append(logits.softmax(-1)[0].cpu().numpy())
            available.append(bool(support(roles, mask)[0]))
            if (i + 1) % 1000 == 0:
                print('single vowel requests', i + 1, 'seconds', round(time.monotonic() - started, 1), flush=True)
    torch.cuda.synchronize()
    seconds = time.monotonic() - started
    print('single vowel complete', len(output), 'seconds', seconds, 'peak_gpu_bytes', torch.cuda.max_memory_allocated(), flush=True)
    return np.asarray(output), np.asarray(available), seconds


def main():
    global SUPPORT_POLICY
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--stage', choices=('calibrate', 'evaluate'), required=True)
    ap.add_argument('--calibration')
    a = ap.parse_args()
    p = json.loads(Path(a.protocol).read_text())
    SUPPORT_POLICY = p['support_policy']
    evaluator.model, evaluator.score_single = model, score_single
    evaluator.main()
    shutil.copy2(__file__, Path(a.out) / Path(__file__).name)


if __name__ == '__main__':
    main()
