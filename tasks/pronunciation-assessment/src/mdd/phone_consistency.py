"""Reference-guided sparse phone losses for length-preserving noisy pairs.

Actual-phone labels supply hard targets. A frozen clean model supplies timing
only, and soft targets only where it agrees confidently with the human label.
Canonical phones are never substituted for actual-phone training targets.
"""

import numpy as np


def actual_error_mask(canonical, actual):
    """Mark substitutions/insertions on the actual-token axis; count deletions.

    Deletions have no actual token to anchor and are handled by sequence CTC.
    Use the same edlib alignment convention as the diagnostic evaluation.
    """
    import edlib
    if not canonical or not actual or any(len(t) != 1 for t in canonical + actual):
        raise ValueError("Expected nonempty single-character phone sequences")
    a, b = "".join(canonical), "".join(actual)
    nice = edlib.getNiceAlignment(edlib.align(a, b, task="path"), a, b)
    flags, deletions = [], 0
    for x, y in zip(nice["query_aligned"], nice["target_aligned"]):
        if y == "-":
            deletions += 1
        else:
            flags.append(x != y)
    if len(flags) != len(actual):
        raise RuntimeError("Actual-token alignment changed length")
    return np.asarray(flags, dtype=bool), deletions


def reference_anchors(log_probs, targets, errors, confidence=0.6, blank=0):
    """One CTC spike per actual token, without spreading phones over blanks."""
    from mdd.gop_feats_v2 import viterbi_align
    segments = viterbi_align(log_probs, targets, blank)
    if segments is None or any(not s for s in segments):
        raise ValueError("Actual reference cannot be CTC-aligned")
    peaks = np.asarray([s[int(np.argmax(log_probs[s, y]))]
                        for s, y in zip(segments, targets)], dtype=np.int64)
    probs = np.exp(log_probs[peaks]).astype(np.float32)
    correct = ((probs.argmax(-1) == np.asarray(targets)) &
               (probs[np.arange(len(targets)), targets] >= confidence))
    return {"peaks": peaks, "targets": np.asarray(targets, dtype=np.int64),
            "errors": np.asarray(errors, dtype=bool), "teacher_probs": probs,
            "teacher_correct": correct, "output_length": len(log_probs)}


def sparse_pair_losses(clean_logits, noisy_logits, anchors, error_weight=4.0):
    """Hard error supervision and gated clean-teacher/noisy KL, per token.

    Tokens are averaged within utterance, then over eligible utterances.
    Incorrect clean-teacher phones never become consistency targets.
    """
    import torch
    import torch.nn.functional as F
    ce, kl = [], []
    zero = (clean_logits.float().sum() + noisy_logits.float().sum()) * 0.0
    for i, anchor in enumerate(anchors):
        device = clean_logits.device
        peaks = torch.as_tensor(anchor["peaks"], device=device, dtype=torch.long)
        targets = torch.as_tensor(anchor["targets"], device=device, dtype=torch.long)
        errors = torch.as_tensor(anchor["errors"], device=device, dtype=torch.bool)
        correct = torch.as_tensor(anchor["teacher_correct"], device=device, dtype=torch.bool)
        if peaks.numel() == 0 or peaks.max() >= clean_logits.shape[1]:
            raise ValueError("Reference spike outside model output")
        clean = clean_logits[i, peaks].float()
        noisy = noisy_logits[i, peaks].float()
        if errors.any():
            ce.append(0.5 * (F.cross_entropy(clean[errors], targets[errors]) +
                             F.cross_entropy(noisy[errors], targets[errors])))
        if correct.any():
            teacher = torch.as_tensor(anchor["teacher_probs"], device=device,
                                      dtype=torch.float32)[correct]
            teacher = teacher / teacher.sum(-1, keepdim=True)
            divergence = F.kl_div(F.log_softmax(noisy[correct], -1), teacher,
                                  reduction="none").sum(-1)
            weights = torch.where(errors[correct], error_weight, 1.0)
            kl.append((divergence * weights).sum() / weights.sum())
    return (torch.stack(ce).mean() if ce else zero,
            torch.stack(kl).mean() if kl else zero)
