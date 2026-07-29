# 62-Class Speaker-Invariant Experiment Summary

## Setup

- Labels: original 61 classes + `한국어_모국어`
- Feature: frozen XLS-R 12-second mean-pooled embedding, 1920 dimensions
- Split: same speaker-disjoint split as the 62-class linear baseline
- Balanced batch rule: class → speaker → utterance
- Contrastive positives: same language, different SpeakerID
- Speaker adversarial loss: speakers are compared only within the same language

## Results

| experiment | overall top1 | overall top5 | overall macro F1 | foreign top1 | foreign top5 | foreign macro F1 |
|---|---:|---:|---:|---:|---:|---:|
| linear baseline | 0.7919 | 0.9310 | 0.2006 | 0.7568 | 0.9194 | 0.1822 |
| fully balanced CE | 0.6601 | 0.9071 | 0.1226 | 0.6030 | 0.8920 | 0.1068 |
| fully balanced + SupCon | 0.7027 | 0.9109 | 0.1307 | 0.6527 | 0.8964 | 0.1158 |
| fully balanced + SupCon + conditional speaker adversarial | 0.6688 | 0.9109 | 0.1213 | 0.6134 | 0.8963 | 0.1067 |
| 80% natural + 20% balanced + SupCon | 0.7879 | 0.9309 | 0.1978 | 0.7521 | 0.9192 | 0.1795 |
| 80% natural + 20% balanced + SupCon + conditional speaker adversarial | 0.7875 | 0.9309 | 0.1978 | 0.7516 | 0.9193 | 0.1795 |

## Interpretation

- Fully balancing all 62 classes strongly over-samples classes represented by one speaker.
  It therefore amplifies speaker memorization rather than learning a general mother-tongue cue.
- Cross-speaker SupCon is better than balanced CE, but it cannot create cross-speaker
  information for classes that contain only one speaker.
- Conservative mixed fine-tuning preserves most baseline accuracy but does not beat the
  baseline on test Macro F1.
- Conditional speaker adversarial training reduces speaker clustering in the learned
  projection, but this does not translate into better held-out classification.
- Validation-selected logit prior adjustment also chose `tau=0`; simple class-prior
  correction did not improve Macro F1.

## Conclusion

The original linear baseline remains the best model under the current frozen feature cache.
The next technically meaningful experiment is to unfreeze the upper XLS-R layers and train
them with cross-speaker SupCon. For one-speaker classes, however, additional independent
speakers are still required for genuine mother-tongue generalization.
