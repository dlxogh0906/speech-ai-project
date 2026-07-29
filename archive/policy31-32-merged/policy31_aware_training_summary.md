# Policy-Aware 31-Class Training Summary

## Split

- Source feature cache: `$NATIONALITY_RUNS/features_fine_811`
- Policy labels: `$NATIONALITY_OUTPUTS/xlsr_nationality_inference_bundle/labels/label_map.json`
- Resplit feature dir: `$NATIONALITY_OUTPUTS/feature_resplit_policy31_seed42`
- Feature condition: XLS-R mean-pooled feature, 16 kHz, `max_sec=12`, dim 1920
- Split order: remap 61-class labels to policy31 first, then speaker-disjoint split
- Seed: 42

| split | samples | speakers | labels present |
|---|---:|---:|---:|
| train | 830,375 | 1,526 | 31 |
| valid | 106,227 | 191 | 27 |
| test | 105,667 | 193 | 29 |

Speaker leakage: 0.

Test-evaluable labels: 29/31.

Test-absent labels:

- `종카어`
- `타밀어`

The previously missing policy labels are now test-evaluable:

- `국가명_불명확`: train 12, valid 0, test 18
- `희소_인도유럽_기타`: train 142, valid 0, test 189

## Results

| model | test top1 | test top5 | test macro-F1 |
|---|---:|---:|---:|
| linear | 0.7216 | 0.9262 | 0.2759 |
| mlp512 | 0.6829 | 0.9129 | 0.2430 |
| avg-logit ensemble | 0.7122 | 0.9219 | 0.2658 |

Linear is still best. MLP512 overfits and underperforms linear.

## Output Dirs

- Split features: `$NATIONALITY_OUTPUTS/feature_resplit_policy31_seed42`
- Training/eval: `$NATIONALITY_OUTPUTS/policy31_aware_seed42_probe`
- Split script: `$NATIONALITY_OUTPUTS/resplit_feature_policy31.py`
