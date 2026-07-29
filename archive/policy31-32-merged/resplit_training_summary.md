# Resplit Feature Training Summary

## Feature Split

- Source feature cache: `$NATIONALITY_RUNS/features_fine_811`
- Resplit feature dir: `$NATIONALITY_OUTPUTS/feature_resplit_61_merged6_seed42_target_counts`
- Feature condition: XLS-R mean-pooled feature, 16 kHz, `max_sec=12`, dim 1920
- Split rule: speaker-disjoint, seed 42, train/valid/test = 0.8/0.1/0.1
- Rare label rule: 1-speaker labels train-only; 2-speaker labels train/test; 3+ speaker labels ratio split

| split | samples | speakers | labels present |
|---|---:|---:|---:|
| train | 825,409 | 1,510 | 61 |
| valid | 98,883 | 197 | 35 |
| test | 117,977 | 203 | 41 |

Speaker leakage: 0.

## Results

| task | model | test classes | top1 | top5 | macro-F1 |
|---|---|---:|---:|---:|---:|
| 61-class merged6 | linear | 41/61 | 0.7559 | 0.9212 | 0.1873 |
| 61-class merged6 | mlp512 | 41/61 | 0.7092 | 0.9079 | 0.1325 |
| 31-class policy remap | linear | 27/31 | 0.7779 | 0.9340 | 0.3038 |
| 31-class policy remap | mlp512 | 27/31 | 0.7189 | 0.9197 | 0.2379 |
| 31-class policy remap | avg-logit ensemble | 27/31 | 0.7671 | 0.9325 | 0.2828 |

## Notes

Linear is best in both 61-class and 31-class runs. MLP512 overfits the cached
features quickly and underperforms linear on test.

The 31-class policy remap here is derived from the 61-class resplit. It has
27/31 test classes because `국가명_불명확`, `종카어`, `타밀어`, and
`희소_인도유럽_기타` have zero test samples after remapping this 61-class split.
This is not identical to a fully policy-aware 31-class split created before
feature extraction.

## Output Dirs

- 61-class linear: `$NATIONALITY_OUTPUTS/probe_61_merged6_resplit_seed42_linear`
- 61-class mlp512: `$NATIONALITY_OUTPUTS/probe_61_merged6_resplit_seed42_mlp512`
- 31-class policy: `$NATIONALITY_OUTPUTS/policy31_resplit_seed42_probe`
