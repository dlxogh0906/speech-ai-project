# Policy31 Duration Sweep - Existing Split vs Other GPU Split

| split | train / valid / test | duration | test n | classes | best val top5 | top1 | top5 | macro F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| existing_policy31_split | train 830,375 / valid 106,227 / test 105,667 | 8s | 105,667 | 29 / 31 | 0.9195 | 0.6965 | 0.9163 | 0.2570 |
| existing_policy31_split | train 830,375 / valid 106,227 / test 105,667 | 12s | 105,667 | 29 / 31 | 0.9246 | 0.7216 | 0.9262 | 0.2759 |
| existing_policy31_split | train 830,375 / valid 106,227 / test 105,667 | 16s | 105,667 | 29 / 31 | 0.9257 | 0.7296 | 0.9289 | 0.2777 |
| othergpu_split | train 844,483 / valid 96,138 / test 101,648 | 8s | 101,648 | 29 / 31 | 0.9212 | 0.7113 | 0.9131 | 0.2319 |
| othergpu_split | train 844,483 / valid 96,138 / test 101,648 | 12s | 101,648 | 29 / 31 | 0.9293 | 0.7334 | 0.9198 | 0.2397 |
| othergpu_split | train 844,483 / valid 96,138 / test 101,648 | 16s | 101,648 | 29 / 31 | 0.9313 | 0.7355 | 0.9219 | 0.2414 |

## Same-Duration Difference

`othergpu_split - existing_policy31_split`

| duration | Δtest n | Δtop1 | Δtop5 | Δmacro F1 |
|---:|---:|---:|---:|---:|
| 8s | -4,019 | +0.0148 | -0.0032 | -0.0251 |
| 12s | -4,019 | +0.0118 | -0.0064 | -0.0362 |
| 16s | -4,019 | +0.0058 | -0.0070 | -0.0363 |

## Short Read

- Both splits show the same trend: `8s < 12s < 16s`.
- Existing split has better top5 and macro F1 at every duration.
- Othergpu split has better top1 at 8s/12s/16s, but lower class-balanced macro F1.
- 16s is best in both splits, but 12s is close and captures most of the duration gain.
