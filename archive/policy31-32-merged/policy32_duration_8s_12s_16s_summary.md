# Policy32 Duration Comparison

Fixed split: train `1,004,483`, valid `116,138`, test `121,648`.

| duration | overall top1 | overall top5 | overall macro F1 | foreign top1 | foreign top5 | foreign macro F1 | Korean precision | Korean recall | Korean F1 | foreign -> Korean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8s | 0.7568 | 0.9268 | 0.2573 | 0.7090 | 0.9124 | 0.2240 | 0.9960 | 0.9997 | 0.9979 | 80 |
| 12s | 0.7767 | 0.9333 | 0.2711 | 0.7328 | 0.9201 | 0.2378 | 0.9987 | 0.9998 | 0.9992 | 27 |
| 16s | 0.7818 | 0.9365 | 0.2759 | 0.7390 | 0.9240 | 0.2426 | 0.9987 | 0.9997 | 0.9992 | 27 |

## Observations

- All aggregate metrics improve monotonically from 8s to 12s to 16s.
- 16s is best; 12s is close and may be the better latency/accuracy tradeoff.
- Korean recall is nearly unchanged, but Korean precision is lower at 8s because more foreign samples are predicted as Korean.
- Independent native-Korean validation is still required because the two corpora have different collection pipelines.
