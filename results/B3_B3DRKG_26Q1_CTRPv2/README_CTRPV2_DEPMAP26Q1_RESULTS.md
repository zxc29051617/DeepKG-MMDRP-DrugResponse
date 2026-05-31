# CTRPv2 + DepMap 26Q1 Expression Result

## Purpose

This experiment keeps the original proposal target:

- CTRPv2 drug response
- AAC
- log2(IC50)

Only the expression matrix is updated:

- Old: DepMap 21Q2 expression
- New: DepMap Public 26Q1 expression

This is the recommended "updated expression but same target" comparison.

## Dataset After Matching

- Cell line-drug pairs: 39,654
- Matched DRKG drugs: 84
- Cell lines: 766
- Expression features: 19,215
- DRKG feature: 1-hop TransE 200d / 50 epochs

## Training Setting

- Model: B3 AttentiveFP multitask
- Comparison: base vs +DRKG
- Split: 3 seeds x 5-fold cell-line CV
- Seeds: 42, 43, 44
- Epochs: 20
- Alpha: 0.8
- LDS: off

## Result

| Model | AAC RMSE | AAC Pearson | AAC Spearman | log2IC50 RMSE | log2IC50 Pearson | log2IC50 Spearman | Combined RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base | 0.1060 | 0.8108 | 0.7726 | 2.3699 | 0.7925 | 0.7856 | 0.7852 |
| +DRKG | 0.1045 | 0.8157 | 0.7836 | 2.3464 | 0.7965 | 0.7935 | 0.7770 |

## Paired Statistics

| Metric | Improved Runs | p-value |
|---|---:|---:|
| AAC MAE | 13 / 15 | 0.0170 |
| AAC RMSE | 10 / 15 | 0.0417 |
| AAC Pearson | 11 / 15 | 0.0616 |
| AAC Spearman | 13 / 15 | 0.0023 |
| log2IC50 MAE | 12 / 15 | 0.0147 |
| log2IC50 RMSE | 9 / 15 | 0.1189 |
| log2IC50 Pearson | 9 / 15 | 0.1542 |
| log2IC50 Spearman | 12 / 15 | 0.0252 |
| Combined RMSE | 9 / 15 | 0.1053 |

## Compared With Previous 21Q2 Result

| Data Version | Model | AAC RMSE | AAC Pearson | log2IC50 RMSE | log2IC50 Pearson | Combined RMSE |
|---|---|---:|---:|---:|---:|---:|
| DepMap 21Q2 + CTRPv2 | Base | 0.1066 | 0.8084 | 2.3930 | 0.7888 | 0.7925 |
| DepMap 21Q2 + CTRPv2 | +DRKG | 0.1047 | 0.8151 | 2.3390 | 0.7987 | 0.7750 |
| DepMap 26Q1 + CTRPv2 | Base | 0.1060 | 0.8108 | 2.3699 | 0.7925 | 0.7852 |
| DepMap 26Q1 + CTRPv2 | +DRKG | 0.1045 | 0.8157 | 2.3464 | 0.7965 | 0.7770 |

Interpretation:

- Updating expression to DepMap 26Q1 does not make the experiment clearly worse.
- The 26Q1 base model improves over the 21Q2 base model.
- The 26Q1 +DRKG model remains better than the 26Q1 base model.
- Compared with the previous 21Q2 +DRKG model, 26Q1 +DRKG has slightly better AAC RMSE/Pearson, but slightly worse log2IC50 RMSE/Pearson and combined RMSE.
- The difference is small, so this should be reported as broadly comparable rather than a clear win or loss.

## Files

- Metrics: `cv_summary.csv`
- Per-run metrics: `cv_metrics.csv`
- Paired stats: `paired_stats.csv`
- 21Q2 vs 26Q1 comparison: `compare_21q2_vs_26q1.csv`
- Run config: `run_config.json`
