# B0/B1/B2 Baseline Results

## Data

This baseline run uses the same data setting as the current B3/B3+DRKG comparison:

- Drug response: CTRPv2
- Targets: AAC + log2(IC50)
- Expression: DepMap Public 26Q1
- Matched drugs: 84
- Cell lines: 766
- Cell line-drug pairs: 39,654
- Split: 3 seeds x 5-fold cell-line CV
- Seeds: 42, 43, 44

## Baseline Definitions

| Baseline | Definition |
|---|---|
| B0 | Per-drug median trivial baseline |
| B1 | Elastic Net linear baseline |
| B2 | MMDRP-base style dual-modal model, expression + Morgan fingerprint, no LMF / LDS / GNN |
| B3 | AttentiveFP + LMF multitask model |
| B3+DRKG | B3 plus DRKG TransE embedding as a third modality |

## Results

| Model | AAC RMSE | AAC Pearson | AAC Spearman | log2IC50 RMSE | log2IC50 Pearson | log2IC50 Spearman | Combined RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | 0.1185 | 0.7571 | 0.7278 | 2.5613 | 0.7519 | 0.7526 | 0.8513 |
| B1 | 0.1284 | 0.7191 | 0.6507 | 2.8476 | 0.7073 | 0.6788 | 0.9441 |
| B2 | 0.1058 | 0.8090 | 0.7735 | 2.3380 | 0.7971 | 0.7931 | 0.7754 |
| B3 | 0.1060 | 0.8108 | 0.7726 | 2.3699 | 0.7925 | 0.7856 | 0.7852 |
| B3+DRKG | 0.1045 | 0.8157 | 0.7836 | 2.3464 | 0.7965 | 0.7935 | 0.7770 |

## Interpretation

- B3+DRKG is the best model by AAC RMSE, AAC Pearson, and AAC Spearman.
- B3+DRKG clearly beats B0 in this same-data setting.
- B2 is very competitive and has the best log2(IC50) RMSE and combined RMSE.
- B1 underperforms in this run; it was implemented as an elastic-net-penalized linear model via SGDRegressor for scalability on the full expression + Morgan feature space.

Recommended report wording:

> Under the DepMap 26Q1 + CTRPv2 setting, adding DRKG to B3 improves AAC prediction and outperforms the per-drug median baseline. However, B2 remains competitive on log2(IC50) and combined RMSE, suggesting that further tuning of the B3/B3+DRKG multitask balance may be needed.

## Files

- `cv_summary.csv`: B0/B1/B2 summary
- `cv_metrics.csv`: per seed/fold metrics
- `paired_stats_vs_b0.csv`: B1/B2 paired comparison against B0
- `b0_b1_b2_b3_drkg_overview.csv`: combined B0-B3+DRKG overview
- `run_config.json`: run configuration
- Source script: `C:\mmdrp_exp\src\run_b0_b1_b2_baselines.py`
