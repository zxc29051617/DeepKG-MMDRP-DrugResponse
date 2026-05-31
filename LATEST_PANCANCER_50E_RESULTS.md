# Pan-Cancer DRKG 50 Epoch Results

This experiment uses a pan-cancer DRKG subgraph constructed from 20 representative cancer driver genes:

TP53, KRAS, PIK3CA, PTEN, APC, BRAF, EGFR, ERBB2, MYC, CDKN2A, RB1, NRAS, HRAS, SMAD4, CTNNB1, BRCA1, BRCA2, ARID1A, IDH1, IDH2.

## Input Setup

- Drug response: CTRPv2 AAC and log2(IC50)
- Cell-line features: DepMap 26Q1 expression
- DRKG feature: pan-cancer driver-gene 1-hop DRKG subgraph, TransE 200d, 50 epochs
- Split: grouped by cell line
- Repeats: 3 seeds x 5 folds = 15 runs per model
- Epoch limit: 50 with early stopping where implemented
- Matched data after DRKG filtering: 44,049 cell line-drug pairs, 766 cell lines, 92 drugs

## Overall Results

| Model | AAC RMSE | AAC Pearson | log2IC50 RMSE | log2IC50 Pearson | Combined RMSE |
|---|---:|---:|---:|---:|---:|
| B0 | 0.1179 | 0.7769 | 2.5169 | 0.7638 | 0.8376 |
| B1 | 0.1167 | 0.7818 | 2.4992 | 0.7700 | 0.8314 |
| B2 | 0.1055 | 0.8256 | 2.3084 | 0.8063 | 0.7664 |
| B3 | 0.1046 | 0.8290 | 2.3209 | 0.8044 | 0.7695 |
| B3+DRKG | 0.1032 | 0.8330 | 2.2903 | 0.8094 | 0.7594 |

## NSCLC-Only Results

| Model | AAC RMSE | AAC Pearson | log2IC50 RMSE | log2IC50 Pearson | Combined RMSE |
|---|---:|---:|---:|---:|---:|
| B0 | 0.1114 | 0.7613 | 2.4595 | 0.7552 | 0.8158 |
| B1 | 0.1166 | 0.7294 | 2.5314 | 0.7347 | 0.8410 |
| B2 | 0.1087 | 0.7750 | 2.3732 | 0.7735 | 0.7881 |
| B3 | 0.1084 | 0.7754 | 2.3933 | 0.7696 | 0.7939 |
| B3+DRKG | 0.1077 | 0.7793 | 2.3543 | 0.7786 | 0.7817 |

## Interpretation

B3+PanCancer-DRKG is the best model in the overall pan-cancer evaluation and also remains best in the NSCLC-only subset. Compared with B3 base, B3+DRKG improves overall combined RMSE by 0.0101 and NSCLC-only combined RMSE by 0.0122.

The pan-cancer DRKG result is the appropriate result to cite when discussing overall pan-cancer performance. The original 8-gene NSCLC DRKG result remains the more disease-specific lung-cancer-oriented experiment.
