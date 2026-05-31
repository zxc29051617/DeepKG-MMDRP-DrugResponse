# 50 Epoch NSCLC-Only Evaluation Results

## Setting

- Data: DepMap 26Q1 expression + CTRPv2 AAC/log2(IC50) + DRKG 1-hop TransE 200d
- Split: cell-line-level GroupKFold style split
- Runs: 3 seeds x 5 folds = 15 runs per model
- Epoch limit: 50 for B1/B2/B3/B3+DRKG; B0 has no training epoch
- NSCLC definition: DepMap 26Q1 lineage_2 == Non-Small Cell Lung Cancer
- Evaluation: predictions are made on held-out validation/test fold cell lines, then NSCLC rows are selected only for metric calculation.

## NSCLC-Only Main Result

| evaluation_subset   | model_display   |   runs |   rows_mean |   cell_lines_mean |   drugs_mean |   AAC_RMSE_mean |   AAC_Pearson_mean |   AAC_Spearman_mean |   log2IC50_RMSE_mean |   log2IC50_Pearson_mean |   log2IC50_Spearman_mean |   combined_RMSE_mean |
|:--------------------|:----------------|-------:|------------:|------------------:|-------------:|----------------:|-------------------:|--------------------:|---------------------:|------------------------:|-------------------------:|---------------------:|
| NSCLC-only          | B0              |     15 |        1039 |              20.2 |           82 |          0.112  |             0.744  |              0.7279 |               2.5015 |                  0.7488 |                   0.7573 |               0.8289 |
| NSCLC-only          | B1              |     15 |        1039 |              20.2 |           82 |          0.1168 |             0.7134 |              0.6837 |               2.579  |                  0.7284 |                   0.716  |               0.8554 |
| NSCLC-only          | B2              |     15 |        1039 |              20.2 |           82 |          0.1108 |             0.7501 |              0.7318 |               2.4309 |                  0.7627 |                   0.76   |               0.8068 |
| NSCLC-only          | B3              |     15 |        1039 |              20.2 |           82 |          0.1107 |             0.753  |              0.7333 |               2.4591 |                  0.7592 |                   0.7581 |               0.8152 |
| NSCLC-only          | B3+DRKG         |     15 |        1039 |              20.2 |           82 |          0.108  |             0.7642 |              0.7433 |               2.4001 |                  0.7701 |                   0.765  |               0.7956 |

## Overall Pan-Cancer Result

| evaluation_subset   | model_display   |   runs |   rows_mean |   cell_lines_mean |   drugs_mean |   AAC_RMSE_mean |   AAC_Pearson_mean |   AAC_Spearman_mean |   log2IC50_RMSE_mean |   log2IC50_Pearson_mean |   log2IC50_Spearman_mean |   combined_RMSE_mean |
|:--------------------|:----------------|-------:|------------:|------------------:|-------------:|----------------:|-------------------:|--------------------:|---------------------:|------------------------:|-------------------------:|---------------------:|
| overall             | B0              |     15 |      7930.8 |             153.2 |      83.8667 |          0.1185 |             0.7571 |              0.7278 |               2.5613 |                  0.7519 |                   0.7526 |               0.8513 |
| overall             | B1              |     15 |      7930.8 |             153.2 |      83.8667 |          0.1157 |             0.7673 |              0.7185 |               2.5207 |                  0.7621 |                   0.751  |               0.8372 |
| overall             | B2              |     15 |      7930.8 |             153.2 |      83.8667 |          0.1056 |             0.8098 |              0.774  |               2.3328 |                  0.7982 |                   0.7935 |               0.7738 |
| overall             | B3              |     15 |      7930.8 |             153.2 |      83.8667 |          0.1054 |             0.8123 |              0.7763 |               2.3555 |                  0.7954 |                   0.79   |               0.7805 |
| overall             | B3+DRKG         |     15 |      7930.8 |             153.2 |      83.8667 |          0.1043 |             0.8166 |              0.7837 |               2.3328 |                  0.799  |                   0.7948 |               0.7728 |

## Main Interpretation

- B3+DRKG is best on NSCLC-only AAC RMSE, AAC Pearson, log2(IC50) RMSE, log2(IC50) Pearson, and combined RMSE.
- B3+DRKG improves over B3 on NSCLC-only combined RMSE in 13/15 runs.
- This matches the target design: pan-cancer training with NSCLC-focused evaluation.
