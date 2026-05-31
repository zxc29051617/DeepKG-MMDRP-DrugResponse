# MMDRP + DRKG Reproducible Package

這個資料夾是本次正式主線實驗的乾淨復現包：

```text
DepMap 26Q1 gene expression
+ CTRPv2 AAC / log2(IC50)
+ DRKG 1-hop lung-cancer TransE 200d feature
```

## 1. 資料來源

### DRKG

官方來源：

```text
https://github.com/gnn4dr/DRKG
https://dgl-data.s3-us-west-2.amazonaws.com/dataset/DRKG/drkg.tar.gz
```

本包使用的官方 DRKG 原始檔：

```text
DRKG_pipeline/official_data/drkg.tsv
DRKG_pipeline/official_data/entity2src.tsv
DRKG_pipeline/official_data/relation_glossary.tsv
DRKG_pipeline/official_data/embed/entities.tsv
DRKG_pipeline/official_data/embed/relations.tsv
```

### DepMap 26Q1 Expression

來源：

```text
DepMap Portal -> Custom Downloads -> Expression (Short-read) Public 26Q1
```

本包原始檔：

```text
Data/raw/depmap_26q1/DepMap_26Q1_Expression_custom.csv
```

MMDRP 實際讀取的檔案：

```text
Data/DRP_Training_Data/DepMap_26Q1_Expression.hdf
Data/DRP_Training_Data/DepMap_26Q1_Expression.csv
```

### CTRPv2

來源：

```text
PharmacoGx::downloadPSet("CTRPv2_2015")
```

本包原始檔：

```text
Data/raw/ctrpv2/CTRPv2.rds
```

MMDRP 實際讀取的檔案：

```text
Data/DRP_Training_Data/CTRP_AAC_IC50_SMILES.hdf
Data/DRP_Training_Data/CTRP_AAC_IC50_SMILES.csv
```

## 2. 資料夾結構

```text
Data/
  DRP_Training_Data/
    CTRP_AAC_IC50_SMILES.hdf
    DepMap_26Q1_Expression.hdf
  DRKG_Features/
    driver_gene_h1_200d_50e/
      mmdrp_drkg_transe_features.npy
      mmdrp_drkg_transe_index.csv
  raw/
    ctrpv2/
    depmap_26q1/

DRKG_pipeline/
  official_data/
  seeds/
  scripts/
  outputs/

src/
  run_attentivefp_multiseed_cv.py
  run_b0_b1_b2_baselines.py
  run_attentivefp_multitask_drkg_optuna.py
  prepare_ctrp_multitask_smiles_hdf.py
  prepare_depmap26q1_expression_hdf.py

config/
  best_config.json

results/
  B3_B3DRKG_26Q1_CTRPv2/
  B0_B1_B2_26Q1_CTRPv2/
```

`config/best_config.json` 是 B3/B3+DRKG 使用的 AttentiveFP/LMF 超參數設定，來自前一步 Optuna 搜尋結果。`run_all.ps1` 會明確讀取這份檔案。

## 3. 最快復現方式

在 PowerShell 執行：

```powershell
cd "C:\Users\user\Desktop\醫電整合fianl project\MMDRP_DRKG_repro_package_20260527"
.\run_all.ps1
```

如果 HDF5 或 Python 對中文路徑有問題，請先把整個資料夾複製到純英文路徑，例如：

```text
C:\MMDRP_DRKG_repro_package_20260527
```

再執行：

```powershell
cd "C:\MMDRP_DRKG_repro_package_20260527"
.\run_all.ps1
```

## 4. 目前已完成的正式結果

正式主線資料：

```text
DepMap 26Q1 + CTRPv2 AAC/log2(IC50) + DRKG 200d
3 seeds x 5-fold cell-line split
50 epochs
alpha = 0.8
no LDS
```

目前 50 epochs 結果摘要：

| Model | AAC RMSE | AAC Pearson | log2IC50 RMSE | log2IC50 Pearson | Combined RMSE |
|---|---:|---:|---:|---:|---:|
| B0 | 0.1185 | 0.7571 | 2.5613 | 0.7519 | 0.8513 |
| B1 | 0.1157 | 0.7673 | 2.5207 | 0.7621 | 0.8372 |
| B2 | 0.1056 | 0.8098 | 2.3328 | 0.7982 | 0.7738 |
| B3 | 0.1054 | 0.8123 | 2.3555 | 0.7954 | 0.7805 |
| B3+DRKG | 0.1043 | 0.8166 | 2.3328 | 0.7990 | 0.7728 |

NSCLC-only 主要結果：

| Model | AAC RMSE | AAC Pearson | log2IC50 RMSE | log2IC50 Pearson | Combined RMSE |
|---|---:|---:|---:|---:|---:|
| B0 | 0.1120 | 0.7440 | 2.5015 | 0.7488 | 0.8289 |
| B1 | 0.1168 | 0.7134 | 2.5790 | 0.7284 | 0.8554 |
| B2 | 0.1108 | 0.7501 | 2.4309 | 0.7627 | 0.8068 |
| B3 | 0.1107 | 0.7530 | 2.4591 | 0.7592 | 0.8152 |
| B3+DRKG | 0.1080 | 0.7642 | 2.4001 | 0.7701 | 0.7956 |

簡短結論：

```text
B3+DRKG 在 NSCLC-only evaluation 中為最佳模型，
並且在 overall pan-cancer evaluation 中也有最佳 AAC RMSE、AAC Pearson 與 combined RMSE。
因此目前可主張：以 NSCLC driver gene 子圖訓練出的 DRKG embedding
作為第三模態，可以改善 MMDRP 在肺癌細胞株評估上的藥物反應預測。
```

## 5. 從官方 DRKG 重新產生 KG feature

如果要從官方 DRKG raw data 重建 DRKG feature，執行：

```powershell
.\run_drkg_pipeline.ps1
```

這會重跑：

```text
official drkg.tsv
-> 8 個 NSCLC driver genes
-> 1-hop lung-cancer subgraph
-> TransE 200d / 50 epochs
-> DrugBank mapping
-> MMDRP DRKG feature matrix
```

注意：TransE 重訓會花時間。若只是要重跑 MMDRP，直接使用本包已整理好的：

```text
Data/DRKG_Features/driver_gene_h1_200d_50e/
```

## 6. 從 raw data 重新產生 MMDRP processed data

通常不需要重跑，因為本包已經放好 `.hdf`。如果要從 raw data 重建：

```powershell
$env:MMDRP_ROOT = (Get-Location).Path
python .\src\prepare_depmap26q1_expression_hdf.py
python .\src\prepare_ctrp_multitask_smiles_hdf.py
```
