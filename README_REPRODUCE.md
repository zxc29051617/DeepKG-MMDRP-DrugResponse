# Reproduction Guide

This repository contains the code, small feature files, summary tables, figures, and attribution outputs for the final pan-cancer MMDRP + DRKG experiment.

Large raw datasets, official DRKG dumps, fold-level checkpoints, and full prediction tables are intentionally excluded from Git tracking. They must be restored separately before a full rerun.

## 1. Main Experiment

Final reported setting:

```text
Expression input: DepMap 26Q1 gene expression
Drug response labels: CTRPv2 AAC and log2(IC50)
Drug structure input: CTRPv2 SMILES
KG input: DRKG TransE embeddings from a pan-cancer driver-gene subgraph
DRKG seed genes: 20 pan-cancer driver genes
DRKG traversal: 1-hop
TransE embedding dimension: 200
TransE epochs: 50
Prediction models: B0, B1, B2, B3, B3+DRKG
Prediction training: 3 seeds x 5 folds
Split type: cell-line split
Internal validation: 10% of training rows for early stopping
Main model epoch budget: maximum 50 epochs with early stopping
Multitask targets: AAC and log2(IC50)
Multitask loss alpha: 0.8
LDS: disabled
```

The 20 pan-cancer DRKG seed genes are stored in:

```text
DRKG_pipeline/seeds/pancancer_driver_genes.tsv
```

Seed gene list:

```text
TP53, KRAS, PIK3CA, PTEN, APC, BRAF, EGFR, ERBB2,
MYC, CDKN2A, RB1, NRAS, HRAS, SMAD4, CTNNB1,
BRCA1, BRCA2, ARID1A, IDH1, IDH2
```

## 2. Data Alignment Pipeline

The final model input is the intersection of CTRPv2 response labels, DepMap expression profiles, and DRKG compound embeddings.

```text
CTRPv2 response table
  columns: cpd_name, ccl_name, primary_disease, area_above_curve, ic50, log2_ic50, cpd_smiles
  226,455 cell line-drug pairs
  544 drugs
  791 cell lines

-> filter to log2_ic50 in [-15, 15]
  220,795 cell line-drug pairs

-> align with DepMap 26Q1 expression by normalized ccl_name / stripped_cell_line_name
  213,843 cell line-drug pairs
  544 drugs
  767 cell lines

-> align with DRKG compound embeddings by normalized cpd_name
  44,049 final cell line-drug pairs
  92 drugs
  766 cell lines
```

The code may load the DRKG mapping before the DepMap expression matrix, but the final dataset is the same intersection:

```text
CTRPv2 ∩ DepMap 26Q1 expression ∩ DRKG compound embedding
```

The 20 pan-cancer seed genes are used only for DRKG subgraph construction and TransE training. They do not directly select CTRPv2 drugs. DrugBank ID is used only as an intermediate compound mapping key; the prediction model receives the 200-dimensional DRKG embedding vector.

## 3. Repository Layout

```text
src/
  prepare_depmap26q1_expression_hdf.py
  prepare_ctrp_multitask_smiles_hdf.py
  run_b0_b1_b2_baselines.py
  run_attentivefp_multiseed_cv.py
  run_attentivefp_multitask_drkg_optuna.py
  summarize_nsclc_prediction_metrics.py

analysis/
  visualization/run_pancancer_visualization.py
  attribution/run_pancancer_ig_attribution.py

DRKG_pipeline/
  scripts/extract_driver_gene_subgraph.py
  scripts/train_transe.py
  scripts/map_drugbank_to_drkg_embeddings.py
  seeds/pancancer_driver_genes.tsv

Data/DRKG_Features/
  pancancer_driver_h1_200d_50e/

results/
  pancancer_50e_evaluation/
  pancancer_visualization_figures/
  pancancer_ig_attribution/

config/
  best_config.json
```

## 4. Required External Data

These files are not tracked in GitHub because they are large.

Required for full training reruns:

```text
Data/DRP_Training_Data/DepMap_26Q1_Expression.hdf
Data/DRP_Training_Data/CTRP_AAC_IC50_SMILES.hdf
```

Required for rerunning the DRKG feature-generation pipeline:

```text
DRKG_pipeline/official_data/drkg.tsv
DRKG_pipeline/official_data/entity2src.tsv
DRKG_pipeline/official_data/relation_glossary.tsv
DRKG_pipeline/official_data/embed/entities.tsv
DRKG_pipeline/official_data/embed/relations.tsv
Data/raw/ctrpv2/week1_final_drug_smiles_recovered_from_mapping.csv
```

Source notes:

```text
DepMap 26Q1 expression:
  DepMap Portal -> Custom Downloads -> Expression, Public 26Q1

CTRPv2:
  PharmacoGx::downloadPSet("CTRPv2_2015")

DRKG:
  https://github.com/gnn4dr/DRKG
  https://dgl-data.s3-us-west-2.amazonaws.com/dataset/DRKG/drkg.tar.gz
```

## 5. Install Dependencies

```powershell
pip install -r requirements.txt
```

For HDF5-heavy commands on Windows, use an ASCII-only path such as:

```text
C:\MMDRP_DRKG_repro_package_20260531
```

This avoids PyTables/HDF5 path issues with non-ASCII folder names.

## 6. Rerun the Pan-Cancer Prediction Experiment

After restoring the required HDF5 files, run:

```powershell
.\run_pancancer_all.ps1
```

This runs:

```text
B3 and B3+DRKG:
  src/run_attentivefp_multiseed_cv.py

B0, B1, and B2:
  src/run_b0_b1_b2_baselines.py

Overall and NSCLC-only summaries:
  src/summarize_nsclc_prediction_metrics.py
```

Rerun outputs are written to:

```text
results/reproduced_pancancer_B3_B3DRKG_26Q1_CTRPv2_50e/
results/reproduced_pancancer_B0_B1_B2_26Q1_CTRPv2_50e/
results/reproduced_pancancer_50e_evaluation/
```

## 7. Rerun the Pan-Cancer DRKG Pipeline

After restoring official DRKG files and the drug mapping CSV, run:

```powershell
.\run_drkg_pipeline.ps1
```

Pipeline:

```text
official DRKG
-> 20 pan-cancer driver genes
-> 1-hop driver-gene subgraph
-> TransE 200d / 50 epochs
-> CTRPv2 drug-to-DRKG compound mapping
-> MMDRP-ready DRKG feature matrix
```

Rerun outputs are written to:

```text
DRKG_pipeline/outputs/reproduced_pancancer_driver_subgraph_h1/
Data/DRKG_Features/reproduced_pancancer_driver_h1_200d_50e/
```

## 8. Generate Figures and Attribution Outputs

Generate result figures:

```powershell
python analysis/visualization/run_pancancer_visualization.py
```

Run Integrated Gradients attribution:

```powershell
python analysis/attribution/run_pancancer_ig_attribution.py
```

Default outputs:

```text
results/pancancer_visualization_figures/
results/pancancer_ig_attribution/
```

## 9. Final Reported Results

Main result files:

```text
LATEST_PANCANCER_50E_RESULTS.md
LATEST_PANCANCER_50E_MODEL_COMPARISON.csv
results/pancancer_50e_evaluation/overall_summary.csv
results/pancancer_50e_evaluation/nsclc_only_summary.csv
```

The GitHub repository includes summary tables and figures, but not the large fold-level checkpoints or full prediction CSV files.
