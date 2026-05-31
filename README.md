# Knowledge Graph-Augmented Multimodal Drug Response Prediction Using DepMap Expression, CTRPv2 Sensitivity Profiles, and DRKG Embeddings

This repository contains a modified MMDRP-based drug response prediction pipeline that adds Drug Repurposing Knowledge Graph (DRKG) embeddings as a third modality.

The final model compares baseline and multimodal variants using DepMap 26Q1 gene expression, CTRPv2 AAC/log2(IC50) drug sensitivity labels, molecular drug structure features, and DRKG TransE embeddings.

Deep learning components in this repository include:

- An expression MLP branch for cell-line gene expression.
- An AttentiveFP-based graph neural network branch for drug molecular structure.
- A DRKG embedding MLP branch for knowledge graph features.
- Low-rank multimodal fusion (LMF) for combining expression, drug structure, and KG modalities.
- Multitask regression for AAC and log2(IC50).
- Integrated Gradients attribution for model interpretation.

## Project Summary

The goal is to test whether adding knowledge graph information improves drug response prediction.

Input modalities:

- Cell-line gene expression from DepMap 26Q1.
- Drug molecular structure from CTRPv2 SMILES.
- Knowledge graph embedding from DRKG.

Prediction targets:

- AAC, area above the dose-response curve.
- log2(IC50).

Main comparison:

```text
B0: per-drug median baseline
B1: Elastic Net baseline
B2: MMDRP-base-style expression + Morgan fingerprint model
B3: expression + AttentiveFP drug branch + LMF multitask model
B3+DRKG: B3 plus DRKG embedding as a third modality
```

## Deep Learning Architecture

The main B3+DRKG model is a three-modality deep learning architecture:

```text
DepMap expression
  -> expression MLP
  -> 128-dimensional representation

Drug SMILES
  -> molecular graph
  -> AttentiveFP GNN
  -> 128-dimensional representation

DRKG TransE embedding
  -> KG MLP
  -> 128-dimensional representation

expression representation
+ drug representation
+ KG representation
  -> low-rank multimodal fusion (LMF)
  -> multitask regression head
  -> AAC and log2(IC50)
```

This allows the model to jointly learn from cell-line omics, molecular drug structure, and biomedical knowledge graph context.

## Repository Layout

```text
src/
  Data preparation, baseline training, B3/B3+DRKG training, metric summarization.

analysis/
  Visualization and Integrated Gradients attribution scripts.

DRKG_pipeline/
  DRKG subgraph extraction, TransE training, and DrugBank-to-DRKG mapping scripts.

Data/DRKG_Features/
  Small MMDRP-ready DRKG feature matrices and mapping tables.

results/
  Summary metrics, final comparison tables, generated figures, and attribution outputs.

config/
  Model configuration used for B3/B3+DRKG.
```

Large raw datasets, HDF5 files, official DRKG dumps, and fold-level checkpoints are intentionally excluded from Git tracking.

## Final Experiment Setting

```text
DepMap: 26Q1 expression
Drug response: CTRPv2 AAC + log2(IC50)
DRKG embedding: pan-cancer driver-gene 1-hop subgraph
Embedding model: TransE
Embedding dimension: 200
TransE epochs: 50
Prediction training: 3 seeds x 5 folds
Split type: cell-line split
Main model epochs: 50
LDS: disabled
Multitask loss alpha: 0.8
```

The pan-cancer DRKG seed genes are:

```text
TP53, KRAS, PIK3CA, PTEN, APC, BRAF, EGFR, ERBB2,
MYC, CDKN2A, RB1, NRAS, HRAS, SMAD4, CTNNB1,
BRCA1, BRCA2, ARID1A, IDH1, IDH2
```

## Final Results

Overall validation results:

| Model | AAC RMSE | AAC Pearson | log2IC50 RMSE | log2IC50 Pearson | Combined RMSE |
|---|---:|---:|---:|---:|---:|
| B0 | 0.1179 | 0.7769 | 2.5169 | 0.7638 | 0.8376 |
| B1 | 0.1167 | 0.7818 | 2.4992 | 0.7700 | 0.8314 |
| B2 | 0.1055 | 0.8256 | 2.3084 | 0.8063 | 0.7664 |
| B3 | 0.1046 | 0.8290 | 2.3209 | 0.8044 | 0.7695 |
| B3+DRKG | **0.1032** | **0.8330** | **2.2903** | **0.8094** | **0.7594** |

NSCLC-only validation results:

| Model | AAC RMSE | AAC Pearson | log2IC50 RMSE | log2IC50 Pearson | Combined RMSE |
|---|---:|---:|---:|---:|---:|
| B0 | 0.1114 | 0.7613 | 2.4595 | 0.7552 | 0.8158 |
| B1 | 0.1166 | 0.7294 | 2.5314 | 0.7347 | 0.8410 |
| B2 | 0.1087 | 0.7750 | 2.3732 | 0.7735 | 0.7881 |
| B3 | 0.1084 | 0.7754 | 2.3933 | 0.7696 | 0.7939 |
| B3+DRKG | **0.1077** | **0.7793** | **2.3543** | **0.7786** | **0.7817** |

Detailed result files:

```text
LATEST_PANCANCER_50E_RESULTS.md
LATEST_PANCANCER_50E_MODEL_COMPARISON.csv
results/pancancer_50e_evaluation/
```

## Visualization and Attribution

Generate final figures:

```powershell
python analysis/visualization/run_pancancer_visualization.py
```

Run Integrated Gradients attribution:

```powershell
python analysis/attribution/run_pancancer_ig_attribution.py
```

Outputs:

```text
results/pancancer_visualization_figures/
results/pancancer_ig_attribution/
```

More details:

```text
analysis/README_ANALYSIS.md
README_ATTRIBUTION_FILES.md
```

## Reproducibility

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run the pan-cancer experiment pipeline:

```powershell
.\run_pancancer_all.ps1
```

Run the DRKG pipeline:

```powershell
.\run_drkg_pipeline.ps1
```

For HDF5-heavy commands on Windows, use an ASCII-only path such as:

```text
C:\MMDRP_DRKG_repro_package_20260531
```

Detailed reproduction notes are in:

```text
README_REPRODUCE.md
```

## Data Availability

The following large files are not intended to be tracked in Git:

- DepMap 26Q1 expression CSV/HDF5 files.
- CTRPv2 PharmacoSet-derived CSV/HDF5 files.
- Official DRKG dump files.
- Full fold-level PyTorch checkpoints.
- Full prediction CSV files.

They should be provided separately through a release archive, cloud storage, or regenerated using the documented scripts.

## Attribution

This project is based on the original MMDRP implementation:

- Original repository: https://github.com/LincolnSteinLab/MMDRP
- Original license: MIT License

Major modifications in this repository include:

- Updating the expression input to DepMap 26Q1.
- Preparing CTRPv2 AAC/log2(IC50) multitask labels.
- Adding DRKG TransE embeddings as a third modality.
- Adding B0/B1/B2/B3/B3+DRKG comparison experiments.
- Adding pan-cancer and NSCLC-only evaluation.
- Adding visualization and Integrated Gradients attribution scripts.

See `NOTICE.md` and `LICENSE` for attribution and license details.
