# Knowledge Graph-Augmented Multimodal Drug Response Prediction Using DepMap Expression, CTRPv2 Sensitivity Profiles, and DRKG Embeddings

This repository contains a modified MMDRP-based drug response prediction pipeline that adds Drug Repurposing Knowledge Graph (DRKG) embeddings as a third modality.

This project was done for the NTU course 醫學電資整合創意專題(二) (114-2, Spring 2026). 

Team: [HSIEN-TING,WANG](https://github.com/zxc29051617), [HU-CHIN-YU](https://github.com/HU-CHIN-YU),[HUANG, YU-WEI](https://github.com/yuwei03).

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
  Model hyperparameters used for B3/B3+DRKG. The final reported run overrides
  the Optuna-selected alpha with alpha = 0.8 from the command line.
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
Internal validation: 10% of training cell lines for early stopping
Main model epoch budget: maximum 50 epochs with early stopping
LDS: disabled
Multitask loss alpha: 0.8
```

The pan-cancer DRKG seed genes are:

```text
TP53, KRAS, PIK3CA, PTEN, APC, BRAF, EGFR, ERBB2,
MYC, CDKN2A, RB1, NRAS, HRAS, SMAD4, CTNNB1,
BRCA1, BRCA2, ARID1A, IDH1, IDH2
```

## Data Alignment Pipeline

The prediction dataset is built as an intersection of CTRPv2 response labels, DepMap expression profiles, and DRKG compound embeddings.

```text
CTRPv2 drug response
  AAC / log2(IC50) labels
  cpd_name, ccl_name, cpd_smiles
  226,455 cell line-drug pairs
  544 drugs
  791 cell lines

-> log2(IC50) range filter [-15, 15]
  220,795 cell line-drug pairs

-> match DepMap 26Q1 expression by normalized cell-line name
  213,843 cell line-drug pairs
  544 drugs
  767 cell lines

-> match DRKG compound embedding by normalized drug name
  44,049 final cell line-drug pairs
  92 drugs
  766 cell lines
```

The DRKG 20-gene seed list is used to construct the knowledge-graph subgraph and train TransE embeddings. Only compounds inside this 1-hop subgraph get an embedding, so a CTRPv2 drug can only be kept if DRKG has at least one edge between it and a seed gene. A CTRPv2 drug enters B3+DRKG only if it has a valid response label, a SMILES string, a matched DepMap cell-line expression profile, and a matched DRKG compound embedding.

DrugBank ID is used only as an intermediate mapping key for linking CTRPv2 compounds to DRKG compound entities. The model input is the final 200-dimensional DRKG TransE vector, not the DrugBank ID itself.

## Final Results

Overall held-out fold results, averaged over 15 seed-fold runs:

| Model | AAC RMSE | AAC Pearson | log2IC50 RMSE | log2IC50 Pearson | Combined RMSE |
|---|---:|---:|---:|---:|---:|
| B0 | 0.1179 | 0.7769 | 2.5169 | 0.7638 | 0.8376 |
| B1 | 0.1167 | 0.7818 | 2.4992 | 0.7700 | 0.8314 |
| B2 | 0.1055 | 0.8256 | 2.3084 | 0.8063 | 0.7664 |
| B3 | 0.1046 | 0.8290 | 2.3209 | 0.8044 | 0.7695 |
| B3+DRKG | **0.1032** | **0.8330** | **2.2903** | **0.8094** | **0.7594** |

NSCLC-only held-out fold results, averaged over the NSCLC subset of the same held-out folds:

| Model | AAC RMSE | AAC Pearson | log2IC50 RMSE | log2IC50 Pearson | Combined RMSE |
|---|---:|---:|---:|---:|---:|
| B0 | 0.1114 | 0.7613 | 2.4595 | 0.7552 | 0.8158 |
| B1 | 0.1166 | 0.7294 | 2.5314 | 0.7347 | 0.8410 |
| B2 | 0.1087 | 0.7750 | 2.3732 | 0.7735 | 0.7881 |
| B3 | 0.1084 | 0.7754 | 2.3933 | 0.7696 | 0.7939 |
| B3+DRKG | **0.1077** | **0.7793** | **2.3543** | **0.7786** | **0.7817** |

Combined RMSE is 0.7 × AAC RMSE + 0.3 × log2(IC50) RMSE, computed on the original scales. The same score is used for early stopping. All five models use the same 44,049 pairs and the same cell-line splits.

Notes on these results:

- B3+DRKG has the lowest mean combined RMSE: 0.0101 lower than B3 and 0.0070 lower than B2, and better than B3 in 12 of the 15 seed-fold runs.
- The 15 runs share training data, so a plain paired t-test (p = 0.002 against B3) overstates significance. With the Nadeau–Bengio correction for repeated cross-validation, p is about 0.11.
- Hyperparameters were tuned for B3+DRKG and reused for the other models, and no separate held-out set was kept.
- Combined RMSE is dominated by the log2(IC50) term.

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

The B3/B3+DRKG training entrypoint used by that pipeline is:

```text
src/run_b3_b3drkg_pancancer.py
```

Run the DRKG pipeline:

```powershell
.\run_drkg_pipeline.ps1
```

For HDF5-heavy commands on Windows, keep the repository in a folder whose path contains only ASCII characters
(PyTables/HDF5 cannot open paths with non-ASCII folder names).

Detailed reproduction notes are in:

```text
README_REPRODUCE.md
```

Important: this GitHub repository does not include the large raw/processed datasets or full fold-level checkpoints. To rerun the full training pipeline, first restore or regenerate the required data files documented in `README_REPRODUCE.md`, especially:

```text
Data/DRP_Training_Data/DepMap_26Q1_Expression.hdf
Data/DRP_Training_Data/CTRP_AAC_IC50_SMILES.hdf
DRKG_pipeline/official_data/drkg.tsv
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

Data sources:

- DepMap 26Q1 Public expression (Broad Institute): https://depmap.org/portal/
- CTRPv2 drug response (Broad Institute; Basu et al. 2013, Seashore-Ludlow et al. 2015, Rees et al. 2016), loaded with PharmacoGx.
- DRKG: https://github.com/gnn4dr/DRKG (Apache-2.0). DRKG combines sources with their own licenses, some of them non-commercial (for example DrugBank, CC BY-NC 4.0). These terms also apply to the DRKG-derived files in `Data/DRKG_Features/`.

See `NOTICE.md` and `LICENSE` for attribution and license details.
