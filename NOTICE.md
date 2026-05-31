# Notice and Attribution

This repository is a modified research package based on the original MMDRP codebase:

- Original repository: https://github.com/LincolnSteinLab/MMDRP
- Original project: Multi-Modal Drug Response Prediction (MMDRP)
- Original license: MIT License

## Major Modifications in This Package

- Updated the cell-line feature input to DepMap 26Q1 gene expression.
- Prepared CTRPv2 AAC and log2(IC50) multitask drug response inputs.
- Added DRKG preprocessing for driver-gene-oriented knowledge graph subgraphs.
- Added TransE 200-dimensional / 50-epoch DRKG embedding generation.
- Added DrugBank/DRKG compound mapping to produce MMDRP-compatible KG features.
- Added B0/B1/B2/B3/B3+DRKG experiment scripts and 3 seeds x 5 folds evaluation.
- Added pancancer and NSCLC-only metric summarization.
- Added pancancer visualization and Integrated Gradients attribution scripts.

## Data Availability

Large external datasets and generated checkpoints are intentionally excluded from Git tracking.

Expected large inputs include:

- DepMap 26Q1 expression data.
- CTRPv2 PharmacoSet-derived drug response data.
- Official DRKG dump.
- Generated fold-level model checkpoints and full prediction CSV files.

For reproducibility, place these files in the paths documented in `README_REPRODUCE.md` or provide them through a separate release/archive/cloud link.

## License Note

Because this package modifies and redistributes code derived from MMDRP, keep the original MMDRP MIT license notice in the repository. If this repository is published publicly, include a `LICENSE` file compatible with the original MIT license and cite the original MMDRP repository in the README.
