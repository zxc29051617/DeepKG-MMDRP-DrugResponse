# Attribution / IG Files

This repository includes the attribution scripts and default attribution outputs. The local full package can also include full pan-cancer fold outputs for downstream Integrated Gradients (IG) or attribution analysis, but large checkpoints and fold-level prediction files are intentionally excluded from GitHub.

## Main Fold Outputs

- `results/pancancer_B3_B3DRKG_26Q1_CTRPv2_50e_full_folds/` (local package only; ignored by Git)
  - 15 `split_indices.npz`
  - 30 PyTorch model checkpoints:
    - `cv_base.pt`
    - `cv_plus_drkg.pt`
  - 30 training histories
  - 30 test prediction CSV files

- `results/pancancer_B0_B1_B2_26Q1_CTRPv2_50e_full_folds/` (local package only; ignored by Git)
  - 15 `split_indices.npz`
  - 15 B2 PyTorch model checkpoints:
    - `B2_mmdrp_base.pt`
  - 45 test prediction CSV files
  - `expression_feature_names.csv`
  - `morgan_feature_names.csv`

## Required Inputs

- `Data/DRP_Training_Data/DepMap_26Q1_Expression.hdf`
- `Data/DRP_Training_Data/CTRP_AAC_IC50_SMILES.hdf`
- `Data/DRKG_Features/pancancer_driver_h1_200d_50e/mmdrp_drkg_transe_features.npy`
- `Data/DRKG_Features/pancancer_driver_h1_200d_50e/mmdrp_drkg_transe_index.csv`
- `config/best_config.json`
- `src/`

## Notes

- B0 has no model checkpoint because it is a per-drug median baseline.
- B1 is an Elastic Net baseline, not a PyTorch model, so no `.pt` checkpoint is stored.
- B2, B3, and B3+DRKG have `.pt` checkpoints.
- For B3+DRKG attribution, use `cv_plus_drkg.pt` with the matching fold's `split_indices.npz`.
- Ready-to-run attribution and visualization scripts are under `analysis/`.
- The default IG script output is `results/pancancer_ig_attribution/`.
- The default figure script output is `results/pancancer_visualization_figures/`.
