# MMDRP + DRKG run status

## Current DRKG output

The DRKG side is ready to be used as the MMDRP third modality.

- Main feature matrix:
  - `outputs/driver_gene_subgraph_h1/drugbank_mmdrp_input_200d_50e/mmdrp_drkg_transe_features.npy`
  - shape: `(104, 200)`
- Row index / mapping table:
  - `outputs/driver_gene_subgraph_h1/drugbank_mmdrp_input_200d_50e/mmdrp_drkg_transe_index.csv`
- Human-readable merged table:
  - `outputs/driver_gene_subgraph_h1/drugbank_mmdrp_input_200d_50e/mmdrp_drkg_transe_merged.csv`

This output was built from the 1-hop lung-cancer driver-gene DRKG subgraph and TransE 200-dimensional embeddings trained for 50 epochs.

## MMDRP run blocker

MMDRP cannot be run yet on this machine because the cloned MMDRP repository does not include the required preprocessed training data.

The missing training data directory is expected to contain files such as:

- `CTRP_AAC_MORGAN.hdf` or `CTRP_AAC_MORGAN_512.hdf`
- `DepMap_21Q2_Expression.hdf`
- optionally `DepMap_21Q2_CopyNumber.hdf`, `DepMap_21Q2_Mutations_by_Cell.hdf`, etc.
- pretrained autoencoder checkpoints under `optimal_autoencoders/` if using the original response-only MMDRP pipeline

The current cloned repo only contains code plus old config/result folders. It does not contain the actual CTRPv2/DepMap training HDF files.

The Python environment is also missing MMDRP dependencies:

- `ray`
- `torch_geometric`
- `rdkit`

## Intended comparison

Once the MMDRP training data and dependencies are available, run the same train/validation/test split for:

1. MMDRP-base:
   - original MMDRP inputs, for example drug Morgan fingerprint + gene expression
2. MMDRP+DRKG:
   - original MMDRP inputs + `mmdrp_drkg_transe_features.npy`
   - align each row by `cpd_name` / DrugBank / DRKG compound entity using `mmdrp_drkg_transe_index.csv`
3. Compare:
   - MAE
   - RMSE
   - Pearson correlation
   - Spearman correlation

The comparison is only valid if both models use the same cell-line/drug samples and the same split.
