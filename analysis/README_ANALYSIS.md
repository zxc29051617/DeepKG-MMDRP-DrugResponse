# Pancancer Analysis Scripts

This folder contains post-training analysis scripts for the final pancancer MMDRP + DRKG experiment.

## Folder Layout

```text
analysis/
  attribution/
    run_pancancer_ig_attribution.py
  visualization/
    run_pancancer_visualization.py
```

## 1. Visualization

Script:

```text
analysis/visualization/run_pancancer_visualization.py
```

Purpose:

- Generate publication/report figures from the completed 3 seeds x 5 folds pancancer experiment.
- Compare B0, B1, B2, B3, and B3+DRKG.
- Plot RMSE, correlation, fold-level variation, B3 vs B3+DRKG deltas, prediction scatter plots, and B3/B3+DRKG training curves.

Default inputs:

```text
results/pancancer_50e_evaluation/
results/pancancer_B3_B3DRKG_26Q1_CTRPv2_50e_full_folds/
results/pancancer_B0_B1_B2_26Q1_CTRPv2_50e_full_folds/
```

Default output:

```text
results/pancancer_visualization_figures/
```

Run:

```powershell
python analysis/visualization/run_pancancer_visualization.py
```

Important outputs:

```text
01_pancancer_aac_rmse_bar.png/.pdf
02_pancancer_log2ic50_rmse_bar.png/.pdf
03_pancancer_combined_rmse_bar.png/.pdf
04_pancancer_rmse_three_panel.png/.pdf
05_pancancer_correlation_metrics.png/.pdf
06_seed_fold_combined_rmse_line.png/.pdf
07_seed_fold_combined_rmse_heatmap.png/.pdf
08_b3drkg_delta_combined_rmse_by_fold.png/.pdf
09_AAC_prediction_scatter.png/.pdf
10_log2IC50_prediction_scatter.png/.pdf
11_AAC_residual_boxplot.png/.pdf
12_log2IC50_residual_boxplot.png/.pdf
13_training_curve_valid_combined_rmse.png/.pdf
14_training_curve_mean_train_loss.png/.pdf
15_train_loss_each_seed_fold.png/.pdf
16_valid_combined_rmse_each_seed_fold.png/.pdf
pancancer_rmse_significance_vs_b3drkg.csv
b3_vs_b3drkg_fold_deltas.csv
training_curve_availability.csv
visualization_manifest.json
```

## 2. Integrated Gradients Attribution

Script:

```text
analysis/attribution/run_pancancer_ig_attribution.py
```

Purpose:

- Run Integrated Gradients on a trained B3 or B3+DRKG checkpoint.
- Estimate which expression genes and DRKG embedding dimensions contribute most to AAC and log2IC50 predictions.

Default checkpoint:

```text
results/pancancer_B3_B3DRKG_26Q1_CTRPv2_50e_full_folds/seed42_fold0/cv_plus_drkg.pt
```

Default inputs:

```text
Data/DRKG_Features/pancancer_driver_h1_200d_50e/
Data/DRP_Training_Data/DepMap_26Q1_Expression.hdf
results/pancancer_B3_B3DRKG_26Q1_CTRPv2_50e_full_folds/
```

Default output:

```text
results/pancancer_ig_attribution/
```

Run default B3+DRKG attribution:

```powershell
python analysis/attribution/run_pancancer_ig_attribution.py
```

Run NSCLC-only random sample attribution:

```powershell
python analysis/attribution/run_pancancer_ig_attribution.py --sample-strategy nsclc_random
```

Run a specific seed/fold:

```powershell
python analysis/attribution/run_pancancer_ig_attribution.py --seed 43 --fold 2 --model cv_plus_drkg
```

Important outputs:

```text
ig_top_gene_attributions.csv
ig_top_drkg_dimension_attributions.csv
ig_sample_attributions.csv
ig_top_genes_AAC.png/.pdf
ig_top_genes_log2IC50.png/.pdf
ig_top_drkg_dims_AAC.png/.pdf
ig_top_drkg_dims_log2IC50.png/.pdf
ig_gene_sample_heatmap_AAC.png/.pdf
ig_gene_sample_heatmap_log2IC50.png/.pdf
ig_modality_contribution.png/.pdf
ig_manifest.json
```

## Notes

- These scripts are post-training analysis scripts. They do not retrain B0/B1/B2/B3/B3+DRKG.
- The package root is detected automatically by searching upward for `Data/`, `results/`, and `src/`, so the scripts can be run from the package root or directly by full path.
- For the cleanest rerun experience, use an ASCII-only path such as `C:\MMDRP_DRKG_repro_package_20260531` before running HDF5-heavy commands.
- On this machine, `C:\MMDRP_DRKG_repro_package_20260531` was created as a junction pointing to the desktop package so HDF5 can read files through an ASCII path without duplicating the 4GB package.
