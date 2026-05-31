import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_attentivefp_multitask_drkg_optuna import (
    DATA_DIR,
    DEFAULT_DRKG_DIR,
    ROOT,
    lds_weights,
    normalize_name,
    set_seed,
    smiles_to_graph,
    train_model,
)


NSCLC_LABEL = "Non-Small Cell Lung Cancer"


def parse_csv_numbers(value, cast=float):
    if isinstance(value, list):
        return value
    return [cast(item.strip()) for item in str(value).split(",") if item.strip()]


def add_depmap_lineage(pair_table: pd.DataFrame) -> pd.DataFrame:
    lineage_path = ROOT / "Data" / "raw" / "depmap_latest_drug_screen" / "DepMap_26Q1_Expression_custom.csv"
    if not lineage_path.exists():
        pair_table = pair_table.copy()
        pair_table["lineage_1"] = pair_table.get("lineage_1", np.nan)
        pair_table["lineage_2"] = pair_table.get("lineage_2", np.nan)
        pair_table["lineage_3"] = pair_table.get("lineage_3", np.nan)
        pair_table["is_nsclc"] = False
        return pair_table

    lineage_cols = ["cell_line_display_name", "lineage_1", "lineage_2", "lineage_3"]
    lineage = pd.read_csv(lineage_path, usecols=lineage_cols)
    lineage["cell_norm"] = lineage["cell_line_display_name"].map(normalize_name)
    lineage = lineage.drop_duplicates("cell_norm")

    pair_table = pair_table.merge(
        lineage[["cell_norm", "lineage_1", "lineage_2", "lineage_3"]],
        on="cell_norm",
        how="left",
    )
    pair_table["is_nsclc"] = pair_table["lineage_2"].eq(NSCLC_LABEL)
    return pair_table


def load_base_data(drkg_dir: Path, ic50_min: float, ic50_max: float, expression_hdf: Path):
    pair_table = pd.read_hdf(DATA_DIR / "CTRP_AAC_IC50_SMILES.hdf", "df")
    rows_before_ic50_filter = len(pair_table)
    pair_table = pair_table[pair_table["log2_ic50"].between(ic50_min, ic50_max)].copy()

    drkg_index = pd.read_csv(drkg_dir / "mmdrp_drkg_transe_index.csv")
    drkg_features = np.load(drkg_dir / "mmdrp_drkg_transe_features.npy").astype(np.float32)
    drkg_index["drug_norm"] = drkg_index["cpd_name"].map(normalize_name)
    norm_to_feature_row = (
        drkg_index[["drug_norm", "feature_row_id"]]
        .drop_duplicates("drug_norm")
        .set_index("drug_norm")["feature_row_id"]
        .to_dict()
    )

    pair_table["drug_norm"] = pair_table["cpd_name"].map(normalize_name)
    pair_table = pair_table[pair_table["drug_norm"].isin(norm_to_feature_row)].copy()

    expression_df = pd.read_hdf(expression_hdf, "df")
    expression_df["cell_norm"] = expression_df["stripped_cell_line_name"].map(normalize_name)
    expression_df = expression_df.drop_duplicates("cell_norm")
    expression_map = {name: i for i, name in enumerate(expression_df["cell_norm"])}

    pair_table["cell_norm"] = pair_table["ccl_name"].map(normalize_name)
    pair_table = pair_table[pair_table["cell_norm"].isin(expression_map)].copy()
    pair_table = pair_table.reset_index(drop=True)
    pair_table = add_depmap_lineage(pair_table)

    info_cols = {
        "DepMap_ID",
        "stripped_cell_line_name",
        "primary_disease",
        "cell_norm",
        "lineage_1",
        "lineage_2",
        "lineage_3",
        "lineage_4",
        "lineage_6",
    }
    gene_cols = [
        col
        for col in expression_df.columns
        if col not in info_cols and pd.api.types.is_numeric_dtype(expression_df[col])
    ]
    expression_raw = expression_df[gene_cols].to_numpy(dtype=np.float32)

    unique_drugs = (
        pair_table[["cpd_name", "cpd_smiles", "drug_norm"]]
        .drop_duplicates("cpd_name")
        .reset_index(drop=True)
    )
    drug_to_idx = {drug: i for i, drug in enumerate(unique_drugs["cpd_name"])}
    graphs = [smiles_to_graph(smiles) for smiles in unique_drugs["cpd_smiles"]]
    kg = np.vstack(
        [drkg_features[norm_to_feature_row[drug_norm]] for drug_norm in unique_drugs["drug_norm"]]
    ).astype(np.float32)

    cell_indices = pair_table["cell_norm"].map(expression_map).to_numpy(dtype=np.int64)
    drug_indices = pair_table["cpd_name"].map(drug_to_idx).to_numpy(dtype=np.int64)
    targets_raw = pair_table[["area_above_curve", "log2_ic50"]].to_numpy(dtype=np.float32)

    return {
        "pair_table": pair_table,
        "expression_raw": expression_raw,
        "graphs": graphs,
        "kg": kg,
        "cell_indices": cell_indices,
        "drug_indices": drug_indices,
        "targets_raw": targets_raw,
        "summary": {
            "rows_before_ic50_filter": int(rows_before_ic50_filter),
            "rows": int(len(pair_table)),
            "ic50_log2_filter": [float(ic50_min), float(ic50_max)],
            "unique_drugs_with_drkg": int(pair_table["cpd_name"].nunique()),
            "unique_cell_lines": int(pair_table["ccl_name"].nunique()),
            "nsclc_cell_lines": int(pair_table.loc[pair_table["is_nsclc"], "ccl_name"].nunique()),
            "nsclc_pairs": int(pair_table["is_nsclc"].sum()),
            "expression_dim": int(expression_raw.shape[1]),
            "expression_hdf": str(expression_hdf),
            "kg_dim": int(kg.shape[1]),
            "targets": ["AAC", "log2IC50"],
        },
    }


def grouped_cv_splits(group_indices, n_folds: int, seed: int, valid_fraction: float):
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(group_indices)
    rng.shuffle(unique_groups)
    test_folds = np.array_split(unique_groups, n_folds)
    all_rows = np.arange(len(group_indices))

    for fold_id, test_groups in enumerate(test_folds):
        remaining_groups = np.setdiff1d(unique_groups, test_groups, assume_unique=False)
        rng_fold = np.random.default_rng(seed * 1000 + fold_id)
        rng_fold.shuffle(remaining_groups)
        n_valid = max(1, int(round(len(remaining_groups) * valid_fraction)))
        valid_groups = remaining_groups[:n_valid]
        train_groups = remaining_groups[n_valid:]

        train_rows = all_rows[np.isin(group_indices, train_groups)]
        valid_rows = all_rows[np.isin(group_indices, valid_groups)]
        test_rows = all_rows[np.isin(group_indices, test_groups)]
        yield fold_id, train_rows, valid_rows, test_rows


def build_prepared_for_split(base, train_rows, valid_rows, test_rows, lds_max_weight: float, use_lds: bool):
    expression = base["expression_raw"].copy()
    train_expression_rows = np.unique(base["cell_indices"][train_rows])
    expression_mean = expression[train_expression_rows].mean(axis=0, keepdims=True)
    expression_std = expression[train_expression_rows].std(axis=0, keepdims=True)
    expression = ((expression - expression_mean) / (expression_std + 1e-6)).astype(np.float32)

    targets_raw = base["targets_raw"]
    target_mean = targets_raw[train_rows].mean(axis=0, keepdims=True)
    target_std = targets_raw[train_rows].std(axis=0, keepdims=True) + 1e-6
    targets_scaled = ((targets_raw - target_mean) / target_std).astype(np.float32)

    if use_lds:
        weights = np.vstack(
            [
                lds_weights(targets_raw[train_rows, 0], targets_raw[:, 0], max_weight=lds_max_weight),
                lds_weights(targets_raw[train_rows, 1], targets_raw[:, 1], max_weight=lds_max_weight),
            ]
        ).T.astype(np.float32)
    else:
        weights = np.ones_like(targets_raw, dtype=np.float32)

    return {
        "pair_table": base["pair_table"],
        "expression": expression,
        "graphs": base["graphs"],
        "kg": base["kg"],
        "cell_indices": base["cell_indices"],
        "drug_indices": base["drug_indices"],
        "targets_raw": targets_raw,
        "targets_scaled": targets_scaled,
        "target_mean": target_mean.astype(np.float32),
        "target_std": target_std.astype(np.float32),
        "weights": weights,
        "splits": {"train": train_rows, "valid": valid_rows, "test": test_rows},
    }


def summarize_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "AAC_MAE",
        "AAC_RMSE",
        "AAC_Pearson",
        "AAC_Spearman",
        "log2IC50_MAE",
        "log2IC50_RMSE",
        "log2IC50_Pearson",
        "log2IC50_Spearman",
        "combined_RMSE",
    ]
    rows = []
    group_cols = ["split_type", "use_lds", "model", "alpha"]
    for key, group in metrics_df.groupby(group_cols, dropna=False):
        split_type, use_lds, model, alpha = key
        row = {
            "split_type": split_type,
            "use_lds": use_lds,
            "model": model,
            "alpha": alpha,
            "runs": int(len(group)),
        }
        for col in metric_cols:
            row[f"{col}_mean"] = float(group[col].mean())
            row[f"{col}_std"] = float(group[col].std(ddof=1)) if len(group) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["split_type", "use_lds", "alpha", "model"]).reset_index(drop=True)


def save_split_indices(out_dir: Path, train_rows, valid_rows, test_rows):
    np.savez_compressed(
        out_dir / "split_indices.npz",
        train_rows=train_rows,
        valid_rows=valid_rows,
        test_rows=test_rows,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--alphas", default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--valid-fraction", type=float, default=0.1)
    parser.add_argument("--split-type", choices=["cell_line", "drug"], default="cell_line")
    parser.add_argument("--ic50-min", type=float, default=-15.0)
    parser.add_argument("--ic50-max", type=float, default=15.0)
    parser.add_argument("--lds-max-weight", type=float, default=10.0)
    parser.add_argument("--no-lds", action="store_true")
    parser.add_argument("--models", default="base,drkg")
    parser.add_argument(
        "--expression-hdf",
        default=str(DATA_DIR / "DepMap_26Q1_Expression.hdf"),
    )
    parser.add_argument(
        "--best-config",
        default=str(ROOT / "results" / "attentivefp_multitask_drkg_optuna" / "best_config.json"),
    )
    parser.add_argument("--drkg-dir", default=str(DEFAULT_DRKG_DIR))
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "results" / "05_multiseed_cv_loss_weight"),
    )
    args = parser.parse_args()

    seeds = parse_csv_numbers(args.seeds, int)
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_config = json.loads(Path(args.best_config).read_text(encoding="utf-8"))
    if args.alphas is None:
        alphas = [float(base_config["alpha"])]
    else:
        alphas = parse_csv_numbers(args.alphas, float)

    base = load_base_data(Path(args.drkg_dir), args.ic50_min, args.ic50_max, Path(args.expression_hdf))
    run_config = {
        **base["summary"],
        "seeds": seeds,
        "n_folds": args.n_folds,
        "max_folds": args.max_folds,
        "alphas": alphas,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "valid_fraction": args.valid_fraction,
        "split_type": args.split_type,
        "lds_max_weight": args.lds_max_weight,
        "use_lds": not args.no_lds,
        "models": models,
        "base_config": base_config,
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    print(json.dumps(run_config, indent=2))

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    all_metrics = []
    group_indices = base["cell_indices"] if args.split_type == "cell_line" else base["drug_indices"]
    for seed in seeds:
        set_seed(seed)
        splits = list(grouped_cv_splits(group_indices, args.n_folds, seed, args.valid_fraction))
        if args.max_folds is not None:
            splits = splits[: args.max_folds]

        for fold_id, train_rows, valid_rows, test_rows in splits:
            fold_dir = out_dir / f"seed{seed}_fold{fold_id}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            save_split_indices(fold_dir, train_rows, valid_rows, test_rows)
            prepared = build_prepared_for_split(
                base,
                train_rows,
                valid_rows,
                test_rows,
                args.lds_max_weight,
                not args.no_lds,
            )
            for alpha in alphas:
                config = {**base_config, "alpha": float(alpha), "patience": base_config.get("patience", 5)}
                for model_key in models:
                    use_kg = model_key.lower() in {"drkg", "plus_drkg", "+drkg"}
                    model_name = f"cv_{'plus_drkg' if use_kg else 'base'}"
                    run_seed = seed * 10000 + fold_id * 100 + int(round(alpha * 1000)) + (1 if use_kg else 0)
                    set_seed(run_seed)
                    print(
                        f"Running seed={seed} fold={fold_id} alpha={alpha:.4f} "
                        f"model={'DRKG' if use_kg else 'base'}"
                    )
                    result, best_valid = train_model(
                        model_name,
                        use_kg,
                        prepared,
                        config,
                        fold_dir,
                        args.epochs,
                        args.batch_size,
                        device,
                        prediction_metadata={
                            "seed": seed,
                            "fold": fold_id,
                            "alpha": float(alpha),
                            "split_type": args.split_type,
                            "use_lds": not args.no_lds,
                            "model": model_name,
                        },
                    )
                    result.update(
                        {
                            "seed": seed,
                            "fold": fold_id,
                            "alpha": float(alpha),
                            "split_type": args.split_type,
                            "use_lds": not args.no_lds,
                            "best_valid_combined_RMSE": float(best_valid),
                            "train_rows": int(len(train_rows)),
                            "valid_rows": int(len(valid_rows)),
                            "test_rows": int(len(test_rows)),
                        }
                    )
                    all_metrics.append(result)
                    metrics_df = pd.DataFrame(all_metrics)
                    metrics_df.to_csv(out_dir / "cv_metrics.csv", index=False)
                    summarize_metrics(metrics_df).to_csv(out_dir / "cv_summary.csv", index=False)

    metrics_df = pd.DataFrame(all_metrics)
    summary_df = summarize_metrics(metrics_df)
    metrics_df.to_csv(out_dir / "cv_metrics.csv", index=False)
    summary_df.to_csv(out_dir / "cv_summary.csv", index=False)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
