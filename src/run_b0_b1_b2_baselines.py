import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.linear_model import SGDRegressor
from torch.utils.data import DataLoader, Dataset

from run_attentivefp_multiseed_cv import (
    DATA_DIR,
    DEFAULT_DRKG_DIR,
    ROOT,
    build_prepared_for_split,
    grouped_cv_splits,
    load_base_data,
    parse_csv_numbers,
    summarize_metrics,
)
from run_attentivefp_multitask_drkg_optuna import metric_dict, set_seed


RDLogger.DisableLog("rdApp.*")


def morgan_fingerprints(unique_drugs: pd.DataFrame, fp_size: int = 2048, radius: int = 2) -> np.ndarray:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)
    fps = np.zeros((len(unique_drugs), fp_size), dtype=np.float32)
    for i, smiles in enumerate(unique_drugs["cpd_smiles"]):
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            raise ValueError(f"Invalid SMILES for {unique_drugs.iloc[i]['cpd_name']}: {smiles}")
        fp = generator.GetFingerprint(mol)
        arr = np.zeros((fp_size,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps[i] = arr.astype(np.float32)
    return fps


def get_unique_drugs_and_fps(base, fp_size: int, radius: int):
    unique_drugs = (
        base["pair_table"][["cpd_name", "cpd_smiles", "drug_norm"]]
        .drop_duplicates("cpd_name")
        .reset_index(drop=True)
    )
    drug_to_idx = {drug: i for i, drug in enumerate(unique_drugs["cpd_name"])}
    expected = base["pair_table"]["cpd_name"].map(drug_to_idx).to_numpy(dtype=np.int64)
    if not np.array_equal(expected, base["drug_indices"]):
        raise RuntimeError("Drug index order mismatch between baseline script and B3 loader.")
    return unique_drugs, morgan_fingerprints(unique_drugs, fp_size=fp_size, radius=radius)


def save_feature_metadata(expression_hdf: Path, fp_size: int, out_dir: Path):
    expression_df = pd.read_hdf(expression_hdf, "df")
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
    pd.DataFrame({"feature_index": np.arange(len(gene_cols)), "gene": gene_cols}).to_csv(
        out_dir / "expression_feature_names.csv",
        index=False,
    )
    pd.DataFrame({"feature_index": np.arange(fp_size), "morgan_bit": np.arange(fp_size)}).to_csv(
        out_dir / "morgan_feature_names.csv",
        index=False,
    )


def compute_metrics(targets_raw: np.ndarray, preds_raw: np.ndarray):
    metrics = {}
    metrics.update(metric_dict(targets_raw[:, 0], preds_raw[:, 0], "AAC"))
    metrics.update(metric_dict(targets_raw[:, 1], preds_raw[:, 1], "log2IC50"))
    metrics["combined_RMSE"] = 0.7 * metrics["AAC_RMSE"] + 0.3 * metrics["log2IC50_RMSE"]
    return metrics


def save_test_predictions(prepared, preds_raw: np.ndarray, out_path: Path, metadata: dict):
    candidate_cols = [
        "cpd_name",
        "ccl_name",
        "primary_disease",
        "lineage_1",
        "lineage_2",
        "lineage_3",
        "is_nsclc",
        "area_above_curve",
        "log2_ic50",
    ]
    prediction_cols = [col for col in candidate_cols if col in prepared["pair_table"].columns]
    test_rows_idx = np.asarray(prepared["splits"]["test"], dtype=np.int64)
    rows = prepared["pair_table"].iloc[test_rows_idx][prediction_cols].copy()
    rows.insert(0, "row_id", test_rows_idx)
    for key, value in metadata.items():
        rows[key] = value
    rows["pred_AAC"] = preds_raw[:, 0]
    rows["pred_log2IC50"] = preds_raw[:, 1]
    rows.to_csv(out_path, index=False)


def run_b0(prepared):
    train_rows = prepared["splits"]["train"]
    test_rows = prepared["splits"]["test"]
    train_drugs = prepared["drug_indices"][train_rows]
    targets = prepared["targets_raw"]
    global_median = np.median(targets[train_rows], axis=0)

    medians = {}
    for drug_idx in np.unique(train_drugs):
        medians[int(drug_idx)] = np.median(targets[train_rows][train_drugs == drug_idx], axis=0)

    preds = np.vstack(
        [medians.get(int(drug_idx), global_median) for drug_idx in prepared["drug_indices"][test_rows]]
    ).astype(np.float32)
    return compute_metrics(targets[test_rows], preds), preds


def build_linear_batch(prepared, morgan: np.ndarray, row_indices: np.ndarray):
    expression = prepared["expression"][prepared["cell_indices"][row_indices]]
    drug_fp = morgan[prepared["drug_indices"][row_indices]]
    return np.hstack([expression, drug_fp]).astype(np.float32, copy=False)


def predict_linear(models, prepared, morgan, row_indices, batch_size):
    preds_scaled = []
    for start in range(0, len(row_indices), batch_size):
        rows = row_indices[start : start + batch_size]
        x_batch = build_linear_batch(prepared, morgan, rows)
        preds_scaled.append(np.column_stack([model.predict(x_batch) for model in models]))
    preds_scaled = np.vstack(preds_scaled).astype(np.float32)
    return preds_scaled * prepared["target_std"] + prepared["target_mean"]


def run_b1_elastic_net(prepared, morgan, seed, epochs, batch_size, alpha, l1_ratio, eta0):
    train_rows = np.asarray(prepared["splits"]["train"], dtype=np.int64)
    valid_rows = np.asarray(prepared["splits"]["valid"], dtype=np.int64)
    test_rows = np.asarray(prepared["splits"]["test"], dtype=np.int64)
    models = [
        SGDRegressor(
            loss="squared_error",
            penalty="elasticnet",
            alpha=alpha,
            l1_ratio=l1_ratio,
            learning_rate="constant",
            eta0=eta0,
            random_state=seed + target_id,
            max_iter=1,
            tol=None,
            average=True,
        )
        for target_id in range(2)
    ]

    rng = np.random.default_rng(seed)
    best_models = None
    best_score = np.inf
    stale = 0
    patience = 3

    for epoch in range(1, epochs + 1):
        shuffled = train_rows.copy()
        rng.shuffle(shuffled)
        for start in range(0, len(shuffled), batch_size):
            rows = shuffled[start : start + batch_size]
            x_batch = build_linear_batch(prepared, morgan, rows)
            y_batch = prepared["targets_scaled"][rows]
            for target_id, model in enumerate(models):
                model.partial_fit(x_batch, y_batch[:, target_id])

        valid_preds = predict_linear(models, prepared, morgan, valid_rows, batch_size)
        valid_metrics = compute_metrics(prepared["targets_raw"][valid_rows], valid_preds)
        score = valid_metrics["combined_RMSE"]
        print(
            f"B1 ElasticNet epoch {epoch:02d} | "
            f"valid_AAC_RMSE={valid_metrics['AAC_RMSE']:.4f} "
            f"| valid_IC50_RMSE={valid_metrics['log2IC50_RMSE']:.4f}"
        )
        if score < best_score:
            best_score = score
            best_models = copy.deepcopy(models)
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    models = best_models if best_models is not None else models
    test_preds = predict_linear(models, prepared, morgan, test_rows, batch_size)
    return compute_metrics(prepared["targets_raw"][test_rows], test_preds), test_preds


class PairDataset(Dataset):
    def __init__(self, row_indices):
        self.row_indices = np.asarray(row_indices, dtype=np.int64)

    def __len__(self):
        return len(self.row_indices)

    def __getitem__(self, idx):
        return int(self.row_indices[idx])


class PairCollator:
    def __init__(self, prepared, morgan):
        self.prepared = prepared
        self.morgan = morgan

    def __call__(self, row_indices):
        row_indices = np.asarray(row_indices, dtype=np.int64)
        return {
            "row_indices": torch.from_numpy(row_indices),
            "expression": torch.from_numpy(self.prepared["expression"][self.prepared["cell_indices"][row_indices]]),
            "drug_fp": torch.from_numpy(self.morgan[self.prepared["drug_indices"][row_indices]]),
            "targets_scaled": torch.from_numpy(self.prepared["targets_scaled"][row_indices]),
            "targets_raw": torch.from_numpy(self.prepared["targets_raw"][row_indices]),
        }


class DualModalMLP(nn.Module):
    def __init__(self, expression_dim, fp_dim, dropout=0.2):
        super().__init__()
        self.expression_encoder = nn.Sequential(
            nn.Linear(expression_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
        )
        self.drug_encoder = nn.Sequential(
            nn.Linear(fp_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

    def forward(self, expression, drug_fp):
        return self.head(torch.cat([self.expression_encoder(expression), self.drug_encoder(drug_fp)], dim=1))


def move_batch(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def evaluate_b2(model, loader, prepared, device, return_predictions=False):
    model.eval()
    preds_scaled = []
    targets_raw = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            pred = model(batch["expression"], batch["drug_fp"])
            preds_scaled.append(pred.detach().cpu().numpy())
            targets_raw.append(batch["targets_raw"].detach().cpu().numpy())
    preds_scaled = np.vstack(preds_scaled)
    targets_raw = np.vstack(targets_raw)
    preds_raw = preds_scaled * prepared["target_std"] + prepared["target_mean"]
    metrics = compute_metrics(targets_raw, preds_raw)
    if return_predictions:
        return metrics, preds_raw
    return metrics


def run_b2_mmdrp_base(
    prepared,
    morgan,
    seed,
    epochs,
    batch_size,
    device,
    lr,
    weight_decay,
    dropout,
    model_out_path=None,
):
    set_seed(seed)
    loaders = {
        split: DataLoader(
            PairDataset(rows),
            batch_size=batch_size,
            shuffle=(split == "train"),
            collate_fn=PairCollator(prepared, morgan),
        )
        for split, rows in prepared["splits"].items()
    }
    model = DualModalMLP(
        expression_dim=prepared["expression"].shape[1],
        fp_dim=morgan.shape[1],
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state = None
    best_score = np.inf
    stale = 0
    patience = 5

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in loaders["train"]:
            batch = move_batch(batch, device)
            optimizer.zero_grad()
            pred = model(batch["expression"], batch["drug_fp"])
            loss = torch.mean((pred - batch["targets_scaled"]) ** 2)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        valid_metrics = evaluate_b2(model, loaders["valid"], prepared, device)
        score = valid_metrics["combined_RMSE"]
        print(
            f"B2 MMDRP-base epoch {epoch:02d} | loss={total_loss / max(len(loaders['train']), 1):.4f} "
            f"| valid_AAC_RMSE={valid_metrics['AAC_RMSE']:.4f} "
            f"| valid_IC50_RMSE={valid_metrics['log2IC50_RMSE']:.4f}"
        )
        if score < best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    if model_out_path is not None:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_class": "DualModalMLP",
                "expression_dim": int(prepared["expression"].shape[1]),
                "fp_dim": int(morgan.shape[1]),
                "target_mean": prepared["target_mean"],
                "target_std": prepared["target_std"],
                "config": {
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "dropout": dropout,
                    "epochs": epochs,
                    "batch_size": batch_size,
                },
            },
            model_out_path,
        )
    return evaluate_b2(model, loaders["test"], prepared, device, return_predictions=True)


def paired_stats(metrics_df: pd.DataFrame):
    from scipy import stats

    rows = []
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
    pivot_index = ["seed", "fold", "split_type", "use_lds", "alpha"]
    base = metrics_df[metrics_df["model"].eq("B0_drug_median")].set_index(pivot_index).sort_index()
    for model_name in ["B1_elastic_net", "B2_mmdrp_base"]:
        other = metrics_df[metrics_df["model"].eq(model_name)].set_index(pivot_index).sort_index()
        for metric in metric_cols:
            diff = other[metric] - base[metric]
            lower = metric.endswith("MAE") or metric.endswith("RMSE") or metric == "combined_RMSE"
            improved = int((diff < 0).sum() if lower else (diff > 0).sum())
            rows.append(
                {
                    "comparison": f"{model_name}_minus_B0",
                    "metric": metric,
                    "B0_mean": float(base[metric].mean()),
                    "model_mean": float(other[metric].mean()),
                    "diff_mean": float(diff.mean()),
                    "improved_runs_vs_B0": improved,
                    "total_runs": int(len(diff)),
                    "paired_ttest_p": float(stats.ttest_rel(other[metric], base[metric]).pvalue),
                }
            )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--b1-epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--valid-fraction", type=float, default=0.1)
    parser.add_argument("--split-type", choices=["cell_line", "drug"], default="cell_line")
    parser.add_argument("--ic50-min", type=float, default=-15.0)
    parser.add_argument("--ic50-max", type=float, default=15.0)
    parser.add_argument("--models", default="B0,B1,B2")
    parser.add_argument("--expression-hdf", default=str(DATA_DIR / "DepMap_26Q1_Expression.hdf"))
    parser.add_argument("--drkg-dir", default=str(DEFAULT_DRKG_DIR))
    parser.add_argument("--out-dir", default=str(ROOT / "results" / "20_b0_b1_b2_ctrpv2_depmap26q1"))
    parser.add_argument("--fp-size", type=int, default=2048)
    parser.add_argument("--fp-radius", type=int, default=2)
    parser.add_argument("--elastic-alpha", type=float, default=1e-4)
    parser.add_argument("--elastic-l1-ratio", type=float, default=0.15)
    parser.add_argument("--elastic-eta0", type=float, default=1e-5)
    parser.add_argument("--b2-lr", type=float, default=0.0021642898458891)
    parser.add_argument("--b2-weight-decay", type=float, default=1.1020244852224831e-06)
    parser.add_argument("--b2-dropout", type=float, default=0.15772477848027838)
    args = parser.parse_args()

    seeds = parse_csv_numbers(args.seeds, int)
    models = [item.strip().upper() for item in args.models.split(",") if item.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = load_base_data(Path(args.drkg_dir), args.ic50_min, args.ic50_max, Path(args.expression_hdf))
    unique_drugs, morgan = get_unique_drugs_and_fps(base, fp_size=args.fp_size, radius=args.fp_radius)
    save_feature_metadata(Path(args.expression_hdf), args.fp_size, out_dir)

    run_config = {
        **base["summary"],
        "baseline_definitions": {
            "B0": "per-drug median trivial baseline",
            "B1": "Elastic Net linear baseline via SGDRegressor penalty=elasticnet",
            "B2": "dual-modal MMDRP-base style MLP with expression + Morgan fingerprint, no LMF/LDS/GNN",
        },
        "seeds": seeds,
        "n_folds": args.n_folds,
        "max_folds": args.max_folds,
        "epochs": args.epochs,
        "b1_epochs": args.b1_epochs,
        "batch_size": args.batch_size,
        "valid_fraction": args.valid_fraction,
        "split_type": args.split_type,
        "models": models,
        "fingerprint": {"type": "Morgan", "fp_size": args.fp_size, "radius": args.fp_radius},
        "elastic_net": {"alpha": args.elastic_alpha, "l1_ratio": args.elastic_l1_ratio, "eta0": args.elastic_eta0},
        "b2_config": {
            "lr": args.b2_lr,
            "weight_decay": args.b2_weight_decay,
            "dropout": args.b2_dropout,
        },
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    unique_drugs.to_csv(out_dir / "baseline_drug_index.csv", index=False)
    print(json.dumps(run_config, indent=2))

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
            prepared = build_prepared_for_split(
                base,
                train_rows,
                valid_rows,
                test_rows,
                lds_max_weight=10.0,
                use_lds=False,
            )
            fold_dir = out_dir / f"seed{seed}_fold{fold_id}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                fold_dir / "split_indices.npz",
                train_rows=train_rows,
                valid_rows=valid_rows,
                test_rows=test_rows,
            )
            for model_key in models:
                print(f"Running seed={seed} fold={fold_id} model={model_key}")
                if model_key == "B0":
                    metrics, preds_raw = run_b0(prepared)
                    model_name = "B0_drug_median"
                elif model_key == "B1":
                    metrics, preds_raw = run_b1_elastic_net(
                        prepared,
                        morgan,
                        seed=seed * 100 + fold_id,
                        epochs=args.b1_epochs,
                        batch_size=args.batch_size,
                        alpha=args.elastic_alpha,
                        l1_ratio=args.elastic_l1_ratio,
                        eta0=args.elastic_eta0,
                    )
                    model_name = "B1_elastic_net"
                elif model_key == "B2":
                    metrics, preds_raw = run_b2_mmdrp_base(
                        prepared,
                        morgan,
                        seed=seed * 100 + fold_id,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        device=device,
                        lr=args.b2_lr,
                        weight_decay=args.b2_weight_decay,
                        dropout=args.b2_dropout,
                        model_out_path=fold_dir / "B2_mmdrp_base.pt",
                    )
                    model_name = "B2_mmdrp_base"
                else:
                    raise ValueError(f"Unknown model: {model_key}")

                save_test_predictions(
                    prepared,
                    preds_raw,
                    fold_dir / f"{model_name}_test_predictions.csv",
                    {
                        "seed": seed,
                        "fold": fold_id,
                        "alpha": 0.8,
                        "split_type": args.split_type,
                        "use_lds": False,
                        "model": model_name,
                    },
                )

                row = {
                    "model": model_name,
                    "seed": seed,
                    "fold": fold_id,
                    "split_type": args.split_type,
                    "use_lds": False,
                    "alpha": 0.8,
                    "train_rows": int(len(train_rows)),
                    "valid_rows": int(len(valid_rows)),
                    "test_rows": int(len(test_rows)),
                    **metrics,
                }
                all_metrics.append(row)
                metrics_df = pd.DataFrame(all_metrics)
                metrics_df.to_csv(out_dir / "cv_metrics.csv", index=False)
                summarize_metrics(metrics_df).to_csv(out_dir / "cv_summary.csv", index=False)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(out_dir / "cv_metrics.csv", index=False)
    summarize_metrics(metrics_df).to_csv(out_dir / "cv_summary.csv", index=False)
    paired_stats(metrics_df).to_csv(out_dir / "paired_stats_vs_b0.csv", index=False)
    print(summarize_metrics(metrics_df).to_string(index=False))


if __name__ == "__main__":
    main()
