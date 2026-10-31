import argparse
import json
import math
import os
import random
import re
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem, RDLogger
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.nn.models import AttentiveFP


RDLogger.DisableLog("rdApp.*")

ROOT = Path(os.environ.get("MMDRP_ROOT", Path(__file__).resolve().parents[1]))
DATA_DIR = ROOT / "Data" / "DRP_Training_Data"
DEFAULT_DRKG_DIR = Path(
    os.environ.get(
        "DRKG_FEATURE_DIR",
        ROOT / "Data" / "DRKG_Features" / "pancancer_driver_h1_200d_50e",
    )
)


SYMBOLS = ["As", "B", "Br", "C", "Cl", "F", "I", "N", "O", "P", "Pt", "S", "other"]
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
    Chem.rdchem.HybridizationType.UNSPECIFIED,
    "other",
]
STEREOS = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
]
NODE_DIM = len(SYMBOLS) + 6 + 1 + 1 + len(HYBRIDIZATIONS) + 1 + 5 + 1 + 2
EDGE_DIM = 10


def normalize_name(value) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def one_hot(index: int, size: int):
    values = [0.0] * size
    values[max(0, min(index, size - 1))] = 1.0
    return values


def smiles_to_graph(smiles: str) -> Data:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    atom_features = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        symbol_idx = SYMBOLS.index(symbol) if symbol in SYMBOLS else SYMBOLS.index("other")

        hybridization = atom.GetHybridization()
        hybrid_idx = HYBRIDIZATIONS.index(hybridization) if hybridization in HYBRIDIZATIONS else HYBRIDIZATIONS.index("other")

        chirality_type = [0.0, 0.0]
        if atom.HasProp("_CIPCode"):
            code = atom.GetProp("_CIPCode")
            if code in ["R", "S"]:
                chirality_type[["R", "S"].index(code)] = 1.0

        atom_features.append(
            one_hot(symbol_idx, len(SYMBOLS))
            + one_hot(atom.GetDegree(), 6)
            + [float(atom.GetFormalCharge())]
            + [float(atom.GetNumRadicalElectrons())]
            + one_hot(hybrid_idx, len(HYBRIDIZATIONS))
            + [1.0 if atom.GetIsAromatic() else 0.0]
            + one_hot(atom.GetTotalNumHs(), 5)
            + [1.0 if atom.HasProp("_ChiralityPossible") else 0.0]
            + chirality_type
        )

    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        bond_type = bond.GetBondType()
        stereo = bond.GetStereo()
        stereo_idx = STEREOS.index(stereo) if stereo in STEREOS else 1
        edge_attr = [
            1.0 if bond_type == Chem.rdchem.BondType.SINGLE else 0.0,
            1.0 if bond_type == Chem.rdchem.BondType.DOUBLE else 0.0,
            1.0 if bond_type == Chem.rdchem.BondType.TRIPLE else 0.0,
            1.0 if bond_type == Chem.rdchem.BondType.AROMATIC else 0.0,
            1.0 if bond.GetIsConjugated() else 0.0,
            1.0 if bond.IsInRing() else 0.0,
        ] + one_hot(stereo_idx, len(STEREOS))

        edge_indices.extend([[begin, end], [end, begin]])
        edge_attrs.extend([edge_attr, edge_attr])

    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, EDGE_DIM), dtype=torch.float32)

    return Data(
        x=torch.tensor(atom_features, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=edge_attr,
    )


def gaussian_kernel(size: int = 5, sigma: float = 2.0) -> np.ndarray:
    radius = size // 2
    xs = np.arange(-radius, radius + 1)
    kernel = np.exp(-(xs**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def lds_weights(train_values, all_values, n_bins=100, kernel_size=5, sigma=2.0, max_weight=10.0):
    train_values = np.asarray(train_values, dtype=float)
    all_values = np.asarray(all_values, dtype=float)
    lower, upper = float(np.min(train_values)), float(np.max(train_values))
    if math.isclose(lower, upper):
        return np.ones_like(all_values, dtype=np.float32)

    edges = np.linspace(lower, upper, n_bins + 1)
    train_bins = np.clip(np.digitize(train_values, edges[1:-1]), 0, n_bins - 1)
    all_bins = np.clip(np.digitize(all_values, edges[1:-1]), 0, n_bins - 1)
    hist = np.bincount(train_bins, minlength=n_bins).astype(float)
    smooth = np.convolve(hist, gaussian_kernel(kernel_size, sigma), mode="same")
    raw_weights = 1.0 / (smooth[all_bins] + 1e-6)
    train_raw = 1.0 / (smooth[train_bins] + 1e-6)
    raw_weights = raw_weights / np.mean(train_raw)
    if max_weight is not None:
        raw_weights = np.clip(raw_weights, 0.0, max_weight)
    return raw_weights.astype(np.float32)


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray, prefix: str):
    err = y_pred - y_true
    pearson = float(np.corrcoef(y_true, y_pred)[0, 1]) if np.std(y_true) > 0 and np.std(y_pred) > 0 else np.nan
    spearman = float(pd.Series(y_true).corr(pd.Series(y_pred), method="spearman"))
    return {
        f"{prefix}_MAE": float(np.mean(np.abs(err))),
        f"{prefix}_RMSE": float(np.sqrt(np.mean(err**2))),
        f"{prefix}_Pearson": pearson,
        f"{prefix}_Spearman": spearman,
    }


class MultitaskGraphDataset(Dataset):
    def __init__(self, row_indices, prepared):
        self.row_indices = np.asarray(row_indices, dtype=np.int64)
        self.prepared = prepared

    def __len__(self):
        return len(self.row_indices)

    def __getitem__(self, idx):
        return int(self.row_indices[idx])


class GraphBatchCollator:
    def __init__(self, prepared, use_kg: bool):
        self.prepared = prepared
        self.use_kg = use_kg

    def __call__(self, row_indices):
        row_indices = np.asarray(row_indices, dtype=np.int64)
        cell_indices = self.prepared["cell_indices"][row_indices]
        drug_indices = self.prepared["drug_indices"][row_indices]
        unique_drugs, inverse = np.unique(drug_indices, return_inverse=True)

        graph_batch = Batch.from_data_list([self.prepared["graphs"][i] for i in unique_drugs])

        batch = {
            "row_indices": torch.from_numpy(row_indices),
            "expression": torch.from_numpy(self.prepared["expression"][cell_indices]),
            "graph_batch": graph_batch,
            "graph_inverse": torch.from_numpy(inverse.astype(np.int64)),
            "targets_scaled": torch.from_numpy(self.prepared["targets_scaled"][row_indices]),
            "targets_raw": torch.from_numpy(self.prepared["targets_raw"][row_indices]),
            "weights": torch.from_numpy(self.prepared["weights"][row_indices]),
        }
        if self.use_kg:
            batch["kg"] = torch.from_numpy(self.prepared["kg"][drug_indices])
        return batch


class MultitaskAttentiveFPLMF(nn.Module):
    def __init__(
        self,
        expression_dim,
        kg_dim=None,
        gnn_hidden=96,
        branch_dim=128,
        fusion_dim=128,
        rank=8,
        gnn_layers=2,
        gnn_timesteps=2,
        dropout=0.2,
    ):
        super().__init__()
        self.use_kg = kg_dim is not None

        self.expression_encoder = nn.Sequential(
            nn.Linear(expression_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, branch_dim),
            nn.ReLU(),
        )
        self.drug_gnn = AttentiveFP(
            in_channels=NODE_DIM,
            hidden_channels=gnn_hidden,
            out_channels=branch_dim,
            edge_dim=EDGE_DIM,
            num_layers=gnn_layers,
            num_timesteps=gnn_timesteps,
            dropout=dropout,
        )

        input_dims = [branch_dim, branch_dim]
        if self.use_kg:
            self.kg_encoder = nn.Sequential(
                nn.Linear(kg_dim, branch_dim),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(branch_dim, branch_dim),
                nn.ReLU(),
            )
            input_dims.append(branch_dim)

        self.lmf = LowRankFusion(input_dims=input_dims, output_dim=fusion_dim, rank=rank)
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

    def forward(self, expression, graph_batch, graph_inverse, kg=None):
        exp_out = self.expression_encoder(expression)
        drug_unique = self.drug_gnn(
            graph_batch.x,
            graph_batch.edge_index,
            graph_batch.edge_attr,
            graph_batch.batch,
        )
        drug_out = drug_unique[graph_inverse]
        encoded = [exp_out, drug_out]
        if self.use_kg:
            encoded.append(self.kg_encoder(kg))
        return self.head(self.lmf(encoded))


class LowRankFusion(nn.Module):
    def __init__(self, input_dims, output_dim=128, rank=8):
        super().__init__()
        self.factors = nn.ParameterList(
            [nn.Parameter(torch.empty(rank, input_dim + 1, output_dim)) for input_dim in input_dims]
        )
        self.fusion_weights = nn.Parameter(torch.empty(1, rank))
        self.fusion_bias = nn.Parameter(torch.zeros(1, output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        for factor in self.factors:
            nn.init.xavier_normal_(factor)
        nn.init.xavier_normal_(self.fusion_weights)

    def forward(self, encoded_modalities):
        batch_size = encoded_modalities[0].shape[0]
        fused = None
        for encoded, factor in zip(encoded_modalities, self.factors):
            ones = torch.ones(batch_size, 1, device=encoded.device, dtype=encoded.dtype)
            with_bias = torch.cat([ones, encoded], dim=1)
            projected = torch.matmul(with_bias, factor)
            fused = projected if fused is None else fused * projected
        return torch.matmul(self.fusion_weights, fused.permute(1, 0, 2)).squeeze(1) + self.fusion_bias


def move_batch(batch, device):
    moved = {}
    for key, value in batch.items():
        if key == "graph_batch":
            moved[key] = value.to(device)
        elif torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def multitask_loss(pred, targets_scaled, weights, alpha):
    mse = (pred - targets_scaled) ** 2
    weighted = mse * weights
    return alpha * weighted[:, 0].mean() + (1.0 - alpha) * weighted[:, 1].mean()


def evaluate(model, loader, prepared, device):
    model.eval()
    preds_scaled = []
    targets_raw = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            pred = model(
                batch["expression"],
                batch["graph_batch"],
                batch["graph_inverse"],
                batch.get("kg"),
            )
            preds_scaled.append(pred.detach().cpu().numpy())
            targets_raw.append(batch["targets_raw"].detach().cpu().numpy())

    preds_scaled = np.vstack(preds_scaled)
    targets_raw = np.vstack(targets_raw)
    preds_raw = preds_scaled * prepared["target_std"] + prepared["target_mean"]

    metrics = {}
    metrics.update(metric_dict(targets_raw[:, 0], preds_raw[:, 0], "AAC"))
    metrics.update(metric_dict(targets_raw[:, 1], preds_raw[:, 1], "log2IC50"))
    metrics["combined_RMSE"] = 0.7 * metrics["AAC_RMSE"] + 0.3 * metrics["log2IC50_RMSE"]
    return metrics, targets_raw, preds_raw


def train_model(
    model_name,
    use_kg,
    prepared,
    config,
    out_dir,
    max_epochs,
    batch_size,
    device,
    trial=None,
    train_rows_override=None,
    prediction_metadata=None,
):
    train_rows = prepared["splits"]["train"] if train_rows_override is None else train_rows_override
    loaders = {
        "train": DataLoader(
            MultitaskGraphDataset(train_rows, prepared),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=GraphBatchCollator(prepared, use_kg),
        ),
        "valid": DataLoader(
            MultitaskGraphDataset(prepared["splits"]["valid"], prepared),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=GraphBatchCollator(prepared, use_kg),
        ),
        "test": DataLoader(
            MultitaskGraphDataset(prepared["splits"]["test"], prepared),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=GraphBatchCollator(prepared, use_kg),
        ),
    }

    model = MultitaskAttentiveFPLMF(
        expression_dim=prepared["expression"].shape[1],
        kg_dim=prepared["kg"].shape[1] if use_kg else None,
        gnn_hidden=config["gnn_hidden"],
        branch_dim=128,
        fusion_dim=config["fusion_dim"],
        rank=config["rank"],
        gnn_layers=config["gnn_layers"],
        gnn_timesteps=config["gnn_timesteps"],
        dropout=config["dropout"],
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    alpha = config["alpha"]
    best_state = None
    best_score = np.inf
    history = []
    patience = config.get("patience", 5)
    stale_epochs = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in loaders["train"]:
            batch = move_batch(batch, device)
            optimizer.zero_grad()
            pred = model(
                batch["expression"],
                batch["graph_batch"],
                batch["graph_inverse"],
                batch.get("kg"),
            )
            loss = multitask_loss(pred, batch["targets_scaled"], batch["weights"], alpha)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        valid_metrics, _, _ = evaluate(model, loaders["valid"], prepared, device)
        score = valid_metrics["combined_RMSE"]
        row = {
            "model": model_name,
            "epoch": epoch,
            "train_loss": total_loss / max(len(loaders["train"]), 1),
            **{f"valid_{k}": v for k, v in valid_metrics.items()},
        }
        history.append(row)
        print(
            f"{model_name} epoch {epoch:02d} | loss={row['train_loss']:.4f} "
            f"| valid_AAC_RMSE={valid_metrics['AAC_RMSE']:.4f} "
            f"| valid_IC50_RMSE={valid_metrics['log2IC50_RMSE']:.4f}"
        )

        if trial is not None:
            trial.report(score, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if score < best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics, targets_raw, preds_raw = evaluate(model, loaders["test"], prepared, device)

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out_dir / f"{model_name}.pt")
        pd.DataFrame(history).to_csv(out_dir / f"{model_name}_history.csv", index=False)
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
        test_rows = prepared["pair_table"].iloc[prepared["splits"]["test"]][prediction_cols].copy()
        test_rows.insert(0, "row_id", prepared["splits"]["test"])
        if prediction_metadata:
            for key, value in prediction_metadata.items():
                test_rows[key] = value
        test_rows["pred_AAC"] = preds_raw[:, 0]
        test_rows["pred_log2IC50"] = preds_raw[:, 1]
        test_rows.to_csv(out_dir / f"{model_name}_test_predictions.csv", index=False)

    return {"model": model_name, "split": "test", **test_metrics}, best_score


def prepare_data(seed: int, drkg_dir: Path, ic50_min: float, ic50_max: float, lds_max_weight: float):
    print("Loading CTRP AAC + IC50 data...")
    pair_table = pd.read_hdf(DATA_DIR / "CTRP_AAC_IC50_SMILES.hdf", "df")
    rows_before_ic50_filter = len(pair_table)
    pair_table = pair_table[
        pair_table["log2_ic50"].between(ic50_min, ic50_max)
    ].copy()

    print("Loading DRKG drug embeddings...")
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

    print("Loading DepMap expression...")
    expression_df = pd.read_hdf(DATA_DIR / "DepMap_26Q1_Expression.hdf", "df")
    expression_df["cell_norm"] = expression_df["stripped_cell_line_name"].map(normalize_name)
    expression_df = expression_df.drop_duplicates("cell_norm")
    expression_map = {name: i for i, name in enumerate(expression_df["cell_norm"])}

    pair_table["cell_norm"] = pair_table["ccl_name"].map(normalize_name)
    pair_table = pair_table[pair_table["cell_norm"].isin(expression_map)].copy()
    pair_table = pair_table.reset_index(drop=True)

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
    gene_cols = [col for col in expression_df.columns if col not in info_cols]
    expression = expression_df[gene_cols].to_numpy(dtype=np.float32)

    unique_drugs = pair_table[["cpd_name", "cpd_smiles", "drug_norm"]].drop_duplicates("cpd_name").reset_index(drop=True)
    drug_to_idx = {drug: i for i, drug in enumerate(unique_drugs["cpd_name"])}
    graphs = [smiles_to_graph(smiles) for smiles in unique_drugs["cpd_smiles"]]
    kg = np.vstack(
        [drkg_features[norm_to_feature_row[drug_norm]] for drug_norm in unique_drugs["drug_norm"]]
    ).astype(np.float32)

    cell_indices = pair_table["cell_norm"].map(expression_map).to_numpy(dtype=np.int64)
    drug_indices = pair_table["cpd_name"].map(drug_to_idx).to_numpy(dtype=np.int64)
    targets_raw = pair_table[["area_above_curve", "log2_ic50"]].to_numpy(dtype=np.float32)

    rng = np.random.default_rng(seed)
    unique_cells = np.unique(cell_indices)
    rng.shuffle(unique_cells)
    n_train = int(len(unique_cells) * 0.8)
    n_valid = int(len(unique_cells) * 0.1)
    train_cells = set(unique_cells[:n_train])
    valid_cells = set(unique_cells[n_train : n_train + n_valid])

    all_rows = np.arange(len(pair_table))
    train_rows = all_rows[np.isin(cell_indices, list(train_cells))]
    valid_rows = all_rows[np.isin(cell_indices, list(valid_cells))]
    test_rows = np.setdiff1d(all_rows, np.concatenate([train_rows, valid_rows]), assume_unique=False)

    train_expression_rows = np.array(sorted(train_cells), dtype=np.int64)
    expression_mean = expression[train_expression_rows].mean(axis=0, keepdims=True)
    expression_std = expression[train_expression_rows].std(axis=0, keepdims=True)
    expression = ((expression - expression_mean) / (expression_std + 1e-6)).astype(np.float32)

    target_mean = targets_raw[train_rows].mean(axis=0, keepdims=True)
    target_std = targets_raw[train_rows].std(axis=0, keepdims=True) + 1e-6
    targets_scaled = ((targets_raw - target_mean) / target_std).astype(np.float32)

    weights = np.vstack(
        [
            lds_weights(targets_raw[train_rows, 0], targets_raw[:, 0], max_weight=lds_max_weight),
            lds_weights(targets_raw[train_rows, 1], targets_raw[:, 1], max_weight=lds_max_weight),
        ]
    ).T.astype(np.float32)

    return {
        "pair_table": pair_table,
        "expression": expression,
        "graphs": graphs,
        "kg": kg,
        "cell_indices": cell_indices,
        "drug_indices": drug_indices,
        "targets_raw": targets_raw,
        "targets_scaled": targets_scaled,
        "target_mean": target_mean.astype(np.float32),
        "target_std": target_std.astype(np.float32),
        "weights": weights,
        "splits": {"train": train_rows, "valid": valid_rows, "test": test_rows},
        "summary": {
            "rows": int(len(pair_table)),
            "rows_before_ic50_filter": int(rows_before_ic50_filter),
            "ic50_log2_filter": [float(ic50_min), float(ic50_max)],
            "unique_drugs_with_drkg": int(pair_table["cpd_name"].nunique()),
            "unique_cell_lines": int(pair_table["ccl_name"].nunique()),
            "expression_dim": int(expression.shape[1]),
            "node_feature_dim": NODE_DIM,
            "edge_feature_dim": EDGE_DIM,
            "kg_dim": int(kg.shape[1]),
            "train_rows": int(len(train_rows)),
            "valid_rows": int(len(valid_rows)),
            "test_rows": int(len(test_rows)),
            "split_type": "cell_line_holdout",
            "targets": ["AAC", "log2IC50"],
            "loss": "alpha * LDS-weighted MSE(AAC) + (1-alpha) * LDS-weighted MSE(log2IC50)",
        },
    }


def sample_config(trial):
    return {
        "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-4, log=True),
        "dropout": trial.suggest_float("dropout", 0.05, 0.35),
        "gnn_hidden": trial.suggest_categorical("gnn_hidden", [64, 96, 128]),
        "rank": trial.suggest_categorical("rank", [4, 8, 16]),
        "fusion_dim": trial.suggest_categorical("fusion_dim", [64, 128, 256]),
        "gnn_layers": trial.suggest_int("gnn_layers", 2, 3),
        "gnn_timesteps": trial.suggest_int("gnn_timesteps", 2, 3),
        "alpha": trial.suggest_float("alpha", 0.6, 0.85),
        "patience": 3,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--trial-epochs", type=int, default=3)
    parser.add_argument("--final-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--search-train-rows", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ic50-min", type=float, default=-15.0)
    parser.add_argument("--ic50-max", type=float, default=15.0)
    parser.add_argument("--lds-max-weight", type=float, default=10.0)
    parser.add_argument("--out-dir", default=str(ROOT / "results" / "attentivefp_multitask_drkg_optuna"))
    parser.add_argument("--drkg-dir", default=str(DEFAULT_DRKG_DIR))
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prepared = prepare_data(args.seed, Path(args.drkg_dir), args.ic50_min, args.ic50_max, args.lds_max_weight)
    summary = {
        **prepared["summary"],
        "drug_branch": "AttentiveFP GNN",
        "fusion": "LMF",
        "kg_branch": "DRKG TransE 200d -> MLP 128d -> LMF",
        "optuna_trials_requested": args.n_trials,
        "trial_epochs": args.trial_epochs,
        "final_epochs": args.final_epochs,
        "search_train_rows": args.search_train_rows,
        "lds_max_weight": args.lds_max_weight,
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    rng = np.random.default_rng(args.seed)
    train_rows = prepared["splits"]["train"]
    if args.search_train_rows and len(train_rows) > args.search_train_rows:
        search_rows = rng.choice(train_rows, size=args.search_train_rows, replace=False)
    else:
        search_rows = train_rows

    def objective(trial):
        config = sample_config(trial)
        _, score = train_model(
            "trial_b3_attentivefp_plus_drkg",
            True,
            prepared,
            config,
            None,
            args.trial_epochs,
            args.batch_size,
            device,
            trial=trial,
            train_rows_override=search_rows,
        )
        return score

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
    )
    study.optimize(objective, n_trials=args.n_trials, gc_after_trial=True)

    trials_df = study.trials_dataframe()
    trials_df.to_csv(out_dir / "optuna_trials.csv", index=False)
    best_config = {**study.best_params, "patience": 5}
    (out_dir / "best_config.json").write_text(json.dumps(best_config, indent=2), encoding="utf-8")
    print("Best config:")
    print(json.dumps(best_config, indent=2))

    results = []
    base_result, _ = train_model(
        "b3_attentivefp_multitask_base",
        False,
        prepared,
        best_config,
        out_dir,
        args.final_epochs,
        args.batch_size,
        device,
    )
    results.append(base_result)

    drkg_result, _ = train_model(
        "b3_attentivefp_multitask_plus_drkg",
        True,
        prepared,
        best_config,
        out_dir,
        args.final_epochs,
        args.batch_size,
        device,
    )
    results.append(drkg_result)

    metrics = pd.DataFrame(results)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
