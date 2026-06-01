#!/usr/bin/env python
"""
Integrated Gradients attribution for pancancer B3 or B3+DRKG checkpoints.

Default checkpoint:
  results/pancancer_B3_B3DRKG_26Q1_CTRPv2_50e_full_folds/seed42_fold0/cv_plus_drkg.pt

Outputs:
  results/pancancer_ig_attribution
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


BASE_COLS = [
    "#f3877f",
    "#c2e8bc",
    "#8ed3c7",
    "#d5a3b1",
    "#e1b27a",
    "#c2c0d6",
    "#94a8c2",
    "#e2e1c3",
    "#bd89bd",
    "#fccde5",
    "#984ea3",
    "#ccebc5",
    "#ffed6f",
    "#9ebcda",
    "#fb8072",
]

INFO_COLS = {
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


def package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def setup_local_imports(root: Path):
    src = root / "src"
    os.environ["MMDRP_ROOT"] = str(root)
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def parse_targets(value: str):
    mapping = {"AAC": 0, "log2IC50": 1, "IC50": 1}
    targets = []
    for item in value.split(","):
        key = item.strip()
        if not key:
            continue
        if key not in mapping:
            raise ValueError(f"Unknown target {key}. Use AAC and/or log2IC50.")
        targets.append((key if key != "IC50" else "log2IC50", mapping[key]))
    return targets


def get_gene_names(expression_hdf: Path):
    expression_df = pd.read_hdf(expression_hdf, "df")
    return [
        col
        for col in expression_df.columns
        if col not in INFO_COLS and pd.api.types.is_numeric_dtype(expression_df[col])
    ]


def select_rows(prepared, args):
    test_rows = np.asarray(prepared["splits"]["test"], dtype=np.int64)
    if args.sample_strategy == "random":
        rng = np.random.default_rng(args.random_seed)
        chosen = rng.choice(test_rows, size=min(args.max_samples, len(test_rows)), replace=False)
        return np.sort(chosen)

    if args.sample_strategy == "nsclc_random":
        table = prepared["pair_table"]
        nsclc_rows = test_rows[table.iloc[test_rows]["is_nsclc"].to_numpy(dtype=bool)]
        if len(nsclc_rows) == 0:
            raise ValueError("No NSCLC rows found in this pancancer test fold.")
        rng = np.random.default_rng(args.random_seed)
        chosen = rng.choice(nsclc_rows, size=min(args.max_samples, len(nsclc_rows)), replace=False)
        return np.sort(chosen)

    pred_file = args.fold_dir / f"{args.model}_test_predictions.csv"
    if not pred_file.exists():
        raise FileNotFoundError(f"Missing prediction file for error-based sampling: {pred_file}")
    preds = pd.read_csv(pred_file)
    preds["combined_abs_error"] = (
        0.7 * (preds["pred_AAC"] - preds["area_above_curve"]).abs()
        + 0.3 * (preds["pred_log2IC50"] - preds["log2_ic50"]).abs()
    )
    chosen = preds.sort_values("combined_abs_error", ascending=False)["row_id"].head(args.max_samples)
    return chosen.to_numpy(dtype=np.int64)


def load_model(prepared, config, checkpoint: Path, model_key: str, device):
    from run_attentivefp_multitask_drkg_optuna import MultitaskAttentiveFPLMF

    use_kg = model_key == "cv_plus_drkg"
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
    try:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, use_kg


def make_batch(prepared, row_indices, use_kg, device):
    from run_attentivefp_multitask_drkg_optuna import GraphBatchCollator, move_batch

    collator = GraphBatchCollator(prepared, use_kg=use_kg)
    batch = collator(row_indices.tolist())
    batch = move_batch(batch, device)
    return batch


def integrated_gradients(model, batch, target_idx, steps, use_kg):
    expression = batch["expression"].detach()
    exp_baseline = torch.zeros_like(expression)
    exp_delta = expression - exp_baseline
    exp_grad_sum = torch.zeros_like(expression)

    if use_kg:
        kg = batch["kg"].detach()
        kg_baseline = torch.zeros_like(kg)
        kg_delta = kg - kg_baseline
        kg_grad_sum = torch.zeros_like(kg)
    else:
        kg = kg_baseline = kg_delta = kg_grad_sum = None

    for step in range(1, steps + 1):
        scale = float(step) / float(steps)
        exp_step = (exp_baseline + scale * exp_delta).detach().requires_grad_(True)
        if use_kg:
            kg_step = (kg_baseline + scale * kg_delta).detach().requires_grad_(True)
            outputs = model(exp_step, batch["graph_batch"], batch["graph_inverse"], kg_step)
            grads = torch.autograd.grad(outputs[:, target_idx].sum(), [exp_step, kg_step])
            exp_grad_sum += grads[0].detach()
            kg_grad_sum += grads[1].detach()
        else:
            outputs = model(exp_step, batch["graph_batch"], batch["graph_inverse"], None)
            grad = torch.autograd.grad(outputs[:, target_idx].sum(), exp_step)[0]
            exp_grad_sum += grad.detach()

    exp_attr = exp_delta * exp_grad_sum / float(steps)
    if use_kg:
        kg_attr = kg_delta * kg_grad_sum / float(steps)
    else:
        kg_attr = None
    return exp_attr.detach().cpu().numpy(), None if kg_attr is None else kg_attr.detach().cpu().numpy()


def attribution_tables(
    exp_attr,
    kg_attr,
    row_indices,
    prepared,
    gene_names,
    target_name,
    target_std,
    top_n,
):
    scale = float(target_std[0, 0 if target_name == "AAC" else 1])
    exp_attr_raw = exp_attr * scale
    gene_df = pd.DataFrame(
        {
            "target": target_name,
            "feature_type": "expression_gene",
            "feature": gene_names,
            "mean_abs_attribution": np.mean(np.abs(exp_attr_raw), axis=0),
            "mean_signed_attribution": np.mean(exp_attr_raw, axis=0),
        }
    ).sort_values("mean_abs_attribution", ascending=False)

    table = prepared["pair_table"].iloc[row_indices].copy()
    sample_df = pd.DataFrame(
        {
            "target": target_name,
            "row_id": row_indices,
            "cpd_name": table["cpd_name"].to_numpy(),
            "ccl_name": table["ccl_name"].to_numpy(),
            "primary_disease": table.get("primary_disease", pd.Series([""] * len(table))).to_numpy(),
            "mean_abs_expression_ig": np.mean(np.abs(exp_attr_raw), axis=1),
            "sum_abs_expression_ig": np.sum(np.abs(exp_attr_raw), axis=1),
        }
    )

    kg_df = None
    if kg_attr is not None:
        kg_attr_raw = kg_attr * scale
        kg_df = pd.DataFrame(
            {
                "target": target_name,
                "feature_type": "drkg_embedding_dim",
                "feature": [f"DRKG_dim_{i:03d}" for i in range(kg_attr_raw.shape[1])],
                "mean_abs_attribution": np.mean(np.abs(kg_attr_raw), axis=0),
                "mean_signed_attribution": np.mean(kg_attr_raw, axis=0),
            }
        ).sort_values("mean_abs_attribution", ascending=False)
        sample_df["mean_abs_drkg_ig"] = np.mean(np.abs(kg_attr_raw), axis=1)
        sample_df["sum_abs_drkg_ig"] = np.sum(np.abs(kg_attr_raw), axis=1)

    return gene_df.head(top_n), kg_df.head(top_n) if kg_df is not None else None, sample_df


def style_axes(ax, title=None, xlabel=None):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color="#e8e8e8", linewidth=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, fontsize=12, weight="bold", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel)


def plot_top_features(df, title, out_path, color, dpi):
    if df is None or df.empty:
        return
    data = df.sort_values("mean_abs_attribution", ascending=True)
    fig_height = max(5.0, 0.24 * len(data) + 1.4)
    fig, ax = plt.subplots(figsize=(9.5, fig_height))
    ax.barh(data["feature"], data["mean_abs_attribution"], color=color, edgecolor="#3d3d3d", linewidth=0.5)
    style_axes(ax, title, "Mean absolute Integrated Gradients attribution")
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_signed_features(df, title, out_path, dpi):
    if df is None or df.empty:
        return
    data = df.copy().sort_values("mean_abs_attribution", ascending=True)
    fig_height = max(5.0, 0.24 * len(data) + 1.4)
    colors = np.where(data["mean_signed_attribution"] >= 0, BASE_COLS[0], BASE_COLS[6])
    fig, ax = plt.subplots(figsize=(10.0, fig_height))
    ax.barh(data["feature"], data["mean_signed_attribution"], color=colors, edgecolor="#3d3d3d", linewidth=0.5)
    ax.axvline(0, color="#333333", linewidth=1)
    style_axes(ax, title, "Mean signed Integrated Gradients attribution")
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_sample_feature_heatmap(attr, feature_names, top_features, row_indices, target_name, out_path, dpi, max_samples=40):
    if attr.size == 0 or top_features.empty:
        return
    feature_to_idx = {name: i for i, name in enumerate(feature_names)}
    selected_features = [f for f in top_features["feature"].tolist() if f in feature_to_idx]
    if not selected_features:
        return
    feat_idx = [feature_to_idx[f] for f in selected_features]
    sample_count = min(max_samples, attr.shape[0])
    heat = attr[:sample_count, :][:, feat_idx].T
    limit = np.nanpercentile(np.abs(heat), 98)
    if not np.isfinite(limit) or limit == 0:
        limit = float(np.max(np.abs(heat)) + 1e-8)

    fig_width = max(10.0, sample_count * 0.24)
    fig_height = max(7.0, len(selected_features) * 0.18 + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(heat, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set_yticks(np.arange(len(selected_features)))
    ax.set_yticklabels(selected_features, fontsize=7)
    ax.set_xticks(np.arange(sample_count))
    ax.set_xticklabels([str(int(x)) for x in row_indices[:sample_count]], rotation=90, fontsize=6)
    ax.set_title(f"Sample-level IG Heatmap: Top Genes ({target_name})", fontsize=12, weight="bold", pad=10)
    ax.set_xlabel("row_id")
    ax.set_ylabel("Gene")
    fig.colorbar(im, ax=ax, fraction=0.018, pad=0.02, label="IG attribution")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_sample_modality_scatter(sample_df, target_name, out_path, dpi):
    if "sum_abs_drkg_ig" not in sample_df.columns:
        return
    fig, ax = plt.subplots(figsize=(7.2, 5.5))
    ax.scatter(
        sample_df["sum_abs_expression_ig"],
        sample_df["sum_abs_drkg_ig"],
        s=36,
        alpha=0.75,
        color=BASE_COLS[4],
        edgecolor="#333333",
        linewidth=0.4,
    )
    style_axes(ax, f"Sample-level Modality Attribution ({target_name})")
    ax.set_xlabel("Sum absolute expression IG")
    ax.set_ylabel("Sum absolute DRKG IG")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_modality_contribution(sample_tables, out_path, dpi):
    merged = pd.concat(sample_tables, ignore_index=True)
    cols = ["sum_abs_expression_ig"]
    labels = ["Expression"]
    colors = [BASE_COLS[2]]
    if "sum_abs_drkg_ig" in merged.columns:
        cols.append("sum_abs_drkg_ig")
        labels.append("DRKG embedding")
        colors.append(BASE_COLS[4])
    rows = []
    for target, group in merged.groupby("target"):
        total = {label: float(group[col].mean()) for label, col in zip(labels, cols)}
        rows.append({"target": target, **total})
    data = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    x = np.arange(len(data))
    bottom = np.zeros(len(data))
    for label, color in zip(labels, colors):
        values = data[label].to_numpy(dtype=float)
        ax.bar(x, values, bottom=bottom, color=color, edgecolor="#3d3d3d", linewidth=0.6, label=label)
        bottom += values
    ax.set_xticks(x)
    ax.set_xticklabels(data["target"])
    ax.set_ylabel("Mean sum absolute IG attribution")
    ax.set_title("Integrated Gradients Modality Contribution", fontsize=12, weight="bold", pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    if len(labels) > 1:
        frac = data.copy()
        totals = frac[labels].sum(axis=1).replace(0, np.nan)
        for label in labels:
            frac[label] = frac[label] / totals
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        x = np.arange(len(frac))
        bottom = np.zeros(len(frac))
        for label, color in zip(labels, colors):
            values = frac[label].to_numpy(dtype=float)
            ax.bar(x, values, bottom=bottom, color=color, edgecolor="#3d3d3d", linewidth=0.6, label=label)
            for i, value in enumerate(values):
                if np.isfinite(value) and value > 0.06:
                    ax.text(i, bottom[i] + value / 2, f"{value:.0%}", ha="center", va="center", fontsize=9)
            bottom += np.nan_to_num(values)
        ax.set_xticks(x)
        ax.set_xticklabels(frac["target"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Fraction of total absolute IG attribution")
        ax.set_title("Integrated Gradients Modality Contribution Fraction", fontsize=12, weight="bold", pad=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=len(labels))
        fig.tight_layout()
        frac_path = out_path.with_name(out_path.stem + "_fraction").with_suffix(".png")
        fig.savefig(frac_path, dpi=dpi, bbox_inches="tight")
        fig.savefig(frac_path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)


def parse_args():
    root = package_root()
    parser = argparse.ArgumentParser(description="Run pancancer Integrated Gradients attribution.")
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument(
        "--full-fold-dir",
        type=Path,
        default=root / "results" / "pancancer_B3_B3DRKG_26Q1_CTRPv2_50e_full_folds",
    )
    parser.add_argument(
        "--drkg-dir",
        type=Path,
        default=root / "Data" / "DRKG_Features" / "pancancer_driver_h1_200d_50e",
    )
    parser.add_argument(
        "--expression-hdf",
        type=Path,
        default=root / "Data" / "DRP_Training_Data" / "DepMap_26Q1_Expression.hdf",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--model", choices=["cv_base", "cv_plus_drkg"], default="cv_plus_drkg")
    parser.add_argument("--targets", default="AAC,log2IC50")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--random-seed", type=int, default=2026)
    parser.add_argument(
        "--sample-strategy",
        choices=["random", "highest_error", "nsclc_random"],
        default="random",
    )
    parser.add_argument("--ic50-min", type=float, default=-15.0)
    parser.add_argument("--ic50-max", type=float, default=15.0)
    parser.add_argument("--lds-max-weight", type=float, default=10.0)
    parser.add_argument("--use-lds", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=root / "results" / "pancancer_ig_attribution",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fold_dir = args.full_fold_dir / f"seed{args.seed}_fold{args.fold}"
    checkpoint = args.fold_dir / f"{args.model}.pt"
    split_path = args.fold_dir / "split_indices.npz"
    run_config_path = args.full_fold_dir / "run_config.json"

    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split indices: {split_path}")
    if not run_config_path.exists():
        raise FileNotFoundError(f"Missing run config: {run_config_path}")
    if not args.drkg_dir.exists():
        raise FileNotFoundError(f"Missing pancancer DRKG feature directory: {args.drkg_dir}")

    setup_local_imports(args.root)
    from run_attentivefp_multiseed_cv import build_prepared_for_split, load_base_data

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if args.device == "auto" and not torch.cuda.is_available():
        device = torch.device("cpu")

    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    base_config = run_config["base_config"]
    base = load_base_data(args.drkg_dir, args.ic50_min, args.ic50_max, args.expression_hdf)
    split = np.load(split_path)
    prepared = build_prepared_for_split(
        base,
        split["train_rows"],
        split["valid_rows"],
        split["test_rows"],
        args.lds_max_weight,
        args.use_lds,
    )
    model, use_kg = load_model(prepared, base_config, checkpoint, args.model, device)
    row_indices = select_rows(prepared, args)
    batch = make_batch(prepared, row_indices, use_kg, device)
    gene_names = get_gene_names(args.expression_hdf)
    targets = parse_targets(args.targets)

    gene_tables = []
    kg_tables = []
    sample_tables = []
    for target_name, target_idx in targets:
        exp_attr, kg_attr = integrated_gradients(model, batch, target_idx, args.steps, use_kg)
        gene_top, kg_top, sample_df = attribution_tables(
            exp_attr,
            kg_attr,
            row_indices,
            prepared,
            gene_names,
            target_name,
            prepared["target_std"],
            args.top_n,
        )
        gene_tables.append(gene_top)
        sample_tables.append(sample_df)
        if kg_top is not None:
            kg_tables.append(kg_top)

        plot_top_features(
            gene_top,
            f"Top Expression Gene IG Attributions ({target_name})",
            args.out_dir / f"ig_top_genes_{target_name}.png",
            BASE_COLS[2],
            args.dpi,
        )
        plot_signed_features(
            gene_top,
            f"Signed Expression Gene IG Attributions ({target_name})",
            args.out_dir / f"ig_signed_top_genes_{target_name}.png",
            args.dpi,
        )
        scaled_exp_attr = exp_attr * float(prepared["target_std"][0, target_idx])
        plot_sample_feature_heatmap(
            scaled_exp_attr,
            gene_names,
            gene_top,
            row_indices,
            target_name,
            args.out_dir / f"ig_gene_sample_heatmap_{target_name}.png",
            args.dpi,
        )
        plot_sample_modality_scatter(
            sample_df,
            target_name,
            args.out_dir / f"ig_sample_modality_scatter_{target_name}.png",
            args.dpi,
        )
        if kg_top is not None:
            plot_top_features(
                kg_top,
                f"Top DRKG Embedding IG Attributions ({target_name})",
                args.out_dir / f"ig_top_drkg_dims_{target_name}.png",
                BASE_COLS[4],
                args.dpi,
            )
            plot_signed_features(
                kg_top,
                f"Signed DRKG Embedding IG Attributions ({target_name})",
                args.out_dir / f"ig_signed_top_drkg_dims_{target_name}.png",
                args.dpi,
            )

    pd.concat(gene_tables, ignore_index=True).to_csv(args.out_dir / "ig_top_gene_attributions.csv", index=False)
    if kg_tables:
        pd.concat(kg_tables, ignore_index=True).to_csv(args.out_dir / "ig_top_drkg_dimension_attributions.csv", index=False)
    pd.concat(sample_tables, ignore_index=True).to_csv(args.out_dir / "ig_sample_attributions.csv", index=False)
    plot_modality_contribution(sample_tables, args.out_dir / "ig_modality_contribution.png", args.dpi)

    manifest = {
        "scope": "pancancer",
        "seed": args.seed,
        "fold": args.fold,
        "model": args.model,
        "checkpoint": str(checkpoint),
        "drkg_dir": str(args.drkg_dir),
        "expression_hdf": str(args.expression_hdf),
        "sample_strategy": args.sample_strategy,
        "sample_count": int(len(row_indices)),
        "steps": args.steps,
        "targets": [name for name, _ in targets],
        "device": str(device),
        "output_dir": str(args.out_dir),
    }
    (args.out_dir / "ig_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved pancancer IG attribution outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()
