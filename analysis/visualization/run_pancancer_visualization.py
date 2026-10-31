#!/usr/bin/env python
"""
Pancancer-only visualization script for the final MMDRP-DRKG repro package.

Default inputs:
  results/pancancer_50e_evaluation
  results/pancancer_B3_B3DRKG_26Q1_CTRPv2_50e_full_folds

Outputs:
  results/pancancer_visualization_figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
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

MODEL_ORDER = ["B0", "B1", "B2", "B3", "B3+DRKG"]
MODEL_COLORS = {
    "B0": BASE_COLS[0],
    "B1": BASE_COLS[1],
    "B2": BASE_COLS[2],
    "B3": BASE_COLS[3],
    "B3+DRKG": BASE_COLS[4],
}
MODEL_LABELS = {
    "B0_drug_median": "B0",
    "B1_elastic_net": "B1",
    "B2_mmdrp_base": "B2",
    "cv_base": "B3",
    "cv_plus_drkg": "B3+DRKG",
}
SIGNIFICANCE_NOTE = "Symbols compare each model against B3+DRKG; one-sided paired sign-flip test over 15 seed-fold runs."


def package_root() -> Path:
    for candidate in Path(__file__).absolute().parents:
        if (candidate / "Data").is_dir() and (candidate / "results").is_dir() and (candidate / "src").is_dir():
            return candidate
    raise RuntimeError("Could not locate package root containing Data/, results/, and src/.")


def repo_relative(path, root) -> str:
    """Return path relative to the repository root when possible."""
    try:
        return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return str(path)


def ensure_out(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def style_axes(ax, title=None, ylabel=None, xlabel=None):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e8e8e8", linewidth=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, fontsize=13, weight="bold", pad=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10)


def savefig(fig, out_path: Path, dpi: int):
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def add_bar_labels(ax, fmt="{:.3f}", pad=0.01):
    ylim = ax.get_ylim()
    span = ylim[1] - ylim[0]
    for patch in ax.patches:
        height = patch.get_height()
        if not np.isfinite(height):
            continue
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            height + span * pad,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0,
        )
    ax.set_ylim(ylim[0], ylim[1] + span * 0.08)


def significance_symbol(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "ns"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def paired_signflip_pvalue(diffs: np.ndarray) -> float:
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    if len(diffs) == 0:
        return np.nan
    observed = float(np.mean(diffs))
    if observed <= 0:
        return 1.0

    total = 1 << len(diffs)
    extreme = 0
    for mask in range(total):
        signs = np.ones(len(diffs), dtype=float)
        for bit in range(len(diffs)):
            if (mask >> bit) & 1:
                signs[bit] = -1.0
        stat = float(np.mean(signs * diffs))
        if stat >= observed - 1e-12:
            extreme += 1
    return (extreme + 1.0) / (total + 1.0)


def compute_reference_significance(
    metrics: pd.DataFrame,
    metric_cols,
    reference: str = "B3+DRKG",
    evaluation_subset: str = "overall",
) -> pd.DataFrame:
    data = metrics.copy()
    if "evaluation_subset" in data.columns:
        data = data[data["evaluation_subset"].eq(evaluation_subset)].copy()
    data = normalize_model_labels(data)

    rows = []
    for metric_col in metric_cols:
        pivot = data.pivot_table(index=["seed", "fold"], columns="model_display", values=metric_col)
        if reference not in pivot.columns:
            continue
        for model in pivot.columns:
            if model == reference:
                continue
            paired = pivot[[model, reference]].dropna()
            if paired.empty:
                continue
            diffs = paired[model].to_numpy(dtype=float) - paired[reference].to_numpy(dtype=float)
            p_value = paired_signflip_pvalue(diffs)
            rows.append(
                {
                    "metric": metric_col,
                    "comparison": f"{model}_minus_{reference}",
                    "model": model,
                    "reference": reference,
                    "runs": int(len(diffs)),
                    "model_mean": float(paired[model].mean()),
                    "reference_mean": float(paired[reference].mean()),
                    "delta_mean": float(np.mean(diffs)),
                    "improved_runs_for_reference": int(np.sum(diffs > 0)),
                    "one_sided_signflip_p": float(p_value),
                    "symbol": significance_symbol(p_value),
                }
            )
    return pd.DataFrame(rows)


def add_significance_labels(ax, data: pd.DataFrame, metric_col: str, sig_df: pd.DataFrame):
    if sig_df is None or sig_df.empty:
        return
    sig_map = sig_df[sig_df["metric"].eq(metric_col)].set_index("model")["symbol"].to_dict()
    ylim = ax.get_ylim()
    span = ylim[1] - ylim[0]
    for patch, model in zip(ax.patches, data["model_display"]):
        if model == "B3+DRKG":
            label = "ref"
        else:
            label = sig_map.get(model, "")
        if not label:
            continue
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            patch.get_height() + span * 0.055,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
            weight="bold" if label != "ns" else "normal",
            color="#222222",
        )
    ax.set_ylim(ylim[0], ylim[1] + span * 0.12)


def normalize_model_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "model_display" not in df.columns:
        df["model_display"] = df["model"].map(MODEL_LABELS).fillna(df["model"])
    else:
        df["model_display"] = df["model_display"].fillna(df["model"].map(MODEL_LABELS))
    df["model_display"] = df["model_display"].replace(MODEL_LABELS)
    return df


def ordered_models(df: pd.DataFrame):
    present = list(dict.fromkeys(df["model_display"].tolist()))
    ordered = [m for m in MODEL_ORDER if m in present]
    ordered.extend([m for m in present if m not in ordered])
    return ordered


def plot_metric_bars(summary: pd.DataFrame, out_dir: Path, dpi: int, sig_df: pd.DataFrame):
    metrics = [
        ("AAC_RMSE_mean", "AAC RMSE", "01_pancancer_aac_rmse_bar.png"),
        ("log2IC50_RMSE_mean", "log2IC50 RMSE", "02_pancancer_log2ic50_rmse_bar.png"),
        ("combined_RMSE_mean", "Combined RMSE", "03_pancancer_combined_rmse_bar.png"),
    ]
    models = ordered_models(summary)

    for col, label, filename in metrics:
        data = summary.set_index("model_display").loc[models].reset_index()
        fig, ax = plt.subplots(figsize=(8.8, 5.2))
        colors = [MODEL_COLORS.get(m, BASE_COLS[i % len(BASE_COLS)]) for i, m in enumerate(data["model_display"])]
        ax.bar(data["model_display"], data[col], color=colors, edgecolor="#3d3d3d", linewidth=0.8)
        style_axes(ax, f"Pancancer Model Comparison: {label}", label, "")
        ax.tick_params(axis="x", labelrotation=0)
        add_bar_labels(ax)
        add_significance_labels(ax, data, col.replace("_mean", ""), sig_df)
        fig.text(0.01, 0.005, SIGNIFICANCE_NOTE, ha="left", va="bottom", fontsize=8, color="#555555")
        savefig(fig, out_dir / filename, dpi)

    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.6), sharex=False)
    for ax, (col, label, _) in zip(axes, metrics):
        data = summary.set_index("model_display").loc[models].reset_index()
        colors = [MODEL_COLORS.get(m, BASE_COLS[i % len(BASE_COLS)]) for i, m in enumerate(data["model_display"])]
        ax.bar(data["model_display"], data[col], color=colors, edgecolor="#3d3d3d", linewidth=0.7)
        style_axes(ax, label, label, "")
        ax.tick_params(axis="x", labelrotation=32)
        add_bar_labels(ax)
        add_significance_labels(ax, data, col.replace("_mean", ""), sig_df)
    fig.text(0.01, 0.005, SIGNIFICANCE_NOTE, ha="left", va="bottom", fontsize=8, color="#555555")
    savefig(fig, out_dir / "04_pancancer_rmse_three_panel.png", dpi)


def plot_correlations(summary: pd.DataFrame, out_dir: Path, dpi: int):
    corr_cols = [
        ("AAC_Pearson_mean", "AAC Pearson"),
        ("AAC_Spearman_mean", "AAC Spearman"),
        ("log2IC50_Pearson_mean", "log2IC50 Pearson"),
        ("log2IC50_Spearman_mean", "log2IC50 Spearman"),
    ]
    models = ordered_models(summary)
    data = summary.set_index("model_display").loc[models].reset_index()
    x = np.arange(len(corr_cols))
    width = 0.16

    fig, ax = plt.subplots(figsize=(12, 5.4))
    for i, model in enumerate(models):
        vals = [float(data.loc[data["model_display"] == model, col].iloc[0]) for col, _ in corr_cols]
        ax.bar(
            x + (i - (len(models) - 1) / 2) * width,
            vals,
            width=width,
            label=model,
            color=MODEL_COLORS.get(model, BASE_COLS[i % len(BASE_COLS)]),
            edgecolor="#3d3d3d",
            linewidth=0.5,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in corr_cols], rotation=15, ha="right")
    style_axes(ax, "Pancancer Correlation Metrics", "Correlation", "")
    ax.set_ylim(0.65, 0.86)
    ax.legend(ncol=len(models), frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    savefig(fig, out_dir / "05_pancancer_correlation_metrics.png", dpi)


def plot_fold_lines(metrics: pd.DataFrame, out_dir: Path, dpi: int):
    metrics = metrics[metrics["evaluation_subset"].eq("overall")].copy()
    metrics = normalize_model_labels(metrics)
    models = ordered_models(metrics)
    labels = []

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    for i, model in enumerate(models):
        data = metrics[metrics["model_display"].eq(model)].sort_values(["seed", "fold"])
        x = np.arange(len(data))
        labels = [f"{int(s)}-{int(f)}" for s, f in zip(data["seed"], data["fold"])]
        ax.plot(
            x,
            data["combined_RMSE"],
            marker="o",
            linewidth=2,
            markersize=4,
            color=MODEL_COLORS.get(model, BASE_COLS[i % len(BASE_COLS)]),
            label=model,
        )
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    style_axes(ax, "Combined RMSE Across Seeds and Folds", "Combined RMSE", "Seed-Fold")
    ax.legend(frameon=False, ncol=len(models), loc="upper center", bbox_to_anchor=(0.5, 1.13))
    savefig(fig, out_dir / "06_seed_fold_combined_rmse_line.png", dpi)

    pivot = metrics.pivot_table(index="model_display", columns=["seed", "fold"], values="combined_RMSE")
    pivot = pivot.reindex(models)
    fig, ax = plt.subplots(figsize=(12.5, 4.6))
    im = ax.imshow(pivot.values, cmap="YlGnBu_r", aspect="auto")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{int(s)}-{int(f)}" for s, f in pivot.columns], rotation=45, ha="right", fontsize=8)
    for r in range(pivot.shape[0]):
        for c in range(pivot.shape[1]):
            val = pivot.values[r, c]
            if np.isfinite(val):
                ax.text(c, r, f"{val:.3f}", ha="center", va="center", fontsize=7, color="#202020")
    ax.set_title("Combined RMSE Heatmap by Seed-Fold", fontsize=13, weight="bold", pad=12)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Combined RMSE")
    savefig(fig, out_dir / "07_seed_fold_combined_rmse_heatmap.png", dpi)


def plot_b3_delta(metrics: pd.DataFrame, out_dir: Path, dpi: int):
    metrics = metrics[metrics["evaluation_subset"].eq("overall")].copy()
    metrics = normalize_model_labels(metrics)
    base = metrics[metrics["model_display"].eq("B3")]
    plus = metrics[metrics["model_display"].eq("B3+DRKG")]
    merged = base.merge(
        plus,
        on=["seed", "fold"],
        suffixes=("_B3", "_B3DRKG"),
    )
    if merged.empty:
        return
    merged["delta_combined_RMSE"] = merged["combined_RMSE_B3DRKG"] - merged["combined_RMSE_B3"]
    merged["delta_AAC_RMSE"] = merged["AAC_RMSE_B3DRKG"] - merged["AAC_RMSE_B3"]
    merged["delta_log2IC50_RMSE"] = merged["log2IC50_RMSE_B3DRKG"] - merged["log2IC50_RMSE_B3"]
    merged["seed_fold"] = merged.apply(lambda r: f"{int(r['seed'])}-{int(r['fold'])}", axis=1)

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    colors = [MODEL_COLORS["B3+DRKG"] if v <= 0 else MODEL_COLORS["B3"] for v in merged["delta_combined_RMSE"]]
    ax.bar(merged["seed_fold"], merged["delta_combined_RMSE"], color=colors, edgecolor="#3d3d3d", linewidth=0.6)
    ax.axhline(0, color="#333333", linewidth=1)
    style_axes(ax, "B3+DRKG Improvement Over B3 by Seed-Fold", "Delta Combined RMSE (B3+DRKG - B3)", "Seed-Fold")
    ax.tick_params(axis="x", labelrotation=45)
    savefig(fig, out_dir / "08_b3drkg_delta_combined_rmse_by_fold.png", dpi)
    merged.to_csv(out_dir / "b3_vs_b3drkg_fold_deltas.csv", index=False)


def downsample_predictions(preds: pd.DataFrame, max_points: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    for model, group in preds.groupby("model_display", dropna=False):
        if len(group) > max_points:
            idx = rng.choice(group.index.to_numpy(), size=max_points, replace=False)
            parts.append(group.loc[idx])
        else:
            parts.append(group)
    return pd.concat(parts, ignore_index=True)


def plot_predictions(preds: pd.DataFrame, out_dir: Path, dpi: int, max_points: int, seed: int):
    preds = normalize_model_labels(preds)
    preds = downsample_predictions(preds, max_points=max_points, seed=seed)
    models = ordered_models(preds)
    pairs = [
        ("area_above_curve", "pred_AAC", "AAC", "09_prediction_scatter_aac.png"),
        ("log2_ic50", "pred_log2IC50", "log2IC50", "10_prediction_scatter_log2ic50.png"),
    ]
    for true_col, pred_col, label, filename in pairs:
        ncols = min(3, len(models))
        nrows = int(np.ceil(len(models) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.4 * nrows), squeeze=False)
        for ax, model in zip(axes.ravel(), models):
            data = preds[preds["model_display"].eq(model)]
            ax.scatter(
                data[true_col],
                data[pred_col],
                s=7,
                alpha=0.22,
                color=MODEL_COLORS.get(model, "#6f6f6f"),
                edgecolor="none",
            )
            lower = min(data[true_col].min(), data[pred_col].min())
            upper = max(data[true_col].max(), data[pred_col].max())
            ax.plot([lower, upper], [lower, upper], color="#444444", linewidth=1, linestyle="--")
            style_axes(ax, model, f"Predicted {label}", f"Observed {label}")
        for ax in axes.ravel()[len(models):]:
            ax.axis("off")
        fig.suptitle(f"Pancancer Observed vs Predicted {label}", fontsize=14, weight="bold", y=1.01)
        savefig(fig, out_dir / filename, dpi)

    residual_defs = [
        ("area_above_curve", "pred_AAC", "AAC residual", "11_residual_distribution_aac.png"),
        ("log2_ic50", "pred_log2IC50", "log2IC50 residual", "12_residual_distribution_log2ic50.png"),
    ]
    for true_col, pred_col, label, filename in residual_defs:
        fig, ax = plt.subplots(figsize=(10.8, 5.2))
        for i, model in enumerate(models):
            data = preds[preds["model_display"].eq(model)]
            residual = data[pred_col] - data[true_col]
            ax.hist(
                residual,
                bins=55,
                density=True,
                histtype="step",
                linewidth=2,
                color=MODEL_COLORS.get(model, BASE_COLS[i % len(BASE_COLS)]),
                label=model,
            )
        ax.axvline(0, color="#333333", linewidth=1, linestyle="--")
        style_axes(ax, f"Pancancer {label} Distribution", "Density", "Prediction - Observed")
        ax.legend(frameon=False, ncol=len(models), loc="upper center", bbox_to_anchor=(0.5, 1.12))
        savefig(fig, out_dir / filename, dpi)


def collect_histories(full_fold_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(full_fold_dir.glob("seed*_fold*/*_history.csv")):
        model = path.name.replace("_history.csv", "")
        folder = path.parent.name
        try:
            seed = int(folder.split("_")[0].replace("seed", ""))
            fold = int(folder.split("_")[1].replace("fold", ""))
        except Exception:
            continue
        df = pd.read_csv(path)
        df["seed"] = seed
        df["fold"] = fold
        df["model"] = model
        df["model_display"] = MODEL_LABELS.get(model, model)
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def plot_training_curves(full_fold_dir: Path, out_dir: Path, dpi: int):
    hist = collect_histories(full_fold_dir)
    if hist.empty:
        return
    col = "valid_combined_RMSE"
    if col not in hist.columns:
        return
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    for i, model in enumerate(["B3", "B3+DRKG"]):
        data = hist[hist["model_display"].eq(model)]
        if data.empty:
            continue
        grouped = data.groupby("epoch", as_index=False)[col].mean()
        ax.plot(
            grouped["epoch"],
            grouped[col],
            linewidth=2.5,
            color=MODEL_COLORS.get(model, BASE_COLS[i]),
            label=model,
        )
    style_axes(ax, "Mean Validation Combined RMSE During Training", "Validation Combined RMSE", "Epoch")
    ax.legend(frameon=False)
    savefig(fig, out_dir / "13_training_curve_valid_combined_rmse.png", dpi)

    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    for i, model in enumerate(["B3", "B3+DRKG"]):
        data = hist[hist["model_display"].eq(model)]
        if data.empty or "train_loss" not in data.columns:
            continue
        grouped = data.groupby("epoch", as_index=False)["train_loss"].mean()
        ax.plot(
            grouped["epoch"],
            grouped["train_loss"],
            linewidth=2.5,
            color=MODEL_COLORS.get(model, BASE_COLS[i]),
            label=model,
        )
    style_axes(ax, "Mean Training Loss During Training", "Training Loss", "Epoch")
    ax.legend(frameon=False)
    savefig(fig, out_dir / "14_training_curve_mean_train_loss.png", dpi)

    fold_dir = out_dir / "training_curves_by_fold"
    fold_dir.mkdir(parents=True, exist_ok=True)
    plot_fold_grids(hist, out_dir, fold_dir, dpi)


def plot_fold_grids(hist: pd.DataFrame, out_dir: Path, fold_dir: Path, dpi: int):
    seed_folds = sorted(hist[["seed", "fold"]].drop_duplicates().itertuples(index=False, name=None))
    if not seed_folds:
        return

    metrics = [
        ("train_loss", "Training Loss", "15_train_loss_each_seed_fold.png"),
        ("valid_combined_RMSE", "Validation Combined RMSE", "16_valid_combined_rmse_each_seed_fold.png"),
    ]
    for metric_col, ylabel, filename in metrics:
        if metric_col not in hist.columns:
            continue
        fig, axes = plt.subplots(3, 5, figsize=(18, 9.8), sharex=False, sharey=False)
        for ax, (seed, fold) in zip(axes.ravel(), seed_folds):
            sub = hist[(hist["seed"].eq(seed)) & (hist["fold"].eq(fold))]
            for i, model in enumerate(["B3", "B3+DRKG"]):
                data = sub[sub["model_display"].eq(model)]
                if data.empty:
                    continue
                ax.plot(
                    data["epoch"],
                    data[metric_col],
                    linewidth=1.8,
                    color=MODEL_COLORS.get(model, BASE_COLS[i]),
                    label=model,
                )
            ax.set_title(f"seed {seed}, fold {fold}", fontsize=10, weight="bold")
            ax.grid(axis="y", color="#eeeeee", linewidth=0.7)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(axis="both", labelsize=8)
        handles, labels = axes.ravel()[0].get_legend_handles_labels()
        fig.suptitle(f"{ylabel} by Seed-Fold", fontsize=15, weight="bold", y=0.985)
        fig.legend(handles, labels, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.955))
        fig.text(0.5, 0.025, "Epoch", ha="center", fontsize=11)
        fig.subplots_adjust(top=0.88, bottom=0.08, left=0.045, right=0.995, hspace=0.36, wspace=0.18)
        out_path = out_dir / filename
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)

        for seed, fold in seed_folds:
            sub = hist[(hist["seed"].eq(seed)) & (hist["fold"].eq(fold))]
            fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
            for ax, col, title in [
                (axes[0], "train_loss", "Training Loss"),
                (axes[1], "valid_combined_RMSE", "Validation Combined RMSE"),
            ]:
                if col not in sub.columns:
                    ax.axis("off")
                    continue
                for i, model in enumerate(["B3", "B3+DRKG"]):
                    data = sub[sub["model_display"].eq(model)]
                    if data.empty:
                        continue
                    ax.plot(
                        data["epoch"],
                        data[col],
                        linewidth=2,
                        color=MODEL_COLORS.get(model, BASE_COLS[i]),
                        label=model,
                    )
                style_axes(ax, title, "Value", "Epoch")
            axes[0].legend(frameon=False)
            fig.suptitle(f"Training Diagnostics: seed {seed}, fold {fold}", fontsize=13, weight="bold")
            savefig(fig, fold_dir / f"seed{seed}_fold{fold}_training_diagnostics.png", dpi)


def write_training_curve_availability(full_fold_dir: Path, baseline_full_fold_dir: Path, out_dir: Path):
    b3_hist = collect_histories(full_fold_dir)
    rows = [
        {
            "model": "B0",
            "history_available": False,
            "reason": "Per-drug median statistical baseline; no gradient-based training loss.",
        },
        {
            "model": "B1",
            "history_available": False,
            "reason": "ElasticNet final checkpoints/predictions are available, but per-epoch history was not saved in this package.",
        },
        {
            "model": "B2",
            "history_available": False,
            "reason": "B2 checkpoints/predictions are available, but per-epoch history CSV was not saved in this package.",
        },
        {
            "model": "B3",
            "history_available": bool((b3_hist["model_display"].eq("B3")).any()) if not b3_hist.empty else False,
            "reason": "History CSV files found under pancancer B3/B3+DRKG full-fold results.",
        },
        {
            "model": "B3+DRKG",
            "history_available": bool((b3_hist["model_display"].eq("B3+DRKG")).any()) if not b3_hist.empty else False,
            "reason": "History CSV files found under pancancer B3/B3+DRKG full-fold results.",
        },
    ]
    pd.DataFrame(rows).to_csv(out_dir / "training_curve_availability.csv", index=False)


def write_manifest(out_dir: Path, summary: pd.DataFrame, args):
    best_row = summary.sort_values("combined_RMSE_mean", ascending=True).iloc[0].to_dict()
    manifest = {
        "scope": "pancancer overall only",
        "evaluation_dir": repo_relative(args.evaluation_dir, args.root),
        "full_fold_dir": repo_relative(args.full_fold_dir, args.root),
        "baseline_full_fold_dir": repo_relative(args.baseline_full_fold_dir, args.root),
        "output_dir": repo_relative(out_dir, args.root),
        "best_by_combined_RMSE": {
            "model": best_row.get("model_display"),
            "combined_RMSE_mean": best_row.get("combined_RMSE_mean"),
            "AAC_RMSE_mean": best_row.get("AAC_RMSE_mean"),
            "log2IC50_RMSE_mean": best_row.get("log2IC50_RMSE_mean"),
        },
    }
    (out_dir / "visualization_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def parse_args():
    root = package_root()
    parser = argparse.ArgumentParser(description="Create pancancer-only result figures.")
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument(
        "--evaluation-dir",
        type=Path,
        default=root / "results" / "pancancer_50e_evaluation",
    )
    parser.add_argument(
        "--full-fold-dir",
        type=Path,
        default=root / "results" / "pancancer_B3_B3DRKG_26Q1_CTRPv2_50e_full_folds",
    )
    parser.add_argument(
        "--baseline-full-fold-dir",
        type=Path,
        default=root / "results" / "pancancer_B0_B1_B2_26Q1_CTRPv2_50e_full_folds",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=root / "results" / "pancancer_visualization_figures",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--max-scatter-points-per-model", type=int, default=6000)
    parser.add_argument("--random-seed", type=int, default=2026)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = ensure_out(args.out_dir)
    summary_path = args.evaluation_dir / "overall_summary.csv"
    metrics_path = args.evaluation_dir / "prediction_metrics_by_fold.csv"
    predictions_path = args.evaluation_dir / "all_test_predictions.csv"

    if not summary_path.exists():
        raise FileNotFoundError(f"Missing pancancer overall summary: {summary_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing fold metrics: {metrics_path}")

    summary = normalize_model_labels(pd.read_csv(summary_path))
    if "evaluation_subset" in summary.columns:
        summary = summary[summary["evaluation_subset"].eq("overall")].copy()
    metrics = pd.read_csv(metrics_path)
    predictions = pd.read_csv(predictions_path) if predictions_path.exists() else None
    rmse_metrics = ["AAC_RMSE", "log2IC50_RMSE", "combined_RMSE"]
    sig_df = compute_reference_significance(metrics, rmse_metrics, reference="B3+DRKG", evaluation_subset="overall")
    sig_df.to_csv(out_dir / "pancancer_rmse_significance_vs_b3drkg.csv", index=False)

    plot_metric_bars(summary, out_dir, args.dpi, sig_df)
    plot_correlations(summary, out_dir, args.dpi)
    plot_fold_lines(metrics, out_dir, args.dpi)
    plot_b3_delta(metrics, out_dir, args.dpi)
    if predictions is not None:
        plot_predictions(predictions, out_dir, args.dpi, args.max_scatter_points_per_model, args.random_seed)
    else:
        print(f"Skipping figures 09-12: {predictions_path} not found.")
    if args.full_fold_dir.exists():
        plot_training_curves(args.full_fold_dir, out_dir, args.dpi)
        write_training_curve_availability(args.full_fold_dir, args.baseline_full_fold_dir, out_dir)
    else:
        print(f"Skipping figures 13-16: {args.full_fold_dir} not found.")
    write_manifest(out_dir, summary, args)

    print(f"Saved pancancer figures to: {out_dir}")


if __name__ == "__main__":
    main()
