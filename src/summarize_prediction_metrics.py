import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


NSCLC_LABEL = "Non-Small Cell Lung Cancer"
MODEL_LABELS = {
    "B0_drug_median": "B0",
    "B1_elastic_net": "B1",
    "B2_mmdrp_base": "B2",
    "cv_base": "B3",
    "cv_plus_drkg": "B3+DRKG",
}
METRIC_COLS = [
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


def parse_dirs(value: str) -> list[Path]:
    return [Path(item.strip()) for item in value.split(",") if item.strip()]


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> dict[str, float]:
    err = y_pred - y_true
    pearson = float(np.corrcoef(y_true, y_pred)[0, 1]) if np.std(y_true) > 0 and np.std(y_pred) > 0 else np.nan
    spearman = float(pd.Series(y_true).corr(pd.Series(y_pred), method="spearman"))
    return {
        f"{prefix}_MAE": float(np.mean(np.abs(err))),
        f"{prefix}_RMSE": float(np.sqrt(np.mean(err**2))),
        f"{prefix}_Pearson": pearson,
        f"{prefix}_Spearman": spearman,
    }


def compute_metrics(frame: pd.DataFrame) -> dict[str, float]:
    metrics = {}
    metrics.update(
        metric_dict(
            frame["area_above_curve"].to_numpy(dtype=float),
            frame["pred_AAC"].to_numpy(dtype=float),
            "AAC",
        )
    )
    metrics.update(
        metric_dict(
            frame["log2_ic50"].to_numpy(dtype=float),
            frame["pred_log2IC50"].to_numpy(dtype=float),
            "log2IC50",
        )
    )
    metrics["combined_RMSE"] = 0.7 * metrics["AAC_RMSE"] + 0.3 * metrics["log2IC50_RMSE"]
    return metrics


def read_predictions(prediction_dirs: list[Path]) -> pd.DataFrame:
    files = []
    for directory in prediction_dirs:
        files.extend(sorted(directory.rglob("*_test_predictions.csv")))
    if not files:
        raise FileNotFoundError("No *_test_predictions.csv files found.")

    frames = []
    for path in files:
        frame = pd.read_csv(path)
        frame["prediction_file"] = str(path)
        frames.append(frame)
    predictions = pd.concat(frames, ignore_index=True)

    if "is_nsclc" not in predictions.columns:
        predictions["is_nsclc"] = predictions["lineage_2"].eq(NSCLC_LABEL)
    else:
        predictions["is_nsclc"] = predictions["is_nsclc"].astype(str).str.lower().isin(["true", "1", "yes"])

    predictions["model_display"] = predictions["model"].map(MODEL_LABELS).fillna(predictions["model"])
    key_cols = ["model", "seed", "fold", "row_id"]
    if all(col in predictions.columns for col in key_cols):
        predictions = predictions.drop_duplicates(key_cols, keep="last").reset_index(drop=True)
    return predictions


def per_fold_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["model", "model_display", "seed", "fold", "alpha", "split_type", "use_lds"]
    for key, group in predictions.groupby(group_cols, dropna=False):
        base = dict(zip(group_cols, key))
        for subset_name, subset in [
            ("overall", group),
            ("NSCLC-only", group[group["is_nsclc"]]),
        ]:
            if subset.empty:
                continue
            rows.append(
                {
                    **base,
                    "evaluation_subset": subset_name,
                    "rows": int(len(subset)),
                    "cell_lines": int(subset["ccl_name"].nunique()),
                    "drugs": int(subset["cpd_name"].nunique()),
                    **compute_metrics(subset),
                }
            )
    return pd.DataFrame(rows)


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["evaluation_subset", "model", "model_display", "alpha", "split_type", "use_lds"]
    for key, group in metrics.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, key))
        row["runs"] = int(len(group))
        row["rows_mean"] = float(group["rows"].mean())
        row["cell_lines_mean"] = float(group["cell_lines"].mean())
        row["drugs_mean"] = float(group["drugs"].mean())
        for col in METRIC_COLS:
            row[f"{col}_mean"] = float(group[col].mean())
            row[f"{col}_std"] = float(group[col].std(ddof=1)) if len(group) > 1 else 0.0
        rows.append(row)
    summary = pd.DataFrame(rows)
    model_order = {"B0": 0, "B1": 1, "B2": 2, "B3": 3, "B3+DRKG": 4}
    summary["model_order"] = summary["model_display"].map(model_order).fillna(99)
    return summary.sort_values(["evaluation_subset", "model_order"]).drop(columns=["model_order"]).reset_index(drop=True)


def paired_stats(metrics: pd.DataFrame) -> pd.DataFrame:
    from scipy import stats

    rows = []
    index_cols = ["evaluation_subset", "seed", "fold", "alpha", "split_type", "use_lds"]
    pivot = metrics.set_index(index_cols + ["model_display"]).sort_index()
    comparisons = [("B3+DRKG", "B3"), ("B3+DRKG", "B0"), ("B3+DRKG", "B2"), ("B3", "B2")]
    for target, baseline in comparisons:
        for subset_name in sorted(metrics["evaluation_subset"].unique()):
            try:
                target_df = pivot.xs((subset_name, target), level=("evaluation_subset", "model_display"))
                base_df = pivot.xs((subset_name, baseline), level=("evaluation_subset", "model_display"))
            except KeyError:
                continue
            common = target_df.index.intersection(base_df.index)
            if len(common) == 0:
                continue
            target_df = target_df.loc[common]
            base_df = base_df.loc[common]
            for metric in METRIC_COLS:
                diff = target_df[metric] - base_df[metric]
                lower = metric.endswith("MAE") or metric.endswith("RMSE")
                rows.append(
                    {
                        "evaluation_subset": subset_name,
                        "comparison": f"{target}_minus_{baseline}",
                        "metric": metric,
                        "target_mean": float(target_df[metric].mean()),
                        "baseline_mean": float(base_df[metric].mean()),
                        "diff_mean": float(diff.mean()),
                        "improved_runs": int((diff < 0).sum() if lower else (diff > 0).sum()),
                        "total_runs": int(len(diff)),
                        "paired_ttest_p": float(stats.ttest_rel(target_df[metric], base_df[metric]).pvalue),
                    }
                )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-dirs", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--save-all-predictions", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions = read_predictions(parse_dirs(args.prediction_dirs))
    fold_metrics = per_fold_metrics(predictions)
    summary = summarize(fold_metrics)
    paired = paired_stats(fold_metrics)

    fold_metrics.to_csv(out_dir / "prediction_metrics_by_fold.csv", index=False)
    summary.to_csv(out_dir / "prediction_metrics_summary.csv", index=False)
    paired.to_csv(out_dir / "paired_prediction_stats.csv", index=False)
    summary[summary["evaluation_subset"].eq("overall")].to_csv(out_dir / "overall_summary.csv", index=False)
    summary[summary["evaluation_subset"].eq("NSCLC-only")].to_csv(out_dir / "nsclc_only_summary.csv", index=False)
    if args.save_all_predictions:
        predictions.to_csv(out_dir / "all_test_predictions.csv", index=False)

    manifest = {
        "prediction_rows": int(len(predictions)),
        "prediction_files": int(predictions["prediction_file"].nunique()),
        "overall_rows": int((~predictions["is_nsclc"].isna()).sum()),
        "nsclc_rows": int(predictions["is_nsclc"].sum()),
        "nsclc_cell_lines": int(predictions.loc[predictions["is_nsclc"], "ccl_name"].nunique()),
        "nsclc_drugs": int(predictions.loc[predictions["is_nsclc"], "cpd_name"].nunique()),
        "outputs": [
            "prediction_metrics_by_fold.csv",
            "prediction_metrics_summary.csv",
            "overall_summary.csv",
            "nsclc_only_summary.csv",
            "paired_prediction_stats.csv",
        ],
    }
    (out_dir / "summary_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
