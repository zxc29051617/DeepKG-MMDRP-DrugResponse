import os
from pathlib import Path

import pandas as pd


ROOT = Path(os.environ.get("MMDRP_ROOT", Path(__file__).resolve().parents[1]))
RAW_DIR = ROOT / "Data" / "raw" / "depmap_latest_drug_screen"
OUT_DIR = ROOT / "Data" / "DRP_Training_Data"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    expression_path = RAW_DIR / "DepMap_26Q1_Expression_custom.csv"
    expression = pd.read_csv(expression_path)
    expression = expression.rename(
        columns={
            "depmap_id": "DepMap_ID",
            "cell_line_display_name": "stripped_cell_line_name",
            "lineage_1": "primary_disease",
        }
    )

    metadata_cols = {
        "DepMap_ID",
        "stripped_cell_line_name",
        "primary_disease",
        "lineage_2",
        "lineage_3",
        "lineage_4",
        "lineage_6",
    }
    gene_cols = [col for col in expression.columns if col not in metadata_cols]
    expression_out = expression[
        [
            "DepMap_ID",
            "stripped_cell_line_name",
            "primary_disease",
            "lineage_2",
            "lineage_3",
            "lineage_4",
            "lineage_6",
        ]
        + gene_cols
    ].copy()

    hdf_out = OUT_DIR / "DepMap_26Q1_Expression.hdf"
    csv_out = OUT_DIR / "DepMap_26Q1_Expression.csv"
    expression_out.to_hdf(hdf_out, key="df", mode="w")
    expression_out.to_csv(csv_out, index=False)

    print(f"Saved {hdf_out}")
    print(f"Saved {csv_out}")
    print(f"Expression shape: {expression_out.shape}")


if __name__ == "__main__":
    main()
