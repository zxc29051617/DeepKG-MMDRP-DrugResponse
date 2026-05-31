import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import rdata
from rdkit import Chem, RDLogger


warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

ROOT = Path(os.environ.get("MMDRP_ROOT", Path(__file__).resolve().parents[1]))
RAW_DIR = ROOT / "Data" / "raw"
OUT_DIR = ROOT / "Data" / "DRP_Training_Data"
RDS_PATH = RAW_DIR / "CTRPv2.rds"


def normalize_cell_name(values: pd.Series) -> pd.Series:
    return (
        values.astype(str)
        .str.upper()
        .str.replace("-", "", regex=False)
        .str.replace(" ", "", regex=False)
    )


def valid_smiles(smiles) -> bool:
    if pd.isna(smiles):
        return False
    return Chem.MolFromSmiles(str(smiles)) is not None


def load_depmap26q1_line_info() -> pd.DataFrame:
    """Load cell-line lineage metadata from the same DepMap 26Q1 source as expression."""
    raw_expression_path = RAW_DIR / "depmap_latest_drug_screen" / "DepMap_26Q1_Expression_custom.csv"
    if raw_expression_path.exists():
        line_info = pd.read_csv(
            raw_expression_path,
            usecols=["cell_line_display_name", "lineage_1", "lineage_2", "lineage_3"],
        )
        line_info = line_info.rename(
            columns={
                "cell_line_display_name": "stripped_cell_line_name",
                "lineage_1": "primary_disease",
            }
        )
        return line_info

    processed_expression_path = OUT_DIR / "DepMap_26Q1_Expression.csv"
    if processed_expression_path.exists():
        return pd.read_csv(
            processed_expression_path,
            usecols=["stripped_cell_line_name", "primary_disease"],
        )

    raise FileNotFoundError(
        "Could not find DepMap 26Q1 lineage metadata. Expected "
        f"{raw_expression_path} or {processed_expression_path}."
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading CTRPv2 PharmacoSet RDS...")
    ctrp = rdata.conversion.convert(rdata.parser.parse_file(RDS_PATH))

    info = ctrp.sensitivity["info"].copy()
    profiles = ctrp.sensitivity["profiles"].copy()

    dose_response = info[["cellid", "drugid"]].join(
        profiles[["aac_recomputed", "ic50_recomputed"]],
        how="inner",
    )
    dose_response = dose_response.dropna(subset=["aac_recomputed", "ic50_recomputed"])
    dose_response = dose_response[
        np.isfinite(dose_response["ic50_recomputed"]) & (dose_response["ic50_recomputed"] > 0)
    ].copy()

    dose_response = (
        dose_response.groupby(["drugid", "cellid"], as_index=False)[
            ["aac_recomputed", "ic50_recomputed"]
        ]
        .median()
        .rename(
            columns={
                "drugid": "cpd_name",
                "cellid": "ccl_name",
                "aac_recomputed": "area_above_curve",
                "ic50_recomputed": "ic50",
            }
        )
    )
    dose_response["log2_ic50"] = np.log2(dose_response["ic50"].astype(float))

    drug = ctrp.drug.copy()
    drug_table = pd.DataFrame(
        {
            "canonical_name": drug.index.astype(str),
            "cpd_smiles": drug["cpd_smiles"].astype("string"),
        }
    )

    merged = dose_response.merge(
        drug_table,
        left_on="cpd_name",
        right_on="canonical_name",
        how="inner",
    ).drop(columns=["canonical_name"])

    line_info = load_depmap26q1_line_info()
    lineage_cols = [col for col in ["lineage_1", "lineage_2", "lineage_3"] if col in line_info.columns]
    line_info = line_info[["stripped_cell_line_name", "primary_disease"] + lineage_cols].copy()
    line_info["other_ccl_name"] = normalize_cell_name(line_info["stripped_cell_line_name"])
    line_info = line_info.drop_duplicates("other_ccl_name")

    before_cell_merge = len(merged)
    merged["other_ccl_name"] = normalize_cell_name(merged["ccl_name"])
    merged = merged.merge(
        line_info[["other_ccl_name", "primary_disease"] + lineage_cols],
        on="other_ccl_name",
        how="inner",
    ).drop(columns=["other_ccl_name"])

    merged.loc[
        merged["cpd_name"].eq("Tipifarnib"),
        "cpd_smiles",
    ] = "CN1C=NC=C1C(C2=CC=C(C=C2)Cl)(C3=CC4=C(C=C3)N(C(=O)C=C4C5=CC(=CC=C5)Cl)C)N"

    before_smiles_filter = len(merged)
    merged = merged[merged["cpd_smiles"].map(valid_smiles)].copy()
    output_cols = [
        "cpd_name",
        "ccl_name",
        "primary_disease",
        *lineage_cols,
        "area_above_curve",
        "ic50",
        "log2_ic50",
        "cpd_smiles",
    ]
    merged = merged[output_cols].reset_index(drop=True)

    for column in ["cpd_name", "ccl_name", "primary_disease", "cpd_smiles"]:
        merged[column] = merged[column].astype(object)

    hdf_out = OUT_DIR / "CTRP_AAC_IC50_SMILES.hdf"
    csv_out = OUT_DIR / "CTRP_AAC_IC50_SMILES.csv"
    summary_out = OUT_DIR / "CTRP_AAC_IC50_SMILES_summary.json"

    merged.to_hdf(hdf_out, key="df", mode="w")
    merged.to_csv(csv_out, index=False)

    summary = {
        "rds_path": str(RDS_PATH),
        "rows_before_cell_line_merge": int(before_cell_merge),
        "rows_after_cell_line_merge": int(before_smiles_filter),
        "rows_after_smiles_filter": int(len(merged)),
        "unique_cell_lines": int(merged["ccl_name"].nunique()),
        "unique_drugs": int(merged["cpd_name"].nunique()),
        "targets": ["area_above_curve", "log2_ic50"],
    }
    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved {hdf_out}")
    print(f"Saved {csv_out}")
    print(f"Saved {summary_out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
