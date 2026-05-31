#!/usr/bin/env python
"""Map a drug SMILES table to DRKG DrugBank compound embeddings."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests


DRUGBANK_PATTERN = re.compile(r"\bDB\d{5}\b", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add DrugBank IDs to a drug SMILES CSV via PubChem synonyms, then "
            "match those IDs to DRKG compound embeddings."
        )
    )
    parser.add_argument(
        "--drug-csv",
        required=True,
        help="CSV with cpd_name, PubChem_CID, and SMILES columns.",
    )
    parser.add_argument(
        "--embedding-dir",
        required=True,
        help="TransE output directory containing entity_embeddings.npy and node_embedding_index.tsv.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for mapping and merged MMDRP input files.",
    )
    parser.add_argument(
        "--cache",
        default=None,
        help="Optional PubChem synonyms cache JSON. Defaults to <out-dir>/pubchem_synonyms_cache.json.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of PubChem CIDs per synonyms API request.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.25,
        help="Seconds to sleep between PubChem API requests.",
    )
    return parser.parse_args()


def normalize_pubchem_cid(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    return str(int(float(text)))


def load_node_index(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return {row["entity"]: row for row in rows}


def load_cache(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_cache(path: Path, cache: dict[str, list[str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(cache, handle, indent=2, sort_keys=True)
        handle.write("\n")


def fetch_pubchem_synonyms(
    cids: list[str],
    cache: dict[str, list[str]],
    cache_path: Path,
    batch_size: int,
    sleep_seconds: float,
) -> None:
    missing = [cid for cid in cids if cid and cid not in cache]
    for start in range(0, len(missing), batch_size):
        batch = missing[start : start + batch_size]
        url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
            f"{','.join(batch)}/synonyms/JSON"
        )
        response = requests.get(url, timeout=60)
        if response.status_code != 200:
            for cid in batch:
                cache[cid] = []
            save_cache(cache_path, cache)
            continue

        data = response.json()
        for info in data.get("InformationList", {}).get("Information", []):
            cache[str(info["CID"])] = info.get("Synonym", [])
        for cid in batch:
            cache.setdefault(cid, [])
        save_cache(cache_path, cache)
        time.sleep(sleep_seconds)


def extract_drugbank_ids(synonyms: list[str]) -> list[str]:
    ids = set()
    for synonym in synonyms:
        ids.update(match.upper() for match in DRUGBANK_PATTERN.findall(str(synonym)))
    return sorted(ids)


def build_drugbank_mapping(
    drug_df: pd.DataFrame, cache: dict[str, list[str]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapping_rows = []
    annotated_rows = []
    for source_row_id, row in drug_df.iterrows():
        cid = row["PubChem_CID_norm"]
        drugbank_ids = extract_drugbank_ids(cache.get(cid, [])) if cid else []
        annotated = row.to_dict()
        annotated["source_row_id"] = int(source_row_id)
        annotated["DrugBank_IDs"] = ";".join(drugbank_ids)
        annotated["n_drugbank_ids"] = len(drugbank_ids)
        annotated_rows.append(annotated)

        for drugbank_id in drugbank_ids:
            mapping_rows.append(
                {
                    "source_row_id": int(source_row_id),
                    "cpd_name": row.get("cpd_name", ""),
                    "PubChem_CID": cid,
                    "SMILES": row.get("SMILES", ""),
                    "DrugBank_ID": drugbank_id,
                }
            )

    return pd.DataFrame(annotated_rows), pd.DataFrame(mapping_rows)


def merge_with_embeddings(
    drugbank_rows: pd.DataFrame,
    node_by_entity: dict[str, dict[str, str]],
    embeddings: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    index_rows = []
    feature_rows = []
    for _, row in drugbank_rows.iterrows():
        entity = f"Compound::{row['DrugBank_ID']}"
        node = node_by_entity.get(entity)
        if node is None:
            continue
        node_id = int(node["node_id"])
        feature_rows.append(embeddings[node_id])
        index_rows.append(
            {
                "feature_row_id": len(index_rows),
                "source_row_id": int(row["source_row_id"]),
                "cpd_name": row["cpd_name"],
                "PubChem_CID": row["PubChem_CID"],
                "SMILES": row["SMILES"],
                "DrugBank_ID": row["DrugBank_ID"],
                "drkg_entity": entity,
                "drkg_node_id": node_id,
                "drkg_entity_id": node.get("drkg_entity_id", ""),
            }
        )

    if feature_rows:
        features = np.asarray(feature_rows, dtype=np.float32)
    else:
        features = np.empty((0, embeddings.shape[1]), dtype=np.float32)
    return pd.DataFrame(index_rows), features


def write_merged_csv(index_df: pd.DataFrame, features: np.ndarray, path: Path) -> None:
    if index_df.empty:
        index_df.to_csv(path, index=False)
        return
    feature_df = pd.DataFrame(
        features, columns=[f"transe_{idx}" for idx in range(features.shape[1])]
    )
    pd.concat([index_df.reset_index(drop=True), feature_df], axis=1).to_csv(
        path, index=False
    )


def main() -> None:
    args = parse_args()
    drug_csv = Path(args.drug_csv)
    embedding_dir = Path(args.embedding_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.cache) if args.cache else out_dir / "pubchem_synonyms_cache.json"

    drug_df = pd.read_csv(drug_csv)
    required_columns = {"cpd_name", "PubChem_CID", "SMILES"}
    missing_columns = sorted(required_columns.difference(drug_df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    drug_df["PubChem_CID_norm"] = drug_df["PubChem_CID"].apply(normalize_pubchem_cid)
    cids = sorted({cid for cid in drug_df["PubChem_CID_norm"] if cid})

    cache = load_cache(cache_path)
    fetch_pubchem_synonyms(cids, cache, cache_path, args.batch_size, args.sleep)

    annotated_df, drugbank_rows = build_drugbank_mapping(drug_df, cache)
    annotated_path = out_dir / "drugbank_mapping_annotated.csv"
    drugbank_rows_path = out_dir / "drugbank_mapping_long.csv"
    annotated_df.to_csv(annotated_path, index=False)
    drugbank_rows.to_csv(drugbank_rows_path, index=False)

    node_by_entity = load_node_index(embedding_dir / "node_embedding_index.tsv")
    embeddings = np.load(embedding_dir / "entity_embeddings.npy")
    index_df, features = merge_with_embeddings(drugbank_rows, node_by_entity, embeddings)

    features_path = out_dir / "mmdrp_drkg_transe_features.npy"
    index_path = out_dir / "mmdrp_drkg_transe_index.csv"
    merged_csv_path = out_dir / "mmdrp_drkg_transe_merged.csv"
    np.save(features_path, features)
    index_df.to_csv(index_path, index=False)
    write_merged_csv(index_df, features, merged_csv_path)

    unmatched_with_drugbank = drugbank_rows[
        ~drugbank_rows["DrugBank_ID"].isin(index_df.get("DrugBank_ID", pd.Series(dtype=str)))
    ]
    unmatched_path = out_dir / "drugbank_ids_not_in_drkg_embedding.csv"
    unmatched_with_drugbank.to_csv(unmatched_path, index=False)

    summary = {
        "drug_csv": str(drug_csv),
        "embedding_dir": str(embedding_dir),
        "drug_rows": int(len(drug_df)),
        "drug_rows_with_pubchem_cid": int((drug_df["PubChem_CID_norm"] != "").sum()),
        "drug_rows_with_smiles": int(drug_df["SMILES"].notna().sum()),
        "unique_pubchem_cids": int(len(cids)),
        "drug_rows_with_drugbank_id": int((annotated_df["n_drugbank_ids"] > 0).sum()),
        "unique_drugbank_ids_from_pubchem": int(drugbank_rows["DrugBank_ID"].nunique())
        if not drugbank_rows.empty
        else 0,
        "drugbank_mapping_rows": int(len(drugbank_rows)),
        "matched_drug_rows": int(index_df["source_row_id"].nunique())
        if not index_df.empty
        else 0,
        "matched_unique_drugbank_ids": int(index_df["DrugBank_ID"].nunique())
        if not index_df.empty
        else 0,
        "matched_unique_drkg_compounds": int(index_df["drkg_entity"].nunique())
        if not index_df.empty
        else 0,
        "matched_feature_rows": int(len(index_df)),
        "features_shape": list(features.shape),
        "unmatched_drugbank_mapping_rows": int(len(unmatched_with_drugbank)),
        "outputs": {
            "drugbank_mapping_annotated": str(annotated_path),
            "drugbank_mapping_long": str(drugbank_rows_path),
            "mmdrp_feature_matrix": str(features_path),
            "mmdrp_feature_index": str(index_path),
            "mmdrp_merged_csv": str(merged_csv_path),
            "unmatched_drugbank_ids": str(unmatched_path),
            "pubchem_synonyms_cache": str(cache_path),
        },
    }
    summary_path = out_dir / "mapping_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
