#!/usr/bin/env python
"""Inspect trained DRKG subgraph embeddings."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect TransE embedding results.")
    parser.add_argument(
        "--embedding-dir",
        default="outputs/pancancer_driver_subgraph_h1/transe_200d_50e",
        help="Directory containing entity_embeddings.npy and node_embedding_index.tsv.",
    )
    parser.add_argument(
        "--seed-file",
        default="seeds/pancancer_driver_genes.tsv",
        help="Seed TSV used by the extraction step.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Nearest neighbors to write for each seed.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Defaults to <embedding-dir>/inspection.",
    )
    return parser.parse_args()


def read_nodes(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_seeds(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = []
        for row in reader:
            entity = row.get("drkg_entity") or row.get("seed") or row.get("entity")
            if not entity:
                raise ValueError(
                    "Seed file must include one of these columns: drkg_entity, seed, entity"
                )
            row["seed"] = entity
            rows.append(row)
    return rows


def normalize(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


def write_nearest_neighbors(
    path: Path,
    embeddings: np.ndarray,
    nodes: list[dict[str, str]],
    seeds: list[dict[str, str]],
    top_k: int,
) -> dict[str, list[dict[str, str | float | int]]]:
    node_by_entity = {row["entity"]: row for row in nodes}
    normalized = normalize(embeddings)
    output: dict[str, list[dict[str, str | float | int]]] = {}

    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "seed_input_symbol",
            "seed_official_symbol",
            "seed_entity",
            "rank",
            "neighbor_node_id",
            "neighbor_entity",
            "neighbor_entity_type",
            "cosine_similarity",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

        for seed in seeds:
            seed_entity = seed["seed"]
            if seed_entity not in node_by_entity:
                output[seed_entity] = []
                continue
            seed_id = int(node_by_entity[seed_entity]["node_id"])
            scores = normalized @ normalized[seed_id]
            scores[seed_id] = -np.inf
            top_indices = np.argsort(-scores)[:top_k]

            rows = []
            for rank, node_id in enumerate(top_indices, start=1):
                node = nodes[int(node_id)]
                row = {
                    "seed_input_symbol": seed.get("input_symbol", ""),
                    "seed_official_symbol": seed.get("official_symbol", ""),
                    "seed_entity": seed_entity,
                    "rank": rank,
                    "neighbor_node_id": node["node_id"],
                    "neighbor_entity": node["entity"],
                    "neighbor_entity_type": node["entity_type"],
                    "cosine_similarity": float(scores[node_id]),
                }
                rows.append(row)
                writer.writerow(row)
            output[seed_entity] = rows

    return output


def write_seed_similarity(
    path: Path,
    embeddings: np.ndarray,
    nodes: list[dict[str, str]],
    seeds: list[dict[str, str]],
) -> list[dict[str, str | float]]:
    node_by_entity = {row["entity"]: row for row in nodes}
    normalized = normalize(embeddings)
    rows: list[dict[str, str | float]] = []

    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "seed_a_input_symbol",
            "seed_a_official_symbol",
            "seed_a_entity",
            "seed_b_input_symbol",
            "seed_b_official_symbol",
            "seed_b_entity",
            "cosine_similarity",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

        for left in seeds:
            for right in seeds:
                left_entity = left["seed"]
                right_entity = right["seed"]
                if left_entity not in node_by_entity or right_entity not in node_by_entity:
                    continue
                left_id = int(node_by_entity[left_entity]["node_id"])
                right_id = int(node_by_entity[right_entity]["node_id"])
                score = float(normalized[left_id] @ normalized[right_id])
                row = {
                    "seed_a_input_symbol": left.get("input_symbol", ""),
                    "seed_a_official_symbol": left.get("official_symbol", ""),
                    "seed_a_entity": left_entity,
                    "seed_b_input_symbol": right.get("input_symbol", ""),
                    "seed_b_official_symbol": right.get("official_symbol", ""),
                    "seed_b_entity": right_entity,
                    "cosine_similarity": score,
                }
                rows.append(row)
                writer.writerow(row)
    return rows


def main() -> None:
    args = parse_args()
    embedding_dir = Path(args.embedding_dir)
    out_dir = Path(args.out_dir) if args.out_dir else embedding_dir / "inspection"
    out_dir.mkdir(parents=True, exist_ok=True)

    nodes = read_nodes(embedding_dir / "node_embedding_index.tsv")
    seeds = read_seeds(Path(args.seed_file))
    embeddings = np.load(embedding_dir / "entity_embeddings.npy")

    neighbors = write_nearest_neighbors(
        out_dir / "seed_nearest_neighbors.tsv",
        embeddings,
        nodes,
        seeds,
        args.top_k,
    )
    seed_similarity = write_seed_similarity(
        out_dir / "seed_cosine_similarity.tsv",
        embeddings,
        nodes,
        seeds,
    )

    summary = {
        "embedding_dir": str(embedding_dir),
        "n_nodes": len(nodes),
        "embedding_shape": list(embeddings.shape),
        "top_k": args.top_k,
        "seed_count": len(seeds),
        "seeds_missing_from_embedding": [
            seed["seed"]
            for seed in seeds
            if seed["seed"] not in {node["entity"] for node in nodes}
        ],
        "outputs": {
            "seed_nearest_neighbors": "seed_nearest_neighbors.tsv",
            "seed_cosine_similarity": "seed_cosine_similarity.tsv",
        },
    }
    with (out_dir / "inspection_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"Wrote nearest neighbors for {len(neighbors)} seeds")
    print(f"Wrote {len(seed_similarity)} seed-pair cosine rows")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
