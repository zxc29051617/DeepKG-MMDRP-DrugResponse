#!/usr/bin/env python
"""Extract a driver-gene-focused subgraph from DRKG."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_SEED_ROWS = [
    {
        "input_symbol": "TP53",
        "official_symbol": "TP53",
        "ncbi_gene_id": "7157",
        "seed": "Gene::7157",
    },
    {
        "input_symbol": "KRAS",
        "official_symbol": "KRAS",
        "ncbi_gene_id": "3845",
        "seed": "Gene::3845",
    },
    {
        "input_symbol": "PIK3CA",
        "official_symbol": "PIK3CA",
        "ncbi_gene_id": "5290",
        "seed": "Gene::5290",
    },
    {
        "input_symbol": "PTEN",
        "official_symbol": "PTEN",
        "ncbi_gene_id": "5728",
        "seed": "Gene::5728",
    },
    {
        "input_symbol": "APC",
        "official_symbol": "APC",
        "ncbi_gene_id": "324",
        "seed": "Gene::324",
    },
    {
        "input_symbol": "BRAF",
        "official_symbol": "BRAF",
        "ncbi_gene_id": "673",
        "seed": "Gene::673",
    },
    {
        "input_symbol": "EGFR",
        "official_symbol": "EGFR",
        "ncbi_gene_id": "1956",
        "seed": "Gene::1956",
    },
    {
        "input_symbol": "ERBB2",
        "official_symbol": "ERBB2",
        "ncbi_gene_id": "2064",
        "seed": "Gene::2064",
    },
    {
        "input_symbol": "MYC",
        "official_symbol": "MYC",
        "ncbi_gene_id": "4609",
        "seed": "Gene::4609",
    },
    {
        "input_symbol": "CDKN2A",
        "official_symbol": "CDKN2A",
        "ncbi_gene_id": "1029",
        "seed": "Gene::1029",
    },
    {
        "input_symbol": "RB1",
        "official_symbol": "RB1",
        "ncbi_gene_id": "5925",
        "seed": "Gene::5925",
    },
    {
        "input_symbol": "NRAS",
        "official_symbol": "NRAS",
        "ncbi_gene_id": "4893",
        "seed": "Gene::4893",
    },
    {
        "input_symbol": "HRAS",
        "official_symbol": "HRAS",
        "ncbi_gene_id": "3265",
        "seed": "Gene::3265",
    },
    {
        "input_symbol": "SMAD4",
        "official_symbol": "SMAD4",
        "ncbi_gene_id": "4089",
        "seed": "Gene::4089",
    },
    {
        "input_symbol": "CTNNB1",
        "official_symbol": "CTNNB1",
        "ncbi_gene_id": "1499",
        "seed": "Gene::1499",
    },
    {
        "input_symbol": "BRCA1",
        "official_symbol": "BRCA1",
        "ncbi_gene_id": "672",
        "seed": "Gene::672",
    },
    {
        "input_symbol": "BRCA2",
        "official_symbol": "BRCA2",
        "ncbi_gene_id": "675",
        "seed": "Gene::675",
    },
    {
        "input_symbol": "ARID1A",
        "official_symbol": "ARID1A",
        "ncbi_gene_id": "8289",
        "seed": "Gene::8289",
    },
    {
        "input_symbol": "IDH1",
        "official_symbol": "IDH1",
        "ncbi_gene_id": "3417",
        "seed": "Gene::3417",
    },
    {
        "input_symbol": "IDH2",
        "official_symbol": "IDH2",
        "ncbi_gene_id": "3418",
        "seed": "Gene::3418",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a driver-gene subgraph from DRKG triples."
    )
    parser.add_argument("--drkg", default="drkg.tsv", help="Path to drkg.tsv.")
    parser.add_argument(
        "--entities",
        default="embed/entities.tsv",
        help="Path to DRKG entity-to-original-id mapping.",
    )
    parser.add_argument(
        "--relations",
        default="embed/relations.tsv",
        help="Path to DRKG relation-to-original-id mapping.",
    )
    parser.add_argument(
        "--entity-src",
        default="entity2src.tsv",
        help="Path to entity source table. Used only to write a subset table.",
    )
    parser.add_argument(
        "--out",
        default="outputs/pancancer_driver_subgraph_h1",
        help="Output directory.",
    )
    parser.add_argument(
        "--seed",
        action="append",
        dest="seeds",
        help="Seed entity. Can be passed more than once. Defaults to driver gene seeds.",
    )
    parser.add_argument(
        "--seed-file",
        default=None,
        help=(
            "Optional TSV seed file. Expected columns include drkg_entity or seed. "
            "Other columns are preserved in seeds.tsv."
        ),
    )
    parser.add_argument(
        "--hops",
        type=int,
        default=1,
        help="Number of graph-expansion hops from the seeds.",
    )
    parser.add_argument(
        "--include-induced",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After hop expansion, keep every DRKG edge whose endpoints are both in the node set.",
    )
    return parser.parse_args()


def read_name_to_id(path: Path) -> dict[str, int]:
    mapping: dict[str, int] = {}
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) >= 2:
                mapping[row[0]] = int(row[1])
    return mapping


def read_seed_rows(path: Path | None, cli_seeds: list[str] | None) -> list[dict[str, str]]:
    if path is not None:
        rows: list[dict[str, str]] = []
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if reader.fieldnames is None:
                raise ValueError(f"Seed file has no header: {path}")
            for row in reader:
                seed = row.get("drkg_entity") or row.get("seed") or row.get("entity")
                if not seed:
                    raise ValueError(
                        "Seed file must include one of these columns: drkg_entity, seed, entity"
                    )
                normalized = {key: value for key, value in row.items() if key is not None}
                normalized["seed"] = seed
                rows.append(normalized)
        if not rows:
            raise ValueError(f"Seed file is empty: {path}")
        return rows

    if cli_seeds:
        return [{"seed": seed} for seed in cli_seeds]

    return [dict(row) for row in DEFAULT_SEED_ROWS]


def iter_triples(path: Path):
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) != 3:
                continue
            yield row[0], row[1], row[2]


def entity_type(entity: str) -> str:
    return entity.split("::", 1)[0]


def relation_source(relation: str) -> str:
    return relation.split("::", 1)[0]


def sort_by_original_id(values: set[str], original_ids: dict[str, int]) -> list[str]:
    missing_floor = 10**18
    return sorted(values, key=lambda value: (original_ids.get(value, missing_floor), value))


def write_tsv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def write_entity_source_subset(path: Path, entity_src_path: Path, nodes: set[str]) -> int:
    if not entity_src_path.exists():
        return 0

    count = 0
    with entity_src_path.open("r", encoding="utf-8", errors="replace", newline="") as source:
        with path.open("w", encoding="utf-8", newline="") as target:
            for line in source:
                entity = line.split("\t", 1)[0].rstrip("\n")
                if entity in nodes:
                    target.write(line)
                    count += 1
    return count


def main() -> None:
    args = parse_args()
    if args.hops < 0:
        raise ValueError("--hops must be >= 0")

    drkg_path = Path(args.drkg)
    entities_path = Path(args.entities)
    relations_path = Path(args.relations)
    entity_src_path = Path(args.entity_src)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    entity_original_ids = read_name_to_id(entities_path)
    relation_original_ids = read_name_to_id(relations_path)

    seed_rows = read_seed_rows(Path(args.seed_file) if args.seed_file else None, args.seeds)
    seeds = list(dict.fromkeys(row["seed"] for row in seed_rows))
    present_seeds = [seed for seed in seeds if seed in entity_original_ids]
    missing_seeds = [seed for seed in seeds if seed not in entity_original_ids]
    if not present_seeds:
        raise ValueError(f"No seed entities were found in {entities_path}")

    nodes = set(present_seeds)
    frontier = set(present_seeds)
    traversal_edges: set[tuple[str, str, str]] = set()
    hop_stats = []

    for depth in range(1, args.hops + 1):
        next_frontier: set[str] = set()
        edges_this_hop = 0
        for head, relation, tail in iter_triples(drkg_path):
            if head in frontier or tail in frontier:
                traversal_edges.add((head, relation, tail))
                edges_this_hop += 1
                if head not in nodes:
                    next_frontier.add(head)
                if tail not in nodes:
                    next_frontier.add(tail)

        nodes.update(next_frontier)
        hop_stats.append(
            {
                "hop": depth,
                "new_nodes": len(next_frontier),
                "frontier_edges_seen": edges_this_hop,
                "total_nodes": len(nodes),
            }
        )
        frontier = next_frontier
        if not frontier:
            break

    if args.include_induced:
        edges = {
            (head, relation, tail)
            for head, relation, tail in iter_triples(drkg_path)
            if head in nodes and tail in nodes
        }
    else:
        edges = traversal_edges

    node_list = sort_by_original_id(nodes, entity_original_ids)
    relation_values = {relation for _, relation, _ in edges}
    relation_list = sort_by_original_id(relation_values, relation_original_ids)

    node_to_local_id = {entity: idx for idx, entity in enumerate(node_list)}
    relation_to_local_id = {relation: idx for idx, relation in enumerate(relation_list)}

    edge_list = sorted(
        edges,
        key=lambda edge: (
            node_to_local_id[edge[0]],
            relation_to_local_id[edge[1]],
            node_to_local_id[edge[2]],
        ),
    )

    write_tsv(
        out_dir / "nodes.tsv",
        ["node_id", "entity", "entity_type", "drkg_entity_id"],
        [
            [
                node_to_local_id[entity],
                entity,
                entity_type(entity),
                entity_original_ids.get(entity, ""),
            ]
            for entity in node_list
        ],
    )
    write_tsv(
        out_dir / "relations.tsv",
        ["relation_id", "relation", "relation_source", "drkg_relation_id"],
        [
            [
                relation_to_local_id[relation],
                relation,
                relation_source(relation),
                relation_original_ids.get(relation, ""),
            ]
            for relation in relation_list
        ],
    )
    write_tsv(
        out_dir / "triples.tsv",
        ["head_id", "relation_id", "tail_id"],
        [
            [
                node_to_local_id[head],
                relation_to_local_id[relation],
                node_to_local_id[tail],
            ]
            for head, relation, tail in edge_list
        ],
    )
    write_tsv(
        out_dir / "triples_names.tsv",
        ["head", "relation", "tail"],
        [[head, relation, tail] for head, relation, tail in edge_list],
    )
    write_tsv(
        out_dir / "seeds.tsv",
        [
            "input_symbol",
            "official_symbol",
            "ncbi_gene_id",
            "seed",
            "present",
            "drkg_entity_id",
        ],
        [
            [
                row.get("input_symbol", ""),
                row.get("official_symbol", ""),
                row.get("ncbi_gene_id", ""),
                row["seed"],
                row["seed"] in entity_original_ids,
                entity_original_ids.get(row["seed"], ""),
            ]
            for row in seed_rows
        ],
    )

    entity_source_rows = write_entity_source_subset(
        out_dir / "entity_sources.tsv", entity_src_path, nodes
    )

    stats = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "drkg_path": str(drkg_path),
        "hops": args.hops,
        "include_induced": args.include_induced,
        "seeds": seeds,
        "seed_rows": seed_rows,
        "present_seeds": present_seeds,
        "missing_seeds": missing_seeds,
        "nodes": len(node_list),
        "relations": len(relation_list),
        "edges": len(edge_list),
        "entity_source_rows": entity_source_rows,
        "hop_stats": hop_stats,
        "entity_type_counts": dict(Counter(entity_type(entity) for entity in node_list)),
        "relation_source_counts": dict(
            Counter(relation_source(relation) for relation in relation_list)
        ),
    }
    with (out_dir / "stats.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"Wrote {len(node_list)} nodes, {len(relation_list)} relations, {len(edge_list)} edges")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
