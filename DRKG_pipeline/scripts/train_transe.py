#!/usr/bin/env python
"""Train TransE embeddings on an extracted DRKG subgraph."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class TransE(nn.Module):
    def __init__(self, n_entities: int, n_relations: int, dim: int, norm: int) -> None:
        super().__init__()
        self.entity = nn.Embedding(n_entities, dim)
        self.relation = nn.Embedding(n_relations, dim)
        self.norm = norm
        bound = 6.0 / math.sqrt(dim)
        nn.init.uniform_(self.entity.weight.data, -bound, bound)
        nn.init.uniform_(self.relation.weight.data, -bound, bound)
        self.normalize_entities()

    def forward(self, triples: torch.Tensor) -> torch.Tensor:
        head = self.entity(triples[:, 0])
        relation = self.relation(triples[:, 1])
        tail = self.entity(triples[:, 2])
        return torch.linalg.vector_norm(head + relation - tail, ord=self.norm, dim=1)

    @torch.no_grad()
    def normalize_entities(self) -> None:
        self.entity.weight.data = F.normalize(self.entity.weight.data, p=2, dim=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TransE on a DRKG subgraph.")
    parser.add_argument(
        "--subgraph-dir",
        default="outputs/pancancer_driver_subgraph_h1",
        help="Directory created by extract_driver_gene_subgraph.py.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Defaults to <subgraph-dir>/transe.",
    )
    parser.add_argument("--dim", type=int, default=200, help="Embedding dimension.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=8192, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--margin", type=float, default=1.0, help="Margin ranking loss.")
    parser.add_argument(
        "--negative-samples",
        type=int,
        default=1,
        help="Negative samples per positive triple.",
    )
    parser.add_argument("--norm", type=int, choices=[1, 2], default=2, help="TransE L1/L2 norm.")
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.05,
        help="Validation ratio. Set 0 to disable validation.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Training device.",
    )
    parser.add_argument(
        "--init-from-drkg",
        action="store_true",
        help="Initialize from full DRKG TransE embeddings using original DRKG ids.",
    )
    parser.add_argument(
        "--full-entity-embeddings",
        default="embed/DRKG_TransE_l2_entity.npy",
        help="Full DRKG entity embedding .npy for --init-from-drkg.",
    )
    parser.add_argument(
        "--full-relation-embeddings",
        default="embed/DRKG_TransE_l2_relation.npy",
        help="Full DRKG relation embedding .npy for --init-from-drkg.",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=10,
        help="Print and record validation loss every N epochs.",
    )
    parser.add_argument(
        "--write-vector-tsv",
        action="store_true",
        help="Also write a TSV with one dense vector column per entity.",
    )
    return parser.parse_args()


def read_nodes(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_relations(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_triples(path: Path) -> np.ndarray:
    triples: list[list[int]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            triples.append(
                [int(row["head_id"]), int(row["relation_id"]), int(row["tail_id"])]
            )
    if not triples:
        raise ValueError(f"No triples found in {path}")
    return np.asarray(triples, dtype=np.int64)


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(value)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_valid_split(
    triples: np.ndarray, valid_ratio: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    if valid_ratio <= 0:
        return triples, np.empty((0, 3), dtype=np.int64)
    if not 0 <= valid_ratio < 1:
        raise ValueError("--valid-ratio must be in [0, 1)")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(triples))
    valid_size = max(1, int(len(triples) * valid_ratio))
    valid_idx = order[:valid_size]
    train_idx = order[valid_size:]
    if len(train_idx) == 0:
        raise ValueError("Validation split consumed all triples")
    return triples[train_idx], triples[valid_idx]


def make_negative_batch(
    batch: torch.Tensor, n_entities: int, negative_samples: int
) -> torch.Tensor:
    repeated = batch.repeat_interleave(negative_samples, dim=0).clone()
    corrupt_head = torch.rand(repeated.shape[0], device=batch.device) < 0.5
    replacements = torch.randint(
        low=0,
        high=n_entities,
        size=(repeated.shape[0],),
        device=batch.device,
        dtype=torch.long,
    )
    repeated[corrupt_head, 0] = replacements[corrupt_head]
    repeated[~corrupt_head, 2] = replacements[~corrupt_head]
    return repeated


def margin_loss(
    model: TransE,
    batch: torch.Tensor,
    n_entities: int,
    negative_samples: int,
    margin: float,
) -> torch.Tensor:
    pos_score = model(batch)
    neg_batch = make_negative_batch(batch, n_entities, negative_samples)
    neg_score = model(neg_batch).view(batch.shape[0], negative_samples)
    return F.relu(margin + pos_score.unsqueeze(1) - neg_score).mean()


@torch.no_grad()
def evaluate_loss(
    model: TransE,
    triples: torch.Tensor,
    n_entities: int,
    negative_samples: int,
    margin: float,
    batch_size: int,
) -> float | None:
    if triples.numel() == 0:
        return None
    losses = []
    for start in range(0, len(triples), batch_size):
        batch = triples[start : start + batch_size]
        losses.append(
            float(margin_loss(model, batch, n_entities, negative_samples, margin).item())
        )
    return float(np.mean(losses))


@torch.no_grad()
def initialize_from_full_drkg(
    model: TransE,
    nodes: list[dict[str, str]],
    relations: list[dict[str, str]],
    entity_embeddings_path: Path,
    relation_embeddings_path: Path,
    dim: int,
) -> None:
    full_entity = np.load(entity_embeddings_path, mmap_mode="r")
    full_relation = np.load(relation_embeddings_path, mmap_mode="r")
    if full_entity.shape[1] != dim or full_relation.shape[1] != dim:
        raise ValueError(
            "--dim must match full DRKG embedding dimension when using --init-from-drkg"
        )

    entity_weight = model.entity.weight.detach().cpu().numpy()
    for row in nodes:
        original_id = row.get("drkg_entity_id", "")
        if original_id != "":
            entity_weight[int(row["node_id"])] = full_entity[int(original_id)]

    relation_weight = model.relation.weight.detach().cpu().numpy()
    for row in relations:
        original_id = row.get("drkg_relation_id", "")
        if original_id != "":
            relation_weight[int(row["relation_id"])] = full_relation[int(original_id)]

    model.entity.weight.data.copy_(torch.from_numpy(entity_weight))
    model.relation.weight.data.copy_(torch.from_numpy(relation_weight))
    model.normalize_entities()


def write_loss_log(path: Path, rows: list[dict[str, float | int | None]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["epoch", "train_loss", "valid_loss"], delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(rows)


def write_embedding_index(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = rows[0].keys() if rows else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def export_entity_type_subsets(
    out_dir: Path,
    nodes: list[dict[str, str]],
    embeddings: np.ndarray,
) -> dict[str, int]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in nodes:
        grouped[row["entity_type"]].append(row)

    counts = {}
    for entity_type, rows in grouped.items():
        local_ids = [int(row["node_id"]) for row in rows]
        subset = embeddings[local_ids]
        file_prefix = f"subset_{safe_name(entity_type)}"
        np.save(out_dir / f"{file_prefix}_embeddings.npy", subset)
        with (out_dir / f"{file_prefix}_nodes.tsv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
        counts[entity_type] = len(rows)
    return counts


def write_vector_tsv(
    path: Path,
    nodes: list[dict[str, str]],
    embeddings: np.ndarray,
) -> None:
    dim = embeddings.shape[1]
    fieldnames = ["node_id", "entity", "entity_type"] + [f"v{i}" for i in range(dim)]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(fieldnames)
        for row in nodes:
            node_id = int(row["node_id"])
            writer.writerow(
                [row["node_id"], row["entity"], row["entity_type"]]
                + [f"{value:.8g}" for value in embeddings[node_id]]
            )


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)

    subgraph_dir = Path(args.subgraph_dir)
    out_dir = Path(args.out_dir) if args.out_dir else subgraph_dir / "transe"
    out_dir.mkdir(parents=True, exist_ok=True)

    nodes = read_nodes(subgraph_dir / "nodes.tsv")
    relations = read_relations(subgraph_dir / "relations.tsv")
    triples = read_triples(subgraph_dir / "triples.tsv")

    n_entities = len(nodes)
    n_relations = len(relations)
    train_triples_np, valid_triples_np = train_valid_split(
        triples, args.valid_ratio, args.seed
    )

    device = choose_device(args.device)
    model = TransE(n_entities, n_relations, args.dim, args.norm).to(device)
    if args.init_from_drkg:
        initialize_from_full_drkg(
            model,
            nodes,
            relations,
            Path(args.full_entity_embeddings),
            Path(args.full_relation_embeddings),
            args.dim,
        )
        model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_triples = torch.from_numpy(train_triples_np).to(device)
    valid_triples = torch.from_numpy(valid_triples_np).to(device)
    loss_rows: list[dict[str, float | int | None]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(train_triples), device=device)
        batch_losses = []
        for start in range(0, len(train_triples), args.batch_size):
            idx = order[start : start + args.batch_size]
            batch = train_triples[idx]
            loss = margin_loss(
                model,
                batch,
                n_entities,
                args.negative_samples,
                args.margin,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.normalize_entities()
            batch_losses.append(float(loss.item()))

        train_loss = float(np.mean(batch_losses))
        should_eval = epoch == 1 or epoch == args.epochs or epoch % args.eval_every == 0
        valid_loss = (
            evaluate_loss(
                model,
                valid_triples,
                n_entities,
                args.negative_samples,
                args.margin,
                args.batch_size,
            )
            if should_eval
            else None
        )
        loss_rows.append(
            {"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss}
        )
        if should_eval:
            valid_text = "n/a" if valid_loss is None else f"{valid_loss:.6f}"
            print(
                f"epoch={epoch} train_loss={train_loss:.6f} valid_loss={valid_text}",
                flush=True,
            )

    model.eval()
    model.normalize_entities()
    entity_embeddings = model.entity.weight.detach().cpu().numpy().astype(np.float32)
    relation_embeddings = model.relation.weight.detach().cpu().numpy().astype(np.float32)

    np.save(out_dir / "entity_embeddings.npy", entity_embeddings)
    np.save(out_dir / "relation_embeddings.npy", relation_embeddings)
    write_embedding_index(out_dir / "node_embedding_index.tsv", nodes)
    write_embedding_index(out_dir / "relation_embedding_index.tsv", relations)
    write_loss_log(out_dir / "loss.tsv", loss_rows)
    subset_counts = export_entity_type_subsets(out_dir, nodes, entity_embeddings)
    if args.write_vector_tsv:
        write_vector_tsv(out_dir / "entity_embeddings.tsv", nodes, entity_embeddings)

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "subgraph_dir": str(subgraph_dir),
        "n_entities": n_entities,
        "n_relations": n_relations,
        "n_triples": int(len(triples)),
        "n_train_triples": int(len(train_triples_np)),
        "n_valid_triples": int(len(valid_triples_np)),
        "dim": args.dim,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "margin": args.margin,
        "negative_samples": args.negative_samples,
        "norm": args.norm,
        "seed": args.seed,
        "device": str(device),
        "init_from_drkg": args.init_from_drkg,
        "entity_type_subset_counts": subset_counts,
        "outputs": {
            "entity_embeddings": "entity_embeddings.npy",
            "relation_embeddings": "relation_embeddings.npy",
            "node_index": "node_embedding_index.tsv",
            "relation_index": "relation_embedding_index.tsv",
            "loss": "loss.tsv",
        },
    }
    with (out_dir / "training_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"Wrote embeddings to {out_dir}")


if __name__ == "__main__":
    main()
