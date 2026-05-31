# Lung Cancer DRKG Subgraph Pipeline

This project contains the DRKG data files and pretrained full-graph embeddings. The
pipeline added here prepares a lung-cancer-focused DRKG subgraph, then trains TransE
embeddings on that subgraph for downstream MMDRP use.

## Seeds

The default seed entities are lung-cancer driver genes:

| Input symbol | Official symbol | NCBI Gene ID | DRKG entity |
| --- | --- | --- | --- |
| EGFR | EGFR | 1956 | `Gene::1956` |
| KRAS | KRAS | 3845 | `Gene::3845` |
| ALK | ALK | 238 | `Gene::238` |
| ROS1 | ROS1 | 6098 | `Gene::6098` |
| BRAF | BRAF | 673 | `Gene::673` |
| MET | MET | 4233 | `Gene::4233` |
| RET | RET | 5979 | `Gene::5979` |
| HER2 | ERBB2 | 2064 | `Gene::2064` |

HER2 is the common protein/clinical name; `ERBB2` is the official gene symbol used
for NCBI Gene ID 2064. The seed table is stored at
`seeds/lung_cancer_driver_genes.tsv`.

## 1. Extract the subgraph

```powershell
python .\scripts\extract_lung_cancer_subgraph.py `
  --seed-file .\seeds\lung_cancer_driver_genes.tsv `
  --hops 1 `
  --out .\outputs\driver_gene_subgraph_h1
```

Default behavior expands one hop from the seed diseases and then writes the induced
subgraph over the discovered node set.

Main outputs:

- `nodes.tsv` - local node ids, original DRKG entity names, entity types, original ids
- `relations.tsv` - local relation ids and original DRKG relation names
- `triples.tsv` - integer triples for model training
- `triples_names.tsv` - readable triples with DRKG names
- `entity_sources.tsv` - source rows for the retained entities
- `stats.json` - extraction metadata and counts

The current 1-hop driver-gene subgraph contains 5,040 nodes, 96 relations, and
670,797 triples.

To build a larger graph:

```powershell
python .\scripts\extract_lung_cancer_subgraph.py `
  --seed-file .\seeds\lung_cancer_driver_genes.tsv `
  --hops 2 `
  --out .\outputs\driver_gene_subgraph_h2
```

Use `--no-include-induced` if you only want traversal edges found during hop
expansion.

The current 2-hop induced driver-gene subgraph contains 66,262 nodes, 107
relations, and 5,787,861 triples. This is close to the full DRKG edge count, so
it is useful as a broad candidate graph but is much less lung-cancer-specific
than the 1-hop graph.

## 2. Train TransE

Train from scratch:

```powershell
python .\scripts\train_transe.py `
  --subgraph-dir .\outputs\driver_gene_subgraph_h1 `
  --epochs 200 `
  --dim 400 `
  --batch-size 4096
```

To match a 200-dimensional, 50-epoch baseline:

```powershell
python .\scripts\train_transe.py `
  --subgraph-dir .\outputs\driver_gene_subgraph_h1 `
  --out-dir .\outputs\driver_gene_subgraph_h1\transe_200d_50e `
  --epochs 50 `
  --dim 200 `
  --batch-size 8192 `
  --eval-every 10
```

This aligned baseline currently has final train loss `0.2974` and final
validation loss `0.2975`.

Initialize from the full DRKG TransE embeddings and fine-tune on the subgraph:

```powershell
python .\scripts\train_transe.py `
  --subgraph-dir .\outputs\driver_gene_subgraph_h1 `
  --epochs 200 `
  --dim 400 `
  --batch-size 4096 `
  --init-from-drkg
```

For the large 2-hop graph, a short baseline run is:

```powershell
python .\scripts\train_transe.py `
  --subgraph-dir .\outputs\driver_gene_subgraph_h2 `
  --out-dir .\outputs\driver_gene_subgraph_h2\transe_20e `
  --epochs 20 `
  --dim 400 `
  --batch-size 32768 `
  --valid-ratio 0.01 `
  --eval-every 5 `
  --init-from-drkg
```

Main outputs are written to `<subgraph-dir>/transe/`:

- `entity_embeddings.npy` - node embeddings in `node_embedding_index.tsv` order
- `relation_embeddings.npy` - relation embeddings in `relation_embedding_index.tsv` order
- `node_embedding_index.tsv` - mapping from embedding rows back to DRKG entities
- `relation_embedding_index.tsv` - mapping from relation rows back to DRKG relations
- `subset_Compound_embeddings.npy` and `subset_Compound_nodes.tsv` - compound-only export
- `loss.tsv` - training loss log
- `training_metadata.json` - training parameters and output manifest

For MMDRP, use `entity_embeddings.npy` plus `node_embedding_index.tsv` for all
subgraph nodes, or the `subset_Compound_*` files if the third modality should be
drug/compound-level only.

## 3. Inspect embeddings

```powershell
python .\scripts\inspect_embeddings.py `
  --embedding-dir .\outputs\driver_gene_subgraph_h1\transe `
  --seed-file .\seeds\lung_cancer_driver_genes.tsv `
  --top-k 20
```

Main outputs are written to `<embedding-dir>/inspection/`:

- `seed_nearest_neighbors.tsv` - nearest entities for each seed by cosine similarity
- `seed_cosine_similarity.tsv` - cosine similarity between the seed genes
- `inspection_summary.json` - inspection metadata

## 4. Build MMDRP DrugBank + DRKG Inputs

Map `week1_final_drug_smiles.csv` to DrugBank IDs through PubChem synonyms, then
match those DrugBank IDs to DRKG compound embeddings.

1-hop:

```powershell
python .\scripts\map_drugbank_to_drkg_embeddings.py `
  --drug-csv C:\Users\user\Downloads\week1_final_drug_smiles.csv `
  --embedding-dir .\outputs\driver_gene_subgraph_h1\transe `
  --out-dir .\outputs\driver_gene_subgraph_h1\drugbank_mmdrp_input `
  --cache .\outputs\driver_gene_subgraph_h1\smiles_match\pubchem_synonyms_cache.json
```

2-hop:

```powershell
python .\scripts\map_drugbank_to_drkg_embeddings.py `
  --drug-csv C:\Users\user\Downloads\week1_final_drug_smiles.csv `
  --embedding-dir .\outputs\driver_gene_subgraph_h2\transe_20e `
  --out-dir .\outputs\driver_gene_subgraph_h2\drugbank_mmdrp_input `
  --cache .\outputs\driver_gene_subgraph_h1\smiles_match\pubchem_synonyms_cache.json
```

Current mapping counts:

| Graph / Embedding | Drug rows with DrugBank ID | Matched drug rows | Feature rows | Feature shape |
| --- | ---: | ---: | ---: | --- |
| 1-hop, 400d/200e | 183 | 103 | 104 | `(104, 400)` |
| 1-hop, 200d/50e | 183 | 103 | 104 | `(104, 200)` |
| 2-hop, 400d/20e | 183 | 172 | 173 | `(173, 400)` |

Main outputs:

- `drugbank_mapping_annotated.csv` - original drug table plus `DrugBank_IDs`
- `drugbank_mapping_long.csv` - one row per drug and DrugBank ID
- `mmdrp_drkg_transe_features.npy` - TransE feature matrix for MMDRP
- `mmdrp_drkg_transe_index.csv` - row-to-drug/DRKG mapping for the feature matrix
- `mmdrp_drkg_transe_merged.csv` - readable merged table with `transe_0` to `transe_399`