# Pan-Cancer DRKG Pipeline

This folder contains the pan-cancer DRKG feature-generation outputs used by the final MMDRP pan-cancer experiment.

## Seed Genes

Seed file:

- `seeds/pancancer_driver_genes.tsv`

The 20 pan-cancer driver genes are:

TP53, KRAS, PIK3CA, PTEN, APC, BRAF, EGFR, ERBB2, MYC, CDKN2A, RB1, NRAS, HRAS, SMAD4, CTNNB1, BRCA1, BRCA2, ARID1A, IDH1, IDH2.

## Output Folder

Main output:

- `outputs/pancancer_driver_subgraph_h1/`

Important files:

- `stats.json`: subgraph statistics
- `triples.tsv`: TransE training triples
- `triples_names.tsv`: human-readable triples
- `transe_200d_50e/entity_embeddings.npy`: trained entity embeddings
- `transe_200d_50e/node_embedding_index.tsv`: entity-to-row index
- `transe_200d_50e/loss.tsv`: TransE training loss
- `drugbank_mmdrp_input_200d_50e/mmdrp_drkg_transe_features.npy`: MMDRP KG feature matrix
- `drugbank_mmdrp_input_200d_50e/mmdrp_drkg_transe_index.csv`: MMDRP drug-to-feature index

## Statistics

The pan-cancer DRKG subgraph uses:

- 20 seed genes
- 1-hop traversal
- 11,067 nodes
- 1,766,128 triples
- 97 relation types
- TransE dimension: 200
- TransE epochs: 50
- MMDRP mapped KG feature shape: 112 x 200

## MMDRP Usage

The MMDRP scripts use the copied feature folder:

- `Data/DRKG_Features/pancancer_driver_h1_200d_50e/`

That folder contains the same MMDRP-ready feature matrix and index needed by:

- `run_pancancer_all.ps1`
- `src/run_attentivefp_multiseed_cv.py`
- `src/run_b0_b1_b2_baselines.py`
