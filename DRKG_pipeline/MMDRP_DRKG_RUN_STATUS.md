# DRKG Pipeline Status

The current DRKG pipeline used by the final experiment is pan-cancer oriented.

## Current Main Output

The MMDRP-side feature folder is:

```text
Data/DRKG_Features/pancancer_driver_h1_200d_50e/
```

Important files:

```text
mmdrp_drkg_transe_features.npy
mmdrp_drkg_transe_index.csv
mmdrp_drkg_transe_merged.csv
mapping_summary.json
```

## Pan-Cancer DRKG Construction

Seed file:

```text
DRKG_pipeline/seeds/pancancer_driver_genes.tsv
```

Seed genes:

```text
TP53, KRAS, PIK3CA, PTEN, APC, BRAF, EGFR, ERBB2,
MYC, CDKN2A, RB1, NRAS, HRAS, SMAD4, CTNNB1,
BRCA1, BRCA2, ARID1A, IDH1, IDH2
```

Subgraph and embedding settings:

```text
Traversal: 1-hop
Nodes: 11,067
Triples: 1,766,128
Relations: 97
Embedding model: TransE
Embedding dimension: 200
TransE epochs: 50
MMDRP KG feature shape: 112 x 200
```

## MMDRP Usage

The pan-cancer MMDRP scripts load:

```text
Data/DRKG_Features/pancancer_driver_h1_200d_50e/mmdrp_drkg_transe_features.npy
Data/DRKG_Features/pancancer_driver_h1_200d_50e/mmdrp_drkg_transe_index.csv
```

These features are used by B3+DRKG as the third modality, together with DepMap expression and drug molecular graph features.
