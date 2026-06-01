# Input Statistics

## Processed CTRPv2 Response

- Cell line-drug pairs: 226,455
- Unique cell lines: 791
- Unique drugs: 544
- Targets: AAC (`area_above_curve`) and `log2_ic50`

## DepMap 26Q1 Expression

- Expression rows: 1,719
- Gene expression features: 19,215
- Expression matrix used: `(1719, 19215)`

## Pan-Cancer DRKG 1-Hop Subgraph

- Seed genes: 20 pan-cancer driver genes
- Seed list: TP53, KRAS, PIK3CA, PTEN, APC, BRAF, EGFR, ERBB2, MYC, CDKN2A, RB1, NRAS, HRAS, SMAD4, CTNNB1, BRCA1, BRCA2, ARID1A, IDH1, IDH2
- Nodes: 11,067
- Triples: 1,766,128
- Relations: 97
- Hop setting: 1-hop induced subgraph
- Embedding model: TransE
- Embedding dimension: 200
- TransE epochs: 50

## DRKG Feature

- Feature matrix: `(112, 200)`
- Index rows: 112
- Unique DRKG compounds: 112
- Unique DrugBank IDs: 112
- Feature folder: `Data/DRKG_Features/pancancer_driver_h1_200d_50e/`

## Final Model Input After Intersection

- Final cell line-drug pairs: 44,049
- Final unique cell lines: 766
- Final unique drugs: 92
- Expression input dim: 19,215
- Drug structure input: SMILES -> molecular graph -> AttentiveFP
- KG input dim: 200
- Prediction targets: AAC and log2(IC50)

## NSCLC Subset Prepared

- Definition: `lineage_2 == 'Non-Small Cell Lung Cancer'` using DepMap 26Q1 metadata
- NSCLC cell lines in matched data: 101
- NSCLC pairs in matched data: 5,816
- Drugs in NSCLC subset: 92
