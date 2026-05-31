# Input Statistics
## Processed CTRPv2 Response
- Cell line-drug pairs: 226,455
- Unique cell lines: 791
- Unique drugs: 544
- Targets: AAC (`area_above_curve`) and `log2_ic50`
- Rows after log2(IC50) filter [-15, 15]: 220,795

## DepMap 26Q1 Expression
- Expression rows: 1,719
- Unique normalized cell lines: 1,719
- Gene expression features: 19,215
- Expression matrix used: (1,719, 19,215)


## DRKG Lung Cancer 1-Hop Subgraph
- Seed genes: EGFR, KRAS, ALK, ROS1, BRAF, MET, RET, HER2/ERBB2
- Nodes: 5,040
- Triples: 670,797
- Relations: 96
- Hop setting: 1-hop induced subgraph
- Entity types: 2,875 genes, 771 compounds, 281 diseases, 744 biological processes, 180 pathways, and others

## DRKG Feature
- Feature matrix: (104, 200)
- Index rows: 104
- Unique DRKG compounds: 104
- Unique DrugBank IDs: 104

## Final Model Input After Intersection
- Final cell line-drug pairs: 39,654
- Final unique cell lines: 766
- Final unique drugs: 84
- Expression input dim: 19,215
- Drug structure input: SMILES -> molecular graph -> AttentiveFP
- KG input dim: 200
- Prediction targets: AAC and log2(IC50)
- Pairs per cell line min / median / max: 3 / 53.0 / 77
- Pairs per drug min / median / max: 8 / 533.5 / 732

## NSCLC Subset Prepared
- definition: lineage_2 == 'Non-Small Cell Lung Cancer' using DepMap 26Q1 metadata
- all_lung_cell_lines_in_matched_data: 135
- nsclc_cell_lines_in_matched_data: 101
- nsclc_pairs_in_matched_data: 5195
- drugs_in_nsclc_subset: 84
- output_csv: C:\mmdrp_exp\results\19_ctrpv2_depmap26q1_no_lds_alpha08_20e\nsclc_cell_lines_depmap26q1_ctrpv2_drkg_matched.csv
