$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$DrkgRoot = Join-Path $Root "DRKG_pipeline"
$Official = Join-Path $DrkgRoot "official_data"
$SeedFile = Join-Path $DrkgRoot "seeds\pancancer_driver_genes.tsv"
$SubgraphOut = Join-Path $DrkgRoot "outputs\reproduced_pancancer_driver_subgraph_h1"
$TranseOut = Join-Path $SubgraphOut "transe_200d_50e"
$FeatureOut = Join-Path $Root "Data\DRKG_Features\reproduced_pancancer_driver_h1_200d_50e"
$DrugCsv = Join-Path $Root "Data\raw\ctrpv2\week1_final_drug_smiles_recovered_from_mapping.csv"
$Cache = Join-Path $DrkgRoot "cache\pubchem_synonyms_cache.json"

Write-Host "Extracting 1-hop pan-cancer driver-gene DRKG subgraph..."
python (Join-Path $DrkgRoot "scripts\extract_driver_gene_subgraph.py") `
  --drkg (Join-Path $Official "drkg.tsv") `
  --entities (Join-Path $Official "embed\entities.tsv") `
  --relations (Join-Path $Official "embed\relations.tsv") `
  --entity-src (Join-Path $Official "entity2src.tsv") `
  --seed-file $SeedFile `
  --hops 1 `
  --out $SubgraphOut

Write-Host ""
Write-Host "Training TransE 200d / 50 epochs..."
python (Join-Path $DrkgRoot "scripts\train_transe.py") `
  --subgraph-dir $SubgraphOut `
  --out-dir $TranseOut `
  --epochs 50 `
  --dim 200 `
  --batch-size 8192 `
  --eval-every 10

Write-Host ""
Write-Host "Mapping drugs to DRKG compound embeddings..."
python (Join-Path $DrkgRoot "scripts\map_drugbank_to_drkg_embeddings.py") `
  --drug-csv $DrugCsv `
  --embedding-dir $TranseOut `
  --out-dir $FeatureOut `
  --cache $Cache

Write-Host ""
Write-Host "Done. Reproduced DRKG feature folder:"
Write-Host $FeatureOut
