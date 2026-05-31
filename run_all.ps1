$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:MMDRP_ROOT = $Root
$env:DRKG_FEATURE_DIR = Join-Path $Root "Data\DRKG_Features\driver_gene_h1_200d_50e"

$ExpressionHdf = Join-Path $Root "Data\DRP_Training_Data\DepMap_26Q1_Expression.hdf"
$DrkgDir = $env:DRKG_FEATURE_DIR
$BestConfig = Join-Path $Root "config\best_config.json"
$B3Out = Join-Path $Root "results\reproduced_B3_B3DRKG_26Q1_CTRPv2_50e"
$BaselineOut = Join-Path $Root "results\reproduced_B0_B1_B2_26Q1_CTRPv2_50e"
$SummaryOut = Join-Path $Root "results\reproduced_50e_nsclc_evaluation"
$PredictionDirs = "$B3Out,$BaselineOut"

Write-Host "Root: $Root"
Write-Host "Expression: $ExpressionHdf"
Write-Host "DRKG features: $DrkgDir"
Write-Host "B3 best config: $BestConfig"

Write-Host ""
Write-Host "Running B3 base vs B3 + DRKG..."
python (Join-Path $Root "src\run_attentivefp_multiseed_cv.py") `
  --expression-hdf $ExpressionHdf `
  --best-config $BestConfig `
  --drkg-dir $DrkgDir `
  --seeds "42,43,44" `
  --n-folds 5 `
  --epochs 50 `
  --batch-size 512 `
  --models "base,drkg" `
  --alphas 0.8 `
  --no-lds `
  --out-dir $B3Out

Write-Host ""
Write-Host "Running B0 / B1 / B2 baselines..."
python (Join-Path $Root "src\run_b0_b1_b2_baselines.py") `
  --expression-hdf $ExpressionHdf `
  --drkg-dir $DrkgDir `
  --seeds "42,43,44" `
  --n-folds 5 `
  --epochs 50 `
  --b1-epochs 50 `
  --models "B0,B1,B2" `
  --out-dir $BaselineOut

Write-Host ""
Write-Host "Summarizing overall and NSCLC-only metrics..."
python (Join-Path $Root "src\summarize_nsclc_prediction_metrics.py") `
  --prediction-dirs $PredictionDirs `
  --out-dir $SummaryOut `
  --save-all-predictions

Write-Host ""
Write-Host "Done. Check the reproduced result folders under:"
Write-Host (Join-Path $Root "results")
