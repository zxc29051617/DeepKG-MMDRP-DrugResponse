$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Write-Host "run_all.ps1 delegates to the final pan-cancer experiment pipeline."
& (Join-Path $Root "run_pancancer_all.ps1")
