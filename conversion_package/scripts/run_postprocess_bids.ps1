$ErrorActionPreference = "Stop"

# Your environment sets PYTHONPATH to a broken Python install (C:\Python312\Lib\site-packages).
# Clear it so the repo's venv resolves packages correctly.
$env:PYTHONPATH = ""

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$packageRoot = Resolve-Path (Join-Path $scriptRoot "..")
$repoRoot = Resolve-Path (Join-Path $packageRoot "..")

$venv = Join-Path $repoRoot ".venv_bids311\Scripts\python.exe"
if (-not (Test-Path $venv)) {
    $venv = Join-Path $repoRoot ".venv_bids\Scripts\python.exe"
}
if (-not (Test-Path $venv)) {
    throw "Python venv not found in $repoRoot (.venv_bids311 or .venv_bids)."
}

$scriptPath = Join-Path $scriptRoot "postprocess_bids.py"
& $venv $scriptPath @args
