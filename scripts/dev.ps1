param(
    [ValidateSet('serve', 'check', 'agents')]
    [string]$Task = 'serve'
)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $repoRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw 'Create the .venv and install requirements/dev.lock first; see README.md.'
}
Push-Location $repoRoot
try {
    if ($Task -eq 'check') {
        & $pythonPath scripts/check.py
    } else {
        & $pythonPath -m src.cli $Task
    }
    if ($LASTEXITCODE -ne 0) { throw "Task failed with exit code $LASTEXITCODE" }
} finally {
    Pop-Location
}
