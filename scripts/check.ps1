# Stage 1 verification: lint, unit tests, seed SQL, ingest static docs.
# Stops at the first failing step with a non-zero exit code.
# Usage (from the repo root):  .\scripts\check.ps1

# Python logs to stderr; rely on exit codes, not PowerShell's stderr handling.
$ErrorActionPreference = "Continue"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Host "Python venv not found at $python" -ForegroundColor Red
    exit 1
}

$steps = @(
    @{ Name = "ruff check";   Args = @("-m", "ruff", "check", ".") },
    @{ Name = "pytest";       Args = @("-m", "pytest", "-q") },
    @{ Name = "seed --reset"; Args = @("-m", "app.data.seed", "--reset") },
    @{ Name = "ingest --reset"; Args = @("-m", "app.data.ingest", "--reset") }
)

foreach ($step in $steps) {
    Write-Host ""
    Write-Host "==> $($step.Name)" -ForegroundColor Cyan
    & $python @($step.Args)
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: $($step.Name) (exit $LASTEXITCODE)" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

Write-Host ""
Write-Host "All checks passed." -ForegroundColor Green
exit 0
