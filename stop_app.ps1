$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$RuntimeDir = Join-Path $ProjectRoot ".app-startup"
$StatePath = Join-Path $RuntimeDir "processes.json"

function Get-Descendants {
    param([int]$ParentId)

    $Children = @(Get-CimInstance Win32_Process -Filter "ParentProcessId = $ParentId" -ErrorAction SilentlyContinue)
    foreach ($Child in $Children) {
        Get-Descendants -ParentId ([int]$Child.ProcessId)
        $Child
    }
}

function Test-ExpectedServiceCommand {
    param(
        [string]$CommandLine,
        [string]$Service
    )

    $Normalized = $CommandLine.Replace("\", "/")
    if ($Service -eq "backend") {
        return $Normalized.Contains("uvicorn app.main:app") -and $Normalized.Contains("--port 8000")
    }
    if ($Service -eq "frontend") {
        return $Normalized.Contains("streamlit") `
            -and $Normalized.Contains("app/ui/streamlit_app.py") `
            -and $Normalized.Contains("8501")
    }
    return $false
}

if (-not (Test-Path -LiteralPath $StatePath -PathType Leaf)) {
    Write-Host "No launcher-owned process state found. No processes were stopped."
    exit 0
}

try {
    $TrackedProcesses = @(Get-Content -LiteralPath $StatePath -Raw | ConvertFrom-Json)
}
catch {
    throw "Cannot read launcher process state at $StatePath. No processes were stopped."
}

$Remaining = @()
foreach ($Entry in $TrackedProcesses) {
    $ProcessId = 0
    if (-not [int]::TryParse([string]$Entry.ProcessId, [ref]$ProcessId)) {
        Write-Warning "Ignoring invalid process ID in launcher state."
        continue
    }

    $Process = Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessId" -ErrorAction SilentlyContinue
    if ($null -eq $Process) {
        continue
    }

    $CommandLine = [string]$Process.CommandLine
    $NormalizedCommandLine = $CommandLine.Replace("\", "/")
    $ExecutableMatches = [string]::Equals(
        [string]$Process.ExecutablePath,
        $PythonExe,
        [StringComparison]::OrdinalIgnoreCase
    )
    $RootMatches = $NormalizedCommandLine.IndexOf(
        $ProjectRoot.Replace("\", "/"),
        [StringComparison]::OrdinalIgnoreCase
    ) -ge 0
    $ServiceMatches = Test-ExpectedServiceCommand -CommandLine $CommandLine -Service $Entry.Service

    if (-not ($ExecutableMatches -and $RootMatches -and $ServiceMatches)) {
        Write-Warning "Process $ProcessId no longer matches the tracked $($Entry.Service) command; leaving it untouched and dropping its stale record."
        continue
    }

    foreach ($Descendant in (Get-Descendants -ParentId $ProcessId)) {
        $DescendantCommandLine = [string]$Descendant.CommandLine
        $DescendantRootMatches = $DescendantCommandLine.Replace("\", "/").IndexOf(
            $ProjectRoot.Replace("\", "/"),
            [StringComparison]::OrdinalIgnoreCase
        ) -ge 0
        if ($DescendantRootMatches -and (Test-ExpectedServiceCommand `
                    -CommandLine $DescendantCommandLine -Service $Entry.Service)) {
            Stop-Process -Id $Descendant.ProcessId -ErrorAction Stop
            Write-Host "Stopped launcher-owned $($Entry.Service) worker process (PID $($Descendant.ProcessId))."
        }
    }

    Stop-Process -Id $ProcessId -ErrorAction Stop
    Write-Host "Stopped launcher-started $($Entry.Service) process (PID $ProcessId)."
}

if ($Remaining.Count -gt 0) {
    $Remaining | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath $StatePath -Encoding UTF8
}
elseif (Test-Path -LiteralPath $StatePath) {
    Remove-Item -LiteralPath $StatePath -Force
}
Write-Host "Only processes started by start_app.ps1 were targeted."
