$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$RuntimeDir = Join-Path $ProjectRoot ".app-startup"
$StatePath = Join-Path $RuntimeDir "processes.json"
$BackendUrl = "http://127.0.0.1:8000"
$BackendHealthUrl = "$BackendUrl/health"
$FrontendUrl = "http://127.0.0.1:8501"
$FrontendHealthUrl = "$FrontendUrl/_stcore/health"
$StartupTimeoutSeconds = 30

function Test-PortOpen {
    param([int]$Port)

    $Client = New-Object System.Net.Sockets.TcpClient
    try {
        $Async = $Client.BeginConnect("127.0.0.1", $Port, $null, $null)
        if (-not $Async.AsyncWaitHandle.WaitOne(500)) {
            return $false
        }
        $Client.EndConnect($Async)
        return $true
    }
    catch {
        return $false
    }
    finally {
        $Client.Close()
    }
}

function Test-HealthyEndpoint {
    param([string]$Url)

    try {
        $Response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 2
        return ($Response.StatusCode -eq 200)
    }
    catch {
        return $false
    }
}

function Save-TrackedProcesses {
    $TrackedProcesses | ConvertTo-Json -Depth 3 |
        Set-Content -LiteralPath $StatePath -Encoding UTF8
}

function Wait-ForEndpoint {
    param(
        [string]$Url,
        [System.Diagnostics.Process]$Process,
        [string]$ServiceName,
        [string]$ErrorLog
    )

    $Deadline = (Get-Date).AddSeconds($StartupTimeoutSeconds)
    while ((Get-Date) -lt $Deadline) {
        if (Test-HealthyEndpoint -Url $Url) {
            return $true
        }
        if ($Process.HasExited) {
            $Details = ""
            if (Test-Path $ErrorLog) {
                $Details = (Get-Content -Path $ErrorLog -Tail 20) -join [Environment]::NewLine
            }
            throw "$ServiceName exited before becoming healthy. Log: $ErrorLog`n$Details"
        }
        Start-Sleep -Seconds 1
    }
    throw "$ServiceName did not become healthy within $StartupTimeoutSeconds seconds. Log: $ErrorLog"
}

if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    throw "Virtual-environment Python was not found: $PythonExe"
}

New-Item -ItemType Directory -Path $RuntimeDir -Force | Out-Null
$TrackedProcesses = @()
if (Test-Path -LiteralPath $StatePath) {
    try {
        $LoadedState = Get-Content -LiteralPath $StatePath -Raw | ConvertFrom-Json
        if ($null -ne $LoadedState) {
            $TrackedProcesses = @($LoadedState)
        }
    }
    catch {
        throw "Cannot read launcher process state at $StatePath. Inspect or remove the invalid state file before starting."
    }
}

$BackendStarted = $false
if (Test-PortOpen -Port 8000) {
    if (-not (Test-HealthyEndpoint -Url $BackendHealthUrl)) {
        throw "Port 8000 is occupied, but $BackendHealthUrl is not healthy. No backend was started."
    }
    Write-Host "Reusing the healthy API already listening on port 8000."
}
else {
    $BackendOut = Join-Path $RuntimeDir "backend.stdout.log"
    $BackendErr = Join-Path $RuntimeDir "backend.stderr.log"
    $BackendArgs = @(
        "-m", "uvicorn", "--app-dir", "`"$ProjectRoot`"", "app.main:app",
        "--host", "127.0.0.1",
        "--port", "8000"
    )
    $BackendProcess = Start-Process `
        -FilePath $PythonExe `
        -ArgumentList $BackendArgs `
        -WorkingDirectory $ProjectRoot `
        -RedirectStandardOutput $BackendOut `
        -RedirectStandardError $BackendErr `
        -WindowStyle Hidden `
        -PassThru
    $TrackedProcesses += [pscustomobject]@{
        ProcessId = $BackendProcess.Id
        Service = "backend"
        StartedAt = (Get-Date).ToString("o")
    }
    Save-TrackedProcesses
    $BackendStarted = $true

    try {
        Wait-ForEndpoint -Url $BackendHealthUrl -Process $BackendProcess `
            -ServiceName "FastAPI backend" -ErrorLog $BackendErr | Out-Null
    }
    catch {
        Write-Error $_.Exception.Message
        exit 1
    }
}

$FrontendStarted = $false
if (Test-PortOpen -Port 8501) {
    if (-not (Test-HealthyEndpoint -Url $FrontendHealthUrl)) {
        throw "Port 8501 is occupied, but $FrontendHealthUrl is not healthy. No frontend was started."
    }
    Write-Host "Reusing the healthy Streamlit application already listening on port 8501."
}
else {
    $FrontendOut = Join-Path $RuntimeDir "frontend.stdout.log"
    $FrontendErr = Join-Path $RuntimeDir "frontend.stderr.log"
    $FrontendArgs = @(
        "-m", "streamlit", "run", "`"$ProjectRoot\app\ui\streamlit_app.py`"",
        "--server.address", "127.0.0.1",
        "--server.port", "8501",
        "--server.headless", "true"
    )
    $FrontendProcess = Start-Process `
        -FilePath $PythonExe `
        -ArgumentList $FrontendArgs `
        -WorkingDirectory $ProjectRoot `
        -RedirectStandardOutput $FrontendOut `
        -RedirectStandardError $FrontendErr `
        -WindowStyle Hidden `
        -PassThru
    $TrackedProcesses += [pscustomobject]@{
        ProcessId = $FrontendProcess.Id
        Service = "frontend"
        StartedAt = (Get-Date).ToString("o")
    }
    Save-TrackedProcesses
    $FrontendStarted = $true

    try {
        Wait-ForEndpoint -Url $FrontendHealthUrl -Process $FrontendProcess `
            -ServiceName "Streamlit frontend" -ErrorLog $FrontendErr | Out-Null
    }
    catch {
        Write-Error $_.Exception.Message
        if ($BackendStarted) {
            Write-Warning "The backend was started by this script and remains available at $BackendUrl. Run .\stop_app.ps1 to stop tracked processes."
        }
        exit 1
    }
}

Write-Host ""
Write-Host "Parking Chatbot is ready."
Write-Host "Backend:  $BackendUrl  (health: $BackendHealthUrl)"
Write-Host "Frontend: $FrontendUrl"
try {
    Start-Process $FrontendUrl
}
catch {
    Write-Warning "Services are running, but the browser could not be opened automatically: $($_.Exception.Message)"
}
