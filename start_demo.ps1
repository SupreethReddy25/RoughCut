# RoughCut demo launcher: builds the UI if needed, starts the app and opens the browser.
#   powershell -ExecutionPolicy Bypass -File start_demo.ps1 [-DataDir C:\tmp\rc_demo] [-Port 8000]
param([string]$DataDir = "", [int]$Port = 8000)
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not (Test-Path "$root\frontend\dist\index.html")) {
    Write-Host "Building the UI..."
    Push-Location "$root\frontend"; npm run build; Pop-Location
}
if ($DataDir) { $env:ROUGHCUT_DATA_DIR = $DataDir }
Start-Job -ScriptBlock { param($p) Start-Sleep 4; Start-Process "http://127.0.0.1:$p" } -ArgumentList $Port | Out-Null
Set-Location "$root\backend"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --port $Port
