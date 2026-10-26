Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    Write-Error "Python launcher 'py' was not found. Install Python 3.11 or 3.12 from https://www.python.org/downloads/ and check 'Add python.exe to PATH'."
}

py -3 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host ""
Write-Host "Ready. Activate the venv, then start the plot:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  python plot_rigid_body.py MOTIVE_SERVER_IP"
