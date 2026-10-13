# Start the local vision server (OWLv2 open-vocabulary detector) for the TurtleBot 4 camera.
# Listens on 127.0.0.1:8001 only.
#   powershell -ExecutionPolicy Bypass -File scripts\serve_vision.ps1
param([string]$Device = "", [int]$Port = 8001)
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv-laya\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Error "Laya venv not found. Set it up first: see README.md, 'Set up Laya'."
    exit 1
}
$serverArgs = @((Join-Path $root "vision\serve_vision.py"), "--host", "127.0.0.1", "--port", "$Port")
if ($Device) { $serverArgs += @("--device", $Device) }
Write-Host "Vision (OWLv2) on http://127.0.0.1:$Port"
& $python -I @serverArgs
