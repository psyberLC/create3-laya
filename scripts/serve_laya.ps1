# Start Laya, the local decision-model server, for the Create 3.
# Serves `typed-decisions` (task understanding) and, once trained, the `create3-driving`
# fine-tune from models\create3-driving. Binds to 127.0.0.1 only: laya-serve otherwise listens
# on 0.0.0.0 with no authentication.
#   powershell -ExecutionPolicy Bypass -File scripts\serve_laya.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\serve_laya.ps1 -Device cpu
param(
    [string]$Device = "cuda",
    [int]$Port = 8000,
    [string]$BindHost = "127.0.0.1"
)
$root = Split-Path -Parent $PSScriptRoot
$serve = Join-Path $root ".venv-laya\Scripts\laya-serve.exe"
if (-not (Test-Path $serve)) {
    Write-Error "laya-serve not found. Install it first: see README.md, 'Set up Laya'."
    exit 1
}
# Each checkpoint holds ~2 GB of GPU memory (fp32 weights), and Gazebo needs what is left of an
# 8 GB card: preload only task understanding. The driving fine-tune is registered but loads on
# first use, i.e. only when the fine-tuned driver is chosen. Requests that name no model fall
# back to typed-decisions too, so a bare probe can't load a third checkpoint.
$driving = Join-Path $root "models\create3-driving"
$lazy = ""
if (Test-Path (Join-Path $driving "model.safetensors")) {
    $env:LAYA_EXTRA_MODELS = (@{ "create3-driving" = $driving } | ConvertTo-Json -Compress)
    $lazy = " (create3-driving on demand)"
} else {
    Remove-Item Env:LAYA_EXTRA_MODELS -ErrorAction SilentlyContinue
    Write-Host "No driving fine-tune in models\create3-driving yet; serving task understanding only."
}
$env:LAYA_HOST = $BindHost
$env:LAYA_PORT = "$Port"
$env:LAYA_DEVICE = $Device
$env:LAYA_PRELOAD = "1"
$env:LAYA_MODELS = "typed-decisions"
$env:LAYA_MAX_LOADED = "2"
$env:LAYA_DEFAULT_MODEL = "typed-decisions"
Write-Host "Laya on http://${BindHost}:$Port ($Device): typed-decisions$lazy"
& $serve
