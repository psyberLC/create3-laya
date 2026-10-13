# Run the TurtleBot 4 vision test (depot world) from Windows. Start scripts\serve_laya.ps1
# and scripts\serve_vision.ps1 first.
#   powershell -ExecutionPolicy Bypass -File scripts\tb4_test.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\tb4_test.ps1 "go to where the pallets are stored"
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Tasks)
$root = Split-Path -Parent $PSScriptRoot
$wslRoot = (wsl -d Ubuntu-24.04 -- wslpath -a ($root -replace '\\', '/')).Trim()
# Windows PowerShell runs ForEach-Object once for $null, which would send one empty task.
$quoted = ''
if ($Tasks) { $quoted = ($Tasks | ForEach-Object { "'" + ($_ -replace "'", "'\''") + "'" }) -join ' ' }
wsl -d Ubuntu-24.04 -- bash -lc "'$wslRoot/scripts/wsl/tb4_test.sh' $quoted"
