# Run the Gazebo (native create3_sim) test from Windows. Starts nothing on Windows: run
# scripts\serve_laya.ps1 first so the Laya server is listening on 127.0.0.1:8000.
#   powershell -ExecutionPolicy Bypass -File scripts\gazebo_test.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\gazebo_test.ps1 "go to the east hall" "turn around"
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Tasks)
$root = Split-Path -Parent $PSScriptRoot
$wslRoot = (wsl -d Ubuntu-24.04 -- wslpath -a ($root -replace '\\', '/')).Trim()
# Windows PowerShell runs ForEach-Object once for $null, which would send one empty task.
$quoted = ''
if ($Tasks) { $quoted = ($Tasks | ForEach-Object { "'" + ($_ -replace "'", "'\''") + "'" }) -join ' ' }
wsl -d Ubuntu-24.04 -- bash -lc "'$wslRoot/scripts/wsl/gazebo_test.sh' $quoted"
