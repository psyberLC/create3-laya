@echo off
rem Opens the create3-laya launcher window (no console). Extra arguments go to launcher\app.py,
rem for example:  Launcher.cmd --sim tb4 --start
set "PYW=%~dp0.venv-laya\Scripts\pythonw.exe"
if not exist "%PYW%" (
  echo The launcher runs on the .venv-laya environment, which is missing.
  echo See README.md, "Set up the local models".
  pause
  exit /b 1
)
start "" "%PYW%" "%~dp0launcher\app.py" %*
