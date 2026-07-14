@echo off
setlocal
set "PYTHON=%LOCALAPPDATA%\miniconda3\envs\cisegmentation\python.exe"
if not exist "%PYTHON%" (
  echo Missing cisegmentation environment: %PYTHON%
  exit /b 1
)
"%PYTHON%" "%~dp0run_roundtrip.py" %*
endlocal
