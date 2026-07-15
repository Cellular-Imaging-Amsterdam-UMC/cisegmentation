@echo off
setlocal
set "CONDA_EXE=%LOCALAPPDATA%\miniconda3\Scripts\conda.exe"
if not exist "%CONDA_EXE%" set "CONDA_EXE=conda"
"%CONDA_EXE%" env list | findstr /R /C:"^cisegmentation " >nul
if errorlevel 1 (
  "%CONDA_EXE%" env create -f "%~dp0environment.yml"
) else (
  "%CONDA_EXE%" env update -n cisegmentation -f "%~dp0environment.yml" --prune
)
if errorlevel 1 exit /b 1
"%CONDA_EXE%" run -n cisegmentation python -m pip install --upgrade pip
if errorlevel 1 exit /b 1
"%CONDA_EXE%" run -n cisegmentation python -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 exit /b 1
"%CONDA_EXE%" run -n cisegmentation python -m pip install -r "%~dp0requirements_launcher.txt"
if errorlevel 1 exit /b 1
"%CONDA_EXE%" run -n cisegmentation python "%~dp0tools\cuda_smoke.py"
endlocal
