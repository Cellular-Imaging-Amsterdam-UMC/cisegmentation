@echo off
setlocal
set /p VERSION=<"%~dp0version.txt"
if not defined VERSION set "VERSION=0.1.0"
set "PYTHON_EXE=%LOCALAPPDATA%\miniconda3\envs\cisegmentation\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"
pushd "%~dp0"
"%PYTHON_EXE%" tools\download_models.py --models-dir models
if errorlevel 1 (
  popd
  endlocal & exit /b 1
)
for /f "tokens=*" %%H in ('certutil -hashfile models\.complete.json SHA256 ^| findstr /R /V /C:"hash of" /C:"CertUtil"') do set "MODEL_CACHE_ID=%%H"
if not defined MODEL_CACHE_ID (
  popd
  endlocal & exit /b 1
)
docker image inspect w_cisegmentation-model-cache:%MODEL_CACHE_ID% >nul 2>&1
if errorlevel 1 (
  docker build -f Dockerfile.models --build-arg MODEL_CACHE_ID=%MODEL_CACHE_ID% -t w_cisegmentation-model-cache:%MODEL_CACHE_ID% -t w_cisegmentation-model-cache:latest .
  if errorlevel 1 (
    popd
    endlocal & exit /b 1
  )
)
docker build %* --build-arg MODEL_CACHE_IMAGE=w_cisegmentation-model-cache:%MODEL_CACHE_ID% -t w_cisegmentation:%VERSION% -t w_cisegmentation:latest .
set "EXITCODE=%ERRORLEVEL%"
popd
endlocal & exit /b %EXITCODE%
