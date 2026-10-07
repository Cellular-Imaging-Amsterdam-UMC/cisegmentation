@echo off
setlocal EnableExtensions
set "BUILD_OPTIONS="
:parse_args
if "%~1"=="" goto :args_done
if /I "%~1"=="--no-cache" (
    set "BUILD_OPTIONS=--no-cache"
    shift /1
    goto :parse_args
)
echo ERROR: Only --no-cache is supported. Optional interface images are local builds only.
exit /b 1
:args_done
if not exist "%~dp0..\version.txt" (
    echo ERROR: The workflow version.txt is missing.
    exit /b 1
)
set "VERSION="
set /p VERSION=<"%~dp0..\version.txt"
if not defined VERSION (
    echo ERROR: The workflow version.txt is empty.
    exit /b 1
)
call docker image inspect "w_cisegmentation:%VERSION%" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Build the workflow image first using builddocker.cmd in the repository root.
    exit /b 1
)
call docker build %BUILD_OPTIONS% -f "%~dp0Dockerfile.jupyter" --build-arg BASE_IMAGE=w_cisegmentation:%VERSION% -t w_cisegmentation:%VERSION%-jupyter -t w_cisegmentation:latest-jupyter "%~dp0."
exit /b %ERRORLEVEL%
