@echo off
call "%~dp0builddocker.cmd" %*
if errorlevel 1 exit /b 1
set /p VERSION=<"%~dp0version.txt"
docker build -f "%~dp0Dockerfile.jupyter" --build-arg BASE_IMAGE=w_cisegmentation:%VERSION% -t w_cisegmentation:%VERSION%-jupyter -t w_cisegmentation:latest-jupyter "%~dp0"
