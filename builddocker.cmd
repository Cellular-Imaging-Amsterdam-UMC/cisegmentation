@echo off
setlocal
set /p VERSION=<"%~dp0version.txt"
if not defined VERSION set "VERSION=0.1.0"
pushd "%~dp0"
docker build %* -t w_cisegmentation:%VERSION% -t w_cisegmentation:latest .
set "EXITCODE=%ERRORLEVEL%"
popd
endlocal & exit /b %EXITCODE%
