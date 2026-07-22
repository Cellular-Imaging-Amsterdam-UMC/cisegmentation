@echo off
setlocal
set "PYTHON_EXE=%LOCALAPPDATA%\miniconda3\envs\cisegmentation\python.exe"
if not exist "%PYTHON_EXE%" (
    echo ERROR: The cisegmentation Conda environment was not found.
    echo Create it first by running: create_env.cmd
    exit /b 1
)
"%PYTHON_EXE%" -m pytest %*
endlocal & exit /b %ERRORLEVEL%
