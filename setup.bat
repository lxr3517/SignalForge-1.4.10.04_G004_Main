@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "VENV_DIR=%CD%\.venv"
set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
    echo Creating project virtual environment at "%VENV_DIR%"
    where py >nul 2>&1
    if not errorlevel 1 (
        py -3 -m venv "%VENV_DIR%"
    ) else (
        python -m venv "%VENV_DIR%"
    )
    if errorlevel 1 (
        echo ERROR: Could not create the virtual environment. Install Python 3.11+ and rerun setup.bat.
        exit /b 1
    )
)

echo Installing project dependencies into "%VENV_DIR%"
"%PYTHON_EXE%" -m ensurepip --upgrade >nul 2>&1
"%PYTHON_EXE%" -m pip install --upgrade pip
if errorlevel 1 exit /b 1
"%PYTHON_EXE%" -m pip install -r requirements.txt
if errorlevel 1 exit /b 1

echo.
echo Project virtual environment is ready.
echo Python: "%PYTHON_EXE%"
exit /b 0
