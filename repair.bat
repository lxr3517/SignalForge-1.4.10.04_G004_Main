@echo off
setlocal
cd /d "%~dp0"

echo Repairing SignalForge local runtime...
echo.

call setup.bat
if errorlevel 1 (
    echo.
    echo Repair failed while setting up the project virtual environment.
    pause
    exit /b 1
)

echo.
echo Running import and health checks...
".venv\Scripts\python.exe" tools\repair_check.py
if errorlevel 1 (
    echo.
    echo Repair checks found a problem. Review the messages above.
    pause
    exit /b 1
)

echo.
echo Repair complete. Start the app with run_local.bat.
pause
