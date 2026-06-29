@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Project virtual environment not found. Running setup...
    call setup.bat
    if errorlevel 1 (
        echo.
        echo Setup failed. Run repair.bat for a full rebuild and health check.
        pause
        exit /b 1
    )
)

".venv\Scripts\python.exe" -m uvicorn app.main:app --reload
pause
