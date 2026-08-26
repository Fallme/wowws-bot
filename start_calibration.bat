@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo First-time setup is required. Run install_and_start.bat first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" calibrate_input.py
pause
