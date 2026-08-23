@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" control_server.py
) else (
    py -3 control_server.py
)
if errorlevel 1 pause
