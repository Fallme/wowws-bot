@echo off
setlocal EnableExtensions
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"

rem The only end-user entry point. First use delegates to the installer.
if not exist ".venv\Scripts\python.exe" (
    call "%~dp0install_and_start.bat"
    exit /b %errorlevel%
)

rem Do not start a second server when the panel is already running.
powershell -NoProfile -Command "if (Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue) { exit 0 } exit 1" >nul 2>nul
if errorlevel 1 (
    start "WOWS Control Panel" /min cmd /c ""%~dp0start_control_panel.bat""
    timeout /t 3 /nobreak >nul
)

start "" "http://127.0.0.1:8765/"
exit /b 0
