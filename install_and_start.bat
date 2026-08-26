@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"

echo [1/4] Checking Python...
where py >nul 2>nul
if errorlevel 1 (
    echo Python launcher was not found. Please install Python 3.10 or newer from python.org, then run this file again.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [2/4] Creating local virtual environment...
    py -3.11 -m venv .venv 2>nul
    if errorlevel 1 py -3 -m venv .venv
    if errorlevel 1 (
        echo Unable to create the virtual environment.
        pause
        exit /b 1
    )
)

echo [3/4] Installing required packages...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :install_failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :install_failed

where nvidia-smi >nul 2>nul
if not errorlevel 1 (
    echo NVIDIA GPU detected. Installing optional GPU OCR runtime...
    ".venv\Scripts\python.exe" -m pip install -r requirements-gpu.txt
    if errorlevel 1 echo GPU runtime was unavailable. The control panel will use CPU OCR instead.
) else (
    echo NVIDIA GPU was not detected. CPU OCR will be used.
)

echo [4/4] Starting the local control panel...
start "WOWS Control Panel" /min cmd /c ""%~dp0start_control_panel.bat""
timeout /t 3 /nobreak >nul
start "" "http://127.0.0.1:8765/"
echo The control panel has been opened in your browser.
exit /b 0

:install_failed
echo Installation failed. Check your network connection and run this file again.
pause
exit /b 1
