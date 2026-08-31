@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"

title 战舰世界自动战斗控制台 - 首次安装
echo [1/4] 正在检查 Python...
where py >nul 2>nul
if errorlevel 1 (
    echo 未找到 Python。请安装 64 位 Python 3.11，并勾选 Add Python to PATH，然后重新运行 one_click_run.bat。
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [2/4] 正在创建项目专用运行环境...
    py -3.11 -m venv .venv 2>nul
    if errorlevel 1 py -3 -m venv .venv
    if errorlevel 1 (
        echo 无法创建运行环境，请确认 Python 安装完整。
        pause
        exit /b 1
    )
)

echo [3/4] 正在安装或修复运行依赖，首次执行可能需要几分钟...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :install_failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :install_failed

where nvidia-smi >nul 2>nul
if not errorlevel 1 (
    echo 检测到 NVIDIA 显卡，正在尝试安装 GPU OCR 组件...
    ".venv\Scripts\python.exe" -m pip install -r requirements-gpu.txt
    if errorlevel 1 echo GPU OCR 组件不可用，将自动使用 CPU OCR。
) else (
    echo 未检测到 NVIDIA 显卡，将使用 CPU OCR。
)

echo [4/4] 正在启动本地控制台...
start "WOWS Control Panel" /min cmd /c ""%~dp0start_control_panel.bat""
timeout /t 3 /nobreak >nul
start "" "http://127.0.0.1:8765/"
echo 安装完成，控制台已在浏览器中打开。
exit /b 0

:install_failed
echo 依赖安装失败。请检查网络连接，然后重新运行 one_click_run.bat。
pause
exit /b 1
