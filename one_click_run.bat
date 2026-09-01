@echo off
setlocal EnableExtensions
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"
title 战舰世界自动战斗控制台

rem 普通用户始终只运行本文件。首次使用或依赖损坏时自动进入安装。
if not exist ".venv\Scripts\python.exe" (
    call "%~dp0install_and_start.bat"
    exit /b %errorlevel%
)

rem 更新代码后若新增了依赖，自动修复环境，不要求用户手动运行 pip。
".venv\Scripts\python.exe" -c "import cv2,numpy,yaml,win32api,vgamepad,rapidocr,onnxruntime" >nul 2>nul
if errorlevel 1 (
    echo 检测到运行环境不完整，正在自动修复...
    call "%~dp0install_and_start.bat"
    exit /b %errorlevel%
)

rem 控制台已经运行时只打开网页，不重复启动服务。
powershell -NoProfile -Command "if (Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue) { exit 0 } exit 1" >nul 2>nul
if errorlevel 1 (
    echo 正在启动本地控制台...
    start "WOWS Control Panel" /min cmd /c ""%~dp0start_control_panel.bat""
    echo 如果出现 Windows 用户账户控制提示，请选择“是”...
    powershell -NoProfile -Command "$deadline=(Get-Date).AddSeconds(30); do { if (Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue) { exit 0 }; Start-Sleep -Milliseconds 500 } while ((Get-Date) -lt $deadline); exit 1" >nul 2>nul
    if errorlevel 1 (
        echo 控制台尚未启动，请确认已允许管理员权限后再运行一次。
        pause
        exit /b 1
    )
)

start "" "http://127.0.0.1:8765/"
echo 控制台已打开：http://127.0.0.1:8765/
exit /b 0
