@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0"

set "ROOT=%~dp0"
set "PYTHON=%ROOT%.venv\Scripts\python.exe"
set "START_LOG=%ROOT%data\startup.log"
if not defined WOWS_PANEL_PORT set "WOWS_PANEL_PORT=8899"
set "PORT=%WOWS_PANEL_PORT%"
title 战舰世界自动战斗控制台

if not exist "%ROOT%data" mkdir "%ROOT%data" >nul 2>&1
call :log "启动入口: %ROOT%"

if not exist "%PYTHON%" (
    call :log "未找到项目虚拟环境，开始创建 .venv"
    where py >nul 2>&1
    if errorlevel 1 goto :python_missing
    py -3.11 -m venv "%ROOT%.venv" >> "%START_LOG%" 2>&1
    if errorlevel 1 py -3 -m venv "%ROOT%.venv" >> "%START_LOG%" 2>&1
    if errorlevel 1 goto :venv_failed
)

if not exist "%PYTHON%" goto :venv_failed

call :log "检查项目依赖"
"%PYTHON%" -c "import cv2,numpy,yaml,win32api,vgamepad,rapidocr,onnxruntime" >> "%START_LOG%" 2>&1
if errorlevel 1 (
    call :log "依赖不完整，使用 requirements.txt 修复"
    "%PYTHON%" -m pip install -r "%ROOT%requirements.txt" >> "%START_LOG%" 2>&1
    if errorlevel 1 goto :install_failed
)

>nul 2>&1 (netstat -ano | findstr /c:":%PORT%" | findstr /c:"LISTENING")
if not errorlevel 1 (
    call :log "检测到已有控制台，直接打开网页"
    goto :open_browser
)

call :log "启动本地控制台，日志写入 data\startup.log"
set "SERVER_STDOUT=%ROOT%data\control_server.stdout.log"
set "SERVER_STDERR=%ROOT%data\control_server.stderr.log"
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$p=Start-Process -FilePath '%PYTHON%' -ArgumentList @('control_server.py','--no-browser') -WorkingDirectory '%ROOT%' -RedirectStandardOutput '%SERVER_STDOUT%' -RedirectStandardError '%SERVER_STDERR%' -WindowStyle Hidden -PassThru; if(-not $p){exit 1}"
if errorlevel 1 goto :server_start_failed

set /a tries=0
:wait_server
>nul 2>&1 (netstat -ano | findstr /c:":%PORT%" | findstr /c:"LISTENING")
if not errorlevel 1 goto :open_browser
set /a tries+=1
if !tries! geq 45 goto :server_timeout
powershell.exe -NoProfile -Command "Start-Sleep -Seconds 1"
goto :wait_server

:open_browser
start "" "http://127.0.0.1:%PORT%/"
echo 战舰控制台已启动：http://127.0.0.1:%PORT%/
echo 启动日志：%START_LOG%
exit /b 0

:python_missing
call :fail "未找到 Python 启动器 py。请安装 64 位 Python 3.11 后重试。"
exit /b 1

:venv_failed
call :fail "项目虚拟环境创建失败，请查看 data\startup.log。"
exit /b 1

:install_failed
call :fail "依赖安装失败，请检查网络并查看 data\startup.log。"
exit /b 1

:server_start_failed
call :fail "控制台进程无法启动，请查看 data\startup.log。"
exit /b 1

:server_timeout
call :fail "等待控制台超过 45 秒；可能是端口被占用或服务启动失败。"
exit /b 1

:fail
echo.
echo [启动失败] %~1
echo 详细日志：%START_LOG%
echo.
pause
exit /b 1

:log
>> "%START_LOG%" echo [%date% %time%] %~1
exit /b 0
