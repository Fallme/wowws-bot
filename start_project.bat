@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul
title 战舰世界自动战斗控制台
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\start_project.ps1" %*
set "START_EXIT_CODE=%errorlevel%"
if not "%START_EXIT_CODE%"=="0" (
    echo.
    echo 启动失败，详细信息见上方提示和项目 data\startup.log。
    pause
)
exit /b %START_EXIT_CODE%
