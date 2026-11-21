@echo off
rem Double click this file to start the AR Portal localization server and its panel.
rem It works from any folder and does not need the PowerShell script policy to be changed.
title AR Portal Server
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start-server.ps1" %*
echo.
echo Press any key to close this window.
pause >nul
