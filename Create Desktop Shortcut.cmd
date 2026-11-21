@echo off
rem Double click: puts the shortcut "AR Portal Server" on your desktop. Nothing else is changed.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\make-desktop-shortcut.ps1"
echo.
echo Press any key to close this window.
pause >nul
