@echo off

:: If running inside PowerShell, re-launch in cmd.exe
if defined PSModulePath if not defined __RUN_CMD (
    set "__RUN_CMD=1"
    cmd /c "%~f0" %*
    exit /b %errorlevel%
)

chcp 65001 >nul
cd /d "%~dp0"

:: Check venv exists
if not exist "venv\Scripts\python.exe" (
    echo [ERROR] 仮想環境が見つかりません。先に install.bat を実行してください。
    pause
    exit /b 1
)

:: Set HuggingFace cache
set "HF_HOME=%~dp0.hf_cache"

:: Launch WebUI
echo See-through WebUI を起動しています...
call venv\Scripts\python.exe tools\webui.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] WebUI がエラーで終了しました。
    pause
)
