@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

echo ============================================================
echo   See-through WebUI Installer
echo ============================================================
echo.

cd /d "%~dp0"

:: --- Init log ---
echo See-through WebUI Install Log > install.log
echo Date: %date% %time% >> install.log
echo. >> install.log

:: ============================================================
:: Pre-flight checks
:: ============================================================

:: --- NVIDIA GPU ---
echo [0] Checking NVIDIA GPU ...
nvidia-smi >nul 2>&1
if not %errorlevel%==0 goto :err_no_gpu
echo   OK
echo.

:: --- Disk space ---
echo [0] Checking disk space ...
for /f "tokens=3" %%f in ('dir /-C "%~dp0." 2^>nul ^| findstr /C:"bytes free"') do set "FREE_BYTES=%%f"
if defined FREE_BYTES (
    for /f %%n in ('powershell -Command "[math]::Floor(%FREE_BYTES% / 1GB)"') do set "FREE_GB=%%n"
    if !FREE_GB! LSS 15 goto :err_disk_space
    echo   Free: !FREE_GB! GB
)
echo   OK
echo.

:: ============================================================
:: Find or install Python
:: ============================================================
echo [1] Python ...

set "PYTHON_CMD="

:: --- py launcher ---
where py >nul 2>&1
if not %errorlevel%==0 goto :check_path_python

py -3.12 --version >nul 2>&1
if %errorlevel%==0 (
    set "PYTHON_CMD=py -3.12"
    echo   OK: Python 3.12
    goto :python_ok
)
py -3.11 --version >nul 2>&1
if %errorlevel%==0 (
    set "PYTHON_CMD=py -3.11"
    echo   OK: Python 3.11
    goto :python_ok
)
py -3.10 --version >nul 2>&1
if %errorlevel%==0 (
    set "PYTHON_CMD=py -3.10"
    echo   OK: Python 3.10
    goto :python_ok
)

:check_path_python
where python >nul 2>&1
if not %errorlevel%==0 goto :install_python

for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set "PY_VER_STR=%%v"
echo   PATH: python !PY_VER_STR!
for /f "tokens=1,2 delims=." %%a in ("!PY_VER_STR!") do (
    if "%%a"=="3" if %%b GEQ 10 if %%b LEQ 12 (
        set "PYTHON_CMD=python"
        goto :python_ok
    )
)

:install_python
echo.
echo   Python 3.10+ not found. Installing Python 3.12 ...
echo.
set "PY_INSTALLER=python-3.12.9-amd64.exe"
set "PY_URL=https://www.python.org/ftp/python/3.12.9/%PY_INSTALLER%"

echo   Downloading ...
powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%PY_URL%' -OutFile '%PY_INSTALLER%' -UseBasicParsing"
if not exist "%PY_INSTALLER%" goto :err_python_dl

echo   Installing ...
"%PY_INSTALLER%" /passive InstallAllUsers=0 PrependPath=1 Include_launcher=1 Include_pip=1
if %errorlevel% neq 0 goto :err_python_install
del "%PY_INSTALLER%" 2>nul
set "PATH=%LOCALAPPDATA%\Programs\Python\Python312;%LOCALAPPDATA%\Programs\Python\Python312\Scripts;%PATH%"
set "PYTHON_CMD=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
echo   OK: Python 3.12 installed.

:python_ok
echo   Using: !PYTHON_CMD!
echo   PYTHON_CMD=!PYTHON_CMD! >> install.log
echo.

:: ============================================================
:: Create venv
:: ============================================================
echo [2] Creating venv ...
if exist "venv\Scripts\python.exe" (
    echo   OK: venv exists.
    goto :venv_ok
)
!PYTHON_CMD! -m venv venv
if not exist "venv\Scripts\python.exe" goto :err_venv
echo   OK: venv created.
:venv_ok
echo.

:: ============================================================
:: Hand off to Python setup script
:: ============================================================
echo [3] Running setup ...
echo.
call venv\Scripts\python.exe webui\setup.py
if %errorlevel% neq 0 goto :err_setup
echo.
echo   Log saved to: install.log
pause
exit /b 0

:: ============================================================
:: Error handlers
:: ============================================================

:err_no_gpu
echo.
echo   [ERROR] NVIDIA GPU not detected.
echo   This tool requires an NVIDIA GPU with CUDA support.
echo   https://www.nvidia.com/drivers
echo.
echo   Log: install.log
pause
exit /b 1

:err_disk_space
echo.
echo   [ERROR] Not enough disk space (need 15+ GB).
echo   Free: !FREE_GB! GB
echo.
echo   Log: install.log
pause
exit /b 1

:err_python_dl
echo.
echo   [ERROR] Python download failed.
echo   Install manually: https://www.python.org/downloads/
echo   Log: install.log
pause
exit /b 1

:err_python_install
echo   [ERROR] Python install failed.
del "%PY_INSTALLER%" 2>nul
echo   Log: install.log
pause
exit /b 1

:err_venv
echo.
echo   [ERROR] Failed to create venv.
echo   If conda is active, open a new Command Prompt and try again.
echo   Log: install.log
pause
exit /b 1

:err_setup
echo.
echo   [ERROR] Setup failed. Check install.log for details.
echo   You can retry by running install.bat again.
echo   Log: install.log
pause
exit /b 1
