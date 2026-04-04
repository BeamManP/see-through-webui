@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

echo ============================================================
echo   See-through WebUI Installer
echo ============================================================
echo.

cd /d "%~dp0"

:: --- Config ---
set "PIP_CACHE_DIR=%~dp0.pip_cache"
set "LOGFILE=%~dp0install.log"

:: --- Init log ---
echo See-through WebUI Install Log > "%LOGFILE%"
echo Date: %date% %time% >> "%LOGFILE%"
echo. >> "%LOGFILE%"

:: ============================================================
:: Pre-flight checks
:: ============================================================

:: --- Check 0a: NVIDIA GPU ---
echo [0/7] Checking NVIDIA GPU ...
nvidia-smi >nul 2>&1
if not %errorlevel%==0 goto :err_no_gpu
for /f "tokens=*" %%g in ('nvidia-smi --query-gpu=name --format=csv,noheader 2^>nul') do (
    echo   GPU: %%g
    echo   GPU: %%g >> "%LOGFILE%"
)
echo   OK
echo.

:: --- Check 0b: Disk space (need ~15GB) ---
echo [0/7] Checking disk space ...
for /f "tokens=3" %%f in ('dir /-C "%~dp0." 2^>nul ^| findstr /C:"bytes free"') do set "FREE_BYTES=%%f"
:: Rough check: 15GB = 15000000000 bytes. We check if first 2 digits > 15 (for 10+ digit numbers)
set "FREE_DISPLAY="
if defined FREE_BYTES (
    set "FREE_GB="
    for /f %%n in ('powershell -Command "[math]::Floor(%FREE_BYTES% / 1GB)"') do set "FREE_GB=%%n"
    echo   Free: !FREE_GB! GB >> "%LOGFILE%"
    if !FREE_GB! LSS 15 goto :err_disk_space
    echo   Free: !FREE_GB! GB
)
echo   OK
echo.

:: ============================================================
:: Step 1: Find Python 3.10-3.12
:: ============================================================
echo [1/7] Python ...

set "PYTHON_CMD="

:: Check py launcher (most reliable on Windows)
where py >nul 2>&1
if not %errorlevel%==0 goto :check_path_python

:: Try 3.12
py -3.12 --version >nul 2>&1
if %errorlevel%==0 (
    set "PYTHON_CMD=py -3.12"
    echo   OK: Python 3.12
    goto :python_ok
)

:: Try 3.11
py -3.11 --version >nul 2>&1
if %errorlevel%==0 (
    set "PYTHON_CMD=py -3.11"
    echo   OK: Python 3.11
    goto :python_ok
)

:: Try 3.10
py -3.10 --version >nul 2>&1
if %errorlevel%==0 (
    set "PYTHON_CMD=py -3.10"
    echo   OK: Python 3.10
    goto :python_ok
)

:check_path_python
:: Check python in PATH
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
:: Python not found - download and install
echo.
echo   Python 3.10+ not found. Installing Python 3.12...
echo.

set "PY_INSTALLER=python-3.12.9-amd64.exe"
set "PY_URL=https://www.python.org/ftp/python/3.12.9/%PY_INSTALLER%"

echo   Downloading...
powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%PY_URL%' -OutFile '%PY_INSTALLER%' -UseBasicParsing"
if not exist "%PY_INSTALLER%" goto :err_python_dl

echo   Installing...
"%PY_INSTALLER%" /passive InstallAllUsers=0 PrependPath=1 Include_launcher=1 Include_pip=1
if %errorlevel% neq 0 goto :err_python_install
del "%PY_INSTALLER%" 2>nul
set "PATH=%LOCALAPPDATA%\Programs\Python\Python312;%LOCALAPPDATA%\Programs\Python\Python312\Scripts;%PATH%"
set "PYTHON_CMD=py -3.12"
echo   Python 3.12 installed.

:python_ok
echo   PYTHON_CMD=!PYTHON_CMD! >> "%LOGFILE%"
echo.

:: ============================================================
:: Step 2: Create venv
:: ============================================================
echo [2/7] venv ...
if exist "venv\Scripts\python.exe" (
    echo   OK: venv exists.
    goto :venv_ok
)
%PYTHON_CMD% -m venv venv
if not exist "venv\Scripts\python.exe" goto :err_venv
echo   OK: venv created.
:venv_ok
echo.

:: ============================================================
:: Step 3: Upgrade pip
:: ============================================================
echo [3/7] pip ...
call venv\Scripts\python.exe -m pip install --upgrade pip >> "%LOGFILE%" 2>&1
echo   OK
echo.

:: ============================================================
:: Step 4: Install PyTorch
:: ============================================================
echo [4/7] PyTorch + CUDA 12.8 ...
echo   (first time: several minutes)
call venv\Scripts\pip.exe install ^
    torch==2.8.0+cu128 ^
    torchvision==0.23.0+cu128 ^
    torchaudio==2.8.0+cu128 ^
    --index-url https://download.pytorch.org/whl/cu128 >> "%LOGFILE%" 2>&1
if %errorlevel% neq 0 goto :err_pytorch
echo   OK
echo.

:: ============================================================
:: Step 5: Install dependencies
:: ============================================================
echo [5/7] dependencies ...

:: Install common and annotators as editable packages (local modules)
echo   Installing common/annotators... >> "%LOGFILE%"
call venv\Scripts\pip.exe install -e ./common -e ./annotators >> "%LOGFILE%" 2>&1
if %errorlevel% neq 0 goto :err_deps

:: Install WebUI requirements
echo   Installing webui requirements... >> "%LOGFILE%"
call venv\Scripts\pip.exe install -r webui\requirements.txt >> "%LOGFILE%" 2>&1
if %errorlevel% neq 0 goto :err_deps

:: Handle assets folder (symlink alternative)
if not exist "assets" (
    echo   Copying assets...
    xcopy /E /I /Q "common\assets" "assets" >nul 2>nul
)

echo   OK
echo.

:: ============================================================
:: Step 6: Verify CUDA
:: ============================================================
echo [6/7] Verifying CUDA ...
call venv\Scripts\python.exe -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'  CUDA: {torch.version.cuda}  GPU: {torch.cuda.get_device_name(0)}')" 2>> "%LOGFILE%"
if %errorlevel% neq 0 goto :err_cuda
echo   OK
echo.

:: ============================================================
:: Step 7: Download NF4 models
:: ============================================================
echo [7/7] NF4 models ...
echo   (first time: ~3GB download)
set "HF_HOME=%~dp0.hf_cache"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"

call venv\Scripts\python.exe -c ^
    "import os; os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING']='1'; from huggingface_hub import snapshot_download; print('  LayerDiff NF4...'); snapshot_download('24yearsold/seethroughv0.0.2_layerdiff3d_nf4', cache_dir=r'%HF_HOME%', local_dir_use_symlinks=False); print('  Marigold NF4...'); snapshot_download('24yearsold/seethroughv0.0.1_marigold_nf4', cache_dir=r'%HF_HOME%', local_dir_use_symlinks=False); print('  OK')" 2>> "%LOGFILE%"

if %errorlevel% neq 0 goto :err_model
echo.

:: --- Done ---
echo ============================================================
echo.
echo   Install complete!
echo.
echo   To start: double-click run.bat
echo.
echo ============================================================
echo.
echo   Log saved to: install.log
pause
exit /b 0

:: ============================================================
:: Error handlers (outside parentheses to avoid encoding issues)
:: ============================================================

:err_no_gpu
echo.
echo   [ERROR] NVIDIA GPU not detected.
echo   This tool requires an NVIDIA GPU with CUDA support.
echo   If you have an NVIDIA GPU, update your driver:
echo   https://www.nvidia.com/drivers
echo.
echo   Log saved to: install.log
pause
exit /b 1

:err_disk_space
echo.
echo   [ERROR] Not enough disk space. Need at least 15 GB free.
echo   Current free: !FREE_GB! GB
echo.
echo   Log saved to: install.log
pause
exit /b 1

:err_python_dl
echo.
echo   [ERROR] Download failed. Install Python manually:
echo   https://www.python.org/downloads/
echo   Log saved to: install.log
pause
exit /b 1

:err_python_install
echo   [ERROR] Python install failed. Install manually.
del "%PY_INSTALLER%" 2>nul
echo   Log saved to: install.log
pause
exit /b 1

:err_venv
echo.
echo   [ERROR] Failed to create venv.
echo   If conda is active, open a new Command Prompt and try again.
echo   Log saved to: install.log
pause
exit /b 1

:err_pytorch
echo   [ERROR] PyTorch install failed. Check install.log for details.
echo   Log saved to: install.log
pause
exit /b 1

:err_deps
echo   [ERROR] Dependency install failed. Check install.log for details.
echo   Log saved to: install.log
pause
exit /b 1

:err_cuda
echo.
echo   [ERROR] CUDA is not available after installing PyTorch.
echo   Make sure you have an NVIDIA GPU and an up-to-date driver:
echo   https://www.nvidia.com/drivers
echo   Log saved to: install.log
pause
exit /b 1

:err_model
echo.
echo   [ERROR] Model download failed. Check install.log for details.
echo   You can retry by running install.bat again.
echo   Log saved to: install.log
pause
exit /b 1
