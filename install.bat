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

:: --- Step 1: Find Python 3.10-3.12 ---
echo [1/6] Python を確認しています...

set "PYTHON_CMD="

:: Check py launcher first (most reliable on Windows)
where py >nul 2>&1
if %errorlevel%==0 (
    for /f "tokens=*" %%i in ('py -3.12 --version 2^>nul') do set "PY_VER=%%i"
    if defined PY_VER (
        set "PYTHON_CMD=py -3.12"
        echo   OK: Python 3.12 ^(py launcher^)
        goto :python_ok
    )
    for /f "tokens=*" %%i in ('py -3.11 --version 2^>nul') do set "PY_VER=%%i"
    if defined PY_VER (
        set "PYTHON_CMD=py -3.11"
        echo   OK: Python 3.11 ^(py launcher^)
        goto :python_ok
    )
    for /f "tokens=*" %%i in ('py -3.10 --version 2^>nul') do set "PY_VER=%%i"
    if defined PY_VER (
        set "PYTHON_CMD=py -3.10"
        echo   OK: Python 3.10 ^(py launcher^)
        goto :python_ok
    )
)

:: Check python in PATH
where python >nul 2>&1
if %errorlevel%==0 (
    for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set "PY_VER_STR=%%v"
    echo   PATH: python !PY_VER_STR!
    for /f "tokens=1,2 delims=." %%a in ("!PY_VER_STR!") do (
        if "%%a"=="3" if %%b GEQ 10 if %%b LEQ 12 (
            set "PYTHON_CMD=python"
            goto :python_ok
        )
    )
)

:: Python not found — download and install
echo.
echo   Python 3.12 が見つかりません。自動でインストールします...
echo.

set "PY_INSTALLER=python-3.12.9-amd64.exe"
set "PY_URL=https://www.python.org/ftp/python/3.12.9/%PY_INSTALLER%"

echo   ダウンロード中...
powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%PY_URL%' -OutFile '%PY_INSTALLER%' -UseBasicParsing"
if not exist "%PY_INSTALLER%" (
    echo.
    echo   [ERROR] ダウンロード失敗。手動でインストールしてください:
    echo   https://www.python.org/downloads/
    pause
    exit /b 1
)

echo   インストール中...
"%PY_INSTALLER%" /passive InstallAllUsers=0 PrependPath=1 Include_launcher=1 Include_pip=1
if %errorlevel% neq 0 (
    echo   [ERROR] インストール失敗。手動でインストールしてください。
    del "%PY_INSTALLER%" 2>nul
    pause
    exit /b 1
)
del "%PY_INSTALLER%" 2>nul
set "PATH=%LOCALAPPDATA%\Programs\Python\Python312;%LOCALAPPDATA%\Programs\Python\Python312\Scripts;%PATH%"
set "PYTHON_CMD=py -3.12"
echo   Python 3.12 をインストールしました。

:python_ok
echo.

:: --- Step 2: Create venv ---
echo [2/6] 仮想環境を作成しています...
if not exist "venv\Scripts\python.exe" (
    %PYTHON_CMD% -m venv venv
    if not exist "venv\Scripts\python.exe" (
        echo.
        echo   [ERROR] venv の作成に失敗しました。
        echo   conda が有効な場合は、新しいコマンドプロンプトから実行してください。
        pause
        exit /b 1
    )
    echo   OK: venv を作成しました。
) else (
    echo   OK: venv は既に存在します。
)
echo.

:: --- Step 3: Upgrade pip ---
echo [3/6] pip を更新しています...
call venv\Scripts\python.exe -m pip install --upgrade pip --quiet 2>nul
echo   OK
echo.

:: --- Step 4: Install PyTorch ---
echo [4/6] PyTorch + CUDA 12.8 をインストールしています...
echo   (初回は数分〜10分かかります)
call venv\Scripts\pip.exe install ^
    torch==2.8.0+cu128 ^
    torchvision==0.23.0+cu128 ^
    torchaudio==2.8.0+cu128 ^
    --index-url https://download.pytorch.org/whl/cu128 ^
    --quiet
if %errorlevel% neq 0 (
    echo   [ERROR] PyTorch のインストールに失敗しました。
    echo   ネットワーク接続を確認してください。
    pause
    exit /b 1
)
echo   OK
echo.

:: --- Step 5: Install dependencies ---
echo [5/6] 依存パッケージをインストールしています...

:: Install common and annotators as editable packages (local modules)
call venv\Scripts\pip.exe install -e ./common -e ./annotators --quiet 2>nul

:: Install WebUI requirements
call venv\Scripts\pip.exe install -r webui\requirements.txt --quiet
if %errorlevel% neq 0 (
    echo   [WARNING] 一部パッケージでエラーが出ました。
)

:: Handle assets folder (symlink alternative)
if not exist "assets" (
    echo   assets フォルダをコピーしています...
    xcopy /E /I /Q "common\assets" "assets" >nul 2>nul
)

echo   OK
echo.

:: --- Step 6: Download NF4 models ---
echo [6/6] NF4量子化モデルをダウンロードしています...
echo   (初回は ~3GB、数分かかります)
set "HF_HOME=%~dp0.hf_cache"

call venv\Scripts\python.exe -c ^
    "from huggingface_hub import snapshot_download; print('  LayerDiff NF4...'); snapshot_download('24yearsold/seethroughv0.0.2_layerdiff3d_nf4', cache_dir=r'%HF_HOME%'); print('  Marigold NF4...'); snapshot_download('24yearsold/seethroughv0.0.1_marigold_nf4', cache_dir=r'%HF_HOME%'); print('  OK')"

if %errorlevel% neq 0 (
    echo.
    echo   [WARNING] モデルのダウンロードに失敗しました。
    echo   初回起動時に自動ダウンロードされます。
)
echo.

:: --- Done ---
echo ============================================================
echo.
echo   インストール完了！
echo.
echo   起動方法: run.bat をダブルクリック
echo.
echo ============================================================
pause
