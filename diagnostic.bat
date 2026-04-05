@echo off
chcp 65001 >nul
cd /d "%~dp0"

set "OUT=%~dp0diagnostic.txt"

echo See-through WebUI Diagnostic > "%OUT%"
echo ============================== >> "%OUT%"
echo Date: %date% %time% >> "%OUT%"
echo. >> "%OUT%"

echo --- OS --- >> "%OUT%"
ver >> "%OUT%"
echo. >> "%OUT%"

echo --- Current Directory --- >> "%OUT%"
echo %cd% >> "%OUT%"
echo. >> "%OUT%"

echo --- GPU --- >> "%OUT%"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>&1 >> "%OUT%"
if %errorlevel% neq 0 (
    echo nvidia-smi failed, trying basic: >> "%OUT%"
    nvidia-smi 2>&1 | findstr /i "NVIDIA driver" >> "%OUT%"
)
echo. >> "%OUT%"

echo --- Python (py launcher) --- >> "%OUT%"
where py 2>&1 >> "%OUT%"
py --list 2>&1 >> "%OUT%"
echo. >> "%OUT%"

echo --- Python (PATH) --- >> "%OUT%"
where python 2>&1 >> "%OUT%"
python --version 2>&1 >> "%OUT%"
echo. >> "%OUT%"

echo --- Python312 default path --- >> "%OUT%"
set "PY312=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
echo Expected: %PY312% >> "%OUT%"
if exist "%PY312%" (
    echo EXISTS: YES >> "%OUT%"
    "%PY312%" --version >> "%OUT%" 2>&1
    "%PY312%" -c "import venv; print('venv module: OK')" >> "%OUT%" 2>&1
) else (
    echo EXISTS: NO >> "%OUT%"
)
echo. >> "%OUT%"

echo --- venv folder --- >> "%OUT%"
if exist "venv" (
    echo venv folder: EXISTS >> "%OUT%"
    if exist "venv\Scripts\python.exe" (
        echo venv\Scripts\python.exe: EXISTS >> "%OUT%"
        venv\Scripts\python.exe --version >> "%OUT%" 2>&1
    ) else (
        echo venv\Scripts\python.exe: MISSING >> "%OUT%"
        dir /b venv 2>&1 >> "%OUT%"
    )
) else (
    echo venv folder: NOT FOUND >> "%OUT%"
)
echo. >> "%OUT%"

echo --- Folder contents --- >> "%OUT%"
dir /b 2>&1 >> "%OUT%"
echo. >> "%OUT%"

echo --- Conda check --- >> "%OUT%"
where conda 2>&1 >> "%OUT%"
if defined CONDA_DEFAULT_ENV (
    echo CONDA_DEFAULT_ENV=%CONDA_DEFAULT_ENV% >> "%OUT%"
) else (
    echo CONDA: not active >> "%OUT%"
)
echo. >> "%OUT%"

echo --- Environment Variables --- >> "%OUT%"
echo PATH=%PATH% >> "%OUT%"
echo LOCALAPPDATA=%LOCALAPPDATA% >> "%OUT%"
echo. >> "%OUT%"

echo --- install.log --- >> "%OUT%"
if exist "install.log" (
    type install.log >> "%OUT%"
) else (
    echo install.log not found >> "%OUT%"
)

echo.
echo Done! Results saved to:
echo   %OUT%
echo.
echo Please send this file to the developer.
pause
