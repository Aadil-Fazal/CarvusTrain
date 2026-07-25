@echo off
setlocal enabledelayedexpansion
pushd %~dp0

REM ============================================
REM  Publish CarvusTrain to PyPI
REM ============================================

REM Read version dynamically from version.py
for /f "tokens=2 delims==" %%a in ('findstr /b "__version__" CarvusTrain\version.py') do set VERSION=%%a
set VERSION=%VERSION: =%
set VERSION=%VERSION:"=%
set VERSION=%VERSION:'=%

echo ============================================
echo  CarvusTrain v%VERSION% - PyPI Publisher
echo ============================================
echo.

REM Verify token is set
if "%PYPI_TOKEN%"=="" (
    echo [INFO] PYPI_TOKEN not set. Paste your token below.
    echo.
    set /p PYPI_TOKEN="PyPI API Token: "
    echo.
)

REM Step 1: Clean old builds
echo [1/4] Cleaning old builds...
if exist dist rmdir /s /q dist
if exist build rmdir /s /q build
for /d %%d in (*.egg-info) do rmdir /s /q "%%d"

REM Step 2: Ensure build tools
echo [2/4] Installing build tools...
python -m pip install --quiet build twine
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install build tools.
    pause
    exit /b 1
)

REM Step 3: Build
echo [3/4] Building v%VERSION% wheel...
python -m build
if %errorlevel% neq 0 (
    echo [ERROR] Build failed!
    pause
    exit /b 1
)

REM Step 4: Upload
echo [4/4] Uploading to PyPI...
python -m twine upload dist/* --username __token__ --password "!PYPI_TOKEN!"
if %errorlevel% neq 0 (
    echo [ERROR] Upload failed!
    pause
    exit /b 1
)

echo.
echo ============================================
echo  [SUCCESS] CarvusTrain v%VERSION% published!
echo ============================================
echo.
echo  Anyone can now install:
echo    pip install carvustrain
echo.

pause
