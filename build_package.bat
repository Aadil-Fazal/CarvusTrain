@echo off
setlocal
pushd %~dp0

REM ============================================
REM  Build CarvusTrain package
REM ============================================

REM Read version dynamically from version.py
for /f "tokens=2 delims==" %%a in ('findstr /b "__version__" CarvusTrain\version.py') do set VERSION=%%a
set VERSION=%VERSION: =%
set VERSION=%VERSION:"=%
set VERSION=%VERSION:'=%

echo ============================================
echo  Building CarvusTrain v%VERSION%
echo ============================================

REM Clean old build artifacts
if exist dist rmdir /s /q dist
if exist build rmdir /s /q build
for /d %%d in (*.egg-info) do rmdir /s /q "%%d"

REM Build new wheel
python -m pip install build
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install 'build' package. Check your Python/pip setup.
    pause
    exit /b 1
)
python -m build

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Build failed!
    pause
    exit /b 1
)

echo.
echo [SUCCESS] Wheel built successfully!
dir dist\

echo.
echo ============================================
echo  Installing the package...
echo ============================================

REM Uninstall old version first
python -m pip uninstall -y carvustrain 2>nul

REM Install the new wheel
for %%f in (dist\carvustrain-*.whl) do (
    python -m pip install "%%f"
)

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Installation failed!
    pause
    exit /b 1
)

echo.
echo [SUCCESS] CarvusTrain v%VERSION% installed!
echo.
echo ============================================
echo  Running verification...
echo ============================================

python test_fix.py

echo.
echo ============================================
echo  Done! You can now run:
echo    carvustrain --help
echo ============================================

pause
