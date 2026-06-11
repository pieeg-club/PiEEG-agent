@echo off
REM ============================================================
REM PiEEG Agent: Windows Installer
REM
REM Usage: Double-click or run from Command Prompt
REM
REM What it does:
REM   1. Checks for Python 3.10+
REM   2. Creates a Python virtual environment
REM   3. Installs pieeg-agent dependencies
REM   4. Creates a launch shortcut (pieeg-agent.cmd)
REM
REM Note: Frontend is prebuilt — no Node.js required.
REM ============================================================

setlocal enabledelayedexpansion

REM Change to script directory
cd /d "%~dp0"

echo.
echo ╔══════════════════════════════════════════╗
echo ║       PiEEG Agent Installer (Windows)   ║
echo ╚══════════════════════════════════════════╝
echo.

REM ============================================================
REM Step 1: Check Python
REM ============================================================
echo [1/5] Checking Python...

python --version >nul 2>&1
if errorlevel 1 (
    echo   ✗ Python not found
    echo.
    echo   Please install Python 3.10+ from https://www.python.org/downloads/
    echo   Make sure to check "Add Python to PATH" during installation
    echo.
    pause
    exit /b 1
)

REM Get Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PY_VERSION=%%i
echo   ✓ Python %PY_VERSION%

REM Check Python version is 3.10+
for /f "tokens=1,2 delims=." %%a in ("%PY_VERSION%") do (
    set PY_MAJOR=%%a
    set PY_MINOR=%%b
)

if %PY_MAJOR% LSS 3 (
    echo   ✗ Python %PY_VERSION% is too old. Python 3.10+ required.
    pause
    exit /b 1
)

if %PY_MAJOR% EQU 3 if %PY_MINOR% LSS 10 (
    echo   ✗ Python %PY_VERSION% is too old. Python 3.10+ required.
    pause
    exit /b 1
)

REM ============================================================
REM Step 2: Create venv and install Python packages
REM ============================================================
echo.
echo [2/3] Installing pieeg-agent...

if not exist ".venv" (
    echo   Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo   ✗ Failed to create virtual environment
        pause
        exit /b 1
    )
)

call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo   ✗ Failed to activate virtual environment
    pause
    exit /b 1
)

echo   Upgrading pip...
python -m pip install --upgrade pip -q

echo   Installing dependencies...
pip install -e ".[web,dev]" -q
if errorlevel 1 (
    echo   ✗ Failed to install pieeg-agent
    pause
    exit /b 1
)

echo   ✓ Installed to: %CD%\.venv

REM Check frontend is present
if exist "frontend\dist" (
    echo   ✓ Frontend: prebuilt React app included
) else (
    echo   ⚠ Frontend dist\ not found. Run: cd frontend ^&^& npm install ^&^& npm run build
)

REM ============================================================
REM Step 3: Create launcher
REM ============================================================
echo.
echo [3/3] Creating launcher...

REM Create a batch script to launch pieeg-agent
echo @echo off > pieeg-agent.cmd
echo cd /d "%%~dp0" >> pieeg-agent.cmd
echo call .venv\Scripts\activate.bat >> pieeg-agent.cmd
echo python -m pieeg_agent %%* >> pieeg-agent.cmd

echo   ✓ Created launcher: pieeg-agent.cmd

REM ============================================================
REM Done!
REM ============================================================
echo.
echo === Setup complete! ===
echo.
echo   Start the web interface:
echo     pieeg-agent.cmd web
echo.
echo   Dashboard: http://localhost:8080
echo.
echo   For LSL stream integration, connect to a PiEEG-server or any LSL source.
echo.
echo   Explore commands:
echo     pieeg-agent.cmd --help
echo.
pause
