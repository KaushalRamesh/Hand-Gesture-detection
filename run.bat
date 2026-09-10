@echo off
setlocal EnableExtensions EnableDelayedExpansion

title AI Hand Gesture Control System ^| PowerPoint ^& Video

:: ============================================================
:: AI HAND GESTURE OBJECT DETECTION SYSTEM
:: YOLOv8 + MediaPipe
:: ============================================================

cd /d "%~dp0"

echo ==============================================================================
echo       AI HAND GESTURE OBJECT DETECTION SYSTEM (YOLOv8 + MediaPipe)
echo            Touchless PowerPoint Presentation and Video Control
echo ==============================================================================
echo.

:: ------------------------------------------------------------
:: 1. Locate Python
:: ------------------------------------------------------------

echo [1/5] Checking Python installation...

set "PYTHON_CMD="

where python >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_CMD=python"
)

if not defined PYTHON_CMD (
    where py >nul 2>&1
    if not errorlevel 1 (
        set "PYTHON_CMD=py -3"
    )
)

if not defined PYTHON_CMD (
    echo.
    echo [ERROR] Python was not found!
    echo Please install Python 3.11 or newer.
    echo.
    pause
    exit /b 1
)

echo [INFO] Python command: %PYTHON_CMD%

%PYTHON_CMD% --version

if errorlevel 1 (
    echo [ERROR] Python could not be started.
    pause
    exit /b 1
)

echo.

:: ------------------------------------------------------------
:: 2. Check Python version
:: ------------------------------------------------------------

%PYTHON_CMD% -c "import sys; print('[INFO] Python version:', sys.version)"

%PYTHON_CMD% -c "import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)"

if errorlevel 1 (
    echo.
    echo [ERROR] Python 3.11 or newer is required.
    pause
    exit /b 1
)

echo [OK] Python version is supported.
echo.

:: ------------------------------------------------------------
:: 3. Check virtual environment
:: ------------------------------------------------------------

echo [2/5] Checking environment...

if exist ".venv\Scripts\python.exe" (
    echo [INFO] Using .venv
    set "PYTHON_CMD=.venv\Scripts\python.exe"
) else if exist "venv\Scripts\python.exe" (
    echo [INFO] Using venv
    set "PYTHON_CMD=venv\Scripts\python.exe"
) else (
    echo [INFO] No virtual environment found.
    echo [INFO] Using system Python.
)

echo [INFO] Python executable:
%PYTHON_CMD% -c "import sys; print(sys.executable)"

echo.

:: ------------------------------------------------------------
:: 4. Verify dependencies
:: ------------------------------------------------------------

echo [3/5] Checking dependencies...

%PYTHON_CMD% -c "import cv2; print('[OK] OpenCV:', cv2.__version__)"
if errorlevel 1 goto INSTALL_DEPS

%PYTHON_CMD% -c "import mediapipe; print('[OK] MediaPipe')"
if errorlevel 1 goto INSTALL_DEPS

%PYTHON_CMD% -c "import ultralytics; print('[OK] Ultralytics')"
if errorlevel 1 goto INSTALL_DEPS

%PYTHON_CMD% -c "import pyautogui; print('[OK] PyAutoGUI')"
if errorlevel 1 goto INSTALL_DEPS

%PYTHON_CMD% -c "import PySide6; print('[OK] PySide6')"
if errorlevel 1 goto INSTALL_DEPS

%PYTHON_CMD% -c "import torch; print('[OK] PyTorch:', torch.__version__)"
if errorlevel 1 goto INSTALL_DEPS

echo.
echo [OK] All required dependencies are installed.
goto ASSETS


:INSTALL_DEPS

echo.
echo [INFO] One or more dependencies are missing.
echo [INFO] Installing from requirements.txt...
echo.

if not exist "requirements.txt" (
    echo [ERROR] requirements.txt was not found!
    echo Expected location:
    echo %CD%\requirements.txt
    echo.
    pause
    exit /b 1
)

%PYTHON_CMD% -m pip install --upgrade pip

if errorlevel 1 (
    echo [ERROR] Failed to upgrade pip.
    pause
    exit /b 1
)

%PYTHON_CMD% -m pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo ==============================================================================
    echo [ERROR] Dependency installation failed.
    echo ==============================================================================
    echo.
    echo Try running:
    echo %PYTHON_CMD% -m pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

echo.
echo [OK] Dependencies installed successfully.
echo.


:ASSETS

:: ------------------------------------------------------------
:: 5. Check assets
:: ------------------------------------------------------------

echo [4/5] Checking assets...

if not exist "assets" (
    echo [INFO] Creating assets directory...
    mkdir "assets"
)

if not exist "assets\sample_video.mp4" (
    echo [INFO] Generating bundled demonstration video...

    if not exist "assets\generate_sample_video.py" (
        echo [ERROR] assets\generate_sample_video.py was not found!
        pause
        exit /b 1
    )

    %PYTHON_CMD% "assets\generate_sample_video.py"

    if errorlevel 1 (
        echo.
        echo [ERROR] Failed to generate sample video.
        pause
        exit /b 1
    )
)

if not exist "assets\sample_video.mp4" (
    echo [ERROR] sample_video.mp4 was not created.
    pause
    exit /b 1
)

echo [OK] Assets verified.
echo.

:: ------------------------------------------------------------
:: Launch application
:: ------------------------------------------------------------

echo [5/5] Starting AI Gesture Control Dashboard...
echo ------------------------------------------------------------------------------
echo Application is running.
echo Close the dashboard window or press Ctrl+C to exit.
echo ------------------------------------------------------------------------------
echo.

if not exist "main.py" (
    echo [ERROR] main.py was not found!
    echo Expected:
    echo %CD%\main.py
    echo.
    pause
    exit /b 1
)

%PYTHON_CMD% "main.py"

set "EXIT_CODE=%ERRORLEVEL%"

echo.

if not "%EXIT_CODE%"=="0" (
    echo ==============================================================================
    echo [ERROR] Application exited with error code %EXIT_CODE%.
    echo ==============================================================================
    echo.
    pause
    exit /b %EXIT_CODE%
)

echo Application closed normally.

endlocal
exit /b 0
