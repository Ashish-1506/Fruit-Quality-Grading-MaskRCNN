@echo off
setlocal

title Fruit Quality Grading Dashboard - Setup and Launch
echo =====================================================================
echo  Fruit Quality Grading and Surface Defect Quantification Dashboard
echo =====================================================================
echo.

cd /d "%~dp0"

:: 1. Check Python installation
echo [*] Checking Python installation...
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python is not found in PATH!
    echo Please install Python 3.10 or 3.11 from python.org and add it to your PATH.
    echo.
    pause
    exit /b 1
)

python --version
echo.

:: 2. Create or verify virtual environment
if not exist ".venv\Scripts\activate.bat" (
    echo [*] Creating project-local virtual environment: .venv ...
    python -m venv --system-site-packages .venv
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to create virtual environment!
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created.
) else (
    echo [*] Found existing virtual environment: .venv
)

:: 3. Activate virtual environment
call .venv\Scripts\activate.bat
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to activate virtual environment!
    pause
    exit /b 1
)

:: 4. Install / Verify dependencies
echo [*] Checking required dependencies...
python -c "import streamlit, torch, torchvision, PIL, numpy, pandas, matplotlib" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [*] Installing dependencies from requirements.txt...
    python -m pip install -r requirements.txt
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Dependency installation encountered issues.
        pause
        exit /b 1
    )
) else (
    echo [OK] All dependencies are verified.
)

:: 5. Device check
echo.
echo [*] Checking hardware acceleration...
python -c "import torch; print('    Compute Device:', 'CUDA GPU (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU (Local Execution)')"
echo.

:: 6. Checkpoint check
if not exist "models\best_maskrcnn_cbam_fpn.pth" (
    echo [NOTE] Trained checkpoint not found at:
    echo        models\best_maskrcnn_cbam_fpn.pth
    echo        The dashboard will start with guidance on where to place it.
    echo.
) else (
    echo [OK] Trained checkpoint found at models\best_maskrcnn_cbam_fpn.pth.
    echo.
)

:: 7. Launch Streamlit Dashboard
echo =====================================================================
echo  Launching Streamlit Dashboard...
echo  The application will open in your default browser.
echo  Press Ctrl+C in this window to stop the server.
echo =====================================================================
echo.

python -m streamlit run app.py --server.port=8501 --browser.gatherUsageStats=false

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Streamlit exited with an error code: %ERRORLEVEL%
    pause
)
