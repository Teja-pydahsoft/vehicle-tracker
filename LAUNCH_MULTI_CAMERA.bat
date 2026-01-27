@echo off
echo ========================================
echo  Multi-Camera Vehicle Detection System
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

echo [1/3] Checking dependencies...
pip show flask >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing required packages...
    pip install flask flask-cors opencv-python numpy
)

echo [2/3] Starting Multi-Camera API Server...
echo.
echo Dashboard will be available at: http://localhost:5000
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

REM Start the API server
python multi_camera_api.py

pause
