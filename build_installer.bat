@echo off
echo =========================================================
echo   BUILDING SMART INSTALLER (SMALL AND FAST DOWNLOAD)
echo =========================================================
echo.

:: Clean up previous builds
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

echo [1/1] Building Installer Binary...
:: We only bundle the installer. This file stays small because it doesn't 
:: include Torch/OpenCV yet - it downloads them during installation!
pyinstaller --noconsole --onefile --icon="app_icon.ico" --name "INSTALL_AI_SMART_MONITORING_SYSTEM" ^
    --add-data "yolov8n.pt;." ^
    --add-data "dashboard;dashboard" ^
    --add-data "app_icon.ico;." ^
    --add-data "main.py;." ^
    --add-data "vehicle_counter.py;." ^
    --add-data "multi_camera_api.py;." ^
    --add-data "generate_report.py;." ^
    --add-data "custom_tracker.yaml;." ^
    installer.py

echo.
echo =========================================================
echo            BUILD SUCCESSFUL!
echo =========================================================
echo.
echo Your smart installer is in: dist\INSTALL_AI_SMART_MONITORING_SYSTEM.exe
echo.
echo THIS FILE IS SMALL (~30MB) - Perfect for sharing! 
echo When the client runs it, it will:
echo 1. Ask for an install folder.
echo 2. Download high-performance AI libraries automatically.
echo 3. Create a Desktop Shortcut.
echo.
pause
