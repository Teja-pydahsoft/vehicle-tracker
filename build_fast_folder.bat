@echo off
echo ===================================================
echo   BUILDING PORTABLE FOLDER (FAST BUILD)
echo ===================================================

:: Clean up old builds
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

echo Building...
:: --onedir creates a FOLDER instead of one compressed file (Much faster)
pyinstaller --console --onedir --name "VehicleDetectionSystem" ^
    --add-data "yolov8n.pt;." ^
    --add-data "dashboard;dashboard" ^
    --hidden-import=ultralytics ^
    --hidden-import=pandas ^
    --hidden-import=flask_cors ^
    --hidden-import=engineio.async_drivers.threading ^
    --hidden-import=cv2 ^
    --hidden-import=numpy ^
    --hidden-import=PIL ^
    --hidden-import=easyocr ^
    main.py

echo.
echo ===================================================
echo   BUILD SUCCESSFUL!
echo ===================================================
echo.
echo Your app is in the folder: dist\VehicleDetectionSystem
echo.
echo 1. Open 'dist\VehicleDetectionSystem'
echo 2. Run 'VehicleDetectionSystem.exe'
echo 3. You can ZIP this folder to send to clients.
echo.
pause
