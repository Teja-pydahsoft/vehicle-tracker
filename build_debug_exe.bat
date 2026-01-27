@echo off
echo ===================================================
echo   BUILDING DEBUG EXE (WITH CONSOLE LOGS)
echo ===================================================

if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

echo Building...
:: Note: --console (default) instead of --noconsole allows you to see logs
pyinstaller --onefile --name "VehicleDetection_DEBUG" ^
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
echo   DEBUG BUILD SUCCESSFUL!
echo ===================================================
echo.
echo Your app is: dist\VehicleDetection_DEBUG.exe
echo Run this to see backend logs in the black console window.
pause
