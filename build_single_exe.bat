@echo off
echo =========================================================
echo   BUILDING SINGLE-FILE AI SMART VEHICLE MONITORING SYSTEM
echo =========================================================

if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

echo Building Single EXE with Professional Icon...
pyinstaller --noconsole --onefile --icon="app_icon.ico" --name "AI_Smart_Vehicle_Monitoring_System" ^
    --add-data "yolov8n.pt;." ^
    --add-data "dashboard;dashboard" ^
    --add-data "app_icon.ico;." ^
    --add-data "custom_tracker.yaml;." ^
    --hidden-import=ultralytics ^
    --hidden-import=pandas ^
    --hidden-import=flask ^
    --hidden-import=flask_cors ^
    --hidden-import=cv2 ^
    --hidden-import=numpy ^
    --hidden-import=PIL ^
    --hidden-import=easyocr ^
    --hidden-import=PySide6 ^
    --hidden-import=sqlite3 ^
    --hidden-import=psutil ^
    main.py

:: Create Desktop Shortcut Script (PowerShell)
echo.
echo Creating Desktop Shortcut...
powershell -Command "$s=(New-Object -COM WScript.Shell).CreateShortcut('%USERPROFILE%\Desktop\AI Smart Vehicle Monitoring System.lnk');$s.TargetPath='%~dp0dist\AI_Smart_Vehicle_Monitoring_System.exe';$s.WorkingDirectory='%~dp0dist';$s.IconLocation='%~dp0dist\AI_Smart_Vehicle_Monitoring_System.exe,0';$s.Save()"

echo.
echo =========================================================
echo   BUILD SUCCESSFUL!
echo =========================================================
echo.
echo Your single file app is: dist\AI_Smart_Vehicle_Monitoring_System.exe
echo A shortcut has also been created on your Desktop!
pause
