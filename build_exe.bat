@echo off
echo =========================================================
echo   BUILDING AI SMART VEHICLE MONITORING SYSTEM (PRO)
echo =========================================================

:: Clean up previous builds
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist FinalBuild rmdir /s /q FinalBuild

echo.
echo [1/4] Building Main Desktop App with Professional Icon...
pyinstaller --noconsole --onefile --icon="app_icon.ico" --name "AI_Smart_Vehicle_Monitor" main.py

echo.
echo [2/4] Building Camera Worker Process...
pyinstaller --noconsole --onefile --icon="app_icon.ico" --name "vehicle_counter" --hidden-import=ultralytics --hidden-import=pandas vehicle_counter.py

echo.
echo [3/4] Building API Server...
pyinstaller --noconsole --onefile --name "multi_camera_api" --hidden-import=flask_cors multi_camera_api.py

echo.
echo [4/4] Assembling Final Package...
mkdir FinalBuild

:: Move EXEs
move dist\AI_Smart_Vehicle_Monitor.exe FinalBuild\
move dist\vehicle_counter.exe FinalBuild\
move dist\multi_camera_api.exe FinalBuild\

:: Copy Essential Assets
echo Copying assets and icons...
copy yolov8n.pt FinalBuild\
copy app_icon.ico FinalBuild\
if exist dashboard xcopy /E /I dashboard FinalBuild\dashboard

:: Create Desktop Shortcut Script (PowerShell)
echo.
echo Creating Desktop Shortcut...
powershell -Command "$s=(New-Object -COM WScript.Shell).CreateShortcut('%USERPROFILE%\Desktop\AI Smart Vehicle Monitor.lnk');$s.TargetPath='%~dp0FinalBuild\AI_Smart_Vehicle_Monitor.exe';$s.WorkingDirectory='%~dp0FinalBuild';$s.IconLocation='%~dp0FinalBuild\AI_Smart_Vehicle_Monitor.exe,0';$s.Save()"

:: Cleanup temporary pyinstaller folders
rmdir /s /q build
rmdir /s /q dist

echo.
echo =========================================================
echo            BUILD SUCCESSFUL!
echo =========================================================
echo.
echo Your professional application is ready in the 'FinalBuild' folder.
echo A shortcut has also been created on your Desktop!
echo.
echo INSTRUCTIONS:
echo 1. Keep all files in the 'FinalBuild' folder together.
echo 2. Run 'AI_Smart_Vehicle_Monitor.exe' or use the Desktop icon.
echo.
pause
