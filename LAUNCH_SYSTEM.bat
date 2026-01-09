@echo off
title Smart Gate System Launcher
echo ===================================================
echo    LAUNCHING SMART GATE AI SYSTEM
echo ===================================================
echo.

:: 1. Start the API Server in a new window
echo [1/3] Starting API Server & Web Dashboard...
start "Smart Gate - API & Dashboard" cmd /k "python api_server.py"
timeout /t 3

:: 2. Start the AI Counter UI
echo [2/3] Starting AI Vehicle Counter...
start "Smart Gate - AI Engine" cmd /k "python vehicle_counter.py"
timeout /t 3

:: 3. Start the Remote Access Tunnel
echo [3/3] Starting Remote Access (Cloudfare Tunnel)...
echo.
echo IMPORTANT: Copy the .trycloudflare.com link from the new window!
echo.
start "Smart Gate - Remote Access" cmd /k ".\cloudflared.exe tunnel --url http://localhost:8000"

echo.
echo ===================================================
echo ALL SYSTEMS STARTING...
echo KEEP ALL TERMINAL WINDOWS OPEN TO RUN THE SYSTEM.
echo ===================================================
pause
