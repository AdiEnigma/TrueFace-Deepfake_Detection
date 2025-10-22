@echo off
echo 🚀 TrueFace Browser Extension Installer
echo ========================================
echo.

cd /d "%~dp0"

echo This script will help you install the TrueFace browser extension.
echo.
echo Prerequisites:
echo ✅ TrueFace backend should be running on localhost:8000
echo ✅ Chrome or Edge browser (Chromium-based)
echo.

set /p continue="Continue with installation? (y/n): "
if /i not "%continue%"=="y" goto exit

echo.
echo 📋 Installation Steps:
echo.
echo 1. Opening Chrome Extensions page...
start chrome://extensions/

echo.
echo 2. Follow these steps in Chrome:
echo    a) Enable "Developer mode" (toggle in top-right)
echo    b) Click "Load unpacked" button
echo    c) Select this folder: %CD%
echo    d) Extension should appear in your extensions list
echo.

echo 3. Pin the extension:
echo    a) Click the extensions icon (puzzle piece) in toolbar
echo    b) Pin TrueFace extension for easy access
echo.

echo 4. Configure the extension:
echo    a) Click the TrueFace icon in toolbar
echo    b) Verify "Backend Online" status
echo    c) Adjust settings as needed
echo.

echo 5. Test on a video call:
echo    a) Go to meet.google.com, zoom.us, teams.microsoft.com, or webex.com
echo    b) Join or start a video call
echo    c) Trust scores should appear on participant videos
echo.

pause

echo.
echo 🧪 Testing Backend Connection...
echo.

curl -s http://localhost:8000/health >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ✅ Backend is running and accessible!
) else (
    echo ❌ Backend is not running or not accessible.
    echo.
    echo Please start your TrueFace backend first:
    echo    cd "c:\Users\ADITYA\Desktop\TrueFace\TrueFace-Backend"
    echo    python main.py
    echo.
    echo Then try installing the extension again.
    pause
    goto exit
)

echo.
echo 📁 Extension files in this directory:
dir /b *.json *.js *.html *.css *.md

echo.
echo 🎉 Installation guide complete!
echo.
echo Next steps:
echo 1. Load the extension in Chrome (follow steps above)
echo 2. Test on a video call platform
echo 3. Enjoy real-time deepfake detection!
echo.

:exit
echo.
echo Press any key to exit...
pause > nul
