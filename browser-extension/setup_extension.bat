@echo off
echo 🎨 TrueFace Browser Extension Setup
echo ===================================
echo.

cd /d "%~dp0"

echo Setting up your TrueFace browser extension...
echo.

echo 1. Creating extension icons...
cd icons
python create_icons.py
if %ERRORLEVEL% NEQ 0 (
    echo ⚠️ Could not create icons automatically.
    echo You can use any 16x16, 32x32, 48x48, and 128x128 PNG icons.
    echo Place them in the icons/ folder as icon16.png, icon32.png, etc.
)
cd ..

echo.
echo 2. Validating extension files...

set missing_files=0

if not exist "manifest.json" (
    echo ❌ Missing: manifest.json
    set /a missing_files+=1
)

if not exist "background.js" (
    echo ❌ Missing: background.js
    set /a missing_files+=1
)

if not exist "content.js" (
    echo ❌ Missing: content.js
    set /a missing_files+=1
)

if not exist "popup.html" (
    echo ❌ Missing: popup.html
    set /a missing_files+=1
)

if not exist "popup.js" (
    echo ❌ Missing: popup.js
    set /a missing_files+=1
)

if not exist "popup.css" (
    echo ❌ Missing: popup.css
    set /a missing_files+=1
)

if not exist "overlay.css" (
    echo ❌ Missing: overlay.css
    set /a missing_files+=1
)

if %missing_files% EQU 0 (
    echo ✅ All extension files present!
) else (
    echo ❌ Missing %missing_files% required files!
    echo Please ensure all extension files are in this directory.
    pause
    exit /b 1
)

echo.
echo 3. Checking TrueFace backend...

curl -s http://localhost:8000/health >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ✅ TrueFace backend is running!
) else (
    echo ⚠️ TrueFace backend is not running.
    echo.
    echo Please start your backend first:
    echo    cd "c:\Users\ADITYA\Desktop\TrueFace\TrueFace-Backend"
    echo    python main.py
    echo.
    set /p continue="Continue anyway? (y/n): "
    if /i not "%continue%"=="y" exit /b 1
)

echo.
echo 4. Extension ready for installation!
echo.
echo 📋 Installation Instructions:
echo.
echo   Step 1: Open Chrome and go to chrome://extensions/
echo   Step 2: Enable "Developer mode" (toggle in top-right)
echo   Step 3: Click "Load unpacked"
echo   Step 4: Select this folder: %CD%
echo   Step 5: Pin the extension to your toolbar
echo   Step 6: Configure settings by clicking the TrueFace icon
echo   Step 7: Test on Google Meet, Zoom, Teams, or WebEx
echo.

echo 🚀 Quick Install:
set /p install="Open Chrome extensions page now? (y/n): "
if /i "%install%"=="y" (
    start chrome://extensions/
    echo.
    echo Chrome extensions page opened!
    echo Follow the steps above to install TrueFace.
)

echo.
echo 📁 Extension files in this directory:
echo.
dir /b *.json *.js *.html *.css *.md 2>nul
echo.
if exist "icons" (
    echo Icons:
    dir /b icons\*.png 2>nul
)

echo.
echo 🎉 Setup complete!
echo.
echo Your TrueFace browser extension is ready to install.
echo It will provide real-time deepfake detection on video calls!
echo.

pause
