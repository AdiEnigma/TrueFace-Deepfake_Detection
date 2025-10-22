@echo off
echo 🎯 MesoNet Fine-Tuning Suite
echo =============================
echo.

cd /d "%~dp0"

echo Select fine-tuning option:
echo.
echo 1. Quick Fine-Tuning (Fast, for testing)
echo 2. Advanced Fine-Tuning (Best quality, slower)
echo 3. Compare Existing Models
echo 4. Exit
echo.

set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" goto quick
if "%choice%"=="2" goto advanced
if "%choice%"=="3" goto compare
if "%choice%"=="4" goto exit

echo Invalid choice. Please try again.
pause
goto start

:quick
echo.
echo ⚡ Starting Quick Fine-Tuning...
echo This will take 10-15 minutes
echo.
python quick_fine_tune.py
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ Quick fine-tuning completed!
    echo 📁 Model saved as: models/mesonet_quick_final.h5
) else (
    echo.
    echo ❌ Quick fine-tuning failed!
)
goto end

:advanced
echo.
echo 🎯 Starting Advanced Fine-Tuning...
echo This will take 1-2 hours for best results
echo.
echo Features:
echo - Two-phase training (initial + fine-tuning)
echo - Advanced data augmentation
echo - Comprehensive evaluation
echo - Training visualizations
echo.
set /p confirm="Continue? (y/n): "
if /i not "%confirm%"=="y" goto start

python fine_tune_mesonet_advanced.py
if %ERRORLEVEL% EQU 0 (
    echo.
    echo 🎉 Advanced fine-tuning completed!
    echo 📁 Model saved as: models/mesonet_model.h5
    echo 📊 Check models/ folder for analysis plots
) else (
    echo.
    echo ❌ Advanced fine-tuning failed!
)
goto end

:compare
echo.
echo 🔍 Comparing All Models...
echo.
python compare_models.py
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ Model comparison completed!
    echo 📊 Check models/model_comparison.png for results
) else (
    echo.
    echo ❌ Model comparison failed!
)
goto end

:exit
echo.
echo 👋 Goodbye!
goto end

:end
echo.
echo Press any key to exit...
pause > nul
