@echo off
echo ============================================
echo  ISO 9283 Analyzer -- Windows EXE Builder
echo ============================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [HATA] Python bulunamadi! https://python.org adresinden Python 3.11 yukleyin.
    pause
    exit /b 1
)

REM Install dependencies
echo [1/3] Kutuphaneler yukleniyor...
python -m pip install --upgrade pip
pip install pandas numpy matplotlib openpyxl pyinstaller
if errorlevel 1 (
    echo [HATA] Paket yukleme basarisiz!
    pause
    exit /b 1
)

REM Clean previous build
echo [2/3] Onceki build temizleniyor...
if exist dist rmdir /s /q dist
if exist build rmdir /s /q build

REM Build EXE
echo [3/3] EXE olusturuluyor...
pyinstaller iso9283_analyzer.spec
if errorlevel 1 (
    echo [HATA] PyInstaller build basarisiz!
    pause
    exit /b 1
)

echo.
echo ============================================
echo  BASARILI! EXE dosyasi: dist\ISO9283_Analyzer.exe
echo ============================================
start dist
pause
