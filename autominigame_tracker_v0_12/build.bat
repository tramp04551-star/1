@echo off
setlocal
if not exist .venv\Scripts\python.exe (
  py -3.12 -m venv .venv
  if errorlevel 1 exit /b 1
)
.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 exit /b 1
.venv\Scripts\python.exe -m pip install -r requirements.txt pyinstaller
if errorlevel 1 exit /b 1
.venv\Scripts\python.exe -m PyInstaller --noconfirm --clean --windowed --name AutoMinigameTracker --collect-all cv2 --collect-all mss --add-data "assets;assets" --hidden-import win32gui --hidden-import win32con --hidden-import win32api --hidden-import pytesseract main.py
if errorlevel 1 exit /b 1
echo Build finished. Check dist\AutoMinigameTracker\
pause