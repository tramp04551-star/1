@echo off
setlocal
cd /d "%~dp0"
py -3.11 -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install pyinstaller pytest
python -m compileall -q app.py tracker.py gpu_backend.py result_dialog.py
if errorlevel 1 exit /b 1
pytest -q
if errorlevel 1 exit /b 1
pyinstaller --noconfirm --clean --onedir --windowed --name "MapleStoryMiniGameAssistant" --collect-all cv2 --collect-all pyopencl --collect-all mss --collect-all pyautogui --add-data "assets;assets" app.py
if errorlevel 1 exit /b 1
echo.
echo Build complete: dist\MapleStoryMiniGameAssistant\MapleStoryMiniGameAssistant.exe
pause
