@echo off
setlocal
cd /d "%~dp0"
py -3.11 -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install pyinstaller pytest
pytest -q
if errorlevel 1 exit /b 1
pyinstaller --noconfirm --clean --onedir --windowed --name "冒险岛小游戏助手" --collect-all cv2 --collect-all pyopencl --collect-all mss --collect-all pyautogui app.py
if errorlevel 1 exit /b 1
echo.
echo Build complete: dist\冒险岛小游戏助手\冒险岛小游戏助手.exe
pause
