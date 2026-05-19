@echo off
title Trading System
echo Starting Trading System...
echo.

cd /d "%~dp0"

if not exist ".venv\Scripts\uvicorn.exe" (
    echo ERROR: virtual environment not found at .venv\
    echo Create it with: py -m venv .venv ^&^& .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

start "API Server" cmd /k "chcp 65001 >nul && set PYTHONIOENCODING=utf-8 && set PYTHONUTF8=1 && .venv\Scripts\uvicorn app.main:app --host 127.0.0.1 --port 8001 --ssl-keyfile=key.pem --ssl-certfile=cert.pem"

timeout /t 3 /nobreak >nul

start "Dashboard" cmd /k "set "PYTHONPATH=%~dp0" && .venv\Scripts\streamlit run dashboard/Home.py"

timeout /t 4 /nobreak >nul

start "" http://localhost:8501

echo.
echo Both servers are starting up.
echo Dashboard will open in your browser automatically.
echo.
echo To stop everything, close the two black terminal windows.
pause
