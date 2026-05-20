@echo off
title Trading System
echo Starting Trading System...
echo.

cd /d "%~dp0"

REM --- Sanity checks ---
if not exist ".venv\Scripts\uvicorn.exe" (
    echo ERROR: virtual environment not found at .venv\
    echo Create it with: py -m venv .venv ^&^& .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

if not exist ".venv\Scripts\streamlit.exe" (
    echo ERROR: streamlit not installed in .venv\
    echo Install with: .venv\Scripts\pip install streamlit
    pause
    exit /b 1
)

if not exist "cert.pem" (
    echo ERROR: cert.pem not found. TLS keypair required.
    pause
    exit /b 1
)
if not exist "key.pem" (
    echo ERROR: key.pem not found. TLS keypair required.
    pause
    exit /b 1
)

if not exist "logs\" mkdir logs

REM --- Set env vars in this shell so child windows inherit them ---
REM UTF-8 codepage so Unicode log chars don't crash on Windows consoles.
REM PYTHONPATH lets dashboard pages import the top-level dashboard modules.
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set "PYTHONPATH=%CD%"

REM --- Launch API server window ---
REM cmd /k keeps the window open on crash so we can read the stack trace.
REM The app already writes to logs\trading.log via Python's RotatingFileHandler,
REM so no shell-side tee is needed (avoids nested-quote breakage that previously
REM kept the dashboard window from spawning).
start "API Server" cmd /k ".venv\Scripts\uvicorn.exe app.main:app --host 127.0.0.1 --port 8001 --ssl-keyfile=key.pem --ssl-certfile=cert.pem"

REM --- Wait up to 30s for /health to return 200 ---
REM Uses curl (bundled with Windows 10+) instead of Invoke-WebRequest because
REM Windows PowerShell 5.1's WebRequest can't reliably handshake our self-signed
REM TLS cert, which made the old loop time out even when the API was healthy.
echo Waiting for API to come up on https://127.0.0.1:8001 ...
set /a _tries=0
:waitloop
set /a _tries+=1
curl.exe -sk --max-time 2 -o nul -w "%%{http_code}" https://127.0.0.1:8001/health 2>nul | findstr /b "200" >nul
if %errorlevel%==0 goto apiready
if %_tries% geq 15 goto apifailed
timeout /t 2 /nobreak >nul
goto waitloop

:apifailed
echo.
echo ERROR: API did not come up within 30 seconds.
echo Check the "API Server" window for the stack trace, or read logs\trading.log
echo.
pause
exit /b 1

:apiready
echo API is up.
echo.

REM --- Launch Streamlit dashboard window ---
REM --server.headless=true stops streamlit from auto-opening its own browser tab;
REM we open the browser explicitly below once 8501 is responsive.
start "Dashboard" cmd /k ".venv\Scripts\streamlit.exe run dashboard\Home.py --server.headless=true --browser.gatherUsageStats=false"

REM --- Wait up to 20s for Streamlit to bind 8501 before opening the browser ---
echo Waiting for dashboard on http://localhost:8501 ...
set /a _dtries=0
:dashloop
set /a _dtries+=1
curl.exe -s --max-time 2 -o nul -w "%%{http_code}" http://localhost:8501 2>nul | findstr /b "200" >nul
if %errorlevel%==0 goto dashready
if %_dtries% geq 10 goto dashtimeout
timeout /t 2 /nobreak >nul
goto dashloop

:dashtimeout
echo WARN: dashboard not responding yet -- opening browser anyway.
goto openbrowser

:dashready
echo Dashboard is up.

:openbrowser
start "" http://localhost:8501

echo.
echo Both servers are running.
echo   - API:       https://127.0.0.1:8001  (window: "API Server")
echo   - Dashboard: http://localhost:8501   (window: "Dashboard")
echo.
echo To stop everything, close both terminal windows.
pause
