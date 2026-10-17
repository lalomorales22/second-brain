@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  py -3 -m venv .venv 2>nul || python -m venv .venv
  if errorlevel 1 (
    echo Python 3.10+ is required. Install it from https://python.org and retry.
    exit /b 1
  )
  ".venv\Scripts\python.exe" -m pip install -e ".[media]"
)
".venv\Scripts\python.exe" -c "import pypdf, imageio_ffmpeg" >nul 2>&1 || ".venv\Scripts\python.exe" -m pip install -e ".[media]"
where npm >nul 2>&1 || (
  echo Node.js is required for the Electron window. Install it from https://nodejs.org and retry.
  exit /b 1
)
cd desktop
if not exist "node_modules\electron" call npm install
call npm start
