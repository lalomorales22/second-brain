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
  ".venv\Scripts\python.exe" -m pip install -e ".[media,desktop,youtube]"
)
".venv\Scripts\python.exe" -c "import pypdf, imageio_ffmpeg, webview, yt_dlp" >nul 2>&1 || ".venv\Scripts\python.exe" -m pip install -e ".[media,desktop,youtube]"
".venv\Scripts\python.exe" -m semantic_gravity_memory.desktop %*
