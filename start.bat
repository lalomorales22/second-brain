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
curl -s -o nul --max-time 2 http://127.0.0.1:11434/api/tags >nul 2>&1 || (
  echo Starting Ollama...
  start "" "%LOCALAPPDATA%\Programs\Ollama\ollama app.exe"
  timeout /t 5 /nobreak >nul
)
echo Starting Second Brain at http://127.0.0.1:8487
".venv\Scripts\python.exe" -m semantic_gravity_memory.api.server --db "%~dp0.data\memory.db" %*
