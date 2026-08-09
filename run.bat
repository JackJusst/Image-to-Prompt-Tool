@echo off
cd /d "%~dp0"
echo Starte Image Prompt Tool...

where ollama >nul 2>nul
if not errorlevel 1 (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "try { Invoke-RestMethod -Uri 'http://localhost:11434/api/tags' -TimeoutSec 2 | Out-Null } catch { Start-Process -WindowStyle Hidden -FilePath 'ollama' -ArgumentList 'serve' }"
)

python -m streamlit run app.py
pause
