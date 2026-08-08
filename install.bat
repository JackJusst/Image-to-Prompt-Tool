@echo off
cd /d %~dp0
echo Installiere Python-Pakete...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
echo.
echo Fertig. Danach run.bat starten.
pause
