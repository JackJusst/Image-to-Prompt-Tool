@echo off
cd /d %~dp0
echo Starte Image Prompt Tool...
python -m streamlit run app.py
pause
