@echo off
cd /d "%~dp0"

echo ============================================
echo Lancement de l'application Streamlit...
echo ============================================

venv\Scripts\python.exe -m streamlit run app.py

pause