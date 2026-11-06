@echo off
echo =========================================================
echo   Starting P33 Diffusion Parameter Explorer Application
echo =========================================================

set VENV_PY=.\.venv\Scripts\python.exe
set STREAMLIT=.\.venv\Scripts\streamlit.exe

if not exist "%VENV_PY%" (
    echo [ERROR] Virtual environment python not found at %VENV_PY%
    pause
    exit /b 1
)

echo Launching Streamlit web dashboard on port 8501...
%STREAMLIT% run app.py --server.port 8501 --server.headless false

pause

