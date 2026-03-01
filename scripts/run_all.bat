@echo off

echo Starting BluePy Services...
echo.

if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run: python -m venv venv
    echo Then: pip install -r requirements.txt
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

echo [1/2] Starting Backend API on port 8000...
start "BluePy Backend" cmd /k "venv\Scripts\activate.bat && uvicorn backend.main:app

timeout /t 3 /nobreak > nul

echo [2/2] Starting Frontend on port 8501...
start "BluePy Frontend" cmd /k "venv\Scripts\activate.bat && streamlit run frontend/app.py

echo.
echo ========================================
echo BluePy Services Started!
echo ========================================
echo Frontend: http://localhost:8501
echo Backend API: http://localhost:8000
echo API Docs: http://localhost:8000/docs
echo ========================================
echo.
echo Press any key to stop all services...
pause > nul

taskkill /FI "WindowTitle eq BluePy Backend*" /T /F > nul 2>&1
taskkill /FI "WindowTitle eq BluePy Frontend*" /T /F > nul 2>&1

echo Services stopped.
pause
