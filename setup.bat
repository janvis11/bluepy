@echo off

echo ========================================
echo BluePy - Complete Setup Script
echo ========================================
echo.

echo [1/6] Checking PostgreSQL...
psql
if %ERRORLEVEL% NEQ 0 (
    echo PostgreSQL not found. Attempting Docker Compose setup...
    docker-compose up -d postgres
    timeout /t 10 /nobreak >nul
) else (
    echo PostgreSQL detected!
)

echo.
echo [2/6] Installing Python dependencies...
pip install -r requirements.txt

echo.
echo [3/6] Initializing database schema...
python scripts/init_db.py

echo.
echo [4/6] Generating sample ARGO float data (375 profiles)...
python scripts/sample_data_generator.py

echo.
echo [5/6] Generating vector embeddings...
python scripts/embed_profiles.py

echo.
echo [6/6] Running verification checks...
python -c "from backend.db.database import test_connection; import sys; sys.exit(0 if test_connection() else 1)"
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo Setup Complete! Ready to launch BluePy
    echo ========================================
    echo.
    echo To start the application:
    echo   Backend:  uvicorn backend.main:app
    echo   Frontend: streamlit run frontend/app.py
    echo.
    echo Or run both with: run_all.bat
    echo ========================================
) else (
    echo.
    echo [ERROR] Database connection failed. Please check PostgreSQL configuration.
    exit /b 1
)
