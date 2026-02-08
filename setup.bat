@echo off
:: BluePy Complete Setup Script for Windows
:: This script sets up the database, generates sample data, and runs the application

echo ========================================
echo BluePy - Complete Setup Script
echo ========================================
echo.

:: Check if PostgreSQL is accessible
echo [1/6] Checking PostgreSQL...
psql --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo PostgreSQL not found. Attempting Docker Compose setup...
    docker-compose up -d postgres
    timeout /t 10 /nobreak >nul
) else (
    echo PostgreSQL detected!
)

:: Install Python dependencies
echo.
echo [2/6] Installing Python dependencies...
pip install -r requirements.txt

:: Initialize database
echo.
echo [3/6] Initializing database schema...
python scripts/init_db.py

:: Generate sample data
echo.
echo [4/6] Generating sample ARGO float data (375 profiles)...
python scripts/sample_data_generator.py --num-floats 15 --cycles-per-float 25 --region all

:: Generate embeddings
echo.
echo [5/6] Generating vector embeddings...
python scripts/embed_profiles.py

:: Verification
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
    echo   Backend:  uvicorn backend.main:app --reload --port 8000
    echo   Frontend: streamlit run frontend/app.py --server.port 8501
    echo.
    echo Or run both with: run_all.bat
    echo ========================================
) else (
    echo.
    echo [ERROR] Database connection failed. Please check PostgreSQL configuration.
    exit /b 1
)
