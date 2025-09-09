@echo off

REM Run startup debug test with virtual environment
echo Running Startup Debug Test...
echo ================================

REM Use virtual environment Python directly
.venv\Scripts\python.exe tests\api\test_startup_debug.py

echo.
echo Test completed. Check tests\api\traceback_errors\ for detailed reports.
pause
