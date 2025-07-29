@echo off

REM Start backend server with full debug output
call .venv\Scripts\activate

REM Enable debug logging
set VERBOSE_SSL_LOGGING=true
set ENABLE_CONNECTION_MONITORING=true

REM Start with debug mode
python uvicorn_start.py
cmd /k