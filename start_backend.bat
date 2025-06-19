@echo off

REM Start backend server with memory-optimized settings and reload for development
REM CRITICAL: Use custom uvicorn startup script for PostgreSQL compatibility on Windows
call .venv\Scripts\activate
python uvicorn_start.py
cmd /k 