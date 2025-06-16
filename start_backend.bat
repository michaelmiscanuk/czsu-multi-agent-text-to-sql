@echo off

REM Start backend server with memory-optimized settings and reload for development
call .venv\Scripts\activate
python -m uvicorn api_server:app --host 0.0.0.0 --port 8000 --log-level info --reload
cmd /k 