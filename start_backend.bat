@echo off

REM Start backend server
call .venv\Scripts\activate
uvicorn api_server:app --reload --reload-exclude .venv
cmd /k 