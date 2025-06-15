#!/bin/bash

# Start backend server with memory-optimized settings
source .venv/bin/activate
uvicorn api_server:app --host 0.0.0.0 --port 8000 --workers 1 --limit-max-requests 100 --timeout-keep-alive 5 --limit-concurrency 200 --backlog 50 --reload --reload-exclude .venv
$SHELL 