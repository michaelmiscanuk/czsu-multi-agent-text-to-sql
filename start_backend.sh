#!/bin/bash

# Start backend server with memory-optimized settings
source .venv/bin/activate
uvicorn api.main:app --host 0.0.0.0 --port 8000 --log-level info
$SHELL 