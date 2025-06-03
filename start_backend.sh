#!/bin/bash

# Start backend server
source .venv/bin/activate
uvicorn api_server:app --reload --reload-exclude .venv
$SHELL 