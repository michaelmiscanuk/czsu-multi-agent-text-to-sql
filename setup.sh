#!/bin/bash

# Backend dependency setup script
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install .
pip install .[dev]

python3 unzip_files.py

# Frontend dependency setup
cd frontend
npm install
npm run build
cd ..

$SHELL 