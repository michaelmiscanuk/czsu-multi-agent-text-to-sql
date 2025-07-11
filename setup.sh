#!/bin/bash

if [ ! -d ".venv" ]; then
    echo "Creating new virtual environment..."
    uv venv --python 3.11.9
    echo "Installing backend dependencies for the first time..."
else
    echo "Virtual environment already exists, checking for updates..."
fi

echo "Activating venv..."
source .venv/bin/activate

echo "Installing/Updating backend packages..."
uv pip install .
uv pip install .[dev]

python unzip_files.py

echo "Setting up frontend..."
cd frontend
npm install
npm run build
cd ..

echo "Setup complete!"
read -p "Press enter to continue..." 