@echo off

if not exist ".venv" (
    echo Creating new virtual environment...
    uv venv --python 3.11.9
    echo Installing backend dependencies for the first time...
) else (
    echo Virtual environment already exists, checking for updates...
)

echo Activating venv...
call .venv\Scripts\activate

echo Installing/Updating backend packages...
uv pip install .
uv pip install .[dev]

python unzip_files.py

echo Setting up frontend...
cd frontend
npm install
npm run build
cd ..

echo Setup complete!
pause
