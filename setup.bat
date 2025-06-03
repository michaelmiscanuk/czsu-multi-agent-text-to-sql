@echo off

echo Creating venv...
uv venv --python 3.11.9

echo Activating venv...
call .venv\Scripts\activate

echo Installing backend dependencies...
uv pip install .
uv pip install .[dev]

echo Setting up frontend...
cd frontend
npm install
npm run build
cd ..

echo Setup complete!
pause