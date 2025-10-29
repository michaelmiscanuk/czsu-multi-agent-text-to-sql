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
pip install -e .

python unzip_files.py

echo Setting up VS Code workspace...
if not exist ".vscode" mkdir .vscode

echo Creating .vscode\settings.json...
(
echo {
echo     "python.defaultInterpreterPath": "./.venv/Scripts/python.exe",
echo     "python.terminal.activateEnvironment": true,
echo     "python.analysis.extraPaths": [
echo         "${workspaceFolder}"
echo     ],
echo     "python.envFile": "${workspaceFolder}/.env"
echo }
) > .vscode\settings.json

echo Setting up frontend...
cd frontend
npm install -g yarn
yarn install
@REM npm install --force
@REM npm install
npm run build
cd ..

echo Setup complete!
echo ✅ VS Code settings configured automatically
pause