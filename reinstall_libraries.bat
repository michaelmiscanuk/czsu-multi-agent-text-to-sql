@echo off
setlocal enabledelayedexpansion

echo ========================================
echo  Reinstalling Libraries - Clean Install
echo ========================================
echo.

echo [1/2] Removing old files and folders...

REM Close VS Code instances first to prevent file locks (skip if running from VS Code)
echo   Closing VS Code instances...
powershell -Command "Get-Process 'Code*' | Stop-Process -Force" 2>nul

REM Kill any Python/Node processes that might lock folders
echo   Killing processes that may lock folders...
start /b taskkill /f /im "python.exe" 2>nul
start /b taskkill /f /im "node.exe" 2>nul

REM Give time for processes to terminate
timeout /t 2 /nobreak >nul

REM Small delay to let processes fully terminate
timeout /t 1 /nobreak >nul

REM Delete package-lock.json first (instant)
if exist "frontend\package-lock.json" del /f /q "frontend\package-lock.json" 2>nul

REM Run folder removals in parallel
start /b powershell -Command "if (Test-Path '.venv') { Remove-Item -LiteralPath '.venv' -Force -Recurse }"
start /b powershell -Command "if (Test-Path 'czsu_multi_agent_text_to_sql.egg-info') { Remove-Item -LiteralPath 'czsu_multi_agent_text_to_sql.egg-info' -Force -Recurse }"
start /b powershell -Command "Get-ChildItem -Path '.' -Directory -Filter '__pycache__' -Recurse | Remove-Item -Force -Recurse"
start /b powershell -Command "if (Test-Path 'build') { Remove-Item -LiteralPath 'build' -Force -Recurse }"
start /b powershell -Command "if (Test-Path 'frontend\.next') { Remove-Item -LiteralPath 'frontend\.next' -Force -Recurse }"
start /b powershell -Command "if (Test-Path 'frontend\node_modules') { Remove-Item -LiteralPath 'frontend\node_modules' -Force -Recurse }"

REM Wait for all removals to complete
set wait_count=0
:wait_removals
if exist ".venv" goto still_removing
if exist "czsu_multi_agent_text_to_sql.egg-info" goto still_removing
powershell -Command "if ((Get-ChildItem -Path '.' -Directory -Filter '__pycache__' -Recurse).Count -gt 0) { exit 1 }" >nul 2>&1
if %errorlevel% equ 1 goto still_removing
if exist "build" goto still_removing
if exist "frontend\.next" goto still_removing
if exist "frontend\node_modules" goto still_removing
goto removals_done
:still_removing
set /a wait_count+=1
if !wait_count! gtr 30 goto removals_timeout
timeout /t 1 /nobreak >nul
goto wait_removals
:removals_timeout
:removals_done

echo   Done removing old files.
echo.

echo [2/2] Installing fresh dependencies (parallel)...
echo.

REM Create VS Code settings first (no dependencies)
if not exist ".vscode" mkdir .vscode
(
echo {
echo     "python.defaultInterpreterPath": "${workspaceFolder}/.venv/Scripts/python.exe",
echo     "python.terminal.activateEnvironment": true,
echo     "python.terminal.activateEnvInCurrentTerminal": true,
echo     "python.analysis.extraPaths": [
echo         "${workspaceFolder}"
echo     ],
echo     "python.envFile": "${workspaceFolder}/.env",
echo     "python.terminal.executeInFileDir": false,
echo     "python.pythonPath": "${workspaceFolder}/.venv/Scripts/python.exe",
echo     "code-runner.executorMap": {
echo         "python": "\"$pythonPath\" -u $fullFileName"
echo     },
echo     "code-runner.fileDirectoryAsCwd": false,
echo     "code-runner.respectShebang": false,
echo     "python-envs.pythonProjects": [],
echo     "terminal.integrated.env.windows": {},
echo     "terminal.integrated.cwd": "${workspaceFolder}",
echo     "terminal.integrated.defaultProfile.windows": "Command Prompt",
echo     "terminal.integrated.profiles.windows": {
echo         "Command Prompt": {
echo             "path": "cmd.exe",
echo             "args": ["/K", ".venv\\Scripts\\activate.bat"],
echo             "icon": "terminal-cmd"
echo         },
echo         "PowerShell": {
echo             "source": "PowerShell",
echo             "args": ["-NoExit", "-Command", "& '.venv\\Scripts\\Activate.ps1'"],
echo             "icon": "terminal-powershell"
echo         }
echo     },
echo     "terminal.integrated.automationProfile.windows": {
echo         "path": "cmd.exe",
echo         "args": ["/K", ".venv\\Scripts\\activate.bat"]
echo     }
echo }
) > .vscode\settings.json

REM Start frontend install in a VISIBLE window (using working setup.bat commands)
echo   Starting frontend installation (in new window - watch for completion)...
start "Frontend Install" cmd /k "cd frontend && npm install --verbose --force && npm run build && cd .. && echo. && echo Frontend installation complete! && pause"

REM Backend install in foreground
echo   Installing backend...
uv venv --python 3.11.9
uv pip install --prerelease=allow --python .venv . .[dev]
python unzip_files.py

echo.
echo ========================================
echo  Backend installation complete!
echo  Close the frontend window when it's done.
echo ========================================
echo.
echo Terminal will stay open. Type 'exit' to close.
cmd /k
