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

REM Create empty temp folder for robocopy trick
if not exist "%TEMP%\empty_dir" mkdir "%TEMP%\empty_dir"

REM Force delete .venv with maximum aggression (super fast and silent)
echo   Force deleting .venv...
if exist ".venv" (
    takeown /f ".venv" /r /d y >nul 2>&1
    icacls ".venv" /grant administrators:F /t >nul 2>&1
    robocopy "%TEMP%\empty_dir" ".venv" /mir /njh /njs /ndl /nc /ns /nfl >nul 2>&1
    rd /s /q ".venv" >nul 2>&1
    if exist ".venv" (
        for /d %%i in (".venv\*") do if exist "%%i" rd /s /q "%%i" >nul 2>&1
        rd /s /q ".venv" >nul 2>&1
    )
)
if exist "czsu_multi_agent_text_to_sql.egg-info" start /b "" cmd /c "if exist "czsu_multi_agent_text_to_sql.egg-info" rd /s /q "czsu_multi_agent_text_to_sql.egg-info" 2>nul"
if exist "__pycache__" start /b "" cmd /c "if exist "__pycache__" rd /s /q "__pycache__" 2>nul"
if exist "frontend\.next" start /b "" cmd /c "robocopy "%TEMP%\empty_dir" "frontend\.next" /mir /njh /njs /ndl /nc /ns /nfl >nul 2>&1 & if exist "frontend\.next" rd /s /q "frontend\.next" 2>nul"
if exist "frontend\node_modules" start /b "" cmd /c "robocopy "%TEMP%\empty_dir" "frontend\node_modules" /mir /njh /njs /ndl /nc /ns /nfl >nul 2>&1 & if exist "frontend\node_modules" rd /s /q "frontend\node_modules" 2>nul"

REM Wait for critical folders to be deleted before proceeding (with timeout)
set wait_count=0
:wait_venv
if exist ".venv" (
    set /a wait_count+=1
    if !wait_count! gtr 30 goto venv_timeout
    timeout /t 1 /nobreak >nul
    goto wait_venv
)
:venv_timeout
set wait_count=0
:wait_node
if exist "frontend\node_modules" (
    set /a wait_count+=1
    if !wait_count! gtr 30 goto node_timeout
    timeout /t 1 /nobreak >nul
    goto wait_node
)
:node_timeout

if exist "%TEMP%\empty_dir" rd /s /q "%TEMP%\empty_dir" 2>nul

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
