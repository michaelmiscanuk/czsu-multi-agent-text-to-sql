@echo off
REM Docker build and test script for CZSU Multi-Agent Text-to-SQL API (Windows)
REM Usage: docker-build-test.bat [build|run|test|clean]

set IMAGE_NAME=czsu-multi-agent-api
set CONTAINER_NAME=czsu-api-test
set PORT=8000

if "%1"=="" set ACTION=build
if not "%1"=="" set ACTION=%1

if "%ACTION%"=="build" goto BUILD
if "%ACTION%"=="run" goto RUN
if "%ACTION%"=="test" goto TEST
if "%ACTION%"=="clean" goto CLEAN
if "%ACTION%"=="full" goto FULL
goto USAGE

:BUILD
echo 🏗️  Building Docker image...
docker build -t %IMAGE_NAME% . --no-cache
echo ✅ Build completed successfully!
goto END

:RUN
echo 🚀 Running container...
docker stop %CONTAINER_NAME% 2>nul
docker rm %CONTAINER_NAME% 2>nul
docker run -d --name %CONTAINER_NAME% -p %PORT%:%PORT% -e PYTHONUNBUFFERED=1 -e PORT=%PORT% %IMAGE_NAME%
echo ✅ Container started on http://localhost:%PORT%
echo 📋 View logs: docker logs -f %CONTAINER_NAME%
goto END

:TEST
echo 🧪 Testing container health...
timeout /t 10 /nobreak >nul
curl -f http://localhost:%PORT%/health
if %errorlevel%==0 (
    echo ✅ Health check passed!
) else (
    echo ❌ Health check failed!
    docker logs %CONTAINER_NAME%
    exit /b 1
)
goto END

:CLEAN
echo 🧹 Cleaning up...
docker stop %CONTAINER_NAME% 2>nul
docker rm %CONTAINER_NAME% 2>nul
docker rmi %IMAGE_NAME% 2>nul
echo ✅ Cleanup completed!
goto END

:FULL
echo 🔄 Full build, run, and test cycle...
call %0 clean
call %0 build
call %0 run
call %0 test
echo ✅ Full cycle completed successfully!
goto END

:USAGE
echo Usage: %0 [build^|run^|test^|clean^|full]
echo   build - Build the Docker image
echo   run   - Run the container
echo   test  - Test container health
echo   clean - Clean up containers and images
echo   full  - Complete build, run, and test cycle
exit /b 1

:END