@echo off
echo Running Equinor DataAnalyzer...

REM Check if release build exists
if not exist "build\bin\Release\DataAnalyzer.exe" (
    echo ERROR: Release build not found. Please run build.bat first.
    exit /b 1
)

REM Run from the correct directory
cd build\bin\Release
start "" DataAnalyzer.exe

echo Application started! 