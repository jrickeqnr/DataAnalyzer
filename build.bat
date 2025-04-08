@echo off
setlocal EnableDelayedExpansion
set BUILD_FAILED=0

echo Building DataAnalyzer...

if exist "build" (
    echo Cleaning build directory...
    rmdir /s /q build
    if !ERRORLEVEL! neq 0 (
        echo Error: Failed to clean build directory
        set BUILD_FAILED=1
        goto :end
    )
)

echo Creating build directory...
mkdir build
if !ERRORLEVEL! neq 0 (
    echo Error: Failed to create build directory
    set BUILD_FAILED=1
    goto :end
)
cd build

echo Configuring with CMake (Release mode)...
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="/W4 /wd4127 /wd4458 /wd4100" ..
if !ERRORLEVEL! neq 0 (
    echo Error: CMake configuration failed
    set BUILD_FAILED=1
    goto :end
)

echo Building with CMake (Release configuration)...
cmake --build . --config Release
if !ERRORLEVEL! neq 0 (
    echo Error: Build failed
    set BUILD_FAILED=1
    goto :end
)

echo Checking output directory structure...
dir bin 2>nul
if !ERRORLEVEL! neq 0 (
    echo Warning: bin directory not found
    set BUILD_FAILED=1
    goto :end
)

dir bin\Release 2>nul
if !ERRORLEVEL! neq 0 (
    echo Warning: Release directory not found
    set BUILD_FAILED=1
    goto :end
)

echo Copying assets to output directory...
if not exist "bin\Release\assets" (
    mkdir bin\Release\assets
    if !ERRORLEVEL! neq 0 (
        echo Error: Failed to create assets directory
        set BUILD_FAILED=1
        goto :end
    )
)

xcopy /E /Y /I ..\assets bin\Release\assets
if !ERRORLEVEL! neq 0 (
    echo Error: Failed to copy assets
    set BUILD_FAILED=1
    goto :end
)

:end
if !BUILD_FAILED! equ 1 (
    echo.
    echo Build FAILED! Check the errors above.
    exit /b 1
) else (
    echo.
    echo Build SUCCEEDED!
    echo You can now run the application from build\bin\Release\DataAnalyzer.exe
    exit /b 0
) 