@echo off
echo Building DataAnalyzer...

if exist "build" (
    echo Cleaning build directory...
    rmdir /s /q build
)

echo Creating build directory...
mkdir build
cd build

echo Configuring with CMake (Release mode)...
cmake -DCMAKE_BUILD_TYPE=Release ..

echo Building with CMake (Release configuration)...
cmake --build . --config Release

echo Checking output directory structure...
dir bin
dir bin\Release

echo Copying assets to output directory...
if not exist "bin\Release\assets" (
    mkdir bin\Release\assets
)
xcopy /E /Y /I ..\assets bin\Release\assets

echo Done!
echo.
echo You can now run the application from build\bin\Release\DataAnalyzer.exe 