#!/bin/bash

# Exit on error
set -e

# Create build directory if it doesn't exist
mkdir -p build_windows

# Set up MinGW paths
export MINGW_PREFIX="x86_64-w64-mingw32"
export CC="${MINGW_PREFIX}-gcc"
export CXX="${MINGW_PREFIX}-g++"
export RC="${MINGW_PREFIX}-windres"

# Navigate to build directory
cd build_windows

# Configure with CMake
cmake .. \
    -DCMAKE_SYSTEM_NAME=Windows \
    -DCMAKE_C_COMPILER=${CC} \
    -DCMAKE_CXX_COMPILER=${CXX} \
    -DCMAKE_RC_COMPILER=${RC} \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=install

# Build the project
cmake --build . --config Release -j$(sysctl -n hw.ncpu)

# Install to the install directory
cmake --install . --config Release

echo "Build completed successfully!"
echo "The Windows executable can be found in build_windows/bin/DataAnalyzer.exe" 