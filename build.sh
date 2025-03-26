#!/bin/bash

# Exit on error
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Building DataAnalyzerCpp...${NC}"

# Create build directory if it doesn't exist
if [ ! -d "build" ]; then
    echo "Creating build directory..."
    mkdir build
fi

# Navigate to build directory
cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build the project
echo "Building project..."
cmake --build . --config Release

# Check if build was successful
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Build successful!${NC}"
    echo -e "${GREEN}Executable location: $(pwd)/bin/DataAnalyzer${NC}"
else
    echo -e "${RED}Build failed!${NC}"
    exit 1
fi

# Copy sample data to build directory
echo "Copying sample data..."
mkdir -p bin/data
cp ../sample_data.csv bin/data/

# Create a run script
echo "Creating run script..."
cat > bin/run.sh << 'EOL'
#!/bin/bash
export LD_LIBRARY_PATH="$(dirname "$0")/../lib:$LD_LIBRARY_PATH"
./DataAnalyzer
EOL
chmod +x bin/run.sh

echo -e "${GREEN}Setup complete! You can run the application using:${NC}"
echo -e "${YELLOW}cd build/bin${NC}"
echo -e "${YELLOW}./run.sh${NC}" 