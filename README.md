# DataAnalyzerCpp

A portable C++ data analysis program designed to preprocess time series data and feed it into AI models. This application provides a complete GUI workflow for data loading, outlier detection, model training, and result visualization.

## Features

- **Portable**: No external dependencies like Python or Matplotlib required.
- **Interactive GUI**: Easy to use interface built with Dear ImGui and GLFW.
- **Preprocessing**: Date parsing, frequency detection, outlier handling.
- **Modeling**: Elastic Net regression (L1/L2 regularization mix).
- **Visualization**: Time series plots with ImPlot.
- **Export**: Save predictions and model statistics for further analysis.

## Requirements

- C++17 compatible compiler (GCC, Clang, MSVC)
- OpenGL 3.3+
- CMake 3.14+

## Libraries Used

- **Eigen**: Header-only library for linear algebra and matrix operations.
- **Dear ImGui**: Immediate mode GUI for the user interface.
- **ImPlot**: Plotting library built on top of Dear ImGui.
- **GLFW**: Cross-platform library for creating windows with OpenGL contexts.
- **fast-cpp-csv-parser**: Header-only CSV parsing library.

## Build Instructions

### Clone the repository

```bash
git clone <repository-url>
cd DataAnalyzerCpp
```

### Build with CMake

```bash
mkdir build
cd build
cmake ..
cmake --build .
```

### Run the application

```bash
./bin/DataAnalyzer
```

## Usage Guide

The application follows a sequential workflow through six screens:

1. **File Browser**:
   - Select a CSV file with numeric data and at least one date column.
   - The program will automatically detect date columns and data frequency.

2. **Outlier Detection**:
   - Detect outliers using the IQR (Interquartile Range) method.
   - Choose to fix outliers by interpolation or keep them as is.

3. **Model Selection**:
   - Currently only supports Elastic Net regression.
   - More models will be added in future versions.

4. **Variable Selection**:
   - Select which features (independent variables) to use.
   - Choose a target (dependent variable) for prediction.
   - Enable seasonality to add sin/cos features based on detected frequency.

5. **Hyperparameter Configuration**:
   - Manually set alpha and lambda parameters.
   - Use automatic tuning with grid search.
   - Train the model and view performance statistics.

6. **Results Visualization**:
   - View time series plots of actual vs. predicted values.
   - Export results and model statistics for further analysis.

## CSV File Format

The application expects CSV files with the following characteristics:
- At least one date column (it will try to detect this automatically).
- One or more numeric columns for features and target variables.
- A header row with column names.

Example:
```
Date,Temperature,Humidity,Energy_Consumption
2023-01-01,10.5,65.2,120.3
2023-01-02,11.2,67.5,122.1
...
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Future Development

- Additional models (Random Forest, Neural Networks, etc.)
- More plot types (scatter plots, bar charts, etc.)
- Advanced data transformations
- Support for categorical variables
- Time series forecasting with confidence intervals 