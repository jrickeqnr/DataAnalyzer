# DataAnalyzer

A portable C++ data analysis program designed to replace existing data analysis workflows in Equinor. This application provides a complete GUI workflow for data loading, outlier detection, model training, and result visualization.

## Features

- **Portable**: No external dependencies like Python or Matplotlib required.
- **Interactive GUI**: Easy to use interface built with Dear ImGui and GLFW.
- **Preprocessing**: Date parsing, frequency detection, outlier handling.
- **Multiple Regression Models**:
  - Linear Regression - Standard ordinary least squares regression
  - Elastic Net - Regression with L1 and L2 regularization
  - XGBoost - Extreme Gradient Boosting regression
  - Neural Network - Multi-layer Perceptron regressor
- **Visualization**: Multiple visualizations per model including scatterplot, timeseries, feature importance
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

### Clone the repository with dependencies

```bash
git clone --recurse-submodules git@github.com:jrickeqnr/DataAnalyzer.git
cd DataAnalyzer
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
   - Choose from multiple regression models including:
     - Linear Regression
     - Elastic Net (with L1/L2 regularization mix)
     - XGBoost
     - Gradient Boosting
     - Neural Network

4. **Variable Selection**:
   - Select which features (independent variables) to use.
   - Choose a target (dependent variable) for prediction.

5. **Hyperparameter Configuration**:
   - Manually set model-specific parameters.
   - Use automatic tuning with grid search and cross-validation.
   - Train the model and view performance statistics (RMSE and R-squared).

6. **Results Visualization**:
   - View time series plots of actual vs. predicted values.
   - Export results and model statistics for further analysis.

## Model API Usage

All models follow a common interface defined in `include/model.h`:

```cpp
#include "../include/model.h"

// Create a model (example with ElasticNet)
DataAnalyzer::ElasticNet model(0.5, 1.0);

// Train the model
model.train(X, y);

// Make predictions
Eigen::VectorXd predictions = model.predict(X_test);

// Get statistics
auto stats = model.getStats();
std::cout << "RMSE: " << stats["RMSE"] << std::endl;
std::cout << "R²: " << stats["R²"] << std::endl;

// Tune hyperparameters
std::vector<double> alpha_values = {0.1, 0.5, 0.9};
std::vector<double> lambda_values = {0.1, 1.0, 10.0};
auto [best_alpha, best_lambda] = model.gridSearch(X, y, alpha_values, lambda_values);
```

## CSV File Format

The application expects CSV files with the following characteristics:

- At least one date column (it will try to detect this automatically).
- One or more numeric columns for features and target variables.
- A header row with column names.

Example:

```txt
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

- Additional models and enhancements to existing ones
- More plot types (scatter plots, bar charts, etc.)
- Advanced data transformations
- Support for categorical variables
- Time series forecasting with confidence intervals
