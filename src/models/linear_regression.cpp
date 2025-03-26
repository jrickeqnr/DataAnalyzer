#include "../../include/model.h"
#include <cmath>

namespace DataAnalyzer {

LinearRegression::LinearRegression(bool fit_intercept)
    : fit_intercept_(fit_intercept), intercept_(0.0), rmse_(0.0), r_squared_(0.0) {
}

bool LinearRegression::train(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
    if (X.rows() != y.rows() || X.rows() == 0) {
        return false;
    }
    
    // Prepare data based on whether to fit intercept
    Eigen::MatrixXd X_train;
    if (fit_intercept_) {
        // Add a column of ones for the intercept
        X_train = Eigen::MatrixXd::Ones(X.rows(), X.cols() + 1);
        X_train.rightCols(X.cols()) = X;
    } else {
        X_train = X;
    }
    
    // Solve normal equations: (X^T * X)^(-1) * X^T * y
    // Using QR decomposition for numerical stability
    Eigen::VectorXd solution = X_train.colPivHouseholderQr().solve(y);
    
    if (fit_intercept_) {
        intercept_ = solution(0);
        coefficients_ = solution.tail(solution.size() - 1);
    } else {
        intercept_ = 0.0;
        coefficients_ = solution;
    }
    
    // Compute performance statistics
    computeStats(X, y);
    
    return true;
}

Eigen::VectorXd LinearRegression::predict(const Eigen::MatrixXd& X) const {
    return X * coefficients_ + Eigen::VectorXd::Constant(X.rows(), intercept_);
}

std::map<std::string, double> LinearRegression::getStats() const {
    return {
        {"RMSE", rmse_},
        {"R²", r_squared_}
    };
}

std::string LinearRegression::getDescription() const {
    return "Linear Regression fits a linear model with coefficients to minimize "
           "the residual sum of squares between the observed targets and the targets "
           "predicted by the linear approximation. It's the simplest and fastest "
           "regression method but assumes linearity between features and target.";
}

Eigen::VectorXd LinearRegression::getCoefficients() const {
    return coefficients_;
}

std::map<std::string, double> LinearRegression::getHyperparameters() const {
    return {
        {"fit_intercept", fit_intercept_ ? 1.0 : 0.0}
    };
}

void LinearRegression::computeStats(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
    Eigen::VectorXd predictions = predict(X);
    
    // Calculate Root Mean Squared Error (RMSE)
    rmse_ = std::sqrt((predictions - y).array().square().mean());
    
    // Calculate R-squared
    double y_mean = y.mean();
    double ss_total = (y.array() - y_mean).square().sum();
    double ss_residual = (y - predictions).array().square().sum();
    r_squared_ = 1.0 - (ss_residual / ss_total);
}

} // namespace DataAnalyzer 