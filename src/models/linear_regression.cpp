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
    return stats_;
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
    int n = X.rows();
    int p = X.cols();
    
    // Calculate prediction errors
    Eigen::VectorXd residuals = y - predictions;
    
    // Calculate MSE (Mean Squared Error)
    double mse = residuals.array().square().mean();
    
    // Calculate RMSE (Root Mean Squared Error)
    rmse_ = std::sqrt(mse);
    
    // Calculate MAE (Mean Absolute Error)
    double mae = residuals.array().abs().mean();
    
    // Calculate R-squared
    double y_mean = y.mean();
    double ss_total = (y.array() - y_mean).square().sum();
    double ss_residual = residuals.array().square().sum();
    r_squared_ = 1.0 - (ss_residual / ss_total);
    
    // Calculate Adjusted R-squared
    double adj_r_squared = 1.0 - (1.0 - r_squared_) * (n - 1) / (n - p - 1);
    
    // Calculate standard errors and t-values
    // Add column of ones for intercept if needed
    Eigen::MatrixXd X_with_intercept;
    if (fit_intercept_) {
        X_with_intercept = Eigen::MatrixXd::Ones(n, p + 1);
        X_with_intercept.rightCols(p) = X;
    } else {
        X_with_intercept = X;
    }
    
    // Calculate variance-covariance matrix
    double sigma2 = ss_residual / (n - p - 1); // Unbiased estimate of error variance
    Eigen::MatrixXd cov_matrix = sigma2 * (X_with_intercept.transpose() * X_with_intercept).inverse();
    
    // Calculate standard errors
    Eigen::VectorXd std_errors = cov_matrix.diagonal().array().sqrt();
    
    // Calculate t-values
    Eigen::VectorXd t_values;
    if (fit_intercept_) {
        t_values = Eigen::VectorXd(p + 1);
        t_values(0) = intercept_ / std_errors(0);
        t_values.tail(p) = coefficients_.array() / std_errors.tail(p).array();
    } else {
        t_values = coefficients_.array() / std_errors.array();
    }
    
    // Calculate AIC and BIC
    double aic = n * std::log(ss_residual / n) + 2 * (p + 1);
    double bic = n * std::log(ss_residual / n) + std::log(n) * (p + 1);
    
    // Store all statistics in the stats map
    stats_ = {
        {"RMSE", rmse_},
        {"MSE", mse},
        {"MAE", mae},
        {"R²", r_squared_},
        {"Adjusted R²", adj_r_squared},
        {"AIC", aic},
        {"BIC", bic}
    };
    
    // Store intercept statistics if applicable
    if (fit_intercept_) {
        stats_["Intercept SE"] = std_errors(0);
        stats_["Intercept t-value"] = t_values(0);
        
        // Store coefficient statistics
        for (int i = 0; i < p; ++i) {
            stats_["SE_" + std::to_string(i)] = std_errors(i + 1);
            stats_["t_value_" + std::to_string(i)] = t_values(i + 1);
        }
    } else {
        // Store coefficient statistics without intercept
        for (int i = 0; i < p; ++i) {
            stats_["SE_" + std::to_string(i)] = std_errors(i);
            stats_["t_value_" + std::to_string(i)] = t_values(i);
        }
    }
}

} // namespace DataAnalyzer 