#include "../../include/model.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>

namespace DataAnalyzer {

// ElasticNet Implementation
ElasticNet::ElasticNet(double alpha, double lambda, int max_iter, double tol)
    : alpha_(alpha), lambda_(lambda), max_iter_(max_iter), tol_(tol),
      intercept_(0.0), rmse_(0.0), r_squared_(0.0) {
}

bool ElasticNet::train(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
    if (X.rows() != y.rows() || X.rows() == 0) {
        return false;
    }
    
    // Prepare data (standardize features)
    Eigen::VectorXd X_mean = X.colwise().mean();
    Eigen::VectorXd X_std = ((X.rowwise() - X_mean.transpose()).array().square().colwise().sum() / X.rows()).sqrt();
    
    // Handle zero standard deviation
    for (int i = 0; i < X_std.size(); ++i) {
        if (X_std(i) < 1e-10) {
            X_std(i) = 1.0;
        }
    }
    
    Eigen::MatrixXd X_scaled = (X.rowwise() - X_mean.transpose()).array().rowwise() / X_std.transpose().array();
    
    // Initialize coefficients
    coefficients_ = Eigen::VectorXd::Zero(X.cols());
    double y_mean = y.mean();
    Eigen::VectorXd y_centered = y.array() - y_mean;
    
    // Coordinate descent algorithm
    double prev_loss = std::numeric_limits<double>::max();
    
    for (int iter = 0; iter < max_iter_; ++iter) {
        // Update training progress
        stats_["Training Progress"] = static_cast<double>(iter) / max_iter_;
        
        // Update each coefficient using coordinate descent
        for (int j = 0; j < X.cols(); ++j) {
            // Calculate the soft threshold value
            double rho = X_scaled.col(j).dot(y_centered - X_scaled * coefficients_ + coefficients_(j) * X_scaled.col(j));
            double beta = softThreshold(rho, lambda_ * alpha_);
            
            if (beta != 0) {
                // Update coefficient with L2 regularization
                coefficients_(j) = beta / (1.0 + lambda_ * (1.0 - alpha_));
            } else {
                coefficients_(j) = 0.0;
            }
        }
        
        // Calculate loss function
        Eigen::VectorXd predictions = X_scaled * coefficients_;
        double loss = (y_centered - predictions).squaredNorm() / (2 * X.rows());
        loss += lambda_ * alpha_ * coefficients_.array().abs().sum();
        loss += 0.5 * lambda_ * (1.0 - alpha_) * coefficients_.squaredNorm();
        
        // Check convergence
        if (std::abs(prev_loss - loss) < tol_) {
            break;
        }
        
        prev_loss = loss;
    }
    
    // Rescale coefficients back to original scale
    for (int j = 0; j < X.cols(); ++j) {
        coefficients_(j) /= X_std(j);
    }
    
    // Calculate intercept
    intercept_ = y_mean - X_mean.dot(coefficients_);
    
    // Compute statistics
    computeStats(X, y);
    
    return true;
}

Eigen::VectorXd ElasticNet::predict(const Eigen::MatrixXd& X) const {
    return X * coefficients_ + Eigen::VectorXd::Constant(X.rows(), intercept_);
}

std::map<std::string, double> ElasticNet::getStats() const {
    return stats_;
}

std::string ElasticNet::getDescription() const {
    return "Elastic Net Regression combines L1 and L2 regularization to balance "
           "sparsity and stability. It's effective for feature selection and handling "
           "correlated predictors. The alpha parameter controls the mix of L1 and L2 "
           "regularization, while lambda controls the overall regularization strength.";
}

Eigen::VectorXd ElasticNet::getCoefficients() const {
    return coefficients_;
}

std::map<std::string, double> ElasticNet::getHyperparameters() const {
    return {
        {"alpha", alpha_},
        {"lambda", lambda_},
        {"max_iterations", static_cast<double>(max_iter_)},
        {"tolerance", tol_}
    };
}

std::pair<double, double> ElasticNet::gridSearch(
    const Eigen::MatrixXd& X, 
    const Eigen::VectorXd& y,
    const std::vector<double>& alpha_values,
    const std::vector<double>& lambda_values,
    int k) {
    
    if (X.rows() != y.rows() || alpha_values.empty() || lambda_values.empty() || k <= 1) {
        return {alpha_, lambda_};
    }
    
    // Standardize features
    Eigen::MatrixXd X_std = X;
    Eigen::VectorXd mean = X.colwise().mean();
    Eigen::VectorXd std = ((X.rowwise() - mean.transpose()).array().square().colwise().sum() / (X.rows() - 1)).sqrt();
    
    for (int i = 0; i < X.cols(); ++i) {
        if (std(i) > 1e-10) {  // Avoid division by zero
            X_std.col(i) = (X.col(i).array() - mean(i)) / std(i);
        }
    }
    
    // Standardize target variable
    double y_mean = y.mean();
    double y_std = std::sqrt((y.array() - y_mean).square().sum() / (y.size() - 1));
    Eigen::VectorXd y_std_vec = (y.array() - y_mean) / y_std;
    
    // Prepare k-fold cross-validation indices once
    std::vector<std::vector<int>> fold_indices(k);
    std::vector<int> indices(X.rows());
    std::iota(indices.begin(), indices.end(), 0);
    
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);
    
    int fold_size = X.rows() / k;
    for (int fold = 0; fold < k; ++fold) {
        int start_idx = fold * fold_size;
        int end_idx = (fold == k - 1) ? X.rows() : (fold + 1) * fold_size;
        fold_indices[fold] = std::vector<int>(indices.begin() + start_idx, indices.begin() + end_idx);
    }
    
    // Best hyperparameters
    double best_alpha = alpha_;
    double best_lambda = lambda_;
    double best_rmse = std::numeric_limits<double>::max();
    
    // Grid search with progress tracking
    int total_combinations = alpha_values.size() * lambda_values.size();
    int current_combination = 0;
    
    // Grid search
    for (double alpha : alpha_values) {
        for (double lambda : lambda_values) {
            current_combination++;
            double fold_rmse_sum = 0.0;
            bool early_stop = false;
            int valid_folds = 0;
            
            // Update progress in stats
            stats_["Grid Search Progress"] = static_cast<double>(current_combination) / total_combinations;
            
            // K-fold cross-validation
            for (int fold = 0; fold < k && !early_stop; ++fold) {
                try {
                    // Create validation set
                    const std::vector<int>& val_indices = fold_indices[fold];
                    
                    // Create training set
                    std::vector<int> train_indices;
                    train_indices.reserve(X.rows() - val_indices.size());
                    for (int i = 0; i < k; ++i) {
                        if (i != fold) {
                            train_indices.insert(train_indices.end(), 
                                               fold_indices[i].begin(), 
                                               fold_indices[i].end());
                        }
                    }
                    
                    // Prepare train/val matrices
                    Eigen::MatrixXd X_train(train_indices.size(), X_std.cols());
                    Eigen::VectorXd y_train(train_indices.size());
                    Eigen::MatrixXd X_val(val_indices.size(), X_std.cols());
                    Eigen::VectorXd y_val(val_indices.size());
                    
                    for (size_t i = 0; i < train_indices.size(); ++i) {
                        X_train.row(i) = X_std.row(train_indices[i]);
                        y_train(i) = y_std_vec(train_indices[i]);
                    }
                    
                    for (size_t i = 0; i < val_indices.size(); ++i) {
                        X_val.row(i) = X_std.row(val_indices[i]);
                        y_val(i) = y_std_vec(val_indices[i]);
                    }
                    
                    // Train model with current hyperparameters
                    ElasticNet model(alpha, lambda, max_iter_, tol_);
                    
                    if (!model.train(X_train, y_train)) {
                        // If training fails, skip this combination
                        early_stop = true;
                        break;
                    }
                    
                    // Evaluate on validation set
                    Eigen::VectorXd predictions = model.predict(X_val);
                    
                    // Convert predictions and targets back to original scale
                    predictions = predictions.array() * y_std + y_mean;
                    Eigen::VectorXd y_val_orig = y_val.array() * y_std + y_mean;
                    
                    double fold_rmse = std::sqrt((predictions - y_val_orig).array().square().mean());
                    
                    // Check for numerical stability
                    if (!std::isfinite(fold_rmse) || fold_rmse > 1e6) {
                        early_stop = true;
                        break;
                    }
                    
                    // Early stopping if RMSE is much worse than best
                    if (best_rmse < std::numeric_limits<double>::max() && 
                        fold_rmse > 2.0 * best_rmse) {
                        early_stop = true;
                        break;
                    }
                    
                    fold_rmse_sum += fold_rmse;
                    valid_folds++;
                    
                } catch (const std::exception&) {
                    // If any exception occurs, skip this combination
                    early_stop = true;
                    break;
                }
            }
            
            if (!early_stop && valid_folds > 0) {
                // Average RMSE across valid folds
                double avg_rmse = fold_rmse_sum / valid_folds;
                
                // Update best hyperparameters if RMSE is improved
                if (avg_rmse < best_rmse) {
                    best_rmse = avg_rmse;
                    best_alpha = alpha;
                    best_lambda = lambda;
                    
                    // Store best parameters in stats
                    stats_["Best RMSE"] = best_rmse;
                    stats_["Best Alpha"] = best_alpha;
                    stats_["Best Lambda"] = best_lambda;
                }
            }
        }
    }
    
    return {best_alpha, best_lambda};
}

double ElasticNet::softThreshold(double x, double lambda) const {
    if (x > lambda) {
        return x - lambda;
    } else if (x < -lambda) {
        return x + lambda;
    } else {
        return 0.0;
    }
}

void ElasticNet::computeStats(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
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
    
    // Add column of ones for intercept
    Eigen::MatrixXd X_with_intercept = Eigen::MatrixXd::Ones(n, p + 1);
    X_with_intercept.rightCols(p) = X;
    
    // Calculate variance-covariance matrix
    double sigma2 = ss_residual / (n - p - 1); // Unbiased estimate of error variance
    Eigen::MatrixXd cov_matrix = sigma2 * (X_with_intercept.transpose() * X_with_intercept).inverse();
    
    // Calculate standard errors
    Eigen::VectorXd std_errors = cov_matrix.diagonal().array().sqrt();
    
    // Calculate t-values
    Eigen::VectorXd t_values(p + 1);
    t_values(0) = intercept_ / std_errors(0);
    t_values.tail(p) = coefficients_.array() / std_errors.tail(p).array();
    
    // Calculate AIC and BIC
    double aic = n * std::log(ss_residual / n) + 2 * (p + 1);
    double bic = n * std::log(ss_residual / n) + std::log(n) * (p + 1);
    
    // Calculate L1 and L2 norms of coefficients
    double l1_norm = coefficients_.array().abs().sum();
    double l2_norm = coefficients_.norm();
    
    // Store all statistics in the stats map
    stats_ = {
        {"RMSE", rmse_},
        {"MSE", mse},
        {"MAE", mae},
        {"R²", r_squared_},
        {"Adjusted R²", adj_r_squared},
        {"AIC", aic},
        {"BIC", bic},
        {"L1 Norm", l1_norm},
        {"L2 Norm", l2_norm},
        {"Intercept SE", std_errors(0)},
        {"Intercept t-value", t_values(0)}
    };
    
    // Store coefficient statistics
    for (int i = 0; i < p; ++i) {
        stats_["SE_" + std::to_string(i)] = std_errors(i + 1);
        stats_["t_value_" + std::to_string(i)] = t_values(i + 1);
    }
}

Eigen::VectorXd ElasticNet::getFeatureImportance() const {
    // For ElasticNet, feature importance is the absolute value of coefficients
    return coefficients_.array().abs();
}

} // namespace DataAnalyzer 