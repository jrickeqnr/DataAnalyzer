#include "../include/model.h"
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
    return {
        {"RMSE", rmse_},
        {"R²", r_squared_}
    };
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
    
    // Prepare k-fold cross-validation indices
    std::vector<int> indices(X.rows());
    std::iota(indices.begin(), indices.end(), 0);
    
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);
    
    // Best hyperparameters
    double best_alpha = alpha_;
    double best_lambda = lambda_;
    double best_rmse = std::numeric_limits<double>::max();
    
    // Grid search
    for (double alpha : alpha_values) {
        for (double lambda : lambda_values) {
            double fold_rmse_sum = 0.0;
            
            // K-fold cross-validation
            for (int fold = 0; fold < k; ++fold) {
                // Split data into training and validation sets
                int fold_size = X.rows() / k;
                int start_idx = fold * fold_size;
                int end_idx = (fold == k - 1) ? X.rows() : (fold + 1) * fold_size;
                
                // Create validation set
                std::vector<int> val_indices(indices.begin() + start_idx, indices.begin() + end_idx);
                
                // Create training set (all indices not in validation set)
                std::vector<int> train_indices;
                for (int i = 0; i < X.rows(); ++i) {
                    if (i < start_idx || i >= end_idx) {
                        train_indices.push_back(i);
                    }
                }
                
                // Prepare train/val matrices
                Eigen::MatrixXd X_train(train_indices.size(), X.cols());
                Eigen::VectorXd y_train(train_indices.size());
                Eigen::MatrixXd X_val(val_indices.size(), X.cols());
                Eigen::VectorXd y_val(val_indices.size());
                
                for (size_t i = 0; i < train_indices.size(); ++i) {
                    X_train.row(i) = X.row(train_indices[i]);
                    y_train(i) = y(train_indices[i]);
                }
                
                for (size_t i = 0; i < val_indices.size(); ++i) {
                    X_val.row(i) = X.row(val_indices[i]);
                    y_val(i) = y(val_indices[i]);
                }
                
                // Train model with current hyperparameters
                ElasticNet model(alpha, lambda, max_iter_, tol_);
                model.train(X_train, y_train);
                
                // Evaluate on validation set
                Eigen::VectorXd predictions = model.predict(X_val);
                double fold_rmse = std::sqrt((predictions - y_val).array().square().mean());
                fold_rmse_sum += fold_rmse;
            }
            
            // Average RMSE across all folds
            double avg_rmse = fold_rmse_sum / k;
            
            // Update best hyperparameters if RMSE is improved
            if (avg_rmse < best_rmse) {
                best_rmse = avg_rmse;
                best_alpha = alpha;
                best_lambda = lambda;
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
    
    // Calculate Root Mean Squared Error (RMSE)
    rmse_ = std::sqrt((predictions - y).array().square().mean());
    
    // Calculate R-squared
    double y_mean = y.mean();
    double ss_total = (y.array() - y_mean).square().sum();
    double ss_residual = (y - predictions).array().square().sum();
    r_squared_ = 1.0 - (ss_residual / ss_total);
}

} // namespace DataAnalyzer 