#include "../../include/model.h"
#include "../../include/logger.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <sstream>

namespace DataAnalyzer {

// ElasticNet Implementation
ElasticNet::ElasticNet(double alpha, double lambda, int max_iter, double tol)
    : alpha_(alpha), lambda_(lambda), max_iter_(max_iter), tol_(tol),
      intercept_(0.0), rmse_(0.0), r_squared_(0.0) {
    std::stringstream ss;
    ss << "Initialized ElasticNet with alpha=" << alpha << ", lambda=" << lambda 
       << ", max_iter=" << max_iter << ", tol=" << tol;
    LOG_INFO(ss.str());
}

bool ElasticNet::train(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
    if (X.rows() != y.rows() || X.cols() == 0) {
        LOG_ERROR("Invalid input dimensions for training");
        return false;
    }
    
    LOG_INFO("Starting ElasticNet training...");
    
    // Clear previous model state
    coefficients_.resize(0);
    intercept_ = 0.0;
    rmse_ = 0.0;
    r_squared_ = 0.0;
    
    // Keep only grid search stats if they exist, remove all other stats
    std::map<std::string, double> grid_search_stats;
    for (const auto& pair : stats_) {
        if (pair.first.substr(0, 11) == "Grid Search") {
            grid_search_stats[pair.first] = pair.second;
        }
    }
    stats_.clear();
    stats_ = grid_search_stats;  // Restore only grid search stats
    
    // Initialize training progress
    stats_["Training Progress"] = 0.0;
    
    // Standardize features using population statistics (divide by n)
    Eigen::MatrixXd X_std = X;
    Eigen::VectorXd X_mean = X.colwise().mean();
    Eigen::VectorXd X_std_dev = ((X.rowwise() - X_mean.transpose()).array().square().colwise().sum() / X.rows()).sqrt();
    
    for (int i = 0; i < X.cols(); ++i) {
        if (X_std_dev(i) < 1e-10) {  // Handle near-zero standard deviation
            X_std_dev(i) = 1.0;
            LOG_WARNING("Feature " + std::to_string(i) + " has near-zero standard deviation");
        }
        X_std.col(i) = (X.col(i).array() - X_mean(i)) / X_std_dev(i);
    }
    
    // Standardize target variable using population statistics
    double y_mean = y.mean();
    double y_std = std::sqrt((y.array() - y_mean).square().sum() / y.rows());
    if (y_std < 1e-10) {
        y_std = 1.0;
        LOG_WARNING("Target variable has near-zero standard deviation");
    }
    Eigen::VectorXd y_scaled = (y.array() - y_mean) / y_std;
    
    // Initialize coefficients
    coefficients_ = Eigen::VectorXd::Zero(X.cols());
    double prev_loss = std::numeric_limits<double>::max();
    
    // Coordinate descent
    for (int iter = 0; iter < max_iter_; ++iter) {
        // Update training progress in the stats map for the GUI
        stats_["Training Progress"] = static_cast<double>(iter) / max_iter_;
        
        // Update each coefficient
        for (int j = 0; j < X.cols(); ++j) {
            // Calculate r_j = y - X*beta + x_j*beta_j
            Eigen::VectorXd r = y_scaled - X_std * coefficients_;
            r += X_std.col(j) * coefficients_(j);
            
            // Calculate coordinate descent update
            double rho = X_std.col(j).dot(r);
            double xj_norm = X_std.col(j).squaredNorm();
            
            if (xj_norm < 1e-10) {
                coefficients_(j) = 0.0;
                continue;
            }
            
            // Soft thresholding
            double beta_j = rho / xj_norm;
            if (beta_j > lambda_ * alpha_) {
                beta_j -= lambda_ * alpha_;
            } else if (beta_j < -lambda_ * alpha_) {
                beta_j += lambda_ * alpha_;
            } else {
                beta_j = 0.0;
            }
            
            // Update coefficient
            coefficients_(j) = beta_j;
        }
        
        // Calculate loss
        double loss = (y_scaled - X_std * coefficients_).squaredNorm() / (2 * X.rows());
        loss += lambda_ * alpha_ * coefficients_.array().abs().sum();
        loss += 0.5 * lambda_ * (1.0 - alpha_) * coefficients_.squaredNorm();
        
        // Check convergence
        if (std::abs(prev_loss - loss) < tol_) {
            std::stringstream ss;
            ss << "Converged after " << iter + 1 << " iterations with loss=" << loss;
            LOG_INFO(ss.str());
            break;
        }
        
        prev_loss = loss;
        
        if (iter % 100 == 0) {
            std::stringstream ss;
            ss << "Iteration " << iter << ", loss=" << loss;
            LOG_DEBUG(ss.str());
        }
    }
    
    // Set final training progress to 1.0 (100%)
    stats_["Training Progress"] = 1.0;
    
    // Rescale coefficients back to original scale
    for (int j = 0; j < X.cols(); ++j) {
        coefficients_(j) = coefficients_(j) * y_std / X_std_dev(j);
    }
    
    // Calculate intercept
    intercept_ = y_mean - X_mean.dot(coefficients_);
    
    // Compute statistics
    computeStats(X, y);
    
    std::stringstream ss;
    ss << "Training completed. RMSE=" << rmse_ << ", R2=" << r_squared_;
    LOG_INFO(ss.str());
    
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
        LOG_ERROR("Invalid parameters for grid search");
        return {alpha_, lambda_};
    }
    
    LOG_INFO("Starting grid search for hyperparameters...");
    
    // Create a temporary map for grid search statistics to avoid 
    // polluting the main stats_ object
    std::map<std::string, double> grid_search_stats;
    
    // Standardize features using population statistics (divide by n)
    Eigen::MatrixXd X_std = X;
    Eigen::VectorXd mean = X.colwise().mean();
    Eigen::VectorXd std = ((X.rowwise() - mean.transpose()).array().square().colwise().sum() / X.rows()).sqrt();
    
    for (int i = 0; i < X.cols(); ++i) {
        if (std(i) < 1e-10) {  // Handle near-zero standard deviation
            std(i) = 1.0;
            LOG_WARNING("Feature " + std::to_string(i) + " has near-zero standard deviation");
        }
        X_std.col(i) = (X.col(i).array() - mean(i)) / std(i);
    }
    
    // Standardize target variable using population statistics
    double y_mean = y.mean();
    double y_std = std::sqrt((y.array() - y_mean).square().sum() / y.rows());
    if (y_std < 1e-10) {
        y_std = 1.0;
        LOG_WARNING("Target variable has near-zero standard deviation");
    }
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
    double best_r2 = -std::numeric_limits<double>::max();
    
    // Grid search with progress tracking
    int total_combinations = alpha_values.size() * lambda_values.size();
    int current_combination = 0;
    
    // Initialize progress in the main stats map
    stats_["Grid Search Progress"] = 0.0;
    
    // Grid search
    for (double alpha : alpha_values) {
        for (double lambda : lambda_values) {
            current_combination++;
            double fold_rmse_sum = 0.0;
            double fold_r2_sum = 0.0;
            bool early_stop = false;
            int valid_folds = 0;
            
            // Update progress in both the temporary stats and the main stats object
            // This ensures the GUI can see the progress
            double progress = static_cast<double>(current_combination) / total_combinations;
            grid_search_stats["Grid Search Progress"] = progress;
            stats_["Grid Search Progress"] = progress;
            
            std::stringstream ss;
            ss << "Testing alpha=" << alpha << ", lambda=" << lambda 
               << " (" << current_combination << "/" << total_combinations << ")";
            LOG_DEBUG(ss.str());
            
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
                    
                    // Calculate RMSE and R² on validation set
                    double fold_rmse = std::sqrt((predictions - y_val).array().square().mean());
                    double fold_r2 = 1.0 - ((predictions - y_val).array().square().sum() / 
                                          (y_val.array() - y_val.mean()).square().sum());
                    
                    // Check for numerical stability
                    if (!std::isfinite(fold_rmse) || !std::isfinite(fold_r2) || 
                        fold_rmse > 1e6 || fold_r2 < -1e6) {
                        early_stop = true;
                        break;
                    }
                    
                    // Early stopping if RMSE is much worse than best
                    if (best_rmse < std::numeric_limits<double>::max() && 
                        fold_rmse > 5.0 * best_rmse) {  // Made stricter
                        early_stop = true;
                        break;
                    }
                    
                    fold_rmse_sum += fold_rmse;
                    fold_r2_sum += fold_r2;
                    valid_folds++;
                    
                } catch (const std::exception&) {
                    // If any exception occurs, skip this combination
                    early_stop = true;
                    break;
                }
            }
            
            if (!early_stop && valid_folds > 0) {
                // Average RMSE and R² across valid folds
                double avg_rmse = fold_rmse_sum / valid_folds;
                double avg_r2 = fold_r2_sum / valid_folds;
                
                // Update best hyperparameters if both RMSE and R² are improved
                if (avg_rmse < best_rmse && avg_r2 > best_r2) {
                    best_rmse = avg_rmse;
                    best_r2 = avg_r2;
                    best_alpha = alpha;
                    best_lambda = lambda;
                    
                    std::stringstream ss;
                    ss << "New best parameters found - alpha=" << alpha 
                       << ", lambda=" << lambda << ", RMSE=" << avg_rmse 
                       << ", R2=" << avg_r2;
                    LOG_INFO(ss.str());
                    
                    // Store best parameters in grid search stats (not in the model's stats)
                    grid_search_stats["Best RMSE"] = best_rmse;
                    grid_search_stats["Best R2"] = best_r2;
                    grid_search_stats["Best Alpha"] = best_alpha;
                    grid_search_stats["Best Lambda"] = best_lambda;
                }
            }
        }
    }
    
    // Set final grid search progress to 1.0 (100%)
    stats_["Grid Search Progress"] = 1.0;
    
    // Store the best grid search results in the model's stats
    // This way we only update the model's stats once, with the best values
    stats_["Grid Search Best RMSE"] = grid_search_stats["Best RMSE"];
    stats_["Grid Search Best R2"] = grid_search_stats["Best R2"];
    stats_["Grid Search Best Alpha"] = grid_search_stats["Best Alpha"];
    stats_["Grid Search Best Lambda"] = grid_search_stats["Best Lambda"];
    
    std::stringstream ss;
    ss << "Grid search completed. Best parameters - alpha=" << best_alpha 
       << ", lambda=" << best_lambda << ", RMSE=" << best_rmse 
       << ", R2=" << best_r2;
    LOG_INFO(ss.str());
    
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
    
    // Store grid search results temporarily
    std::map<std::string, double> grid_search_stats;
    for (const auto& pair : stats_) {
        if (pair.first.substr(0, 11) == "Grid Search") {
            grid_search_stats[pair.first] = pair.second;
        }
    }
    
    // Create new stats map with current statistics
    std::map<std::string, double> new_stats = {
        {"RMSE", rmse_},
        {"MSE", mse},
        {"MAE", mae},
        {"R2", r_squared_},
        {"Adjusted R2", adj_r_squared},
        {"AIC", aic},
        {"BIC", bic},
        {"L1 Norm", l1_norm},
        {"L2 Norm", l2_norm},
        {"Intercept SE", std_errors(0)},
        {"Intercept t-value", t_values(0)}
    };
    
    // Store coefficient statistics
    for (int i = 0; i < p; ++i) {
        new_stats["SE_" + std::to_string(i)] = std_errors(i + 1);
        new_stats["t_value_" + std::to_string(i)] = t_values(i + 1);
    }
    
    // Restore grid search stats
    for (const auto& pair : grid_search_stats) {
        new_stats[pair.first] = pair.second;
    }
    
    // Replace stats with new stats
    stats_ = new_stats;
    
    // Log important statistics
    std::stringstream ss;
    ss << "Model statistics - RMSE=" << rmse_ << ", R2=" << r_squared_ 
       << ", Adjusted R2=" << adj_r_squared << ", AIC=" << aic << ", BIC=" << bic;
    LOG_INFO(ss.str());
    
    // Log coefficient statistics
    for (int i = 0; i < p; ++i) {
        if (std::abs(coefficients_(i)) > 1e-6) {  // Only log non-zero coefficients
            std::stringstream coef_ss;
            coef_ss << "Coefficient " << i << " = " << coefficients_(i) 
                   << " (t-value=" << t_values(i + 1) << ")";
            LOG_DEBUG(coef_ss.str());
        }
    }
}

Eigen::VectorXd ElasticNet::getFeatureImportance() const {
    // For ElasticNet, feature importance is the absolute value of coefficients
    return coefficients_.array().abs();
}

} // namespace DataAnalyzer 