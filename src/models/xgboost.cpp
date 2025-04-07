#include "../../include/model.h"
#include "../../include/logger.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <queue>
#include <limits>
#include <sstream>

namespace DataAnalyzer {

XGBoost::XGBoost(int n_estimators, double learning_rate, int max_depth, double subsample,
                double min_child_weight, double lambda, double alpha, double gamma,
                int max_leaves, int early_stopping_rounds)
    : n_estimators_(n_estimators), learning_rate_(learning_rate), max_depth_(max_depth),
      subsample_(subsample), min_child_weight_(min_child_weight), lambda_(lambda),
      alpha_(alpha), gamma_(gamma), max_leaves_(max_leaves), 
      early_stopping_rounds_(early_stopping_rounds), 
      base_prediction_(0.0), rmse_(0.0), r_squared_(0.0) {
    std::stringstream ss;
    ss << "Initialized model with n_estimators=" << n_estimators
       << ", learning_rate=" << learning_rate
       << ", max_depth=" << max_depth
       << ", subsample=" << subsample
       << ", min_child_weight=" << min_child_weight
       << ", lambda=" << lambda
       << ", alpha=" << alpha
       << ", gamma=" << gamma
       << ", max_leaves=" << max_leaves
       << ", early_stopping_rounds=" << early_stopping_rounds;
    LOG_CLASS_INFO("XGBoost", ss.str());
}

bool XGBoost::train(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
    if (X.rows() != y.rows() || X.rows() == 0) {
        LOG_CLASS_ERROR("XGBoost", "Invalid input dimensions: X rows=" + std::to_string(X.rows()) + 
                      ", y size=" + std::to_string(y.rows()));
        return false;
    }
    
    LOG_CLASS_INFO("XGBoost", "Starting training with " + std::to_string(X.rows()) + 
                  " samples and " + std::to_string(X.cols()) + " features");
    
    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Clear previous trees
    trees_.clear();
    
    // Clear previous model state
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
    
    // Standardize features
    Eigen::MatrixXd X_train = standardizeFeatures(X);
    Eigen::VectorXd y_train = y;
    
    // Setup validation set for early stopping if enabled
    Eigen::MatrixXd X_val;
    Eigen::VectorXd y_val;
    
    if (early_stopping_rounds_ > 0) {
        // Use 20% of the data as validation set
        int train_size = static_cast<int>(X.rows() * 0.8);
        
        // Create random indices
        std::vector<int> indices(X.rows());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), gen);
        
        // Create train and validation split
        Eigen::MatrixXd X_train_all = X_train;
        Eigen::VectorXd y_train_all = y_train;
        
        X_train = Eigen::MatrixXd(train_size, X.cols());
        y_train = Eigen::VectorXd(train_size);
        
        X_val = Eigen::MatrixXd(X.rows() - train_size, X.cols());
        y_val = Eigen::VectorXd(X.rows() - train_size);
        
        for (int i = 0; i < train_size; ++i) {
            X_train.row(i) = X_train_all.row(indices[i]);
            y_train(i) = y_train_all(indices[i]);
        }
        
        for (int i = 0; i < X.rows() - train_size; ++i) {
            X_val.row(i) = X_train_all.row(indices[train_size + i]);
            y_val(i) = y_train_all(indices[train_size + i]);
        }
        
        LOG_CLASS_DEBUG("XGBoost", "Using early stopping with validation set of " + 
                      std::to_string(X_val.rows()) + " samples");
    }
    
    // Initial prediction is the mean of the target values
    base_prediction_ = y_train.mean();
    LOG_CLASS_DEBUG("XGBoost", "Initial prediction (base value): " + std::to_string(base_prediction_));
    
    // Initialize residuals and predictions
    Eigen::VectorXd residuals = y_train - Eigen::VectorXd::Constant(y_train.size(), base_prediction_);
    Eigen::VectorXd predictions = Eigen::VectorXd::Constant(y_train.size(), base_prediction_);
    
    // Variables for early stopping
    double best_val_mse = std::numeric_limits<double>::max();
    int rounds_no_improve = 0;
    int best_iteration = 0;
    
    // Build boosting rounds
    for (int i = 0; i < n_estimators_; ++i) {
        // Update training progress
        stats_["Training Progress"] = static_cast<double>(i) / n_estimators_;
        
        // Apply gradient clipping to residuals
        for (int j = 0; j < residuals.size(); ++j) {
            residuals(j) = clipPrediction(residuals(j), 1e4);
        }
        
        // Subsample the data if subsample < 1.0
        Eigen::MatrixXd X_sub;
        Eigen::VectorXd residuals_sub;
        
        if (subsample_ < 1.0) {
            int sample_size = static_cast<int>(X_train.rows() * subsample_);
            X_sub = Eigen::MatrixXd(sample_size, X_train.cols());
            residuals_sub = Eigen::VectorXd(sample_size);
            
            // Random sampling without replacement
            std::vector<int> indices(X_train.rows());
            std::iota(indices.begin(), indices.end(), 0);
            std::shuffle(indices.begin(), indices.end(), gen);
            
            for (int j = 0; j < sample_size; ++j) {
                X_sub.row(j) = X_train.row(indices[j]);
                residuals_sub(j) = residuals(indices[j]);
            }
            LOG_CLASS_DEBUG("XGBoost", "Round " + std::to_string(i+1) + ": Using " + 
                          std::to_string(sample_size) + " samples (subsample=" + 
                          std::to_string(subsample_) + ")");
        } else {
            X_sub = X_train;
            residuals_sub = residuals;
            LOG_CLASS_DEBUG("XGBoost", "Round " + std::to_string(i+1) + ": Using all samples");
        }
        
        // Build a regression tree
        Tree tree;
        buildRegressionTree(X_sub, residuals_sub, tree, 0, max_depth_);
        trees_.push_back(tree);
        
        // Update predictions and residuals with clipping
        for (int j = 0; j < X_train.rows(); ++j) {
            double prediction = predictTree(tree, X_train.row(j));
            prediction = clipPrediction(prediction);
            predictions(j) += learning_rate_ * prediction;
            residuals(j) = y_train(j) - predictions(j);
        }
        
        // Calculate training MSE
        double train_mse = residuals.array().square().mean();
        
        // Log progress for every 10% or for specific iterations
        if (i == 0 || i == n_estimators_ - 1 || (n_estimators_ > 10 && i % (n_estimators_ / 10) == 0)) {
            LOG_CLASS_INFO("XGBoost", "Round " + std::to_string(i+1) + "/" + 
                         std::to_string(n_estimators_) + ": MSE = " + std::to_string(train_mse));
        }
        
        // Early stopping check
        if (early_stopping_rounds_ > 0) {
            // Calculate validation MSE
            Eigen::VectorXd val_predictions = Eigen::VectorXd::Constant(X_val.rows(), base_prediction_);
            for (const auto& t : trees_) {
                for (int j = 0; j < X_val.rows(); ++j) {
                    double pred = predictTree(t, X_val.row(j));
                    val_predictions(j) += learning_rate_ * clipPrediction(pred);
                }
            }
            
            double val_mse = (y_val - val_predictions).array().square().mean();
            
            // Check if validation error improved
            if (val_mse < best_val_mse) {
                best_val_mse = val_mse;
                rounds_no_improve = 0;
                best_iteration = i;
                
                LOG_CLASS_DEBUG("XGBoost", "Round " + std::to_string(i+1) + 
                              ": New best validation MSE = " + std::to_string(val_mse));
            } else {
                rounds_no_improve++;
                
                if (rounds_no_improve >= early_stopping_rounds_) {
                    LOG_CLASS_INFO("XGBoost", "Early stopping at round " + std::to_string(i+1) + 
                                 " (best was round " + std::to_string(best_iteration+1) + 
                                 " with MSE = " + std::to_string(best_val_mse) + ")");
                    
                    // Keep only the best trees
                    if (best_iteration + 1 < i + 1) {
                        trees_.resize(best_iteration + 1);
                    }
                    
                    break;
                }
            }
        }
    }
    
    // Set final training progress to 1.0 (100%)
    stats_["Training Progress"] = 1.0;
    
    // Compute statistics
    computeStats(X, y);
    LOG_CLASS_INFO("XGBoost", "Training complete. RMSE: " + std::to_string(rmse_) + 
                  ", R2: " + std::to_string(r_squared_));
    
    return true;
}

Eigen::VectorXd XGBoost::predict(const Eigen::MatrixXd& X) const {
    LOG_CLASS_DEBUG("XGBoost", "Predicting for " + std::to_string(X.rows()) + " samples");
    
    // Standardize features
    Eigen::MatrixXd X_std = standardizeFeatures(X);
    
    // Initialize predictions with base prediction
    Eigen::VectorXd predictions = Eigen::VectorXd::Constant(X_std.rows(), base_prediction_);
    
    // Sum up the predictions from each tree with clipping
    for (const auto& tree : trees_) {
        for (int i = 0; i < X_std.rows(); ++i) {
            double prediction = predictTree(tree, X_std.row(i));
            prediction = clipPrediction(prediction);
            predictions(i) += learning_rate_ * prediction;
        }
    }
    
    // Final clipping to prevent extreme values
    for (int i = 0; i < predictions.size(); ++i) {
        predictions(i) = clipPrediction(predictions(i));
    }
    
    return predictions;
}

std::map<std::string, double> XGBoost::getStats() const {
    return stats_;
}

std::string XGBoost::getDescription() const {
    return "XGBoost (Extreme Gradient Boosting) is an optimized gradient boosting "
           "library designed for efficient and scalable training. It builds an ensemble "
           "of decision trees sequentially, with each new tree correcting errors made "
           "by the previous ones. XGBoost offers high prediction accuracy and speed, "
           "with regularization to prevent overfitting.";
}

Eigen::VectorXd XGBoost::getCoefficients() const {
    // Tree-based models don't have coefficients in the traditional sense
    // Return an empty vector
    return Eigen::VectorXd();
}

std::map<std::string, double> XGBoost::getHyperparameters() const {
    return {
        {"n_estimators", static_cast<double>(n_estimators_)},
        {"learning_rate", learning_rate_},
        {"max_depth", static_cast<double>(max_depth_)},
        {"subsample", subsample_},
        {"min_child_weight", min_child_weight_},
        {"lambda", lambda_},
        {"alpha", alpha_},
        {"gamma", gamma_},
        {"max_leaves", static_cast<double>(max_leaves_)},
        {"early_stopping_rounds", static_cast<double>(early_stopping_rounds_)}
    };
}

std::tuple<int, double, int, double> XGBoost::gridSearch(
    const Eigen::MatrixXd& X, 
    const Eigen::VectorXd& y,
    const std::vector<int>& n_estimators_values,
    const std::vector<double>& learning_rate_values,
    const std::vector<int>& max_depth_values,
    const std::vector<double>& subsample_values,
    const std::vector<double>& lambda_values,
    const std::vector<double>& alpha_values,
    int k) {
    
    if (X.rows() != y.rows() || n_estimators_values.empty() || 
        learning_rate_values.empty() || max_depth_values.empty() || 
        subsample_values.empty() || k <= 1) {
        LOG_CLASS_ERROR("XGBoost", "Invalid grid search parameters");
        return {n_estimators_, learning_rate_, max_depth_, subsample_};
    }
    
    LOG_CLASS_INFO("XGBoost", "Starting grid search with " + std::to_string(X.rows()) + 
                  " samples, " + std::to_string(X.cols()) + " features, " + 
                  std::to_string(k) + " folds");
    
    // Create a temporary map for grid search statistics to avoid 
    // polluting the main stats_ object
    std::map<std::string, double> grid_search_stats;
    
    // Standardize features once for the entire grid search
    Eigen::MatrixXd X_std = standardizeFeatures(X);
    
    // Prepare k-fold cross-validation indices once
    std::vector<std::vector<int>> fold_indices(k);
    std::vector<int> indices(X_std.rows());
    std::iota(indices.begin(), indices.end(), 0);
    
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);
    
    int fold_size = X_std.rows() / k;
    for (int fold = 0; fold < k; ++fold) {
        int start_idx = fold * fold_size;
        int end_idx = (fold == k - 1) ? X_std.rows() : (fold + 1) * fold_size;
        fold_indices[fold] = std::vector<int>(indices.begin() + start_idx, indices.begin() + end_idx);
    }
    
    // Best hyperparameters
    int best_n_estimators = n_estimators_;
    double best_learning_rate = learning_rate_;
    int best_max_depth = max_depth_;
    double best_subsample = subsample_;
    double best_lambda = lambda_;
    double best_alpha = alpha_;
    double best_rmse = std::numeric_limits<double>::max();
    
    // Grid search with progress tracking
    int total_combinations = n_estimators_values.size() * learning_rate_values.size() * 
                           max_depth_values.size() * subsample_values.size() *
                           lambda_values.size() * alpha_values.size();
    int current_combination = 0;
    
    LOG_CLASS_INFO("XGBoost", "Grid search will evaluate " + 
                  std::to_string(total_combinations) + " combinations");
    
    // Initialize progress in the main stats map
    stats_["Grid Search Progress"] = 0.0;
    
    // Use a priority queue to try the most promising combinations first
    // This is a heuristic - we try smaller, regularized models first
    struct ModelConfig {
        int n_estimators;
        double learning_rate;
        int max_depth;
        double subsample;
        double lambda;
        double alpha;
        double priority;  // Priority score (lower is better for queue)
        
        bool operator<(const ModelConfig& other) const {
            return priority > other.priority;  // Lower priority values come first
        }
    };
    
    std::priority_queue<ModelConfig> configs;
    
    // Add all configurations with a priority score
    for (int n_estimators : n_estimators_values) {
        for (double learning_rate : learning_rate_values) {
            for (int max_depth : max_depth_values) {
                for (double subsample : subsample_values) {
                    for (double lambda : lambda_values) {
                        for (double alpha : alpha_values) {
                            // Calculate priority based on model complexity and regularization
                            // Favor smaller models with more regularization
                            double priority = n_estimators * max_depth / (lambda + 0.1);
                            
                            configs.push({n_estimators, learning_rate, max_depth, 
                                         subsample, lambda, alpha, priority});
                        }
                    }
                }
            }
        }
    }
    
    // Try configurations in order of priority
    while (!configs.empty()) {
        ModelConfig config = configs.top();
        configs.pop();
        
        current_combination++;
        double fold_rmse_sum = 0.0;
        bool early_stop = false;
        
        // Update progress in both the temporary stats and the main stats object
        double progress = static_cast<double>(current_combination) / total_combinations;
        grid_search_stats["Grid Search Progress"] = progress;
        stats_["Grid Search Progress"] = progress;
        
        LOG_CLASS_DEBUG("XGBoost", "Evaluating combination " + std::to_string(current_combination) + 
                      "/" + std::to_string(total_combinations) + 
                      ": n_estimators=" + std::to_string(config.n_estimators) + 
                      ", learning_rate=" + std::to_string(config.learning_rate) + 
                      ", max_depth=" + std::to_string(config.max_depth) + 
                      ", subsample=" + std::to_string(config.subsample) +
                      ", lambda=" + std::to_string(config.lambda) +
                      ", alpha=" + std::to_string(config.alpha));
        
        // K-fold cross-validation
        for (int fold = 0; fold < k && !early_stop; ++fold) {
            // Create validation set
            const std::vector<int>& val_indices = fold_indices[fold];
            
            // Create training set
            std::vector<int> train_indices;
            train_indices.reserve(X_std.rows() - val_indices.size());
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
                y_train(i) = y(train_indices[i]);
            }
            
            for (size_t i = 0; i < val_indices.size(); ++i) {
                X_val.row(i) = X_std.row(val_indices[i]);
                y_val(i) = y(val_indices[i]);
            }
            
            // Train model with current hyperparameters
            // Use a small number of early stopping rounds to speed up training
            XGBoost model(config.n_estimators, config.learning_rate, config.max_depth, 
                         config.subsample, min_child_weight_, config.lambda,
                         config.alpha, gamma_, max_leaves_, 15);  // Early stopping after 15 rounds
            
            try {
                if (!model.train(X_train, y_train)) {
                    // If training fails, skip this combination
                    LOG_CLASS_WARNING("XGBoost", "Training failed for fold " + 
                                   std::to_string(fold+1) + ", skipping combination");
                    early_stop = true;
                    break;
                }
                
                // Evaluate on validation set
                Eigen::VectorXd predictions = model.predict(X_val);
                double fold_rmse = std::sqrt((predictions - y_val).array().square().mean());
                
                // Early stopping if RMSE is much worse than best
                if (fold_rmse > 1.5 * best_rmse) {
                    LOG_CLASS_DEBUG("XGBoost", "Early stopping for poor performance, RMSE=" + 
                                  std::to_string(fold_rmse) + " vs best=" + 
                                  std::to_string(best_rmse));
                    early_stop = true;
                    break;
                }
                
                fold_rmse_sum += fold_rmse;
                LOG_CLASS_DEBUG("XGBoost", "Fold " + std::to_string(fold+1) + 
                              " RMSE: " + std::to_string(fold_rmse));
            } catch (const std::exception& e) {
                // If any exception occurs, skip this combination
                LOG_CLASS_WARNING("XGBoost", "Exception in fold " + std::to_string(fold+1) + 
                               ": " + e.what());
                early_stop = true;
                break;
            }
        }
        
        if (!early_stop) {
            // Average RMSE across all folds
            double avg_rmse = fold_rmse_sum / k;
            
            LOG_CLASS_INFO("XGBoost", "Combination " + std::to_string(current_combination) +  
                         ": Average RMSE = " + std::to_string(avg_rmse));
            
            // Update best hyperparameters if RMSE is improved
            if (avg_rmse < best_rmse) {
                best_rmse = avg_rmse;
                best_n_estimators = config.n_estimators;
                best_learning_rate = config.learning_rate;
                best_max_depth = config.max_depth;
                best_subsample = config.subsample;
                best_lambda = config.lambda;
                best_alpha = config.alpha;
                
                LOG_CLASS_INFO("XGBoost", "New best combination found! RMSE = " + 
                             std::to_string(best_rmse));
                
                // Store best parameters in grid search stats
                grid_search_stats["Best RMSE"] = best_rmse;
                grid_search_stats["Best N Estimators"] = static_cast<double>(best_n_estimators);
                grid_search_stats["Best Learning Rate"] = best_learning_rate;
                grid_search_stats["Best Max Depth"] = static_cast<double>(best_max_depth);
                grid_search_stats["Best Subsample"] = best_subsample;
                grid_search_stats["Best Lambda"] = best_lambda;
                grid_search_stats["Best Alpha"] = best_alpha;
            }
        }
    }
    
    // Set final grid search progress to 1.0 (100%)
    stats_["Grid Search Progress"] = 1.0;
    
    // Store the best grid search results in the model's stats
    stats_["Grid Search Best RMSE"] = grid_search_stats["Best RMSE"];
    stats_["Grid Search Best N Estimators"] = grid_search_stats["Best N Estimators"];
    stats_["Grid Search Best Learning Rate"] = grid_search_stats["Best Learning Rate"];
    stats_["Grid Search Best Max Depth"] = grid_search_stats["Best Max Depth"];
    stats_["Grid Search Best Subsample"] = grid_search_stats["Best Subsample"];
    stats_["Grid Search Best Lambda"] = grid_search_stats["Best Lambda"];
    stats_["Grid Search Best Alpha"] = grid_search_stats["Best Alpha"];
    
    // Update the model's hyperparameters with the best values
    lambda_ = best_lambda;
    alpha_ = best_alpha;
    
    std::stringstream ss;
    ss << "Grid search complete. Best hyperparameters: "
       << "n_estimators=" << best_n_estimators
       << ", learning_rate=" << best_learning_rate
       << ", max_depth=" << best_max_depth
       << ", subsample=" << best_subsample
       << ", lambda=" << best_lambda
       << ", alpha=" << best_alpha
       << ", RMSE=" << best_rmse;
    LOG_CLASS_INFO("XGBoost", ss.str());
    
    return {best_n_estimators, best_learning_rate, best_max_depth, best_subsample};
}

void XGBoost::computeStats(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
    Eigen::VectorXd predictions = predict(X);
    
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
    
    // Calculate feature importance
    Eigen::VectorXd importance = getFeatureImportance();
    
    // Calculate average tree depth and leaf count
    double avg_depth = 0.0;
    double avg_leaves = 0.0;
    double avg_nodes = 0.0;
    
    for (const auto& tree : trees_) {
        // Calculate tree depth
        int max_depth = 0;
        std::vector<bool> is_leaf(tree.feature_indices.size(), true);
        
        for (size_t i = 0; i < tree.feature_indices.size(); ++i) {
            if (tree.feature_indices[i] >= 0) { // Non-leaf node
                is_leaf[i] = false;
                if (i > 0) { // Not root
                    int parent = (i - 1) / 2;
                    int depth = 1;
                    while (parent > 0) {
                        depth++;
                        parent = (parent - 1) / 2;
                    }
                    max_depth = std::max(max_depth, depth);
                }
            }
        }
        
        avg_depth += max_depth;
        
        // Count leaf nodes and total nodes
        int leaf_count = std::count(is_leaf.begin(), is_leaf.end(), true);
        avg_leaves += leaf_count;
        avg_nodes += tree.feature_indices.size();
    }
    
    if (!trees_.empty()) {
        avg_depth /= trees_.size();
        avg_leaves /= trees_.size();
        avg_nodes /= trees_.size();
    }
    
    // Calculate model complexity metrics
    double model_complexity = avg_depth * avg_nodes / (lambda_ + 0.1);
    
    // Store grid search results temporarily
    std::map<std::string, double> grid_search_stats;
    for (const auto& pair : stats_) {
        if (pair.first.substr(0, 11) == "Grid Search") {
            grid_search_stats[pair.first] = pair.second;
        }
    }
    
    // Store training progress
    double training_progress = 1.0;
    if (stats_.find("Training Progress") != stats_.end()) {
        training_progress = stats_["Training Progress"];
    }
    
    // Create new stats map with current statistics
    std::map<std::string, double> new_stats = {
        {"RMSE", rmse_},
        {"MSE", mse},
        {"MAE", mae},
        {"R2", r_squared_},
        {"Number of Trees", static_cast<double>(trees_.size())},
        {"Average Tree Depth", avg_depth},
        {"Average Leaf Nodes", avg_leaves},
        {"Average Nodes per Tree", avg_nodes},
        {"Model Complexity", model_complexity},
        {"Training Loss", mse}, // Using MSE as the training loss
        {"Learning Rate", learning_rate_},
        {"Max Tree Depth", static_cast<double>(max_depth_)},
        {"Subsample Ratio", subsample_},
        {"L2 Regularization (Lambda)", lambda_},
        {"L1 Regularization (Alpha)", alpha_},
        {"Min Split Gain (Gamma)", gamma_},
        {"Min Child Weight", min_child_weight_},
        {"Max Leaves", static_cast<double>(max_leaves_)},
        {"Early Stopping Rounds", static_cast<double>(early_stopping_rounds_)},
        {"Training Progress", training_progress}
    };
    
    // Store feature importance scores with proper naming
    if (importance.size() > 0) {
        for (int i = 0; i < importance.size(); ++i) {
            new_stats["Feature " + std::to_string(i) + " Importance"] = importance(i);
        }
    }
    
    // Restore grid search stats
    for (const auto& pair : grid_search_stats) {
        new_stats[pair.first] = pair.second;
    }
    
    // Replace stats with new stats
    stats_ = new_stats;
}

double XGBoost::computeAverageTreeDepth() const {
    double total_depth = 0.0;
    for (const auto& tree : trees_) {
        int max_depth = 0;
        std::function<void(int, int)> computeDepth = [&](int node_idx, int current_depth) {
            if (node_idx >= 0 && static_cast<size_t>(node_idx) < tree.feature_indices.size()) {
                max_depth = std::max(max_depth, current_depth);
                if (tree.feature_indices[node_idx] >= 0) {
                    computeDepth(tree.left_children[node_idx], current_depth + 1);
                    computeDepth(tree.right_children[node_idx], current_depth + 1);
                }
            }
        };
        computeDepth(0, 0);
        total_depth += max_depth;
    }
    return total_depth / trees_.size();
}

double XGBoost::predictTree(const Tree& tree, const Eigen::VectorXd& x, int node_idx) const {
    // Base case: leaf node
    if (node_idx >= static_cast<int>(tree.feature_indices.size()) || 
        tree.feature_indices[node_idx] < 0) {
        return tree.values[node_idx];
    }
    
    // If feature value <= threshold, go left; otherwise, go right
    int feature_idx = tree.feature_indices[node_idx];
    double threshold = tree.thresholds[node_idx];
    
    if (x(feature_idx) <= threshold) {
        return predictTree(tree, x, tree.left_children[node_idx]);
    } else {
        return predictTree(tree, x, tree.right_children[node_idx]);
    }
}

// Helper method to build a regression tree
void XGBoost::buildRegressionTree(const Eigen::MatrixXd& X,
                                const Eigen::VectorXd& residuals,
                                Tree& tree,
                                [[maybe_unused]] int depth,
                                int max_depth) {
    // Initialize tree structure
    tree.feature_indices.clear();
    tree.thresholds.clear();
    tree.values.clear();
    tree.left_children.clear();
    tree.right_children.clear();
    
    // Create a queue for BFS tree construction
    std::queue<std::tuple<int, std::vector<int>, int>> queue;
    
    // Start with all samples
    std::vector<int> sample_indices(X.rows());
    std::iota(sample_indices.begin(), sample_indices.end(), 0);
    
    // Root node
    int node_id = 0;
    queue.push(std::make_tuple(node_id, sample_indices, 0)); // (node_id, samples, depth)
    
    // Track the number of leaf nodes as we build the tree
    int leaf_count = 0;
    
    while (!queue.empty()) {
        auto [current_node, current_samples, current_depth] = queue.front();
        queue.pop();
        
        // Extract the samples for this node
        Eigen::VectorXd node_targets(current_samples.size());
        for (size_t i = 0; i < current_samples.size(); ++i) {
            node_targets(i) = residuals(current_samples[i]);
        }
        
        // Calculate the gradient sum and hessian (approximated as 1.0 per instance)
        double gradient_sum = node_targets.sum();
        double hessian_sum = current_samples.size() * 1.0;  // Simplified hessian
        
        // Calculate the optimal value for the leaf with regularization
        double leaf_weight = -gradient_sum / (hessian_sum + lambda_);
        
        // Apply L1 regularization
        if (alpha_ > 0) {
            if (leaf_weight > 0) {
                leaf_weight = std::max(0.0, leaf_weight - alpha_ / (hessian_sum + lambda_));
            } else {
                leaf_weight = std::min(0.0, leaf_weight + alpha_ / (hessian_sum + lambda_));
            }
        }
        
        // Clip the leaf value to prevent extreme predictions
        leaf_weight = clipPrediction(leaf_weight);
        
        // Extend the tree if needed
        while (tree.feature_indices.size() <= static_cast<size_t>(current_node)) {
            tree.feature_indices.push_back(-1);
            tree.thresholds.push_back(0.0);
            tree.values.push_back(0.0);
            tree.left_children.push_back(-1);
            tree.right_children.push_back(-1);
        }
        
        // Make this a leaf node if:
        // 1. Max depth is reached
        // 2. Too few samples
        // 3. Hessian sum is less than min_child_weight
        // 4. Max number of leaves is reached (if specified)
        bool should_make_leaf = 
            current_depth >= max_depth || 
            current_samples.size() <= 1 ||
            hessian_sum < min_child_weight_ ||
            (max_leaves_ > 0 && leaf_count >= max_leaves_);
        
        if (should_make_leaf) {
            tree.feature_indices[current_node] = -1;
            tree.values[current_node] = leaf_weight;
            leaf_count++;
            continue;
        }
        
        // Find the best split
        int best_feature = -1;
        double best_threshold = 0.0;
        double best_gain = gamma_;  // Minimum gain threshold
        std::vector<int> best_left_samples, best_right_samples;
        
        // Try each feature
        for (int feature = 0; feature < X.cols(); ++feature) {
            // Get unique values for this feature from the samples
            std::vector<double> feature_values;
            for (int idx : current_samples) {
                feature_values.push_back(X(idx, feature));
            }
            
            std::sort(feature_values.begin(), feature_values.end());
            feature_values.erase(std::unique(feature_values.begin(), feature_values.end()), 
                                feature_values.end());
            
            // Skip if there's only one value
            if (feature_values.size() <= 1) {
                continue;
            }
            
            // Try each threshold (midpoint between consecutive values)
            for (size_t i = 0; i < feature_values.size() - 1; ++i) {
                double threshold = (feature_values[i] + feature_values[i + 1]) / 2.0;
                
                // Split samples
                std::vector<int> left_samples, right_samples;
                for (int idx : current_samples) {
                    if (X(idx, feature) <= threshold) {
                        left_samples.push_back(idx);
                    } else {
                        right_samples.push_back(idx);
                    }
                }
                
                // Skip if one side is empty or doesn't meet min_child_weight
                if (left_samples.empty() || right_samples.empty() ||
                    left_samples.size() < min_child_weight_ || 
                    right_samples.size() < min_child_weight_) {
                    continue;
                }
                
                // Calculate left node statistics
                Eigen::VectorXd left_targets(left_samples.size());
                for (size_t j = 0; j < left_samples.size(); ++j) {
                    left_targets(j) = residuals(left_samples[j]);
                }
                double left_gradient_sum = left_targets.sum();
                double left_hessian_sum = left_samples.size() * 1.0;
                
                // Calculate right node statistics
                Eigen::VectorXd right_targets(right_samples.size());
                for (size_t j = 0; j < right_samples.size(); ++j) {
                    right_targets(j) = residuals(right_samples[j]);
                }
                double right_gradient_sum = right_targets.sum();
                double right_hessian_sum = right_samples.size() * 1.0;
                
                // Calculate the regularized gain (variance reduction with regularization)
                // This is the XGBoost split finding algorithm formula
                double gain = 0.5 * (
                    std::pow(left_gradient_sum, 2) / (left_hessian_sum + lambda_) +
                    std::pow(right_gradient_sum, 2) / (right_hessian_sum + lambda_) -
                    std::pow(gradient_sum, 2) / (hessian_sum + lambda_)
                ) - gamma_;  // Subtract minimum split gain
                
                // Update best split if this is better
                if (gain > best_gain) {
                    best_gain = gain;
                    best_feature = feature;
                    best_threshold = threshold;
                    best_left_samples = left_samples;
                    best_right_samples = right_samples;
                }
            }
        }
        
        // If no good split found (gain <= gamma_), make this a leaf node
        if (best_feature == -1) {
            tree.feature_indices[current_node] = -1;
            tree.values[current_node] = leaf_weight;
            leaf_count++;
            continue;
        }
        
        // Otherwise, create a split node
        tree.feature_indices[current_node] = best_feature;
        tree.thresholds[current_node] = best_threshold;
        tree.values[current_node] = leaf_weight;  // Store the node value anyway
        
        // Create child nodes
        int left_child = tree.feature_indices.size();
        int right_child = left_child + 1;
        
        tree.left_children[current_node] = left_child;
        tree.right_children[current_node] = right_child;
        
        // Add children to the queue
        queue.push(std::make_tuple(left_child, best_left_samples, current_depth + 1));
        queue.push(std::make_tuple(right_child, best_right_samples, current_depth + 1));
    }
}

Eigen::VectorXd XGBoost::getFeatureImportance() const {
    if (trees_.empty()) {
        return Eigen::VectorXd();
    }

    // Get number of features from the first tree's first split
    int n_features = 0;
    for (const auto& tree : trees_) {
        for (int feature_idx : tree.feature_indices) {
            if (feature_idx >= 0) {  // Valid feature index
                n_features = std::max(n_features, feature_idx + 1);
            }
        }
    }
    
    // Initialize feature importance vector with correct size
    Eigen::VectorXd importance = Eigen::VectorXd::Zero(n_features);
    
    // For each tree, accumulate importance scores
    for (const auto& tree : trees_) {
        for (size_t i = 0; i < tree.feature_indices.size(); ++i) {
            int feature_idx = tree.feature_indices[i];
            if (feature_idx >= 0) {  // Skip leaf nodes
                // Add importance based on the number of samples affected by this split
                importance(feature_idx) += 1.0;
            }
        }
    }
    
    // Normalize importance scores
    if (importance.sum() > 0) {
        importance /= importance.sum();  // Make them sum to 1
    }
    
    return importance;
}

Eigen::MatrixXd XGBoost::standardizeFeatures(const Eigen::MatrixXd& X) const {
    Eigen::MatrixXd X_std = X;
    
    // Compute mean and standard deviation for each feature
    Eigen::VectorXd mean = X.colwise().mean();
    Eigen::VectorXd std_dev = ((X.rowwise() - mean.transpose()).array().square().colwise().sum() / 
                         (X.rows() - 1)).sqrt();
    
    // Standardize each feature (z-score normalization)
    for (int i = 0; i < X.cols(); ++i) {
        if (std_dev(i) > 1e-10) {  // Avoid division by zero
            X_std.col(i) = (X.col(i).array() - mean(i)) / std_dev(i);
        }
    }
    
    return X_std;
}

double XGBoost::clipPrediction(double prediction, double max_value) const {
    // Clip prediction to prevent numerical overflow
    if (std::isnan(prediction) || !std::isfinite(prediction)) {
        LOG_CLASS_WARNING("XGBoost", "NaN or Inf prediction detected, using 0 instead");
        return 0.0;
    }
    if (prediction > max_value) {
        LOG_CLASS_WARNING("XGBoost", "Prediction value too large: " + std::to_string(prediction) + 
                         ", clipping to " + std::to_string(max_value));
        return max_value;
    }
    if (prediction < -max_value) {
        LOG_CLASS_WARNING("XGBoost", "Prediction value too small: " + std::to_string(prediction) + 
                         ", clipping to " + std::to_string(-max_value));
        return -max_value;
    }
    return prediction;
}

} // namespace DataAnalyzer