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

GradientBoosting::GradientBoosting(int n_estimators, double learning_rate, int max_depth, int min_samples_split)
    : n_estimators_(n_estimators), learning_rate_(learning_rate), max_depth_(max_depth),
      min_samples_split_(min_samples_split), initial_prediction_(0.0), rmse_(0.0), r_squared_(0.0) {
    std::stringstream ss;
    ss << "Initialized model with n_estimators=" << n_estimators
       << ", learning_rate=" << learning_rate
       << ", max_depth=" << max_depth
       << ", min_samples_split=" << min_samples_split;
    LOG_CLASS_INFO("GradientBoosting", ss.str());
}

bool GradientBoosting::train(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
    if (X.rows() != y.rows() || X.rows() == 0) {
        LOG_CLASS_ERROR("GradientBoosting", "Invalid input dimensions: X rows=" + std::to_string(X.rows()) + 
                        ", y size=" + std::to_string(y.rows()));
        return false;
    }
    
    LOG_CLASS_INFO("GradientBoosting", "Starting training with " + std::to_string(X.rows()) + 
                   " samples and " + std::to_string(X.cols()) + " features");
    
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
    Eigen::MatrixXd X_train = X;
    Eigen::VectorXd mean = X.colwise().mean();
    Eigen::VectorXd std = ((X.rowwise() - mean.transpose()).array().square().colwise().sum() / (X.rows() - 1)).sqrt();
    
    for (int i = 0; i < X.cols(); ++i) {
        if (std(i) > 1e-10) {  // Avoid division by zero
            X_train.col(i) = (X.col(i).array() - mean(i)) / std(i);
        }
    }
    
    // Store standardization parameters for prediction
    feature_means_ = mean;
    feature_stds_ = std;
    
    // Standardize target variable
    double y_mean = y.mean();
    double y_std = std::sqrt((y.array() - y_mean).square().sum() / (y.size() - 1));
    Eigen::VectorXd y_train = (y.array() - y_mean) / y_std;
    
    // Store target standardization parameters
    target_mean_ = y_mean;
    target_std_ = y_std;
    
    LOG_CLASS_DEBUG("GradientBoosting", "Data standardized, target mean=" + 
                    std::to_string(y_mean) + ", target std=" + std::to_string(y_std));
    
    // Initial prediction is the mean of the target
    initial_prediction_ = y_train.mean();
    LOG_CLASS_DEBUG("GradientBoosting", "Initial prediction (base value): " + 
                    std::to_string(initial_prediction_));
    
    // Initialize predictions with the initial value
    Eigen::VectorXd predictions = Eigen::VectorXd::Constant(y_train.size(), initial_prediction_);
    Eigen::VectorXd residuals = y_train - predictions;
    
    // Build boosting rounds
    for (int i = 0; i < n_estimators_; ++i) {
        // Update progress
        stats_["Training Progress"] = static_cast<double>(i + 1) / n_estimators_;
        
        // Build a regression tree to predict the residuals
        DecisionTree tree = buildTree(X_train, residuals, 0);
        
        // Add the tree regardless of its structure
        trees_.push_back(tree);
        
        // Update predictions and residuals
        for (int j = 0; j < X_train.rows(); ++j) {
            double tree_prediction = tree.predict(X_train.row(j));
            predictions(j) += learning_rate_ * tree_prediction;
            residuals(j) = y_train(j) - predictions(j);
        }
        
        // Log progress for every 10% or for specific iterations
        if (i == 0 || i == n_estimators_ - 1 || (n_estimators_ > 10 && i % (n_estimators_ / 10) == 0)) {
            double mse = residuals.array().square().mean();
            LOG_CLASS_INFO("GradientBoosting", "Round " + std::to_string(i+1) + "/" + 
                           std::to_string(n_estimators_) + ": MSE = " + std::to_string(mse));
        }
        
        // Early stopping if residuals are very small
        if (residuals.norm() < 1e-3) {  // Made even less aggressive
            LOG_CLASS_INFO("GradientBoosting", "Early stopping at round " + 
                           std::to_string(i+1) + " due to small residuals");
            break;
        }
    }
    
    // Set final training progress to 1.0 (100%)
    stats_["Training Progress"] = 1.0;
    
    // Compute statistics using original scale data
    computeStats(X, y);
    LOG_CLASS_INFO("GradientBoosting", "Training complete. Trees built: " + 
                   std::to_string(trees_.size()) + ", RMSE: " + std::to_string(rmse_) + 
                   ", R2: " + std::to_string(r_squared_));
    
    return true;
}

Eigen::VectorXd GradientBoosting::predict(const Eigen::MatrixXd& X) const {
    LOG_CLASS_DEBUG("GradientBoosting", "Predicting for " + std::to_string(X.rows()) + " samples");
    
    // Standardize input features
    Eigen::MatrixXd X_std = X;
    for (int i = 0; i < X.cols(); ++i) {
        if (feature_stds_(i) > 1e-10) {  // Avoid division by zero
            X_std.col(i) = (X.col(i).array() - feature_means_(i)) / feature_stds_(i);
        }
    }
    
    Eigen::VectorXd predictions = Eigen::VectorXd::Constant(X.rows(), initial_prediction_);
    
    // Sum up the predictions from each tree
    for (const auto& tree : trees_) {
        for (int i = 0; i < X.rows(); ++i) {
            predictions(i) += learning_rate_ * tree.predict(X_std.row(i));
        }
    }
    
    // Transform predictions back to original scale
    predictions = predictions.array() * target_std_ + target_mean_;
    
    return predictions;
}

std::map<std::string, double> GradientBoosting::getStats() const {
    return stats_;
}

std::string GradientBoosting::getDescription() const {
    return "Gradient Boosting is a machine learning technique that produces a prediction "
           "model in the form of an ensemble of weak prediction models, typically decision trees. "
           "It builds models sequentially, with each new model correcting errors made by "
           "previously trained models. The algorithm is highly effective for both regression "
           "and classification tasks, known for its accuracy and flexibility.";
}

Eigen::VectorXd GradientBoosting::getCoefficients() const {
    // Tree-based models don't have coefficients in the traditional sense
    // Return an empty vector
    return Eigen::VectorXd();
}

std::map<std::string, double> GradientBoosting::getHyperparameters() const {
    return {
        {"n_estimators", static_cast<double>(n_estimators_)},
        {"learning_rate", learning_rate_},
        {"max_depth", static_cast<double>(max_depth_)},
        {"min_samples_split", static_cast<double>(min_samples_split_)}
    };
}

std::tuple<int, double, int, int> GradientBoosting::gridSearch(
    const Eigen::MatrixXd& X, 
    const Eigen::VectorXd& y,
    const std::vector<int>& n_estimators_values,
    const std::vector<double>& learning_rate_values,
    const std::vector<int>& max_depth_values,
    const std::vector<int>& min_samples_split_values,
    int k) {
    
    if (X.rows() != y.rows() || n_estimators_values.empty() || 
        learning_rate_values.empty() || max_depth_values.empty() || 
        min_samples_split_values.empty() || k <= 1) {
        LOG_CLASS_ERROR("GradientBoosting", "Invalid grid search parameters");
        return {n_estimators_, learning_rate_, max_depth_, min_samples_split_};
    }
    
    LOG_CLASS_INFO("GradientBoosting", "Starting grid search with " + std::to_string(X.rows()) + 
                   " samples, " + std::to_string(X.cols()) + " features, " + 
                   std::to_string(k) + " folds");
    
    // Create a temporary map for grid search statistics to avoid 
    // polluting the main stats_ object
    std::map<std::string, double> grid_search_stats;
    
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
    int best_n_estimators = n_estimators_;
    double best_learning_rate = learning_rate_;
    int best_max_depth = max_depth_;
    int best_min_samples_split = min_samples_split_;
    double best_rmse = std::numeric_limits<double>::max();
    
    // Grid search with progress tracking
    int total_combinations = n_estimators_values.size() * learning_rate_values.size() * 
                           max_depth_values.size() * min_samples_split_values.size();
    int current_combination = 0;
    
    LOG_CLASS_INFO("GradientBoosting", "Grid search will evaluate " + 
                   std::to_string(total_combinations) + " combinations");
    
    // Initialize progress in the main stats map
    stats_["Grid Search Progress"] = 0.0;
    
    // Grid search
    for (int n_estimators : n_estimators_values) {
        for (double learning_rate : learning_rate_values) {
            for (int max_depth : max_depth_values) {
                for (int min_samples_split : min_samples_split_values) {
                    current_combination++;
                    double fold_rmse_sum = 0.0;
                    bool early_stop = false;
                    int valid_folds = 0;
                    
                    // Update progress in both the temporary stats and the main stats object
                    double progress = static_cast<double>(current_combination) / total_combinations;
                    grid_search_stats["Grid Search Progress"] = progress;
                    stats_["Grid Search Progress"] = progress;
                    
                    LOG_CLASS_DEBUG("GradientBoosting", "Evaluating combination " + std::to_string(current_combination) + 
                                   "/" + std::to_string(total_combinations) + 
                                   ": n_estimators=" + std::to_string(n_estimators) + 
                                   ", learning_rate=" + std::to_string(learning_rate) + 
                                   ", max_depth=" + std::to_string(max_depth) + 
                                   ", min_samples_split=" + std::to_string(min_samples_split));
                    
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
                            GradientBoosting model(n_estimators, learning_rate, max_depth, min_samples_split);
                            
                            if (!model.train(X_train, y_train)) {
                                // If training fails, skip this combination
                                LOG_CLASS_WARNING("GradientBoosting", "Training failed for fold " + 
                                                std::to_string(fold+1) + ", skipping combination");
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
                            if (!std::isfinite(fold_rmse)) {  // Only check for non-finite values
                                LOG_CLASS_WARNING("GradientBoosting", "Non-finite RMSE in fold " + 
                                                std::to_string(fold+1) + ", skipping combination");
                                early_stop = true;
                                break;
                            }
                            
                            // Early stopping if RMSE is much worse than best
                            if (best_rmse < std::numeric_limits<double>::max() && 
                                fold_rmse > 10.0 * best_rmse) {  // Made more lenient
                                LOG_CLASS_DEBUG("GradientBoosting", "Early stopping for poor performance, RMSE=" + 
                                              std::to_string(fold_rmse) + " vs best=" + 
                                              std::to_string(best_rmse));
                                early_stop = true;
                                break;
                            }
                            
                            fold_rmse_sum += fold_rmse;
                            valid_folds++;
                            
                            LOG_CLASS_DEBUG("GradientBoosting", "Fold " + std::to_string(fold+1) + 
                                          " RMSE: " + std::to_string(fold_rmse));
                            
                        } catch (const std::exception& e) {
                            // If any exception occurs, skip this combination
                            LOG_CLASS_WARNING("GradientBoosting", "Exception in fold " + std::to_string(fold+1) + 
                                            ": " + e.what());
                            early_stop = true;
                            break;
                        }
                    }
                    
                    if (!early_stop && valid_folds > 0) {
                        // Average RMSE across valid folds
                        double avg_rmse = fold_rmse_sum / valid_folds;
                        
                        LOG_CLASS_INFO("GradientBoosting", "Combination " + std::to_string(current_combination) +  
                                      ": Average RMSE = " + std::to_string(avg_rmse));
                        
                        // Update best hyperparameters if RMSE is improved
                        if (avg_rmse < best_rmse) {
                            best_rmse = avg_rmse;
                            best_n_estimators = n_estimators;
                            best_learning_rate = learning_rate;
                            best_max_depth = max_depth;
                            best_min_samples_split = min_samples_split;
                            
                            LOG_CLASS_INFO("GradientBoosting", "New best combination found! RMSE = " + 
                                          std::to_string(best_rmse));
                            
                            // Store best parameters in grid search stats
                            grid_search_stats["Best RMSE"] = best_rmse;
                            grid_search_stats["Best N Estimators"] = static_cast<double>(best_n_estimators);
                            grid_search_stats["Best Learning Rate"] = best_learning_rate;
                            grid_search_stats["Best Max Depth"] = static_cast<double>(best_max_depth);
                            grid_search_stats["Best Min Samples Split"] = static_cast<double>(best_min_samples_split);
                        }
                    }
                }
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
    stats_["Grid Search Best Min Samples Split"] = grid_search_stats["Best Min Samples Split"];
    
    std::stringstream ss;
    ss << "Grid search complete. Best hyperparameters: "
       << "n_estimators=" << best_n_estimators
       << ", learning_rate=" << best_learning_rate
       << ", max_depth=" << best_max_depth
       << ", min_samples_split=" << best_min_samples_split
       << ", RMSE=" << best_rmse;
    LOG_CLASS_INFO("GradientBoosting", ss.str());
    
    return {best_n_estimators, best_learning_rate, best_max_depth, best_min_samples_split};
}

void GradientBoosting::computeStats(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
    Eigen::VectorXd predictions = predict(X);
    // int n = X.rows(); // Unused variable
    
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
    
    // Calculate average tree depth and leaf nodes
    double avg_tree_depth = 0.0;
    double avg_leaf_nodes = 0.0;
    
    for (const auto& tree : trees_) {
        // Calculate maximum depth of this tree
        int max_depth = 0;
        std::function<void(int, int)> computeDepth = [&](int node_idx, int current_depth) {
            if (node_idx >= 0 && static_cast<size_t>(node_idx) < tree.nodes.size()) {
                max_depth = std::max(max_depth, current_depth);
                if (!tree.nodes[node_idx].is_leaf) {
                    computeDepth(tree.nodes[node_idx].left_child, current_depth + 1);
                    computeDepth(tree.nodes[node_idx].right_child, current_depth + 1);
                }
            }
        };
        computeDepth(0, 0);
        avg_tree_depth += max_depth;
        
        // Count leaf nodes
        int leaf_count = 0;
        for (const auto& node : tree.nodes) {
            if (node.is_leaf) {
                leaf_count++;
            }
        }
        avg_leaf_nodes += leaf_count;
    }
    
    if (!trees_.empty()) {
        avg_tree_depth /= trees_.size();
        avg_leaf_nodes /= trees_.size();
    }
    
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
        {"Average Tree Depth", avg_tree_depth},
        {"Average Leaf Nodes", avg_leaf_nodes},
        {"Training Loss", mse}, // Using MSE as the training loss
        {"Learning Rate", learning_rate_},
        {"Max Tree Depth", static_cast<double>(max_depth_)},
        {"Min Samples Split", static_cast<double>(min_samples_split_)},
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

GradientBoosting::DecisionTree GradientBoosting::buildTree(
    const Eigen::MatrixXd& X, const Eigen::VectorXd& y, int depth) {
    
    // Create a new decision tree
    DecisionTree tree;
    tree.nodes.emplace_back(); // Root node
    
    // Recursively build the tree
    buildTreeRecursive(X, y, 0, depth, tree, 
                       std::vector<int>(X.rows()), 0);
    
    return tree;
}

void GradientBoosting::buildTreeRecursive(
    const Eigen::MatrixXd& X, const Eigen::VectorXd& y, 
    int node_idx, [[maybe_unused]] int depth, DecisionTree& tree, 
    const std::vector<int>& sample_indices, int current_depth) {
    
    // Get samples for this node
    std::vector<int> current_samples;
    if (sample_indices.empty()) {
        current_samples.resize(X.rows());
        std::iota(current_samples.begin(), current_samples.end(), 0);
    } else {
        current_samples = sample_indices;
    }
    
    // Calculate mean target value for this node
    double node_value = 0.0;
    double node_variance = 0.0;
    
    // First pass: calculate mean
    for (int idx : current_samples) {
        node_value += y(idx);
    }
    node_value /= current_samples.size();
    
    // Second pass: calculate variance
    for (int idx : current_samples) {
        node_variance += std::pow(y(idx) - node_value, 2);
    }
    node_variance /= current_samples.size();
    
    // Update node value
    tree.nodes[node_idx].value = node_value;
    
    // Stop splitting if:
    // 1. Max depth reached
    // 2. Not enough samples to split (need at least 2)
    // 3. All target values are exactly identical
    if (current_depth >= max_depth_ || 
        current_samples.size() < 2 || 
        node_variance == 0.0) {  // Only stop if variance is exactly 0
        tree.nodes[node_idx].is_leaf = true;
        return;
    }
    
    // Find the best split
    int best_feature = -1;
    double best_threshold = 0.0;
    double best_gain = -std::numeric_limits<double>::infinity();
    std::vector<int> best_left_samples, best_right_samples;
    
    // Try each feature
    for (int feature = 0; feature < X.cols(); ++feature) {
        // Get feature values for this node's samples
        std::vector<std::pair<double, int>> feature_values;
        for (int idx : current_samples) {
            feature_values.emplace_back(X(idx, feature), idx);
        }
        std::sort(feature_values.begin(), feature_values.end());
        
        // Try splits between each adjacent pair of values
        for (size_t i = 0; i < feature_values.size() - 1; ++i) {
            double curr_value = feature_values[i].first;
            double next_value = feature_values[i + 1].first;
            
            // Skip if values are exactly identical
            if (curr_value == next_value) {
                continue;
            }
            
            // Calculate split threshold
            double threshold = (curr_value + next_value) / 2.0;
            
            // Split samples
            std::vector<int> left_samples, right_samples;
            double left_sum = 0.0, right_sum = 0.0;
            
            // Left samples are all samples up to and including i
            for (size_t j = 0; j <= i; ++j) {
                left_samples.push_back(feature_values[j].second);
                left_sum += y(feature_values[j].second);
            }
            
            // Right samples are all samples after i
            for (size_t j = i + 1; j < feature_values.size(); ++j) {
                right_samples.push_back(feature_values[j].second);
                right_sum += y(feature_values[j].second);
            }
            
            // Skip if either side is empty
            if (left_samples.empty() || right_samples.empty()) {
                continue;
            }
            
            // Calculate means
            double left_mean = left_sum / left_samples.size();
            double right_mean = right_sum / right_samples.size();
            
            // Calculate variances
            double left_var = 0.0, right_var = 0.0;
            for (int idx : left_samples) {
                left_var += std::pow(y(idx) - left_mean, 2);
            }
            for (int idx : right_samples) {
                right_var += std::pow(y(idx) - right_mean, 2);
            }
            
            left_var /= left_samples.size();
            right_var /= right_samples.size();
            
            // Calculate gain as variance reduction
            double gain = node_variance - 
                         (left_samples.size() * left_var + right_samples.size() * right_var) 
                         / current_samples.size();
            
            // Update best split if this is better
            // For small datasets, accept any split that reduces variance
            if (gain > best_gain) {  // Removed gain >= 0.0 condition
                best_gain = gain;
                best_feature = feature;
                best_threshold = threshold;
                best_left_samples = left_samples;
                best_right_samples = right_samples;
            }
        }
    }
    
    // If no valid split found, make this a leaf node
    if (best_feature == -1) {
        tree.nodes[node_idx].is_leaf = true;
        return;
    }
    
    // Set up this node as a split node
    tree.nodes[node_idx].is_leaf = false;
    tree.nodes[node_idx].feature_idx = best_feature;
    tree.nodes[node_idx].threshold = best_threshold;
    
    // Create child nodes
    tree.nodes[node_idx].left_child = tree.nodes.size();
    tree.nodes.emplace_back(); // Left child
    
    tree.nodes[node_idx].right_child = tree.nodes.size();
    tree.nodes.emplace_back(); // Right child
    
    // Recursively build child subtrees
    buildTreeRecursive(X, y, tree.nodes[node_idx].left_child, depth, tree, 
                       best_left_samples, current_depth + 1);
    buildTreeRecursive(X, y, tree.nodes[node_idx].right_child, depth, tree, 
                       best_right_samples, current_depth + 1);
}

Eigen::VectorXd GradientBoosting::getFeatureImportance() const {
    if (trees_.empty()) {
        return Eigen::VectorXd();
    }

    // Get number of features from the first tree's first split
    int n_features = 0;
    for (const auto& tree : trees_) {
        for (const auto& node : tree.nodes) {
            if (!node.is_leaf && node.feature_idx >= 0) {
                n_features = std::max(n_features, node.feature_idx + 1);
            }
        }
    }
    
    // Initialize feature importance vector with correct size
    Eigen::VectorXd importance = Eigen::VectorXd::Zero(n_features);
    
    // For each tree, accumulate importance scores
    for (const auto& tree : trees_) {
        for (const auto& node : tree.nodes) {
            if (!node.is_leaf && node.feature_idx >= 0) {
                // Add importance based on the number of splits using this feature
                importance(node.feature_idx) += 1.0;
            }
        }
    }
    
    // Normalize importance scores
    if (importance.sum() > 0) {
        importance /= importance.sum();  // Make them sum to 1
    }
    
    return importance;
}

} // namespace DataAnalyzer 