#include "../../include/model.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <queue>
#include <limits>

namespace DataAnalyzer {

XGBoost::XGBoost(int n_estimators, double learning_rate, int max_depth, double subsample)
    : n_estimators_(n_estimators), learning_rate_(learning_rate), max_depth_(max_depth),
      subsample_(subsample), base_prediction_(0.0), rmse_(0.0), r_squared_(0.0) {
}

bool XGBoost::train(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
    if (X.rows() != y.rows() || X.rows() == 0) {
        return false;
    }
    
    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);
    
    // Clear previous trees
    trees_.clear();
    
    // Create working copies of data
    Eigen::MatrixXd X_train = X;
    Eigen::VectorXd y_train = y;
    
    // Initial prediction is the mean of the target values
    base_prediction_ = y_train.mean();
    
    // Initialize residuals and predictions
    Eigen::VectorXd residuals = y_train - Eigen::VectorXd::Constant(y_train.size(), base_prediction_);
    Eigen::VectorXd predictions = Eigen::VectorXd::Constant(y_train.size(), base_prediction_);
    
    // Build boosting rounds
    for (int i = 0; i < n_estimators_; ++i) {
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
        } else {
            X_sub = X_train;
            residuals_sub = residuals;
        }
        
        // Build a regression tree
        Tree tree;
        buildRegressionTree(X_sub, residuals_sub, tree, 0, max_depth_);
        trees_.push_back(tree);
        
        // Update predictions and residuals
        for (int j = 0; j < X_train.rows(); ++j) {
            double prediction = predictTree(tree, X_train.row(j));
            predictions(j) += learning_rate_ * prediction;
            residuals(j) = y_train(j) - predictions(j);
        }
    }
    
    // Compute statistics
    computeStats(X, y);
    
    return true;
}

Eigen::VectorXd XGBoost::predict(const Eigen::MatrixXd& X) const {
    // Initialize predictions with base prediction
    Eigen::VectorXd predictions = Eigen::VectorXd::Constant(X.rows(), base_prediction_);
    
    // Sum up the predictions from each tree
    for (const auto& tree : trees_) {
        for (int i = 0; i < X.rows(); ++i) {
            predictions(i) += learning_rate_ * predictTree(tree, X.row(i));
        }
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
        {"subsample", subsample_}
    };
}

std::tuple<int, double, int, double> XGBoost::gridSearch(
    const Eigen::MatrixXd& X, 
    const Eigen::VectorXd& y,
    const std::vector<int>& n_estimators_values,
    const std::vector<double>& learning_rate_values,
    const std::vector<int>& max_depth_values,
    const std::vector<double>& subsample_values,
    int k) {
    
    if (X.rows() != y.rows() || n_estimators_values.empty() || 
        learning_rate_values.empty() || max_depth_values.empty() || 
        subsample_values.empty() || k <= 1) {
        return {n_estimators_, learning_rate_, max_depth_, subsample_};
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
    double best_subsample = subsample_;
    double best_rmse = std::numeric_limits<double>::max();
    
    // Grid search with progress tracking
    int total_combinations = n_estimators_values.size() * learning_rate_values.size() * 
                           max_depth_values.size() * subsample_values.size();
    int current_combination = 0;
    
    // Grid search
    for (int n_estimators : n_estimators_values) {
        for (double learning_rate : learning_rate_values) {
            for (int max_depth : max_depth_values) {
                for (double subsample : subsample_values) {
                    current_combination++;
                    double fold_rmse_sum = 0.0;
                    bool early_stop = false;
                    
                    // Update progress in stats
                    stats_["Grid Search Progress"] = static_cast<double>(current_combination) / total_combinations;
                    
                    // K-fold cross-validation
                    for (int fold = 0; fold < k && !early_stop; ++fold) {
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
                            y_train(i) = y(train_indices[i]);
                        }
                        
                        for (size_t i = 0; i < val_indices.size(); ++i) {
                            X_val.row(i) = X_std.row(val_indices[i]);
                            y_val(i) = y(val_indices[i]);
                        }
                        
                        // Train model with current hyperparameters
                        XGBoost model(n_estimators, learning_rate, max_depth, subsample);
                        
                        try {
                            if (!model.train(X_train, y_train)) {
                                // If training fails, skip this combination
                                early_stop = true;
                                break;
                            }
                            
                            // Evaluate on validation set
                            Eigen::VectorXd predictions = model.predict(X_val);
                            double fold_rmse = std::sqrt((predictions - y_val).array().square().mean());
                            
                            // Early stopping if RMSE is much worse than best
                            if (fold_rmse > 2.0 * best_rmse) {
                                early_stop = true;
                                break;
                            }
                            
                            fold_rmse_sum += fold_rmse;
                        } catch (const std::exception&) {
                            // If any exception occurs, skip this combination
                            early_stop = true;
                            break;
                        }
                    }
                    
                    if (!early_stop) {
                        // Average RMSE across all folds
                        double avg_rmse = fold_rmse_sum / k;
                        
                        // Update best hyperparameters if RMSE is improved
                        if (avg_rmse < best_rmse) {
                            best_rmse = avg_rmse;
                            best_n_estimators = n_estimators;
                            best_learning_rate = learning_rate;
                            best_max_depth = max_depth;
                            best_subsample = subsample;
                        }
                    }
                }
            }
        }
    }
    
    return {best_n_estimators, best_learning_rate, best_max_depth, best_subsample};
}

void XGBoost::computeStats(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
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
    
    // Calculate average tree depth and leaf count
    double avg_depth = 0.0;
    double avg_leaves = 0.0;
    
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
        
        // Count leaf nodes
        int leaf_count = std::count(is_leaf.begin(), is_leaf.end(), true);
        avg_leaves += leaf_count;
    }
    
    if (!trees_.empty()) {
        avg_depth /= trees_.size();
        avg_leaves /= trees_.size();
    }
    
    // Store all statistics in the stats map
    stats_ = {
        {"RMSE", rmse_},
        {"MSE", mse},
        {"MAE", mae},
        {"R²", r_squared_},
        {"Number of Trees", static_cast<double>(trees_.size())},
        {"Average Tree Depth", avg_depth},
        {"Average Leaf Nodes", avg_leaves},
        {"Training Loss", mse}, // Using MSE as the training loss
        {"Learning Rate", learning_rate_},
        {"Max Tree Depth", static_cast<double>(max_depth_)},
        {"Subsample Ratio", subsample_}
    };
    
    // Store feature importance scores with proper naming
    if (importance.size() > 0) {
        for (int i = 0; i < importance.size(); ++i) {
            stats_["Feature " + std::to_string(i) + " Importance"] = importance(i);
        }
    }
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
    // Simple implementation of a regression tree builder
    // In a real implementation, this would be more sophisticated
    
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
    
    while (!queue.empty()) {
        auto [current_node, current_samples, current_depth] = queue.front();
        queue.pop();
        
        // Extract the samples for this node
        Eigen::VectorXd node_targets(current_samples.size());
        for (size_t i = 0; i < current_samples.size(); ++i) {
            node_targets(i) = residuals(current_samples[i]);
        }
        
        double node_value = node_targets.mean();
        
        // Extend the tree if needed
        while (tree.feature_indices.size() <= static_cast<size_t>(current_node)) {
            tree.feature_indices.push_back(-1);
            tree.thresholds.push_back(0.0);
            tree.values.push_back(0.0);
            tree.left_children.push_back(-1);
            tree.right_children.push_back(-1);
        }
        
        // If max depth is reached or too few samples, make this a leaf node
        if (current_depth >= max_depth || current_samples.size() <= 1) {
            tree.feature_indices[current_node] = -1;
            tree.values[current_node] = node_value;
            continue;
        }
        
        // Find the best split
        int best_feature = -1;
        double best_threshold = 0.0;
        double best_gain = -std::numeric_limits<double>::infinity();
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
                
                // Skip if one side is empty
                if (left_samples.empty() || right_samples.empty()) {
                    continue;
                }
                
                // Calculate gain (variance reduction)
                double total_var = node_targets.array().square().sum() / node_targets.size() - 
                                  std::pow(node_targets.mean(), 2);
                
                // Left side variance
                Eigen::VectorXd left_targets(left_samples.size());
                for (size_t j = 0; j < left_samples.size(); ++j) {
                    left_targets(j) = residuals(left_samples[j]);
                }
                double left_var = left_targets.array().square().sum() / left_targets.size() - 
                                 std::pow(left_targets.mean(), 2);
                
                // Right side variance
                Eigen::VectorXd right_targets(right_samples.size());
                for (size_t j = 0; j < right_samples.size(); ++j) {
                    right_targets(j) = residuals(right_samples[j]);
                }
                double right_var = right_targets.array().square().sum() / right_targets.size() - 
                                  std::pow(right_targets.mean(), 2);
                
                // Calculate weighted variance reduction
                double left_weight = static_cast<double>(left_samples.size()) / current_samples.size();
                double right_weight = static_cast<double>(right_samples.size()) / current_samples.size();
                double gain = total_var - (left_weight * left_var + right_weight * right_var);
                
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
        
        // If no good split found, make this a leaf node
        if (best_feature == -1) {
            tree.feature_indices[current_node] = -1;
            tree.values[current_node] = node_value;
            continue;
        }
        
        // Otherwise, create a split node
        tree.feature_indices[current_node] = best_feature;
        tree.thresholds[current_node] = best_threshold;
        tree.values[current_node] = node_value;
        
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

} // namespace DataAnalyzer 