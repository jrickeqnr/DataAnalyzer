#include "../../include/model.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <queue>
#include <limits>

namespace DataAnalyzer {

GradientBoosting::GradientBoosting(int n_estimators, double learning_rate, int max_depth, int min_samples_split)
    : n_estimators_(n_estimators), learning_rate_(learning_rate), max_depth_(max_depth),
      min_samples_split_(min_samples_split), initial_prediction_(0.0), rmse_(0.0), r_squared_(0.0) {
}

bool GradientBoosting::train(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
    if (X.rows() != y.rows() || X.rows() == 0) {
        return false;
    }
    
    // Clear previous trees
    trees_.clear();
    
    // Create working copies of data
    Eigen::MatrixXd X_train = X;
    Eigen::VectorXd y_train = y;
    
    // Initial prediction is the mean of the target values
    initial_prediction_ = y_train.mean();
    
    // Initialize residuals
    Eigen::VectorXd residuals = y_train - Eigen::VectorXd::Constant(y_train.size(), initial_prediction_);
    Eigen::VectorXd predictions = Eigen::VectorXd::Constant(y_train.size(), initial_prediction_);
    
    // Build boosting rounds
    for (int i = 0; i < n_estimators_; ++i) {
        // Build a regression tree to predict the residuals
        DecisionTree tree = buildTree(X_train, residuals, 0);
        trees_.push_back(tree);
        
        // Update predictions and residuals
        for (int j = 0; j < X_train.rows(); ++j) {
            double prediction = tree.predict(X_train.row(j));
            predictions(j) += learning_rate_ * prediction;
            residuals(j) = y_train(j) - predictions(j);
        }
    }
    
    // Compute statistics
    computeStats(X, y);
    
    return true;
}

Eigen::VectorXd GradientBoosting::predict(const Eigen::MatrixXd& X) const {
    Eigen::VectorXd predictions = Eigen::VectorXd::Constant(X.rows(), initial_prediction_);
    
    // Sum up the predictions from each tree
    for (const auto& tree : trees_) {
        for (int i = 0; i < X.rows(); ++i) {
            predictions(i) += learning_rate_ * tree.predict(X.row(i));
        }
    }
    
    return predictions;
}

std::map<std::string, double> GradientBoosting::getStats() const {
    return {
        {"RMSE", rmse_},
        {"R²", r_squared_}
    };
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

std::tuple<int, double, int> GradientBoosting::gridSearch(
    const Eigen::MatrixXd& X, 
    const Eigen::VectorXd& y,
    const std::vector<int>& n_estimators_values,
    const std::vector<double>& learning_rate_values,
    const std::vector<int>& max_depth_values,
    int k) {
    
    if (X.rows() != y.rows() || n_estimators_values.empty() || 
        learning_rate_values.empty() || max_depth_values.empty() || k <= 1) {
        return {n_estimators_, learning_rate_, max_depth_};
    }
    
    // Prepare k-fold cross-validation indices
    std::vector<int> indices(X.rows());
    std::iota(indices.begin(), indices.end(), 0);
    
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);
    
    // Best hyperparameters
    int best_n_estimators = n_estimators_;
    double best_learning_rate = learning_rate_;
    int best_max_depth = max_depth_;
    double best_rmse = std::numeric_limits<double>::max();
    
    // Grid search
    for (int n_estimators : n_estimators_values) {
        for (double learning_rate : learning_rate_values) {
            for (int max_depth : max_depth_values) {
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
                    GradientBoosting model(n_estimators, learning_rate, max_depth, min_samples_split_);
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
                    best_n_estimators = n_estimators;
                    best_learning_rate = learning_rate;
                    best_max_depth = max_depth;
                }
            }
        }
    }
    
    return {best_n_estimators, best_learning_rate, best_max_depth};
}

void GradientBoosting::computeStats(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
    Eigen::VectorXd predictions = predict(X);
    
    // Calculate Root Mean Squared Error (RMSE)
    rmse_ = std::sqrt((predictions - y).array().square().mean());
    
    // Calculate R-squared
    double y_mean = y.mean();
    double ss_total = (y.array() - y_mean).square().sum();
    double ss_residual = (y - predictions).array().square().sum();
    r_squared_ = 1.0 - (ss_residual / ss_total);
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
    int node_idx, int depth, DecisionTree& tree, 
    const std::vector<int>& sample_indices, int current_depth) {
    
    // Get samples for this node
    std::vector<int> current_samples;
    if (sample_indices.empty()) {
        // If no sample indices provided, use all samples (root node)
        current_samples.resize(X.rows());
        std::iota(current_samples.begin(), current_samples.end(), 0);
    } else {
        current_samples = sample_indices;
    }
    
    // Calculate mean target value for this node
    double node_value = 0.0;
    for (int idx : current_samples) {
        node_value += y(idx);
    }
    node_value /= current_samples.size();
    
    // Update node value
    tree.nodes[node_idx].value = node_value;
    
    // Stop splitting if max depth reached or too few samples
    if (current_depth >= max_depth_ || 
        static_cast<int>(current_samples.size()) < min_samples_split_) {
        tree.nodes[node_idx].is_leaf = true;
        return;
    }
    
    // Find the best split
    int best_feature = -1;
    double best_threshold = 0.0;
    double best_mse = std::numeric_limits<double>::infinity();
    std::vector<int> best_left_samples, best_right_samples;
    
    // Try each feature
    for (int feature = 0; feature < X.cols(); ++feature) {
        // Get sorted feature values and corresponding targets
        std::vector<std::pair<double, double>> feature_value_target;
        for (int idx : current_samples) {
            feature_value_target.emplace_back(X(idx, feature), y(idx));
        }
        
        std::sort(feature_value_target.begin(), feature_value_target.end());
        
        // Try potential thresholds (midpoints between distinct values)
        for (size_t i = 0; i < feature_value_target.size() - 1; ++i) {
            // Skip if feature values are identical
            if (feature_value_target[i].first == feature_value_target[i + 1].first) {
                continue;
            }
            
            double threshold = (feature_value_target[i].first + feature_value_target[i + 1].first) / 2.0;
            
            // Split samples
            std::vector<int> left_samples, right_samples;
            for (int idx : current_samples) {
                if (X(idx, feature) <= threshold) {
                    left_samples.push_back(idx);
                } else {
                    right_samples.push_back(idx);
                }
            }
            
            // Skip if split is too unbalanced
            if (left_samples.size() < 2 || right_samples.size() < 2) {
                continue;
            }
            
            // Calculate mean for each child
            double left_mean = 0.0, right_mean = 0.0;
            for (int idx : left_samples) left_mean += y(idx);
            for (int idx : right_samples) right_mean += y(idx);
            left_mean /= left_samples.size();
            right_mean /= right_samples.size();
            
            // Calculate MSE for this split
            double mse = 0.0;
            for (int idx : left_samples) mse += std::pow(y(idx) - left_mean, 2);
            for (int idx : right_samples) mse += std::pow(y(idx) - right_mean, 2);
            
            // Update best split if this is better
            if (mse < best_mse) {
                best_mse = mse;
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
    // Initialize feature importance vector
    Eigen::VectorXd importance = Eigen::VectorXd::Zero(trees_[0].nodes.size());
    
    // For each tree, count how many times each feature is used for splitting
    for (const auto& tree : trees_) {
        for (const auto& node : tree.nodes) {
            if (!node.is_leaf && node.feature_idx >= 0) {
                importance(node.feature_idx) += 1.0;
            }
        }
    }
    
    // Normalize by the number of trees
    importance /= trees_.size();
    
    return importance;
}

} // namespace DataAnalyzer 