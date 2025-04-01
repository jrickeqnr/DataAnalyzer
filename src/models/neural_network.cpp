#include "../../include/model.h"
#include "../../include/logger.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <limits>
#include <sstream>

namespace DataAnalyzer {

NeuralNetwork::NeuralNetwork(const std::vector<int>& hidden_layer_sizes,
                             double learning_rate, int max_iter,
                             double alpha)
    : hidden_layer_sizes_(hidden_layer_sizes), learning_rate_(learning_rate),
      max_iter_(max_iter), alpha_(alpha), rmse_(0.0), r_squared_(0.0) {
    
    std::string layers_str = "";
    for (size_t i = 0; i < hidden_layer_sizes.size(); ++i) {
        if (i > 0) layers_str += ", ";
        layers_str += std::to_string(hidden_layer_sizes[i]);
    }
    
    LOG_CLASS_INFO("NeuralNetwork", "Initialized model with hidden_layers=[" + layers_str + 
                   "], learning_rate=" + std::to_string(learning_rate) + 
                   ", max_iterations=" + std::to_string(max_iter) +
                   ", alpha=" + std::to_string(alpha));
}

bool NeuralNetwork::train(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
    if (X.rows() != y.rows() || X.rows() == 0) {
        LOG_CLASS_ERROR("NeuralNetwork", "Invalid input dimensions: X rows=" + std::to_string(X.rows()) + 
                        ", y size=" + std::to_string(y.rows()));
        return false;
    }
    
    LOG_CLASS_INFO("NeuralNetwork", "Starting training with " + std::to_string(X.rows()) + 
                   " samples and " + std::to_string(X.cols()) + " features");
    
    // Store data dimensions
    int n_samples = X.rows();
    int n_features = X.cols();
    
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
    
    // Normalize features
    X_mean_ = X.colwise().mean();
    X_std_ = ((X.rowwise() - X_mean_.transpose()).array().square().colwise().sum() / n_samples).sqrt();
    
    // Handle zero standard deviation
    for (int i = 0; i < X_std_.size(); ++i) {
        if (X_std_(i) < 1e-10) {
            X_std_(i) = 1.0;
        }
    }
    
    Eigen::MatrixXd X_scaled = (X.rowwise() - X_mean_.transpose()).array().rowwise() / X_std_.transpose().array();
    
    // Normalize target
    y_mean_ = y.mean();
    y_std_ = std::sqrt((y.array() - y_mean_).square().sum() / n_samples);
    if (y_std_ < 1e-10) {
        y_std_ = 1.0;
    }
    
    Eigen::VectorXd y_scaled = (y.array() - y_mean_) / y_std_;
    
    LOG_CLASS_DEBUG("NeuralNetwork", "Data normalized, target mean=" + 
                    std::to_string(y_mean_) + ", target std=" + std::to_string(y_std_));
    
    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist(0.0, 0.1);
    
    // Set up the network architecture
    std::vector<int> layer_sizes;
    layer_sizes.push_back(n_features);
    for (int hidden_size : hidden_layer_sizes_) {
        layer_sizes.push_back(hidden_size);
    }
    layer_sizes.push_back(1); // Output layer (regression)
    
    std::string network_structure = "Network structure: input(" + std::to_string(n_features) + ")";
    for (size_t i = 1; i < layer_sizes.size() - 1; ++i) {
        network_structure += " -> hidden" + std::to_string(i) + "(" + std::to_string(layer_sizes[i]) + ")";
    }
    network_structure += " -> output(1)";
    LOG_CLASS_INFO("NeuralNetwork", network_structure);
    
    // Initialize weights and biases
    weights_.resize(layer_sizes.size() - 1);
    biases_.resize(layer_sizes.size() - 1);
    
    for (size_t i = 0; i < weights_.size(); ++i) {
        // He initialization
        double scale = std::sqrt(2.0 / layer_sizes[i]);
        
        weights_[i] = Eigen::MatrixXd(layer_sizes[i + 1], layer_sizes[i]);
        biases_[i] = Eigen::VectorXd::Zero(layer_sizes[i + 1]);
        
        // Initialize weights with random values
        for (int r = 0; r < weights_[i].rows(); ++r) {
            for (int c = 0; c < weights_[i].cols(); ++c) {
                weights_[i](r, c) = dist(gen) * scale;
            }
        }
        
        LOG_CLASS_DEBUG("NeuralNetwork", "Initialized layer " + std::to_string(i+1) + 
                        " weights with shape " + std::to_string(weights_[i].rows()) + "x" + 
                        std::to_string(weights_[i].cols()) + " using He initialization");
    }
    
    // Train the network using mini-batch gradient descent
    int batch_size = std::min(200, n_samples);
    int n_batches = (n_samples + batch_size - 1) / batch_size; // Ceiling division
    
    LOG_CLASS_INFO("NeuralNetwork", "Training with mini-batch gradient descent, batch_size=" + 
                   std::to_string(batch_size) + ", " + std::to_string(n_batches) + " batches per epoch");
    
    double last_loss = 0.0;
    
    // For each epoch
    for (int iter = 0; iter < max_iter_; ++iter) {
        // Update training progress
        stats_["Training Progress"] = static_cast<double>(iter) / max_iter_;
        
        // Shuffle the data
        std::vector<int> indices(n_samples);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), gen);
        
        // Process each mini-batch
        for (int batch = 0; batch < n_batches; ++batch) {
            int start_idx = batch * batch_size;
            int end_idx = std::min(start_idx + batch_size, n_samples);
            int current_batch_size = end_idx - start_idx;
            
            // Prepare batch data
            Eigen::MatrixXd X_batch(current_batch_size, n_features);
            Eigen::VectorXd y_batch(current_batch_size);
            
            for (int i = 0; i < current_batch_size; ++i) {
                X_batch.row(i) = X_scaled.row(indices[start_idx + i]);
                y_batch(i) = y_scaled(indices[start_idx + i]);
            }
            
            // Forward pass and backpropagation
            backpropagate(X_batch, y_batch);
        }
        
        // Calculate and log loss every 10% of iterations or for specific iterations
        if (iter == 0 || iter == max_iter_ - 1 || (max_iter_ > 10 && iter % (max_iter_ / 10) == 0)) {
            // Calculate MSE loss on the entire dataset
            Eigen::VectorXd predictions = forwardPass(X_scaled);
            double mse = (predictions - y_scaled).array().square().mean();
            last_loss = mse;
            
            LOG_CLASS_INFO("NeuralNetwork", "Epoch " + std::to_string(iter+1) + "/" + 
                           std::to_string(max_iter_) + ": MSE = " + std::to_string(mse));
        }
    }
    
    // Set final training progress to 1.0 (100%)
    stats_["Training Progress"] = 1.0;
    
    // Compute final statistics
    computeStats(X, y);
    LOG_CLASS_INFO("NeuralNetwork", "Training complete. RMSE: " + std::to_string(rmse_) + 
                   ", R2: " + std::to_string(r_squared_) + 
                   ", Final loss: " + std::to_string(last_loss));
    
    return true;
}

Eigen::VectorXd NeuralNetwork::predict(const Eigen::MatrixXd& X) const {
    LOG_CLASS_DEBUG("NeuralNetwork", "Predicting for " + std::to_string(X.rows()) + " samples");
    
    // Scale input features
    Eigen::MatrixXd X_scaled = (X.rowwise() - X_mean_.transpose()).array().rowwise() / X_std_.transpose().array();
    
    // Forward pass through the network
    Eigen::VectorXd predictions = forwardPass(X_scaled);
    
    // Rescale predictions back to original scale
    return predictions.array() * y_std_ + y_mean_;
}

std::map<std::string, double> NeuralNetwork::getStats() const {
    return stats_;
}

std::string NeuralNetwork::getDescription() const {
    return "Neural Network Regressor (Multi-layer Perceptron) is a deep learning model "
           "that consists of multiple layers of neurons with non-linear activation functions. "
           "It can model complex non-linear relationships between features and target variables. "
           "This implementation uses ReLU activation for hidden layers and mini-batch gradient descent.";
}

Eigen::VectorXd NeuralNetwork::getCoefficients() const {
    // Neural networks don't have coefficients in the traditional sense
    // Return weights of the first layer as a proxy
    if (!weights_.empty()) {
        Eigen::MatrixXd first_layer_weights = weights_[0];
        Eigen::VectorXd flattened(first_layer_weights.rows() * first_layer_weights.cols());
        
        int idx = 0;
        for (int r = 0; r < first_layer_weights.rows(); ++r) {
            for (int c = 0; c < first_layer_weights.cols(); ++c) {
                flattened(idx++) = first_layer_weights(r, c);
            }
        }
        
        return flattened;
    }
    
    return Eigen::VectorXd();
}

std::map<std::string, double> NeuralNetwork::getHyperparameters() const {
    std::map<std::string, double> hyperparams;
    
    hyperparams["learning_rate"] = learning_rate_;
    hyperparams["max_iterations"] = static_cast<double>(max_iter_);
    hyperparams["alpha"] = alpha_;
    
    // Add hidden layer sizes
    for (size_t i = 0; i < hidden_layer_sizes_.size(); ++i) {
        hyperparams["hidden_layer_" + std::to_string(i + 1) + "_size"] = 
            static_cast<double>(hidden_layer_sizes_[i]);
    }
    
    return hyperparams;
}

std::tuple<std::vector<int>, double, double, int> NeuralNetwork::gridSearch(
    const Eigen::MatrixXd& X, 
    const Eigen::VectorXd& y,
    const std::vector<std::vector<int>>& hidden_layer_sizes_values,
    const std::vector<double>& learning_rate_values,
    const std::vector<double>& alpha_values,
    const std::vector<int>& max_iterations_values,
    int k) {
    
    if (X.rows() != y.rows() || hidden_layer_sizes_values.empty() || 
        learning_rate_values.empty() || alpha_values.empty() || 
        max_iterations_values.empty() || k <= 1) {
        LOG_CLASS_ERROR("NeuralNetwork", "Invalid grid search parameters");
        return {hidden_layer_sizes_, learning_rate_, alpha_, max_iter_};
    }
    
    LOG_CLASS_INFO("NeuralNetwork", "Starting grid search with " + std::to_string(X.rows()) + 
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
    std::vector<int> best_hidden_layer_sizes = hidden_layer_sizes_;
    double best_learning_rate = learning_rate_;
    double best_alpha = alpha_;
    int best_max_iterations = max_iter_;
    double best_rmse = std::numeric_limits<double>::max();
    
    // Grid search with progress tracking
    int total_combinations = hidden_layer_sizes_values.size() * learning_rate_values.size() * 
                           alpha_values.size() * max_iterations_values.size();
    int current_combination = 0;
    
    LOG_CLASS_INFO("NeuralNetwork", "Grid search will evaluate " + 
                   std::to_string(total_combinations) + " combinations");
    
    // Initialize progress in the main stats map
    stats_["Grid Search Progress"] = 0.0;
    
    // Grid search
    for (const auto& hidden_layer_sizes : hidden_layer_sizes_values) {
        // Convert hidden layer sizes to string for logging
        std::string layers_str = "";
        for (size_t i = 0; i < hidden_layer_sizes.size(); ++i) {
            if (i > 0) layers_str += ", ";
            layers_str += std::to_string(hidden_layer_sizes[i]);
        }
        
        for (double learning_rate : learning_rate_values) {
            for (double alpha : alpha_values) {
                for (int max_iterations : max_iterations_values) {
                    current_combination++;
                    double fold_rmse_sum = 0.0;
                    bool early_stop = false;
                    int valid_folds = 0;
                    
                    // Update progress in both the temporary stats and the main stats object
                    double progress = static_cast<double>(current_combination) / total_combinations;
                    grid_search_stats["Grid Search Progress"] = progress;
                    stats_["Grid Search Progress"] = progress;
                    
                    LOG_CLASS_DEBUG("NeuralNetwork", "Evaluating combination " + std::to_string(current_combination) + 
                                   "/" + std::to_string(total_combinations) + 
                                   ": hidden_layers=[" + layers_str + "]" +
                                   ", learning_rate=" + std::to_string(learning_rate) + 
                                   ", alpha=" + std::to_string(alpha) + 
                                   ", max_iterations=" + std::to_string(max_iterations));
                    
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
                            y_train(i) = y_std_vec(train_indices[i]);
                        }
                        
                        for (size_t i = 0; i < val_indices.size(); ++i) {
                            X_val.row(i) = X_std.row(val_indices[i]);
                            y_val(i) = y_std_vec(val_indices[i]);
                        }
                        
                        try {
                            // Train model with current hyperparameters
                            NeuralNetwork model(hidden_layer_sizes, learning_rate, max_iterations, alpha);
                            
                            // Initialize weights with Xavier/Glorot initialization
                            model.weights_.clear();
                            model.biases_.clear();
                            
                            std::vector<int> layer_sizes;
                            layer_sizes.push_back(X_train.cols());  // Input layer
                            layer_sizes.insert(layer_sizes.end(), 
                                            hidden_layer_sizes.begin(), 
                                            hidden_layer_sizes.end());
                            layer_sizes.push_back(1);  // Output layer
                            
                            std::random_device rd;
                            std::mt19937 gen(rd());
                            
                            for (size_t i = 0; i < layer_sizes.size() - 1; ++i) {
                                double limit = std::sqrt(6.0 / (layer_sizes[i] + layer_sizes[i + 1]));
                                std::uniform_real_distribution<> dis(-limit, limit);
                                
                                Eigen::MatrixXd W = Eigen::MatrixXd::Zero(layer_sizes[i + 1], layer_sizes[i]);
                                for (int r = 0; r < W.rows(); ++r) {
                                    for (int c = 0; c < W.cols(); ++c) {
                                        W(r, c) = dis(gen);
                                    }
                                }
                                model.weights_.push_back(W);
                                model.biases_.push_back(Eigen::VectorXd::Zero(layer_sizes[i + 1]));
                            }
                            
                            if (!model.train(X_train, y_train)) {
                                // If training fails, skip this combination
                                LOG_CLASS_WARNING("NeuralNetwork", "Training failed for fold " + 
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
                            if (!std::isfinite(fold_rmse) || fold_rmse > 1e6) {
                                LOG_CLASS_WARNING("NeuralNetwork", "Non-finite RMSE in fold " + 
                                                std::to_string(fold+1) + ", skipping combination");
                                early_stop = true;
                                break;
                            }
                            
                            // Early stopping if RMSE is much worse than best
                            if (best_rmse < std::numeric_limits<double>::max() && 
                                fold_rmse > 2.0 * best_rmse) {
                                LOG_CLASS_DEBUG("NeuralNetwork", "Early stopping for poor performance, RMSE=" + 
                                              std::to_string(fold_rmse) + " vs best=" + 
                                              std::to_string(best_rmse));
                                early_stop = true;
                                break;
                            }
                            
                            fold_rmse_sum += fold_rmse;
                            valid_folds++;
                            
                            LOG_CLASS_DEBUG("NeuralNetwork", "Fold " + std::to_string(fold+1) + 
                                          " RMSE: " + std::to_string(fold_rmse));
                            
                        } catch (const std::exception& e) {
                            // If any exception occurs, skip this combination
                            LOG_CLASS_WARNING("NeuralNetwork", "Exception in fold " + std::to_string(fold+1) + 
                                            ": " + e.what());
                            early_stop = true;
                            break;
                        }
                    }
                    
                    if (!early_stop && valid_folds > 0) {
                        // Average RMSE across valid folds
                        double avg_rmse = fold_rmse_sum / valid_folds;
                        
                        LOG_CLASS_INFO("NeuralNetwork", "Combination " + std::to_string(current_combination) +  
                                      ": Average RMSE = " + std::to_string(avg_rmse));
                        
                        // Update best hyperparameters if RMSE is improved
                        if (avg_rmse < best_rmse) {
                            best_rmse = avg_rmse;
                            best_hidden_layer_sizes = hidden_layer_sizes;
                            best_learning_rate = learning_rate;
                            best_alpha = alpha;
                            best_max_iterations = max_iterations;
                            
                            LOG_CLASS_INFO("NeuralNetwork", "New best combination found! RMSE = " + 
                                          std::to_string(best_rmse));
                            
                            // Store best parameters in grid search stats
                            grid_search_stats["Best RMSE"] = best_rmse;
                            
                            // Store layer sizes as separate values
                            for (size_t i = 0; i < hidden_layer_sizes.size(); ++i) {
                                grid_search_stats["Best Layer " + std::to_string(i) + " Size"] = 
                                    static_cast<double>(hidden_layer_sizes[i]);
                            }
                            
                            grid_search_stats["Best Learning Rate"] = best_learning_rate;
                            grid_search_stats["Best Alpha"] = best_alpha;
                            grid_search_stats["Best Max Iterations"] = static_cast<double>(best_max_iterations);
                        }
                    }
                }
            }
        }
    }
    
    // Set final grid search progress to 1.0 (100%)
    stats_["Grid Search Progress"] = 1.0;
    
    // Store the best grid search results in the model's stats
    for (const auto& pair : grid_search_stats) {
        if (pair.first.substr(0, 4) == "Best") {
            stats_["Grid Search " + pair.first] = pair.second;
        }
    }
    
    // Convert best hidden layer sizes to string for logging
    std::string best_layers_str = "";
    for (size_t i = 0; i < best_hidden_layer_sizes.size(); ++i) {
        if (i > 0) best_layers_str += ", ";
        best_layers_str += std::to_string(best_hidden_layer_sizes[i]);
    }
    
    std::stringstream ss;
    ss << "Grid search complete. Best hyperparameters: "
       << "hidden_layers=[" << best_layers_str << "]"
       << ", learning_rate=" << best_learning_rate
       << ", alpha=" << best_alpha
       << ", max_iterations=" << best_max_iterations
       << ", RMSE=" << best_rmse;
    LOG_CLASS_INFO("NeuralNetwork", ss.str());
    
    return {best_hidden_layer_sizes, best_learning_rate, best_alpha, best_max_iterations};
}

void NeuralNetwork::computeStats(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
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
    
    // Calculate network architecture statistics
    int total_layers = hidden_layer_sizes_.size() + 2; // Input + Hidden + Output
    int total_params = 0;
    
    // Count parameters (weights and biases)
    for (size_t i = 0; i < weights_.size(); ++i) {
        total_params += weights_[i].size() + biases_[i].size();
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
        {"Total Layers", static_cast<double>(total_layers)},
        {"Total Parameters", static_cast<double>(total_params)},
        {"Training Loss", mse}, // Using MSE as the training loss
        {"Learning Rate", learning_rate_},
        {"Alpha", alpha_},
        {"Max Iterations", static_cast<double>(max_iter_)},
        {"Training Progress", training_progress}
    };
    
    // Store hidden layer sizes
    for (size_t i = 0; i < hidden_layer_sizes_.size(); ++i) {
        new_stats["Hidden Layer " + std::to_string(i + 1) + " Size"] = 
            static_cast<double>(hidden_layer_sizes_[i]);
    }
    
    // Restore grid search stats
    for (const auto& pair : grid_search_stats) {
        new_stats[pair.first] = pair.second;
    }
    
    // Replace stats with new stats
    stats_ = new_stats;
}

Eigen::VectorXd NeuralNetwork::forwardPass(const Eigen::MatrixXd& X) const {
    Eigen::MatrixXd activations = X;
    
    // Process all layers except the output layer
    for (size_t i = 0; i < weights_.size() - 1; ++i) {
        // Linear transformation: Z = W * X + b
        Eigen::MatrixXd Z = (activations * weights_[i].transpose()).rowwise() + biases_[i].transpose();
        
        // Apply ReLU activation function
        activations = Z.array().max(0.0);
    }
    
    // Output layer (linear activation for regression)
    size_t last = weights_.size() - 1;
    Eigen::VectorXd output = (activations * weights_[last].transpose()).rowwise() + biases_[last].transpose();
    
    return output;
}

void NeuralNetwork::backpropagate(const Eigen::MatrixXd& X_batch, const Eigen::VectorXd& y_batch) {
    int batch_size = X_batch.rows();
    
    // Forward pass with stored activations
    std::vector<Eigen::MatrixXd> layer_inputs; // Before activation
    std::vector<Eigen::MatrixXd> activations;  // After activation
    
    activations.push_back(X_batch); // Input layer
    
    // Forward pass through hidden layers
    for (size_t i = 0; i < weights_.size() - 1; ++i) {
        Eigen::MatrixXd Z = (activations.back() * weights_[i].transpose()).rowwise() + biases_[i].transpose();
        layer_inputs.push_back(Z);
        
        // Apply ReLU activation
        Eigen::MatrixXd A = Z.array().max(0.0);
        activations.push_back(A);
    }
    
    // Output layer
    size_t last = weights_.size() - 1;
    Eigen::MatrixXd Z_output = (activations.back() * weights_[last].transpose()).rowwise() + biases_[last].transpose();
    layer_inputs.push_back(Z_output);
    activations.push_back(Z_output); // Linear activation for output layer
    
    // Backpropagation
    // Initialize weight and bias gradients
    std::vector<Eigen::MatrixXd> dW(weights_.size());
    std::vector<Eigen::VectorXd> db(biases_.size());
    
    // Output layer error
    Eigen::MatrixXd delta = activations.back() - y_batch;
    
    // Compute gradients for output layer
    dW[last] = (delta.transpose() * activations[activations.size() - 2]) / batch_size;
    db[last] = delta.colwise().sum() / batch_size;
    
    // Add L2 regularization to output layer weights
    dW[last].array() += alpha_ * weights_[last].array();
    
    // Backpropagate error through hidden layers
    for (int i = static_cast<int>(weights_.size()) - 2; i >= 0; --i) {
        // Compute delta for current layer
        delta = (delta * weights_[i + 1]).array() * 
                (layer_inputs[i].array() > 0.0).cast<double>(); // ReLU derivative
        
        // Compute gradients
        dW[i] = (delta.transpose() * activations[i]) / batch_size;
        db[i] = delta.colwise().sum() / batch_size;
        
        // Add L2 regularization
        dW[i].array() += alpha_ * weights_[i].array();
    }
    
    // Update weights and biases using gradient descent
    for (size_t i = 0; i < weights_.size(); ++i) {
        weights_[i] -= learning_rate_ * dW[i];
        biases_[i] -= learning_rate_ * db[i];
    }
}

double NeuralNetwork::relu(double x) const {
    return std::max(0.0, x);
}

double NeuralNetwork::reluDerivative(double x) const {
    return x > 0.0 ? 1.0 : 0.0;
}

Eigen::VectorXd NeuralNetwork::getFeatureImportance() const {
    // For neural networks, we can approximate feature importance
    // by looking at the absolute values of the weights in the first layer
    if (weights_.empty()) {
        return Eigen::VectorXd::Zero(0);
    }
    
    // Get the weights from the first layer
    const Eigen::MatrixXd& first_layer_weights = weights_[0];
    
    // Calculate feature importance as the sum of absolute weights
    // connecting each input feature to all neurons in the first hidden layer
    Eigen::VectorXd importance = first_layer_weights.array().abs().colwise().sum();
    
    // Normalize by the number of neurons in the first hidden layer
    importance /= first_layer_weights.rows();
    
    return importance;
}

} // namespace DataAnalyzer 