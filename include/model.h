#pragma once

#include <Eigen/Dense>
#include <string>
#include <vector>
#include <map>
#include <memory>

namespace DataAnalyzer {

/**
 * @brief Base class for all models
 */
class Model {
public:
    Model() = default;
    virtual ~Model() = default;

    /**
     * @brief Train the model with the given data
     * @param X Feature matrix (each row is a sample, each column is a feature)
     * @param y Target vector
     * @return bool True if training was successful
     */
    virtual bool train(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) = 0;

    /**
     * @brief Predict using the trained model
     * @param X Feature matrix (each row is a sample, each column is a feature)
     * @return Eigen::VectorXd Predictions
     */
    virtual Eigen::VectorXd predict(const Eigen::MatrixXd& X) const = 0;

    /**
     * @brief Get model statistics
     * @return std::map<std::string, double> Map of statistic names to values
     */
    virtual std::map<std::string, double> getStats() const = 0;

    /**
     * @brief Get model description
     * @return std::string Description of the model
     */
    virtual std::string getDescription() const = 0;
    
    /**
     * @brief Get model coefficients
     * @return Eigen::VectorXd Coefficients
     */
    virtual Eigen::VectorXd getCoefficients() const = 0;
    
    /**
     * @brief Get model hyperparameters
     * @return std::map<std::string, double> Map of hyperparameter names to values
     */
    virtual std::map<std::string, double> getHyperparameters() const = 0;

    /**
     * @brief Check if the model supports feature importance
     * @return bool True if the model supports feature importance
     */
    virtual bool hasFeatureImportance() const { return false; }

    /**
     * @brief Get feature importance scores
     * @return Eigen::VectorXd Feature importance scores
     */
    virtual Eigen::VectorXd getFeatureImportance() const {
        throw std::runtime_error("Feature importance not supported by this model");
    }

    // Helper functions to get specific statistics
    double getRMSE() const {
        auto stats = getStats();
        return stats["RMSE"];
    }
    
    double getR2() const {
        auto stats = getStats();
        return stats["R²"];
    }
    
    double getMAE() const {
        auto stats = getStats();
        return stats["MAE"];
    }
};

/**
 * @brief Elastic Net Regression model
 * Combines L1 and L2 regularization
 */
class ElasticNet : public Model {
public:
    /**
     * @brief Construct a new Elastic Net model
     * @param alpha Mixing parameter between L1 and L2 (0 <= alpha <= 1)
     * @param lambda Regularization strength
     * @param max_iter Maximum number of iterations
     * @param tol Tolerance for stopping criteria
     */
    ElasticNet(double alpha = 0.5, double lambda = 1.0, 
               int max_iter = 10000, double tol = 1e-3);
    
    ~ElasticNet() override = default;

    bool train(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) override;
    Eigen::VectorXd predict(const Eigen::MatrixXd& X) const override;
    std::map<std::string, double> getStats() const override;
    std::string getDescription() const override;
    Eigen::VectorXd getCoefficients() const override;
    std::map<std::string, double> getHyperparameters() const override;
    
    /**
     * @brief Perform grid search to find the best hyperparameters
     * @param X Feature matrix
     * @param y Target vector
     * @param alpha_values Vector of alpha values to try
     * @param lambda_values Vector of lambda values to try
     * @param k Number of folds for cross-validation
     * @return std::pair<double, double> Best (alpha, lambda) pair
     */
    std::pair<double, double> gridSearch(
        const Eigen::MatrixXd& X, 
        const Eigen::VectorXd& y,
        const std::vector<double>& alpha_values,
        const std::vector<double>& lambda_values,
        int k = 5);

    bool hasFeatureImportance() const override { return true; }
    Eigen::VectorXd getFeatureImportance() const override;

private:
    double alpha_;    // Mixing parameter between L1 and L2 (0 <= alpha <= 1)
    double lambda_;   // Regularization strength
    int max_iter_;    // Maximum number of iterations
    double tol_;      // Tolerance for stopping criteria
    
    Eigen::VectorXd coefficients_;  // Model coefficients
    double intercept_;               // Model intercept
    
    // Statistics
    double rmse_;     // Root Mean Squared Error
    double r_squared_; // R-squared
    std::map<std::string, double> stats_; // Model statistics
    
    // Private helper methods
    double softThreshold(double x, double lambda) const;
    void computeStats(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);
};

/**
 * @brief Linear Regression model
 * Standard ordinary least squares regression
 */
class LinearRegression : public Model {
public:
    /**
     * @brief Construct a new Linear Regression model
     * @param fit_intercept Whether to calculate the intercept for the model
     */
    LinearRegression(bool fit_intercept = true);
    
    ~LinearRegression() override = default;

    bool train(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) override;
    Eigen::VectorXd predict(const Eigen::MatrixXd& X) const override;
    std::map<std::string, double> getStats() const override;
    std::string getDescription() const override;
    Eigen::VectorXd getCoefficients() const override;
    std::map<std::string, double> getHyperparameters() const override;

private:
    bool fit_intercept_;  // Whether to fit an intercept
    
    Eigen::VectorXd coefficients_;  // Model coefficients
    double intercept_;              // Model intercept
    
    // Statistics
    double rmse_;        // Root Mean Squared Error
    double r_squared_;   // R-squared
    std::map<std::string, double> stats_; // Model statistics
    
    // Private helper method
    void computeStats(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);
};

/**
 * @brief XGBoost model
 * Extreme Gradient Boosting
 */
class XGBoost : public Model {
public:
    /**
     * @brief Construct a new XGBoost model
     * @param n_estimators Number of boosting rounds
     * @param learning_rate Step size shrinkage used to prevent overfitting
     * @param max_depth Maximum depth of a tree
     * @param subsample Subsample ratio of the training instances
     */
    XGBoost(int n_estimators = 100, double learning_rate = 0.1,
            int max_depth = 3, double subsample = 1.0);
    
    ~XGBoost() override = default;

    bool train(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) override;
    Eigen::VectorXd predict(const Eigen::MatrixXd& X) const override;
    std::map<std::string, double> getStats() const override;
    std::string getDescription() const override;
    Eigen::VectorXd getCoefficients() const override;
    std::map<std::string, double> getHyperparameters() const override;
    
    /**
     * @brief Perform grid search to find the best hyperparameters
     * @param X Feature matrix
     * @param y Target vector
     * @param n_estimators_values Vector of n_estimators values to try
     * @param learning_rate_values Vector of learning_rate values to try
     * @param max_depth_values Vector of max_depth values to try
     * @param subsample_values Vector of subsample values to try
     * @param k Number of folds for cross-validation
     * @return std::tuple<int, double, int, double> Best (n_estimators, learning_rate, max_depth, subsample) tuple
     */
    std::tuple<int, double, int, double> gridSearch(
        const Eigen::MatrixXd& X, 
        const Eigen::VectorXd& y,
        const std::vector<int>& n_estimators_values,
        const std::vector<double>& learning_rate_values,
        const std::vector<int>& max_depth_values,
        const std::vector<double>& subsample_values,
        int k = 5);

    bool hasFeatureImportance() const override { return true; }
    Eigen::VectorXd getFeatureImportance() const override;

private:
    // Internal model representation
    struct Tree {
        std::vector<int> feature_indices;
        std::vector<double> thresholds;
        std::vector<double> values;
        std::vector<int> left_children;
        std::vector<int> right_children;
    };

    // Hyperparameters
    int n_estimators_;        // Number of boosting rounds
    double learning_rate_;    // Step size shrinkage
    int max_depth_;          // Maximum depth of a tree
    double subsample_;       // Subsample ratio of training instances
    
    // Model state
    std::vector<Tree> trees_;  // Collection of trees
    double base_prediction_;   // Initial prediction (mean of target values)
    double rmse_;             // Root mean squared error
    double r_squared_;        // R-squared score
    
    // Statistics
    std::map<std::string, double> stats_;
    
    // Private helper methods
    void computeStats(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);
    double predictTree(const Tree& tree, const Eigen::VectorXd& x, int node_idx = 0) const;
    void buildRegressionTree(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, 
                           Tree& tree, int depth, int max_depth);
    double computeAverageTreeDepth() const;
};

/**
 * @brief Gradient Boosting model
 * Gradient Boosting Regression
 */
class GradientBoosting : public Model {
public:
    /**
     * @brief Construct a new Gradient Boosting model
     * @param n_estimators Number of boosting stages
     * @param learning_rate Learning rate shrinks the contribution of each tree
     * @param max_depth Maximum depth of the individual regression estimators
     * @param min_samples_split Minimum number of samples required to split an internal node
     */
    GradientBoosting(int n_estimators = 100, double learning_rate = 0.1,
                     int max_depth = 3, int min_samples_split = 2);
    
    ~GradientBoosting() override = default;

    bool train(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) override;
    Eigen::VectorXd predict(const Eigen::MatrixXd& X) const override;
    std::map<std::string, double> getStats() const override;
    std::string getDescription() const override;
    Eigen::VectorXd getCoefficients() const override;
    std::map<std::string, double> getHyperparameters() const override;
    
    /**
     * @brief Perform grid search to find the best hyperparameters
     * @param X Feature matrix
     * @param y Target vector
     * @param n_estimators_values Vector of n_estimators values to try
     * @param learning_rate_values Vector of learning_rate values to try
     * @param max_depth_values Vector of max_depth values to try
     * @param min_samples_split_values Vector of min_samples_split values to try
     * @param k Number of folds for cross-validation
     * @return std::tuple<int, double, int, int> Best (n_estimators, learning_rate, max_depth, min_samples_split) tuple
     */
    std::tuple<int, double, int, int> gridSearch(
        const Eigen::MatrixXd& X, 
        const Eigen::VectorXd& y,
        const std::vector<int>& n_estimators_values,
        const std::vector<double>& learning_rate_values,
        const std::vector<int>& max_depth_values,
        const std::vector<int>& min_samples_split_values,
        int k = 5);

    bool hasFeatureImportance() const override { return true; }
    Eigen::VectorXd getFeatureImportance() const override;

private:
    struct Node {
        bool is_leaf;
        int feature_idx;
        double threshold;
        double value;
        int left_child;
        int right_child;
        
        Node() : is_leaf(true), feature_idx(-1), threshold(0.0), value(0.0),
                 left_child(-1), right_child(-1) {}
    };
    
    struct DecisionTree {
        std::vector<Node> nodes;
        
        double predict(const Eigen::VectorXd& x) const {
            int current_node = 0;
            while (!nodes[current_node].is_leaf) {
                if (x(nodes[current_node].feature_idx) <= nodes[current_node].threshold) {
                    current_node = nodes[current_node].left_child;
                } else {
                    current_node = nodes[current_node].right_child;
                }
            }
            return nodes[current_node].value;
        }
    };
    
    int n_estimators_;
    double learning_rate_;
    int max_depth_;
    int min_samples_split_;
    double initial_prediction_;
    std::vector<DecisionTree> trees_;
    std::map<std::string, double> stats_;
    double rmse_;
    double r_squared_;
    
    // Feature standardization parameters
    Eigen::VectorXd feature_means_;
    Eigen::VectorXd feature_stds_;
    
    // Target standardization parameters
    double target_mean_;
    double target_std_;
    
    DecisionTree buildTree(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, int depth);
    void buildTreeRecursive(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, 
                           int node_idx, int depth, DecisionTree& tree,
                           const std::vector<int>& sample_indices, int current_depth);
    void computeStats(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);
};

/**
 * @brief Neural Network model
 * Multi-layer Perceptron Regressor
 */
class NeuralNetwork : public Model {
public:
    /**
     * @brief Construct a new Neural Network model
     * @param hidden_layer_sizes Vector with the number of neurons in each hidden layer
     * @param learning_rate Learning rate for weight updates
     * @param max_iter Maximum number of iterations
     * @param alpha L2 regularization term
     */
    NeuralNetwork(const std::vector<int>& hidden_layer_sizes = {100},
                  double learning_rate = 0.001, int max_iter = 200,
                  double alpha = 0.0001);
    
    ~NeuralNetwork() override = default;

    bool train(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) override;
    Eigen::VectorXd predict(const Eigen::MatrixXd& X) const override;
    std::map<std::string, double> getStats() const override;
    std::string getDescription() const override;
    Eigen::VectorXd getCoefficients() const override;
    std::map<std::string, double> getHyperparameters() const override;
    
    /**
     * @brief Perform grid search to find the best hyperparameters
     * @param X Feature matrix
     * @param y Target vector
     * @param hidden_layer_sizes_values Vector of hidden_layer_sizes configurations to try
     * @param learning_rate_values Vector of learning_rate values to try
     * @param alpha_values Vector of alpha values to try
     * @param max_iterations_values Vector of max_iterations values to try
     * @param k Number of folds for cross-validation
     * @return std::tuple<std::vector<int>, double, double, int> Best (hidden_layer_sizes, learning_rate, alpha, max_iterations) tuple
     */
    std::tuple<std::vector<int>, double, double, int> gridSearch(
        const Eigen::MatrixXd& X, 
        const Eigen::VectorXd& y,
        const std::vector<std::vector<int>>& hidden_layer_sizes_values,
        const std::vector<double>& learning_rate_values,
        const std::vector<double>& alpha_values,
        const std::vector<int>& max_iterations_values,
        int k = 5);

    bool hasFeatureImportance() const override { return true; }
    Eigen::VectorXd getFeatureImportance() const override;

private:
    // Hyperparameters
    std::vector<int> hidden_layer_sizes_;  // Number of neurons in each hidden layer
    double learning_rate_;                 // Learning rate
    int max_iter_;                         // Maximum number of iterations
    double alpha_;                         // L2 regularization term
    
    // Model parameters
    std::vector<Eigen::MatrixXd> weights_;   // Weights for each layer
    std::vector<Eigen::VectorXd> biases_;    // Biases for each layer
    
    // Normalization parameters
    Eigen::VectorXd X_mean_;
    Eigen::VectorXd X_std_;
    double y_mean_;
    double y_std_;
    
    // Statistics
    double rmse_;       // Root Mean Squared Error
    double r_squared_;  // R-squared
    std::map<std::string, double> stats_; // Model statistics
    
    // Private helper methods
    void computeStats(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);
    Eigen::VectorXd forwardPass(const Eigen::MatrixXd& X) const;
    void backpropagate(const Eigen::MatrixXd& X_batch, const Eigen::VectorXd& y_batch);
    double relu(double x) const;
    double reluDerivative(double x) const;
};

} // namespace DataAnalyzer 