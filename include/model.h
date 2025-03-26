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
               int max_iter = 1000, double tol = 1e-4);
    
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
    
    // Private helper methods
    double softThreshold(double x, double lambda) const;
    void computeStats(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);
};

// Placeholders for future models
/*
class LinearRegression : public Model {
    // To be implemented
};

class RandomForest : public Model {
    // To be implemented
};

class XGBoost : public Model {
    // To be implemented
};

class NeuralNetwork : public Model {
    // To be implemented
};
*/

} // namespace DataAnalyzer 