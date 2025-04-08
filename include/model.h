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
} // namespace DataAnalyzer 