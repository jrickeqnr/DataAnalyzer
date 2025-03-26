#pragma once

#include <Eigen/Dense>
#include <string>
#include <vector>
#include <memory>
#include "data_handler.h"

namespace DataAnalyzer {

/**
 * @brief Base class for all plot types
 */
class Plot {
public:
    Plot(const std::string& title = "Plot");
    virtual ~Plot() = default;
    
    /**
     * @brief Render the plot
     */
    virtual void render() = 0;
    
    /**
     * @brief Get the plot title
     * @return std::string The plot title
     */
    std::string getTitle() const;
    
    /**
     * @brief Set the plot title
     * @param title The new title
     */
    void setTitle(const std::string& title);
    
    /**
     * @brief Save the plot as an image
     * @param filepath Path to save the image
     * @return bool True if saving was successful
     */
    virtual bool saveImage(const std::string& filepath) const = 0;

protected:
    std::string title_;
};

/**
 * @brief Class for time series plots
 */
class TimeSeries : public Plot {
public:
    /**
     * @brief Construct a new Time Series plot
     * @param title Plot title
     * @param xLabel Label for x-axis
     * @param yLabel Label for y-axis
     */
    TimeSeries(const std::string& title = "Time Series",
              const std::string& xLabel = "Time",
              const std::string& yLabel = "Value");
    
    ~TimeSeries() override = default;
    
    /**
     * @brief Set the data for the plot
     * @param dates Vector of dates for x-axis
     * @param actual Actual values
     * @param predicted Predicted values
     * @return bool True if data was set successfully
     */
    bool setData(const std::vector<Date>& dates,
                const Eigen::VectorXd& actual,
                const Eigen::VectorXd& predicted);
    
    /**
     * @brief Render the plot
     */
    void render() override;
    
    /**
     * @brief Save the plot as an image
     * @param filepath Path to save the image
     * @return bool True if saving was successful
     */
    bool saveImage(const std::string& filepath) const override;
    
    /**
     * @brief Get the string representations of dates for the x-axis
     * @return std::vector<std::string> String representations of dates
     */
    std::vector<std::string> getDateStrings() const;

private:
    std::string xLabel_;
    std::string yLabel_;
    std::vector<Date> dates_;
    Eigen::VectorXd actualValues_;
    Eigen::VectorXd predictedValues_;
    bool hasData_ = false;
};

/**
 * @brief Class for scatter plots of actual vs predicted values
 */
class ScatterPlot : public Plot {
public:
    /**
     * @brief Construct a new Scatter Plot
     * @param title Plot title
     * @param xLabel Label for x-axis
     * @param yLabel Label for y-axis
     */
    ScatterPlot(const std::string& title = "Actual vs Predicted",
               const std::string& xLabel = "Actual",
               const std::string& yLabel = "Predicted");
    
    ~ScatterPlot() override = default;
    
    /**
     * @brief Set the data for the plot
     * @param actual Actual values
     * @param predicted Predicted values
     * @return bool True if data was set successfully
     */
    bool setData(const Eigen::VectorXd& actual,
                const Eigen::VectorXd& predicted);
    
    /**
     * @brief Render the plot
     */
    void render() override;
    
    /**
     * @brief Save the plot as an image
     * @param filepath Path to save the image
     * @return bool True if saving was successful
     */
    bool saveImage(const std::string& filepath) const override;

private:
    std::string xLabel_;
    std::string yLabel_;
    Eigen::VectorXd actualValues_;
    Eigen::VectorXd predictedValues_;
    bool hasData_ = false;
};

/**
 * @brief Class for feature importance plots
 */
class FeatureImportancePlot : public Plot {
public:
    /**
     * @brief Construct a new Feature Importance Plot
     * @param title Plot title
     * @param xLabel Label for x-axis
     * @param yLabel Label for y-axis
     */
    FeatureImportancePlot(const std::string& title = "Feature Importance",
                         const std::string& xLabel = "Features",
                         const std::string& yLabel = "Importance");
    
    ~FeatureImportancePlot() override = default;
    
    /**
     * @brief Set the data for the plot
     * @param featureNames Vector of feature names
     * @param importance Vector of importance values
     * @return bool True if data was set successfully
     */
    bool setData(const std::vector<std::string>& featureNames,
                const Eigen::VectorXd& importance);
    
    /**
     * @brief Render the plot
     */
    void render() override;
    
    /**
     * @brief Save the plot as an image
     * @param filepath Path to save the image
     * @return bool True if saving was successful
     */
    bool saveImage(const std::string& filepath) const override;

private:
    std::string xLabel_;
    std::string yLabel_;
    std::vector<std::string> featureNames_;
    Eigen::VectorXd importanceValues_;
    bool hasData_ = false;
};

/**
 * @brief Class for managing multiple plots
 */
class PlotManager {
public:
    /**
     * @brief Add a plot to the manager
     * @param plot The plot to add
     */
    void addPlot(std::shared_ptr<Plot> plot);
    
    /**
     * @brief Render all plots
     */
    void render();
    
    /**
     * @brief Get the current plot index
     * @return size_t The current plot index
     */
    size_t getCurrentPlotIndex() const;
    
    /**
     * @brief Set the current plot index
     * @param index The new plot index
     */
    void setCurrentPlotIndex(size_t index);
    
    /**
     * @brief Get the number of plots
     * @return size_t The number of plots
     */
    size_t getPlotCount() const;
    
    /**
     * @brief Get a plot by index
     * @param index The index of the plot to get
     * @return std::shared_ptr<Plot> The plot at the given index, or nullptr if invalid
     */
    std::shared_ptr<Plot> getPlot(size_t index) const;

private:
    std::vector<std::shared_ptr<Plot>> plots_;
    size_t currentPlotIndex_ = 0;
};

// Placeholder for future plot types
/*
class BarChart : public Plot {
    // To be implemented
};

class Histogram : public Plot {
    // To be implemented
};
*/

/**
 * @brief Class for residual plots
 */
class ResidualPlot : public Plot {
public:
    /**
     * @brief Construct a new Residual Plot
     * @param title Plot title
     * @param xLabel Label for x-axis
     * @param yLabel Label for y-axis
     */
    ResidualPlot(const std::string& title = "Residual Analysis",
                const std::string& xLabel = "Predicted",
                const std::string& yLabel = "Residuals");
    
    ~ResidualPlot() override = default;
    
    /**
     * @brief Set the data for the plot
     * @param predicted Predicted values
     * @param residuals Residual values
     * @return bool True if data was set successfully
     */
    bool setData(const Eigen::VectorXd& predicted,
                const Eigen::VectorXd& residuals);
    
    /**
     * @brief Render the plot
     */
    void render() override;
    
    /**
     * @brief Save the plot as an image
     * @param filepath Path to save the image
     * @return bool True if saving was successful
     */
    bool saveImage(const std::string& filepath) const override;

private:
    std::string xLabel_;
    std::string yLabel_;
    Eigen::VectorXd predictedValues_;
    Eigen::VectorXd residualValues_;
    bool hasData_ = false;
};

/**
 * @brief Class for coefficient statistics plots
 */
class CoefficientStatsPlot : public Plot {
public:
    /**
     * @brief Construct a new Coefficient Statistics Plot
     * @param title Plot title
     * @param xLabel Label for x-axis
     * @param yLabel Label for y-axis
     */
    CoefficientStatsPlot(const std::string& title = "Coefficient Statistics",
                        const std::string& xLabel = "Features",
                        const std::string& yLabel = "Value");
    
    ~CoefficientStatsPlot() override = default;
    
    /**
     * @brief Set the data for the plot
     * @param featureNames Vector of feature names
     * @param coefficients Vector of coefficient values
     * @param standardErrors Vector of standard errors
     * @param tValues Vector of t-values
     * @return bool True if data was set successfully
     */
    bool setData(const std::vector<std::string>& featureNames,
                const Eigen::VectorXd& coefficients,
                const Eigen::VectorXd& standardErrors,
                const Eigen::VectorXd& tValues);
    
    /**
     * @brief Render the plot
     */
    void render() override;
    
    /**
     * @brief Save the plot as an image
     * @param filepath Path to save the image
     * @return bool True if saving was successful
     */
    bool saveImage(const std::string& filepath) const override;

private:
    std::string xLabel_;
    std::string yLabel_;
    std::vector<std::string> featureNames_;
    Eigen::VectorXd coefficientValues_;
    Eigen::VectorXd standardErrors_;
    Eigen::VectorXd tValues_;
    bool hasData_ = false;
};

} // namespace DataAnalyzer 