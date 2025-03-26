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

// Placeholder for future plot types
/*
class ScatterPlot : public Plot {
    // To be implemented
};

class BarChart : public Plot {
    // To be implemented
};

class Histogram : public Plot {
    // To be implemented
};
*/

} // namespace DataAnalyzer 