#include "../include/plotting.h"
#include <imgui.h>
#include <implot.h>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace DataAnalyzer {

// Plot Implementation
Plot::Plot(const std::string& title)
    : title_(title) {
}

std::string Plot::getTitle() const {
    return title_;
}

void Plot::setTitle(const std::string& title) {
    title_ = title;
}

// TimeSeries Implementation
TimeSeries::TimeSeries(const std::string& title, const std::string& xLabel, const std::string& yLabel)
    : Plot(title), xLabel_(xLabel), yLabel_(yLabel), hasData_(false) {
}

bool TimeSeries::setData(const std::vector<Date>& dates,
                        const Eigen::VectorXd& actual,
                        const Eigen::VectorXd& predicted) {
    if (dates.empty() || dates.size() != actual.size() || actual.size() != predicted.size()) {
        return false;
    }
    
    dates_ = dates;
    actualValues_ = actual;
    predictedValues_ = predicted;
    hasData_ = true;
    
    return true;
}

void TimeSeries::render() {
    if (!hasData_ || dates_.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "No data available to plot");
        return;
    }
    
    // Get date strings for x-axis
    std::vector<std::string> dateStrings = getDateStrings();
    
    // Convert date strings to char* array for ImPlot
    std::vector<const char*> labels(dateStrings.size());
    for (size_t i = 0; i < dateStrings.size(); ++i) {
        labels[i] = dateStrings[i].c_str();
    }
    
    // Convert Eigen vectors to std::vector for ImPlot
    std::vector<double> actual(actualValues_.data(), actualValues_.data() + actualValues_.size());
    std::vector<double> predicted(predictedValues_.data(), predictedValues_.data() + predictedValues_.size());
    
    // Create x-axis values (0, 1, 2, ...)
    std::vector<double> xs(actual.size());
    std::iota(xs.begin(), xs.end(), 0.0);
    
    // Set up plot parameters
    if (ImPlot::BeginPlot(title_.c_str(), ImVec2(-1, -1), ImPlotFlags_NoMouseText)) {
        // Set axis labels
        ImPlot::SetupAxes(xLabel_.c_str(), yLabel_.c_str());
        
        // Set x-axis tick labels to be dates
        ImPlot::SetupAxisTicks(ImAxis_X1, xs.data(), xs.size(), labels.data());
        
        // Plot actual values
        ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 4, ImVec4(0.0f, 0.7f, 0.0f, 1.0f), IMPLOT_AUTO, ImVec4(0.0f, 0.7f, 0.0f, 1.0f));
        ImPlot::PlotLine("Actual", xs.data(), actual.data(), actual.size());
        
        // Plot predicted values
        ImPlot::SetNextMarkerStyle(ImPlotMarker_Square, 4, ImVec4(0.0f, 0.0f, 0.8f, 1.0f), IMPLOT_AUTO, ImVec4(0.0f, 0.0f, 0.8f, 1.0f));
        ImPlot::PlotLine("Predicted", xs.data(), predicted.data(), predicted.size());
        
        ImPlot::EndPlot();
    }
    
    // Display legend/stats
    ImGui::Separator();
    ImGui::Text("Statistics:");
    
    // Calculate RMSE
    double mse = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        mse += std::pow(actual[i] - predicted[i], 2);
    }
    double rmse = std::sqrt(mse / actual.size());
    
    // Calculate Mean Absolute Error (MAE)
    double mae = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        mae += std::abs(actual[i] - predicted[i]);
    }
    mae /= actual.size();
    
    // Calculate Mean Absolute Percentage Error (MAPE)
    double mape = 0.0;
    int validCount = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (std::abs(actual[i]) > 1e-10) {  // Avoid division by zero
            mape += std::abs((actual[i] - predicted[i]) / actual[i]);
            validCount++;
        }
    }
    mape = validCount > 0 ? (mape / validCount) * 100.0 : 0.0;
    
    ImGui::Text("RMSE: %.4f", rmse);
    ImGui::Text("MAE: %.4f", mae);
    ImGui::Text("MAPE: %.2f%%", mape);
}

bool TimeSeries::saveImage(const std::string& filepath) const {
    // In a real implementation, this would save the plot as an image
    // For now, we'll save the data as a CSV file
    try {
        std::filesystem::path path(filepath);
        std::filesystem::create_directories(path.parent_path());
        
        std::ofstream file(filepath);
        if (!file.is_open()) {
            return false;
        }
        
        // Write header
        file << "Date,Actual,Predicted\n";
        
        // Write data
        for (size_t i = 0; i < dates_.size(); ++i) {
            file << dates_[i].toString() << ","
                << actualValues_(i) << ","
                << predictedValues_(i) << "\n";
        }
        
        return true;
    } catch (const std::exception& e) {
        return false;
    }
}

std::vector<std::string> TimeSeries::getDateStrings() const {
    std::vector<std::string> result;
    result.reserve(dates_.size());
    
    // Determine date format based on frequency
    // For now, just use ISO format for all
    for (const auto& date : dates_) {
        result.push_back(date.toString());
    }
    
    // If there are too many dates, reduce the number of labels
    if (result.size() > 20) {
        size_t step = result.size() / 20;
        std::vector<std::string> reduced;
        for (size_t i = 0; i < result.size(); i += step) {
            reduced.push_back(result[i]);
        }
        
        // Add the last date if it's not already included
        if (!reduced.empty() && reduced.back() != result.back()) {
            reduced.push_back(result.back());
        }
        
        // Create empty strings for positions without labels
        std::vector<std::string> final(result.size());
        size_t reducedIdx = 0;
        
        for (size_t i = 0; i < result.size(); ++i) {
            if (i % step == 0 && reducedIdx < reduced.size()) {
                final[i] = reduced[reducedIdx++];
            } else {
                final[i] = "";
            }
        }
        
        return final;
    }
    
    return result;
}

} // namespace DataAnalyzer 