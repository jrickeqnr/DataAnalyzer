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

const std::string& Plot::getTitle() const {
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
    if (dates.empty() || 
        static_cast<Eigen::Index>(dates.size()) != actual.size() || 
        actual.size() != predicted.size()) {
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
    
    // Convert Eigen vectors to std::vector for ImPlot
    std::vector<double> actual(actualValues_.data(), actualValues_.data() + actualValues_.size());
    std::vector<double> predicted(predictedValues_.data(), predictedValues_.data() + predictedValues_.size());
    
    // Convert dates to timestamps for ImPlot's time axis
    std::vector<double> timestamps;
    timestamps.reserve(dates_.size());
    for (const auto& date : dates_) {
        timestamps.push_back(date.toTimestamp());
    }
    
    // Set up plot parameters
    if (ImPlot::BeginPlot(title_.c_str(), ImVec2(-1, -1), ImPlotFlags_NoMouseText)) {
        // Set up axes with time formatting for x-axis
        ImPlot::SetupAxisFormat(ImAxis_X1, "%m/%d/%y");
        ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Time);
        ImPlot::SetupAxis(ImAxis_X1, xLabel_.c_str());
        ImPlot::SetupAxis(ImAxis_Y1, yLabel_.c_str());
        
        // Plot actual values
        ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 4, ImVec4(0.0f, 0.7f, 0.0f, 1.0f), IMPLOT_AUTO, ImVec4(0.0f, 0.7f, 0.0f, 1.0f));
        ImPlot::PlotLine("Actual", timestamps.data(), actual.data(), actual.size());
        
        // Plot predicted values
        ImPlot::SetNextMarkerStyle(ImPlotMarker_Square, 4, ImVec4(0.0f, 0.0f, 0.8f, 1.0f), IMPLOT_AUTO, ImVec4(0.0f, 0.0f, 0.8f, 1.0f));
        ImPlot::PlotLine("Predicted", timestamps.data(), predicted.data(), predicted.size());
        
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

std::string TimeSeries::getType() const {
    return "TimeSeries";
}

// ScatterPlot Implementation
ScatterPlot::ScatterPlot(const std::string& title, const std::string& xLabel, const std::string& yLabel)
    : Plot(title), xLabel_(xLabel), yLabel_(yLabel), hasData_(false) {
}

bool ScatterPlot::setData(const Eigen::VectorXd& actual,
                         const Eigen::VectorXd& predicted) {
    if (actual.size() != predicted.size()) {
        return false;
    }
    
    actualValues_ = actual;
    predictedValues_ = predicted;
    hasData_ = true;
    
    return true;
}

void ScatterPlot::render() {
    if (!hasData_ || actualValues_.size() == 0) {
        ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "No data available to plot");
        return;
    }
    
    // Convert Eigen vectors to std::vector for ImPlot
    std::vector<double> actual(actualValues_.data(), actualValues_.data() + actualValues_.size());
    std::vector<double> predicted(predictedValues_.data(), predictedValues_.data() + predictedValues_.size());
    
    // Set up plot parameters
    if (ImPlot::BeginPlot(title_.c_str(), ImVec2(-1, -1), ImPlotFlags_NoMouseText)) {
        // Set axis labels
        ImPlot::SetupAxes(xLabel_.c_str(), yLabel_.c_str());
        
        // Plot scatter points
        ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 4, ImVec4(0.0f, 0.7f, 0.0f, 1.0f), IMPLOT_AUTO, ImVec4(0.0f, 0.7f, 0.0f, 1.0f));
        ImPlot::PlotScatter("Actual vs Predicted", actual.data(), predicted.data(), actual.size());
        
        // Plot perfect prediction line
        double minVal = std::min(actualValues_.minCoeff(), predictedValues_.minCoeff());
        double maxVal = std::max(actualValues_.maxCoeff(), predictedValues_.maxCoeff());
        std::vector<double> perfectLine = {minVal, maxVal};
        ImPlot::SetNextLineStyle(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), 1.0f);
        ImPlot::PlotLine("Perfect Prediction", perfectLine.data(), perfectLine.data(), perfectLine.size());
        
        ImPlot::EndPlot();
    }
    
    // Display statistics
    ImGui::Separator();
    ImGui::Text("Statistics:");
    
    // Calculate R-squared
    double meanActual = actualValues_.mean();
    double ssTot = 0.0;
    double ssRes = 0.0;
    for (int i = 0; i < actualValues_.size(); ++i) {
        ssTot += std::pow(actualValues_(i) - meanActual, 2);
        ssRes += std::pow(actualValues_(i) - predictedValues_(i), 2);
    }
    double rSquared = 1.0 - (ssRes / ssTot);
    
    // Calculate RMSE
    double rmse = std::sqrt(ssRes / actualValues_.size());
    
    ImGui::Text("R-squared: %.4f", rSquared);
    ImGui::Text("RMSE: %.4f", rmse);
}

std::string ScatterPlot::getType() const {
    return "ScatterPlot";
}

// PlotManager Implementation
void PlotManager::addPlot(std::shared_ptr<Plot> plot) {
    plots_.push_back(plot);
}

void PlotManager::render() {
    if (plots_.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "No plots available");
        return;
    }
    
    // Add plot navigation
    ImGui::BeginGroup();
    if (ImGui::Button("Previous Plot") && currentPlotIndex_ > 0) {
        currentPlotIndex_--;
    }
    ImGui::SameLine();
    ImGui::Text("Plot %zu of %zu", currentPlotIndex_ + 1, plots_.size());
    ImGui::SameLine();
    if (ImGui::Button("Next Plot") && currentPlotIndex_ < plots_.size() - 1) {
        currentPlotIndex_++;
    }
    ImGui::EndGroup();
    
    ImGui::Separator();
    
    // Render current plot
    if (currentPlotIndex_ < plots_.size()) {
        plots_[currentPlotIndex_]->render();
    }
}

size_t PlotManager::getCurrentPlotIndex() const {
    return currentPlotIndex_;
}

void PlotManager::setCurrentPlotIndex(size_t index) {
    if (index < plots_.size()) {
        currentPlotIndex_ = index;
    }
}

size_t PlotManager::getPlotCount() const {
    return plots_.size();
}

std::shared_ptr<Plot> PlotManager::getPlot(size_t index) const {
    if (index < plots_.size()) {
        return plots_[index];
    }
    return nullptr;
}

bool PlotManager::exportPlotData(const std::string& directory) {
    try {
        // Create the directory if it doesn't exist
        std::filesystem::create_directories(directory);
        
        // Export each plot's data to CSV
        for (size_t i = 0; i < plots_.size(); ++i) {
            std::string filename = directory + "/plot_" + plots_[i]->getType() + "_" + std::to_string(i) + ".csv";
            std::ofstream file(filename);
            
            if (!file.is_open()) {
                return false;
            }
            
            // Write plot title and type
            file << "Plot Type: " << plots_[i]->getType() << "\n";
            file << "Plot Title: " << plots_[i]->getTitle() << "\n\n";
            
            // Export data based on plot type
            if (auto* timeSeries = dynamic_cast<TimeSeries*>(plots_[i].get())) {
                file << "Date,Actual Value,Predicted Value\n";
                std::vector<std::string> dateStrings = timeSeries->getDateStrings();
                Eigen::VectorXd actual = timeSeries->getActualValues();
                Eigen::VectorXd predicted = timeSeries->getPredictedValues();
                
                for (Eigen::Index j = 0; j < actual.size(); ++j) {
                    file << dateStrings[j] << "," << actual(j) << "," << predicted(j) << "\n";
                }
            }
            else if (auto* scatterPlot = dynamic_cast<ScatterPlot*>(plots_[i].get())) {
                file << "Actual Value,Predicted Value\n";
                Eigen::VectorXd actual = scatterPlot->getActualValues();
                Eigen::VectorXd predicted = scatterPlot->getPredictedValues();
                
                for (Eigen::Index j = 0; j < actual.size(); ++j) {
                    file << actual(j) << "," << predicted(j) << "\n";
                }
            }
            else if (auto* residualPlot = dynamic_cast<ResidualPlot*>(plots_[i].get())) {
                file << "Predicted Value,Residual\n";
                Eigen::VectorXd predicted = residualPlot->getPredictedValues();
                Eigen::VectorXd residuals = residualPlot->getResidualValues();
                
                for (Eigen::Index j = 0; j < predicted.size(); ++j) {
                    file << predicted(j) << "," << residuals(j) << "\n";
                }
            }
            else if (auto* featurePlot = dynamic_cast<FeatureImportancePlot*>(plots_[i].get())) {
                file << "Feature,Importance\n";
                std::vector<std::string> featureNames = featurePlot->getFeatureNames();
                Eigen::VectorXd importance = featurePlot->getImportanceValues();
                
                for (Eigen::Index j = 0; j < importance.size(); ++j) {
                    file << featureNames[j] << "," << importance(j) << "\n";
                }
            }
            else if (auto* coeffPlot = dynamic_cast<CoefficientStatsPlot*>(plots_[i].get())) {
                file << "Feature,Coefficient,Standard Error,t-value\n";
                std::vector<std::string> featureNames = coeffPlot->getFeatureNames();
                Eigen::VectorXd coefficients = coeffPlot->getCoefficientValues();
                Eigen::VectorXd standardErrors = coeffPlot->getStandardErrors();
                Eigen::VectorXd tValues = coeffPlot->getTValues();
                
                for (Eigen::Index j = 0; j < coefficients.size(); ++j) {
                    file << featureNames[j] << "," << coefficients(j) << "," 
                         << standardErrors(j) << "," << tValues(j) << "\n";
                }
            }
        }
        
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

// ResidualPlot Implementation
ResidualPlot::ResidualPlot(const std::string& title, const std::string& xLabel, const std::string& yLabel)
    : Plot(title), xLabel_(xLabel), yLabel_(yLabel), hasData_(false) {
}

bool ResidualPlot::setData(const Eigen::VectorXd& predicted, const Eigen::VectorXd& residuals) {
    if (predicted.size() != residuals.size() || predicted.size() == 0) {
        return false;
    }
    
    predictedValues_ = predicted;
    residualValues_ = residuals;
    hasData_ = true;
    
    return true;
}

void ResidualPlot::render() {
    if (!hasData_) {
        ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "No data available to plot");
        return;
    }
    
    // Convert Eigen vectors to std::vector for ImPlot
    std::vector<double> predicted(predictedValues_.data(), predictedValues_.data() + predictedValues_.size());
    std::vector<double> residuals(residualValues_.data(), residualValues_.data() + residualValues_.size());
    
    // Set up plot parameters
    if (ImPlot::BeginPlot(title_.c_str(), ImVec2(-1, -1), ImPlotFlags_NoMouseText)) {
        // Set axis labels
        ImPlot::SetupAxes(xLabel_.c_str(), yLabel_.c_str());
        
        // Plot scatter points
        ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 4, ImVec4(0.0f, 0.7f, 0.0f, 1.0f), IMPLOT_AUTO, ImVec4(0.0f, 0.7f, 0.0f, 1.0f));
        ImPlot::PlotScatter("Residuals", predicted.data(), residuals.data(), predicted.size());
        
        // Plot zero line
        double minVal = predictedValues_.minCoeff();
        double maxVal = predictedValues_.maxCoeff();
        std::vector<double> zeroLine = {minVal, maxVal};
        std::vector<double> zeros = {0.0, 0.0};
        ImPlot::SetNextLineStyle(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), 1.0f);
        ImPlot::PlotLine("Zero Line", zeroLine.data(), zeros.data(), zeroLine.size());
        
        ImPlot::EndPlot();
    }
    
    // Display statistics
    ImGui::Separator();
    ImGui::Text("Residual Statistics:");
    
    // Calculate residual statistics
    double meanResidual = residualValues_.mean();
    double stdResidual = std::sqrt((residualValues_.array() - meanResidual).square().sum() / (residualValues_.size() - 1));
    
    ImGui::Text("Mean Residual: %.4f", meanResidual);
    ImGui::Text("Std Dev of Residuals: %.4f", stdResidual);
}

std::string ResidualPlot::getType() const {
    return "ResidualPlot";
}

// CoefficientStatsPlot Implementation
CoefficientStatsPlot::CoefficientStatsPlot(const std::string& title, const std::string& xLabel, const std::string& yLabel)
    : Plot(title), xLabel_(xLabel), yLabel_(yLabel), hasData_(false) {
}

bool CoefficientStatsPlot::setData(const std::vector<std::string>& featureNames,
                                  const Eigen::VectorXd& coefficients,
                                  const Eigen::VectorXd& standardErrors,
                                  const Eigen::VectorXd& tValues) {
    if (featureNames.empty() || 
        static_cast<Eigen::Index>(featureNames.size()) != coefficients.size() ||
        static_cast<Eigen::Index>(featureNames.size()) != standardErrors.size() ||
        static_cast<Eigen::Index>(featureNames.size()) != tValues.size()) {
        return false;
    }
    
    featureNames_ = featureNames;
    coefficientValues_ = coefficients;
    standardErrors_ = standardErrors;
    tValues_ = tValues;
    hasData_ = true;
    
    return true;
}

void CoefficientStatsPlot::render() {
    if (!hasData_ || featureNames_.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "No data available to plot");
        return;
    }
    
    // Convert feature names to char* array for ImPlot
    std::vector<const char*> labels(featureNames_.size());
    for (size_t i = 0; i < featureNames_.size(); ++i) {
        labels[i] = featureNames_[i].c_str();
    }
    
    // Convert Eigen vectors to std::vector for ImPlot
    std::vector<double> coefficients(coefficientValues_.data(), coefficientValues_.data() + coefficientValues_.size());
    std::vector<double> standardErrors(standardErrors_.data(), standardErrors_.data() + standardErrors_.size());
    std::vector<double> tValues(tValues_.data(), tValues_.data() + tValues_.size());
    
    // Create x-axis values (0, 1, 2, ...)
    std::vector<double> xs(coefficients.size());
    std::iota(xs.begin(), xs.end(), 0.0);
    
    // Set up plot parameters for coefficients and standard errors
    if (ImPlot::BeginPlot((title_ + " - Coefficients and Standard Errors").c_str(), ImVec2(-1, -1), ImPlotFlags_NoMouseText)) {
        // Set axis labels
        ImPlot::SetupAxes(xLabel_.c_str(), "Value");
        
        // Set x-axis tick labels to be feature names
        ImPlot::SetupAxisTicks(ImAxis_X1, xs.data(), xs.size(), labels.data());
        
        // Plot coefficients as bars
        ImPlot::SetNextFillStyle(ImVec4(0.0f, 0.7f, 0.0f, 0.5f));
        ImPlot::PlotBars("Coefficients", xs.data(), coefficients.data(), coefficients.size(), 0.4);
        
        // Plot error bars
        for (size_t i = 0; i < coefficients.size(); ++i) {
            double x = xs[i];
            double y = coefficients[i];
            double err = standardErrors[i];
            
            ImPlot::SetNextLineStyle(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), 1.0f);
            std::vector<double> errX = {x, x};
            std::vector<double> errY = {y - err, y + err};
            ImPlot::PlotLine("##ErrorBar", errX.data(), errY.data(), 2);
        }
        
        ImPlot::EndPlot();
    }
    
    // Set up plot parameters for t-values
    if (ImPlot::BeginPlot((title_ + " - T-Values").c_str(), ImVec2(-1, -1), ImPlotFlags_NoMouseText)) {
        // Set axis labels
        ImPlot::SetupAxes(xLabel_.c_str(), "T-Value");
        
        // Set x-axis tick labels to be feature names
        ImPlot::SetupAxisTicks(ImAxis_X1, xs.data(), xs.size(), labels.data());
        
        // Plot t-values as bars
        ImPlot::SetNextFillStyle(ImVec4(0.0f, 0.0f, 0.7f, 0.5f));
        ImPlot::PlotBars("T-Values", xs.data(), tValues.data(), tValues.size(), 0.4);
        
        // Plot significance thresholds (±1.96 for 95% confidence)
        double minX = -0.5;
        double maxX = xs.back() + 0.5;
        std::vector<double> threshX = {minX, maxX};
        std::vector<double> threshY_pos = {1.96, 1.96};
        std::vector<double> threshY_neg = {-1.96, -1.96};
        
        ImPlot::SetNextLineStyle(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), 1.0f);
        ImPlot::PlotLine("95% Confidence", threshX.data(), threshY_pos.data(), 2);
        ImPlot::PlotLine("##NegThreshold", threshX.data(), threshY_neg.data(), 2);
        
        ImPlot::EndPlot();
    }
    
    // Display statistics
    ImGui::Separator();
    ImGui::Text("Coefficient Statistics:");
    
    for (size_t i = 0; i < featureNames_.size(); ++i) {
        ImGui::Text("%s:", featureNames_[i].c_str());
        ImGui::Indent(20.0f);
        ImGui::Text("Coefficient: %.4f", coefficientValues_(i));
        ImGui::Text("Std Error: %.4f", standardErrors_(i));
        ImGui::Text("T-Value: %.4f", tValues_(i));
        ImGui::Text("P-Value: %.4f", 2.0 * (1.0 - std::erf(std::abs(tValues_(i)) / std::sqrt(2.0))));
        ImGui::Unindent(20.0f);
    }
}

std::string CoefficientStatsPlot::getType() const {
    return "CoefficientStatsPlot";
}

} // namespace DataAnalyzer