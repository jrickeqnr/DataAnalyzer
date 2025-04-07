#include "../include/gui.h"
#include <imgui.h>
#include <implot.h>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <filesystem>
#include <thread>
#include <mutex>
#include <sstream>
#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <Eigen/Dense>
#include <GLFW/glfw3.h>
#include "../include/logger.h"
#include "../include/data_handler.h"
#include "../include/model.h"
#include "../include/plotting.h"

namespace fs = std::filesystem;

namespace DataAnalyzer {

void GUI::renderFileBrowser() {
    ImGui::Text("File Browser");
    ImGui::Separator();
    
    ImGui::TextWrapped("Select a CSV file to load data for analysis. The file should contain numeric columns and at least one date column.");
    ImGui::Spacing();
    
    static bool showFileDialog = false;
    
    if (ImGui::Button("Browse Files", ImVec2(120, 0))) {
        showFileDialog = true;
    }
    
    // Modal window for file selection
    if (showFileDialog) {
        ImGui::SetNextWindowSize(ImVec2(600, 400));
        ImGui::SetNextWindowPos(ImVec2(ImGui::GetIO().DisplaySize.x * 0.3f, ImGui::GetIO().DisplaySize.y * 0.2f));
        
        if (ImGui::Begin("Select CSV File", &showFileDialog, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse)) {
            // Set the file dialog window position to be further right
            if (ImGuiFileDialog::fileDialog(selectedFilePath_, "Select CSV File", ".csv")) {
                showFileDialog = false;
                
                // Load the selected file
                if (!selectedFilePath_.empty()) {
                    ImGui::Text("Loading file: %s", selectedFilePath_.c_str());
                    
                    // Load the CSV file
                    if (dataHandler_.loadCSV(selectedFilePath_)) {
                        ImGui::Text("File loaded successfully!");
                        // Reset training progress when new file is loaded
                        trainingProgress_ = 0.0;
                        isTraining_ = false;
                    } else {
                        ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "Error loading file!");
                        selectedFilePath_.clear();
                    }
                }
            }
        }
        ImGui::End();
    }
    
    if (!selectedFilePath_.empty()) {
        ImGui::Text("Selected File: %s", selectedFilePath_.c_str());
        
        // Display data summary in a scrollable region
        ImGui::Spacing();
        ImGui::BeginChild("DataSummary", ImVec2(0, ImGui::GetWindowHeight() * 0.6f), true);
        ImGui::TextWrapped("%s", dataHandler_.getDataSummary().c_str());
        ImGui::EndChild();
        
        // Navigation button aligned to the right
        ImGui::Spacing();
        float windowWidth = ImGui::GetWindowWidth();
        float buttonWidth = 200.0f;
        ImGui::SetCursorPosX(windowWidth - buttonWidth - 20.0f); // 20.0f for padding
        if (ImGui::Button("Next: Outlier Detection", ImVec2(buttonWidth, 0))) {
            setScreen(Screen::OUTLIER_DETECTION);
        }
    }
}

void GUI::renderOutlierDetection() {
    if (selectedFilePath_.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "No file loaded! Please load a CSV file first.");
        if (ImGui::Button("Back to File Browser", ImVec2(200, 0))) {
            setScreen(Screen::FILE_BROWSER);
        }
        return;
    }
    
    ImGui::Text("Outlier Detection");
    ImGui::Separator();
    
    ImGui::TextWrapped("Detect outliers in the data using the IQR method. Outliers are defined as values outside Q1 - k*IQR and Q3 + k*IQR, where k is the sensitivity value.");
    ImGui::Spacing();
    
    // Add sensitivity slider
    static bool sensitivityChanged = false;
    ImGui::TextWrapped("Adjust sensitivity (lower values detect more outliers, higher values detect fewer):");
    if (ImGui::SliderFloat("Sensitivity", &outlierSensitivity_, 0.5f, 3.0f, "%.1f")) {
        sensitivityChanged = true;
    }
    
    static bool detectPressed = false;
    static bool showOutlierPopup = false;
    static std::string popupMessage;
    static int totalOutliers = 0;
    
    // Reset detection if sensitivity changed
    if (sensitivityChanged && detectPressed) {
        detectPressed = false;
        sensitivityChanged = false;
        selectedOutliers_.clear();
    }
    
    if (!detectPressed) {
        if (ImGui::Button("Detect Outliers", ImVec2(150, 0))) {
            // Detect outliers in all numeric columns with specified sensitivity
            outliers_ = dataHandler_.detectOutliers({}, static_cast<double>(outlierSensitivity_));
            detectPressed = true;
            sensitivityChanged = false;
            
            // Initialize selectedOutliers_ with all outliers (all checked by default)
            selectedOutliers_ = outliers_;
            
            // Calculate total outliers and prepare popup
            totalOutliers = 0;
            for (const auto& [col, rows] : outliers_) {
                totalOutliers += rows.size();
            }
            
            if (totalOutliers > 0) {
                popupMessage = "Found " + std::to_string(totalOutliers) + " outliers in the data.\nCheck the details below for more information.";
            } else {
                popupMessage = "No outliers were detected in the data.";
            }
            showOutlierPopup = true;
        }
    }
    
    // Render popup outside of the button click handler
    if (showOutlierPopup) {
        ImGui::OpenPopup("Outlier Detection Results");
        showOutlierPopup = false;
    }
    
    // Always render the popup window
    if (ImGui::BeginPopupModal("Outlier Detection Results", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::Text("%s", popupMessage.c_str());
        ImGui::Separator();
        if (ImGui::Button("OK", ImVec2(120, 0))) {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
    
    // Show results if detection has been performed
    if (!outliers_.empty()) {
        ImGui::Text("Outliers detected in the following columns:");
        ImGui::Spacing();
        
        // Add select all/none buttons
        if (ImGui::Button("Select All")) {
            selectedOutliers_ = outliers_;
        }
        ImGui::SameLine();
        if (ImGui::Button("Select None")) {
            selectedOutliers_.clear();
        }
        ImGui::Spacing();
        
        const std::vector<std::string>& columnNames = dataHandler_.getColumnNames();
        const std::vector<Date>& dates = dataHandler_.getDates();
        static bool outlierFixed = false;
        static Eigen::MatrixXd originalData;
        
        // Store original data before fixing
        if (!outlierFixed) {
            originalData = dataHandler_.getData();
        }
        
        for (const auto& [col, rows] : outliers_) {
            if (col < columnNames.size()) {
                ImGui::Text("Column '%s': %zu outliers", columnNames[col].c_str(), rows.size());
                
                if (ImGui::TreeNode(("Details##" + std::to_string(col)).c_str())) {
                    ImGui::Indent(20.0f);
                    ImGui::Text("Outliers:");
                    
                    // Create a table for outlier details
                    if (ImGui::BeginTable(("OutliersTable##" + std::to_string(col)).c_str(), outlierFixed ? 4 : 3, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
                        ImGui::TableSetupColumn("Selected", ImGuiTableColumnFlags_WidthFixed, 70.0f);
                        ImGui::TableSetupColumn("Date");
                        ImGui::TableSetupColumn(outlierFixed ? "Original Value" : "Value");
                        if (outlierFixed) {
                            ImGui::TableSetupColumn("Fixed Value");
                        }
                        ImGui::TableHeadersRow();
                        
                        for (size_t row : rows) {
                            ImGui::TableNextRow();
                            
                            // Checkbox column - only enabled if not fixed yet
                            ImGui::TableSetColumnIndex(0);
                            if (!outlierFixed) {
                                // Create a unique ID for the checkbox
                                std::string checkboxId = "##select_" + std::to_string(col) + "_" + std::to_string(row);
                                
                                // Check if this outlier is selected
                                bool isSelected = false;
                                auto it = selectedOutliers_.find(col);
                                if (it != selectedOutliers_.end()) {
                                    isSelected = std::find(it->second.begin(), it->second.end(), row) != it->second.end();
                                }
                                
                                // Render checkbox
                                if (ImGui::Checkbox(checkboxId.c_str(), &isSelected)) {
                                    // Update selectedOutliers_ based on checkbox state
                                    if (isSelected) {
                                        // Add to selected outliers if not already there
                                        if (selectedOutliers_.find(col) == selectedOutliers_.end()) {
                                            selectedOutliers_[col] = std::vector<size_t>{row};
                                        } else if (std::find(selectedOutliers_[col].begin(), selectedOutliers_[col].end(), row) == selectedOutliers_[col].end()) {
                                            selectedOutliers_[col].push_back(row);
                                        }
                                    } else {
                                        // Remove from selected outliers
                                        auto it = selectedOutliers_.find(col);
                                        if (it != selectedOutliers_.end()) {
                                            auto& rowVec = it->second;
                                            rowVec.erase(std::remove(rowVec.begin(), rowVec.end(), row), rowVec.end());
                                            
                                            // Remove the column entry if no rows left
                                            if (rowVec.empty()) {
                                                selectedOutliers_.erase(col);
                                            }
                                        }
                                    }
                                }
                            } else {
                                // Show whether this outlier was fixed
                                auto it = selectedOutliers_.find(col);
                                bool wasFixed = it != selectedOutliers_.end() && 
                                    std::find(it->second.begin(), it->second.end(), row) != it->second.end();
                                
                                if (wasFixed) {
                                    ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "Fixed");
                                } else {
                                    ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Skipped");
                                }
                            }
                            
                            // Date column
                            ImGui::TableSetColumnIndex(1);
                            if (row < dates.size()) {
                                ImGui::Text("%s", dates[row].toString().c_str());
                            } else {
                                ImGui::Text("Unknown Date");
                            }
                            
                            // Value column
                            ImGui::TableSetColumnIndex(2);
                            if (row < static_cast<size_t>(originalData.rows()) && col < static_cast<size_t>(originalData.cols())) {
                                ImGui::Text("%.2f", originalData(row, col));
                            } else {
                                ImGui::Text("Invalid Value");
                            }
                            
                            // Fixed value column (only shown after fixing)
                            if (outlierFixed) {
                                ImGui::TableSetColumnIndex(3);
                                
                                // Check if this outlier was selected for fixing
                                auto it = selectedOutliers_.find(col);
                                bool wasFixed = it != selectedOutliers_.end() && 
                                    std::find(it->second.begin(), it->second.end(), row) != it->second.end();
                                
                                if (wasFixed && row < static_cast<size_t>(dataHandler_.getData().rows()) && 
                                    col < static_cast<size_t>(dataHandler_.getData().cols())) {
                                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 1.0f, 0.0f, 1.0f));
                                    ImGui::Text("%.2f", dataHandler_.getData()(row, col));
                                    ImGui::PopStyleColor();
                                } else if (row < static_cast<size_t>(dataHandler_.getData().rows()) && 
                                          col < static_cast<size_t>(dataHandler_.getData().cols())) {
                                    // Show original value for non-fixed outliers
                                    ImGui::Text("%.2f", originalData(row, col));
                                } else {
                                    ImGui::Text("Invalid Value");
                                }
                            }
                        }
                        ImGui::EndTable();
                    }
                    
                    ImGui::Unindent(20.0f);
                    ImGui::TreePop();
                }
            }
        }
        
        ImGui::Spacing();
        if (!outlierFixed) {
            if (ImGui::Button("Fix Selected Outliers", ImVec2(200, 0))) {
                if (selectedOutliers_.empty()) {
                    showErrorPopup("No outliers selected for fixing!");
                } else {
                    if (dataHandler_.fixOutliers(outliers_, selectedOutliers_)) {
                        showSuccessPopup("Selected outliers fixed successfully!");
                        outlierFixed = true;
                    } else {
                        showErrorPopup("Failed to fix outliers!");
                    }
                }
            }
            
            ImGui::SameLine();
            if (ImGui::Button("Reset Selection", ImVec2(150, 0))) {
                // Reset to all outliers selected
                selectedOutliers_ = outliers_;
            }
        } else {
            // Add a button to start over
            if (ImGui::Button("Detect Outliers Again", ImVec2(200, 0))) {
                outlierFixed = false;
                detectPressed = false;
                outliers_.clear();
                selectedOutliers_.clear();
            }
        }
    }
    
    // Navigation buttons
    ImGui::Spacing();
    if (ImGui::Button("Back: File Browser", ImVec2(200, 0))) {
        detectPressed = false; // Reset detection state when navigating away
        outliers_.clear();
        selectedOutliers_.clear();
        setScreen(Screen::FILE_BROWSER);
    }
    
    ImGui::SameLine();
    if (ImGui::Button("Next: Model Selection", ImVec2(200, 0))) {
        detectPressed = false; // Reset detection state when navigating away
        setScreen(Screen::MODEL_SELECTION);
    }
}

void GUI::renderModelSelection() {
    if (selectedFilePath_.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "No file loaded! Please load a CSV file first.");
        if (ImGui::Button("Back to File Browser", ImVec2(200, 0))) {
            setScreen(Screen::FILE_BROWSER);
        }
        return;
    }
    
    ImGui::Text("Model Selection");
    ImGui::Separator();
    
    ImGui::TextWrapped("Select a model for data analysis:");
    ImGui::Spacing();
    
    // Model selection with all available models
    const char* models[] = {"Linear Regression", "Elastic Net", "XGBoost", "Gradient Boosting", "Neural Network"};
    static int selectedModel = 0;
    
    ImGui::Combo("Model", &selectedModel, models, IM_ARRAYSIZE(models));
    selectedModelIndex_ = selectedModel;
    
    // Display model description based on selection
    ImGui::Spacing();
    switch (selectedModel) {
        case 0: // Linear Regression
            ImGui::TextWrapped("Linear Regression is a standard approach for modeling the relationship between a dependent variable and one or more independent variables. It uses ordinary least squares to find the line that minimizes the sum of squared errors between predictions and actual values.");
            break;
        case 1: // Elastic Net
            ImGui::TextWrapped("Elastic Net Regression combines L1 and L2 regularization to balance sparsity and stability. It's effective for feature selection and handling correlated predictors. The α (alpha) parameter controls the mix of L1 and L2 regularization, while λ (lambda) controls the overall regularization strength.");
            break;
        case 2: // XGBoost
            ImGui::TextWrapped("XGBoost (Extreme Gradient Boosting) is an optimized gradient boosting library designed for efficient and scalable training. It builds an ensemble of decision trees sequentially, with each new tree correcting errors made by the previous ones. XGBoost offers high prediction accuracy and speed, with regularization to prevent overfitting.");
            break;
        case 3: // Gradient Boosting
            ImGui::TextWrapped("Gradient Boosting is an ensemble learning technique that builds a series of weak models (typically decision trees) sequentially, where each model tries to correct the errors of the previous ones. It combines these weak models to create a strong predictive model that generally outperforms individual algorithms.");
            break;
        case 4: // Neural Network
            ImGui::TextWrapped("Neural Network is a multi-layer perceptron regressor that uses backpropagation for training. It consists of an input layer, one or more hidden layers, and an output layer, with neurons connected by weights that are adjusted during training to minimize prediction error. Neural networks can capture complex non-linear relationships in the data.");
            break;
    }
    
    // Display model diagram (simplified)
    ImGui::Spacing();
    if (ImGui::CollapsingHeader("Model Diagram", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Indent(20.0f);
        switch (selectedModel) {
            case 0: // Linear Regression
                ImGui::TextWrapped("Input Features -> [Linear Regression] -> Prediction");
                ImGui::TextWrapped("Model: y = b0 + b1*x1 + b2*x2 + ... + bn*xn");
                break;
            case 1: // Elastic Net
                ImGui::TextWrapped("Input Features -> [Elastic Net] -> Prediction");
                ImGui::TextWrapped("Loss = MSE + alpha*lambda*L1 + (1-alpha)*lambda*L2");
                break;
            case 2: // XGBoost
                ImGui::TextWrapped("Input Features -> [Tree 1] -> [Tree 2] -> ... -> [Tree n] -> Sum -> Prediction");
                ImGui::TextWrapped("Each tree corrects errors from previous trees");
                break;
            case 3: // Gradient Boosting
                ImGui::TextWrapped("Input Features -> [Tree 1] -> [Tree 2] -> ... -> [Tree n] -> Sum -> Prediction");
                ImGui::TextWrapped("Each tree fits to the residuals of the previous trees");
                break;
            case 4: // Neural Network
                ImGui::TextWrapped("Input Features -> [Hidden Layer(s)] -> [Output Layer] -> Prediction");
                ImGui::TextWrapped("Neurons connected by weights, trained with backpropagation");
                break;
        }
        ImGui::Unindent(20.0f);
    }
    
    // Navigation buttons
    ImGui::Spacing();
    if (ImGui::Button("Back: Outlier Detection", ImVec2(200, 0))) {
        setScreen(Screen::OUTLIER_DETECTION);
    }
    
    ImGui::SameLine();
    if (ImGui::Button("Next: Variable Selection", ImVec2(200, 0))) {
        setScreen(Screen::VARIABLE_SELECTION);
    }
}

void GUI::renderVariableSelection() {
    if (selectedFilePath_.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "No file loaded! Please load a CSV file first.");
        if (ImGui::Button("Back to File Browser", ImVec2(200, 0))) {
            setScreen(Screen::FILE_BROWSER);
        }
        return;
    }
    
    ImGui::Text("Variable Selection");
    ImGui::Separator();
    
    ImGui::TextWrapped("Select input features (independent variables) and target variable (dependent variable) for the model.");
    ImGui::Spacing();
    
    // Get column names and indices
    const std::vector<std::string>& columnNames = dataHandler_.getColumnNames();
    const std::vector<size_t>& numericIndices = dataHandler_.getNumericColumnIndices();
    const std::vector<size_t>& dateIndices = dataHandler_.getDateColumnIndices();
    
    // Filter out date columns from numeric indices to get only valid model variables
    std::vector<size_t> modelVariables;
    for (size_t idx : numericIndices) {
        // Only include if not a date column
        if (std::find(dateIndices.begin(), dateIndices.end(), idx) == dateIndices.end()) {
            modelVariables.push_back(idx);
        }
    }
    
    // Initialize selected features if empty
    if (selectedFeatures_.empty() && !modelVariables.empty()) {
        // Default: select all valid model variables except the last one
        selectedFeatures_ = std::vector<size_t>(modelVariables.begin(), modelVariables.end() - 1);
        
        // Default target: last valid model variable
        if (!modelVariables.empty()) {
            selectedTargetIndices_.push_back(modelVariables.back());
        }

        // Update DataHandler with initial selections
        dataHandler_.setSelectedFeatures(selectedFeatures_);
        dataHandler_.setSelectedTargets(selectedTargetIndices_);
    }

    // Search functionality
    static char searchBuffer[128] = "";
    ImGui::Text("Search Variables:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(300);
    if (ImGui::InputText("##search", searchBuffer, IM_ARRAYSIZE(searchBuffer))) {
        // Search is handled in real-time as user types
    }
    ImGui::Spacing();

    // Create two columns for input and target selection
    const float padding = 10.0f;
    const float availWidth = ImGui::GetContentRegionAvail().x;
    const float columnWidth = (availWidth - 3 * padding) / 2;

    ImGui::Columns(2, "VariableSelectionColumns", false);
    ImGui::SetColumnWidth(0, columnWidth);
    ImGui::SetColumnWidth(1, columnWidth);

    // Input Features Selection (Left Column)
    ImGui::BeginChild("InputFeaturesFrame", ImVec2(columnWidth, ImGui::GetWindowHeight() * 0.6f), true);
    ImGui::Text("Input Features");
    ImGui::Separator();
    ImGui::TextWrapped("Select variables to use as input features for the model.");
    ImGui::Spacing();

    // Control buttons for input features
    if (ImGui::Button("Select All", ImVec2(-1, 0))) {
        selectedFeatures_.clear();
        selectedFeatures_ = modelVariables;
        // Remove any target variables from the selection
        for (size_t targetIdx : selectedTargetIndices_) {
            selectedFeatures_.erase(
                std::remove(selectedFeatures_.begin(), selectedFeatures_.end(), targetIdx),
                selectedFeatures_.end()
            );
        }
    }
    ImGui::Separator();

    // Scrollable area for input selection with search
    ImGui::BeginChild("InputFeatures", ImVec2(0, -ImGui::GetFrameHeightWithSpacing() * 4), true);
    for (size_t idx : modelVariables) {
        // Apply search filter
        std::string colName = columnNames[idx];
        if (!searchBuffer[0] || colName.find(searchBuffer) != std::string::npos) {
            bool isSelected = std::find(selectedFeatures_.begin(), selectedFeatures_.end(), idx) != selectedFeatures_.end();
            if (ImGui::Checkbox(colName.c_str(), &isSelected)) {
                if (isSelected) {
                    selectedFeatures_.push_back(idx);
                } else {
                    selectedFeatures_.erase(
                        std::remove(selectedFeatures_.begin(), selectedFeatures_.end(), idx),
                        selectedFeatures_.end()
                    );
                }
                dataHandler_.setSelectedFeatures(selectedFeatures_);
            }
            
            // Add tooltip for each feature
            if (ImGui::IsItemHovered()) {
                ImGui::BeginTooltip();
                ImGui::Text("Column: %s", colName.c_str());
                ImGui::Text("Type: Numeric");
                ImGui::EndTooltip();
            }
        }
    }
    ImGui::EndChild();

    // Warning about overlapping selections
    std::vector<size_t> overlapping;
    for (size_t idx : selectedFeatures_) {
        if (std::find(selectedTargetIndices_.begin(), selectedTargetIndices_.end(), idx) != selectedTargetIndices_.end()) {
            overlapping.push_back(idx);
        }
    }
    if (!overlapping.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.0f, 1.0f), "Warning: Some variables are selected as both input and target:");
        std::string overlap_vars;
        for (size_t i = 0; i < overlapping.size(); ++i) {
            overlap_vars += columnNames[overlapping[i]];
            if (i < overlapping.size() - 1) overlap_vars += ", ";
        }
        ImGui::TextWrapped("%s", overlap_vars.c_str());
    }
    ImGui::EndChild();

    ImGui::NextColumn();

    // Target Variable Selection (Right Column) - REDUCED HEIGHT
    ImGui::BeginChild("TargetVariableFrame", ImVec2(columnWidth, ImGui::GetWindowHeight() * 0.4f), true);
    ImGui::Text("Target Variables");
    ImGui::Separator();
    ImGui::TextWrapped("Select variable(s) to predict. For best results, select only one target variable.");
    ImGui::Spacing();

    // Scrollable area for target selection with search
    ImGui::BeginChild("TargetVariables", ImVec2(0, -ImGui::GetFrameHeightWithSpacing()), true);
    for (size_t idx : modelVariables) {
        // Apply search filter
        std::string colName = columnNames[idx];
        if (!searchBuffer[0] || colName.find(searchBuffer) != std::string::npos) {
            bool isTarget = std::find(selectedTargetIndices_.begin(), selectedTargetIndices_.end(), idx) != selectedTargetIndices_.end();
            if (ImGui::Checkbox(colName.c_str(), &isTarget)) {
                if (isTarget) {
                    selectedTargetIndices_.push_back(idx);
                } else {
                    selectedTargetIndices_.erase(
                        std::remove(selectedTargetIndices_.begin(), selectedTargetIndices_.end(), idx),
                        selectedTargetIndices_.end()
                    );
                }
                dataHandler_.setSelectedTargets(selectedTargetIndices_);
            }

            // Add tooltip for each potential target
            if (ImGui::IsItemHovered()) {
                ImGui::BeginTooltip();
                ImGui::Text("Column: %s", colName.c_str());
                ImGui::Text("Type: Numeric");
                ImGui::EndTooltip();
            }
        }
    }
    ImGui::EndChild();

    // Warning about multiple targets or no targets
    if (selectedTargetIndices_.size() > 1) {
        ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.0f, 1.0f), "Warning: Multiple targets selected.");
        ImGui::TextWrapped("This may affect model performance. Consider selecting only one target for better results.");
    } else if (selectedTargetIndices_.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.0f, 1.0f), "Warning: No target variable selected.");
        ImGui::TextWrapped("You must select at least one target variable to proceed.");
    }
    ImGui::EndChild();
    
    // Time Series Options (Right Column - Below Target Selection)
    ImGui::BeginChild("TimeSeriesOptionsFrame", ImVec2(columnWidth, ImGui::GetWindowHeight() * 0.18f), true);
    ImGui::Text("Time Series Options");
    ImGui::Separator();
    
    // Lag Values slider
    ImGui::TextWrapped("Lag Values: Apply previous time periods as features");
    ImGui::SetNextItemWidth(columnWidth - 20.0f);
    if (ImGui::SliderInt("##LagValues", &lagValues_, 0, 12, "%d lags")) {
        // Update the data handler with the new lag values
        dataHandler_.setLagValues(lagValues_);
    }
    if (ImGui::IsItemHovered()) {
        ImGui::BeginTooltip();
        ImGui::TextWrapped("Number of previous time periods to use as features. For example, if set to 3, the values from t-1, t-2, and t-3 will be included as features for predicting the value at time t.");
        ImGui::EndTooltip();
    }
    
    // Seasonality slider
    ImGui::TextWrapped("Seasonality: Apply seasonal lags to target variables");
    ImGui::SetNextItemWidth(columnWidth - 20.0f);
    if (ImGui::SliderInt("##Seasonality", &seasonality_, 0, 12, "%d periods")) {
        // Update the data handler with the new seasonality value
        dataHandler_.setSeasonality(seasonality_);
    }
    if (ImGui::IsItemHovered()) {
        ImGui::BeginTooltip();
        ImGui::TextWrapped("Number of seasonal periods to use. For example, if set to 4 for quarterly data, the value from the same quarter in previous years will be included as features. The system will automatically find the best seasonal pattern within this maximum range.");
        ImGui::EndTooltip();
    }
    
    ImGui::EndChild();

    ImGui::Columns(1);
    
    // Selection Summary
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Text("Selection Summary:");
    
    if (!selectedFeatures_.empty()) {
        ImGui::Text("Input Features (%zu):", selectedFeatures_.size());
        ImGui::SameLine();
        std::string features;
        for (size_t i = 0; i < selectedFeatures_.size(); ++i) {
            features += columnNames[selectedFeatures_[i]];
            if (i < selectedFeatures_.size() - 1) features += ", ";
        }
        ImGui::TextWrapped("%s", features.c_str());
    } else {
        ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "No input features selected!");
    }

    ImGui::Text("Target Variables (%zu):", selectedTargetIndices_.size());
    ImGui::SameLine();
    if (!selectedTargetIndices_.empty()) {
        std::string targets;
        for (size_t i = 0; i < selectedTargetIndices_.size(); ++i) {
            targets += columnNames[selectedTargetIndices_[i]];
            if (i < selectedTargetIndices_.size() - 1) targets += ", ";
        }
        ImGui::TextWrapped("%s", targets.c_str());
    } else {
        ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "No target variables selected!");
    }
    
    // Navigation buttons
    ImGui::Spacing();
    if (ImGui::Button("Back: Model Selection", ImVec2(200, 0))) {
        setScreen(Screen::MODEL_SELECTION);
    }
    
    ImGui::SameLine();
    
    // Validate selection
    bool canProceed = !selectedFeatures_.empty() && !selectedTargetIndices_.empty();
    
    if (canProceed) {
        if (ImGui::Button("Next: Hyperparameters", ImVec2(200, 0))) {
            setScreen(Screen::HYPERPARAMETERS);
        }
    } else {
        ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
        ImGui::Button("Next: Hyperparameters", ImVec2(200, 0));
        ImGui::PopStyleVar();
        
        if (selectedFeatures_.empty()) {
            ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "Please select at least one input feature!");
        }
        if (selectedTargetIndices_.empty()) {
            ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "Please select at least one target variable!");
        }
    }
}

void GUI::renderHyperparameters() {
    if (selectedFilePath_.empty() || selectedFeatures_.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "No variables selected! Please select features first.");
        if (ImGui::Button("Back to Variable Selection", ImVec2(200, 0))) {
            setScreen(Screen::VARIABLE_SELECTION);
        }
        return;
    }
    
    ImGui::Text("Hyperparameter Configuration");
    ImGui::Separator();
    
    // Show different hyperparameter options based on selected model
    const char* modelTypes[] = {"Linear Regression", "Elastic Net", "XGBoost", "Gradient Boosting", "Neural Network"};
    ImGui::TextWrapped("Configure the hyperparameters for the %s model or use automatic tuning.", modelTypes[selectedModelIndex_]);
    ImGui::Spacing();
    
    // Hyperparameter configuration
    ImGui::Checkbox("Auto-tune Hyperparameters", &autoHyperparameters_);
    
    // Status is now handled in the status box below

    // Show best parameters if available during auto-tuning
    if (autoHyperparameters_ && model_) {
        auto stats = model_->getStats();
        if (stats.find("Best RMSE") != stats.end()) {
            ImGui::Separator();
            ImGui::Text("Best Parameters Found:");
            ImGui::Indent(20.0f);
            ImGui::Text("RMSE: %.4f", stats["Best RMSE"]);
            
            if (stats.find("Best Learning Rate") != stats.end()) {
                ImGui::Text("Learning Rate: %.4f", stats["Best Learning Rate"]);
            }
            if (stats.find("Best N Estimators") != stats.end()) {
                ImGui::Text("N Estimators: %d", static_cast<int>(stats["Best N Estimators"]));
            }
            if (stats.find("Best Max Depth") != stats.end()) {
                ImGui::Text("Max Depth: %d", static_cast<int>(stats["Best Max Depth"]));
            }
            if (stats.find("Best Alpha") != stats.end()) {
                ImGui::Text("Alpha: %.4f", stats["Best Alpha"]);
            }
            if (stats.find("Best Lambda") != stats.end()) {
                ImGui::Text("Lambda: %.4f", stats["Best Lambda"]);
            }
            if (stats.find("Best Min Samples Split") != stats.end()) {
                ImGui::Text("Min Samples Split: %d", static_cast<int>(stats["Best Min Samples Split"]));
            }
            ImGui::Unindent(20.0f);
        }
    }
    // Show current metrics if not auto-tuning
    else if (model_) {
        auto stats = model_->getStats();
        if (stats.find("Current RMSE") != stats.end()) {
            ImGui::Text("Current RMSE: %.4f", stats["Current RMSE"]);
        }
        if (stats.find("Current R²") != stats.end()) {
            ImGui::Text("Current R²: %.4f", stats["Current R²"]);
        }
    }
    
    // Different hyperparameters based on model type
    if (!autoHyperparameters_) {
        switch (selectedModelIndex_) {
            case 0: // Linear Regression
                // Linear Regression doesn't have hyperparameters
                ImGui::TextWrapped("Linear Regression does not have tunable hyperparameters.");
                break;
                
            case 1: // Elastic Net
                {
                    float alpha = static_cast<float>(alpha_);
                    float lambda = static_cast<float>(lambda_);
                    
                    if (ImGui::SliderFloat("Alpha (L1/L2 Mix)", &alpha, 0.0f, 1.0f, "%.2f")) {
                        alpha_ = static_cast<double>(alpha);
                    }
                    if (ImGui::SliderFloat("Lambda (Regularization)", &lambda, 0.0f, 10.0f, "%.2f")) {
                        lambda_ = static_cast<double>(lambda);
                    }
                }
                break;
                
            case 2: // XGBoost
                {
                    int n_estimators = static_cast<int>(n_estimators_);
                    float learning_rate = static_cast<float>(learning_rate_);
                    int max_depth = static_cast<int>(max_depth_);
                    float subsample = static_cast<float>(subsample_);
                    
                    if (ImGui::SliderInt("Number of Estimators", &n_estimators, 10, 500)) {
                        n_estimators_ = n_estimators;
                    }
                    if (ImGui::SliderFloat("Learning Rate", &learning_rate, 0.01f, 1.0f, "%.2f")) {
                        learning_rate_ = static_cast<double>(learning_rate);
                    }
                    if (ImGui::SliderInt("Max Depth", &max_depth, 1, 10)) {
                        max_depth_ = max_depth;
                    }
                    if (ImGui::SliderFloat("Subsample", &subsample, 0.1f, 1.0f, "%.2f")) {
                        subsample_ = static_cast<double>(subsample);
                    }
                }
                break;
                
            case 3: // Gradient Boosting
                {
                    int n_estimators = static_cast<int>(n_estimators_);
                    float learning_rate = static_cast<float>(learning_rate_);
                    int max_depth = static_cast<int>(max_depth_);
                    int min_samples_split = static_cast<int>(min_samples_split_);
                    
                    if (ImGui::SliderInt("Number of Estimators", &n_estimators, 5, 500)) {
                        n_estimators_ = n_estimators;
                    }
                    if (ImGui::SliderFloat("Learning Rate", &learning_rate, 0.1, 0.8, "%.2f")) {
                        learning_rate_ = static_cast<double>(learning_rate);
                    }
                    if (ImGui::SliderInt("Max Depth", &max_depth, 2, 5)) {
                        max_depth_ = max_depth;
                    }
                    if (ImGui::SliderInt("Min Samples Split", &min_samples_split, 2, 5)) {
                        min_samples_split_ = static_cast<double>(min_samples_split);
                    }
                }
                break;
                
            case 4: // Neural Network
                {
                    int hidden_layers = static_cast<int>(hidden_layers_);
                    int neurons_per_layer = static_cast<int>(neurons_per_layer_);
                    float learning_rate = static_cast<float>(learning_rate_);
                    int max_iterations = static_cast<int>(max_iterations_);
                    
                    if (ImGui::SliderInt("Hidden Layers", &hidden_layers, 1, 5)) {
                        hidden_layers_ = hidden_layers;
                    }
                    if (ImGui::SliderInt("Neurons per Layer", &neurons_per_layer, 5, 100)) {
                        neurons_per_layer_ = neurons_per_layer;
                    }
                    if (ImGui::SliderFloat("Learning Rate", &learning_rate, 0.001f, 0.5f, "%.3f")) {
                        learning_rate_ = static_cast<double>(learning_rate);
                    }
                    if (ImGui::SliderInt("Max Iterations", &max_iterations, 100, 10000)) {
                        max_iterations_ = max_iterations;
                    }
                }
                break;
        }
    }
    
    // Add the new status display
    ImGui::BeginGroup();
    
    // Create a light blue background for the status message
    ImVec2 statusSize = ImVec2(ImGui::GetContentRegionAvail().x, 0);
    if (isTraining_) {
        ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.1f, 0.2f, 0.3f, 0.3f));  // Light blue background
        statusSize.y = autoHyperparameters_ ? 60.0f : 40.0f;  // Taller for auto-tuning
    } else if (trainingProgress_ >= 1.0) {
        ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.1f, 0.3f, 0.1f, 0.3f));  // Light green background
        statusSize.y = 40.0f;
    } else {
        ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.2f, 0.2f, 0.2f, 0.3f));  // Gray background
        statusSize.y = 40.0f;
    }
    
    ImGui::BeginChild("StatusBox", statusSize, true);
    
    if (isTraining_) {
        if (autoHyperparameters_) {
            auto stats = model_->getStats();
            
            // Main status message
            ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[0]);  // Use default font but we could use a larger one if available
            ImGui::TextColored(ImVec4(1.0f, 1.0f, 1.0f, 1.0f), "Model Training in Progress");
            ImGui::PopFont();
            
            ImGui::Spacing();
            
            // Detailed status based on current phase
            if (stats.find("Grid Search Progress") != stats.end() && trainingProgress_ <= 0.5) {
                // Phase 1 details
                ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), 
                    "Finding optimal hyperparameters by evaluating multiple configurations...");
                
                // If we have best parameters so far, show them
                if (stats.find("Best RMSE") != stats.end()) {
                    ImGui::Text("Current best RMSE: %.4f", stats["Best RMSE"]);
                }
            } else {
                // Phase 2 details
                ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), 
                    "Training final model with optimal parameters...");
            }
        } else {
            // Simple training message for non-auto-tuning
            ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[0]);
            ImGui::TextColored(ImVec4(1.0f, 1.0f, 1.0f, 1.0f), "Training Model...");
            ImGui::PopFont();
            
            ImGui::Spacing();
            ImGui::Text("This may take a moment depending on dataset size and model complexity.");
        }
    } else if (trainingProgress_ >= 1.0) {
        ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[0]);
        ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "Training Completed Successfully!");
        ImGui::PopFont();
        
        ImGui::Spacing();
        ImGui::Text("You can now view the model results below or proceed to the plotting screen.");
    } else {
        ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[0]);
        ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.8f, 1.0f), "Ready to Train Model");
        ImGui::PopFont();
        
        ImGui::Spacing();
        ImGui::Text("Configure settings above and click 'Train Model' to begin.");
    }
    
    ImGui::EndChild();
    ImGui::PopStyleColor();
    ImGui::EndGroup();
    
    ImGui::Spacing();
    
    // Show training progress section (always visible)
    ImGui::Spacing();
    ImGui::Separator();
    
    // Progress bar with phase indication
    if (isTraining_ && autoHyperparameters_) {
        // Auto-tuning progress visualization
        ImGui::BeginGroup();
        
        // Progress label
        ImGui::Spacing();
        ImGui::Text("Training Progress:");
        
        // Create a cleaner two-phase progress indicator
        const float barHeight = 24.0f;
        const float fullWidth = ImGui::GetContentRegionAvail().x;
        const float halfWidth = fullWidth / 2.0f;
        ImVec2 cursorPos = ImGui::GetCursorPos();
        
        // Phase labels - Before drawing bars
        ImGui::SetCursorPos(ImVec2(cursorPos.x + halfWidth * 0.25f, cursorPos.y));
        ImGui::Text("Phase 1: Finding Best Parameters");
        ImGui::SetCursorPos(ImVec2(cursorPos.x + halfWidth + halfWidth * 0.25f, cursorPos.y));
        ImGui::Text("Phase 2: Training Final Model");
        
        // Reset cursor position for drawing bars
        ImGui::SetCursorPos(ImVec2(cursorPos.x, cursorPos.y + ImGui::GetTextLineHeightWithSpacing()));
        
        // Phase 1 bar (left half)
        ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.1f, 0.1f, 0.2f, 1.0f));  // Darker blue background
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ImVec4(0.0f, 0.6f, 1.0f, 1.0f));  // Brighter blue for progress
        
        // If we're in phase 1, show percentage in the bar, otherwise show "Completed"
        char phase1Text[32];
        if (trainingProgress_ <= 0.5f) {
            snprintf(phase1Text, sizeof(phase1Text), "%.0f%%", (trainingProgress_ / 0.5f) * 100.0f);
        } else {
            strcpy(phase1Text, "Completed");
        }
        
        ImGui::PushID("Phase1Bar");
        ImGui::ProgressBar(std::min<float>(trainingProgress_ / 0.5f, 1.0f), ImVec2(halfWidth - 5.0f, barHeight), phase1Text);
        ImGui::PopID();
        ImGui::PopStyleColor(2);
        
        ImGui::SameLine(0, 10);  // Add spacing between bars
        
        // Phase 2 bar (right half)
        ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.1f, 0.2f, 0.1f, 1.0f));  // Darker green background
        
        // Change color based on whether phase 2 has started
        if (trainingProgress_ > 0.5f) {
            ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ImVec4(0.0f, 0.8f, 0.0f, 1.0f));  // Bright green for active
        } else {
            ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ImVec4(0.2f, 0.4f, 0.2f, 0.5f));  // Faded green for inactive
        }
        
        // Show % only if phase 2 is active
        char phase2Text[32] = "";
        if (trainingProgress_ > 0.5f) {
            snprintf(phase2Text, sizeof(phase2Text), "%.0f%%", ((trainingProgress_ - 0.5f) / 0.5f) * 100.0f);
        } else {
            strcpy(phase2Text, "Waiting...");
        }
        
        ImGui::PushID("Phase2Bar");
        ImGui::ProgressBar(
            trainingProgress_ > 0.5f ? (trainingProgress_ - 0.5f) / 0.5f : 0.0f, 
            ImVec2(halfWidth - 5.0f, barHeight), 
            phase2Text
        );
        ImGui::PopID();
        ImGui::PopStyleColor(2);
        
        // Overall percentage
        ImGui::Spacing();
        ImGui::Text("Overall Progress: %.0f%%", trainingProgress_ * 100.0f);
        
        ImGui::EndGroup();
    } else {
        // Standard progress bar for regular training (single phase)
        ImGui::Text("Training Progress:");
        ImGui::Spacing();
        
        // Use a nicer progress bar for regular training
        ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.1f, 0.1f, 0.1f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ImVec4(0.0f, 0.7f, 0.0f, 1.0f));
        
        // Show percentage inside the bar
        char progressText[32];
        snprintf(progressText, sizeof(progressText), "%.0f%%", trainingProgress_ * 100.0f);
        
        ImGui::ProgressBar(trainingProgress_, ImVec2(-1, 24.0f), progressText);
        ImGui::PopStyleColor(2);
    }
    
    // Update progress during training
    if (isTraining_ && model_) {
        auto stats = model_->getStats();
        
        // When using auto-hyperparameter tuning, we have two phases:
        // 1. Grid search progress (finding best parameters)
        // 2. Final model training progress (training with best parameters)
        if (autoHyperparameters_) {
            if (stats.find("Grid Search Progress") != stats.end()) {
                // Grid search phase - use half of progress bar for grid search
                trainingProgress_ = stats["Grid Search Progress"] * 0.5;
            } else if (stats.find("Training Progress") != stats.end()) {
                // Final training phase - start from 50% and go to 100%
                trainingProgress_ = 0.5 + (stats["Training Progress"] * 0.5);
            }
        } else {
            // Regular training (no auto-tuning)
            if (stats.find("Training Progress") != stats.end()) {
                trainingProgress_ = stats["Training Progress"];
            }
        }
        
        // Process Windows messages to prevent "not responding" state
        if (window_) {
            glfwPollEvents();
        }
    }
    
    ImGui::Spacing();
    if (ImGui::Button("Train Model", ImVec2(200, 0))) {
        // Only start training if not already training
        if (!isTraining_) {
            isTraining_ = true;  // Set training state when button is clicked
            trainingProgress_ = 0.0;  // Reset progress
            modelTrainingSuccess_ = false;  // Reset success state
            shouldStopTraining_ = false;  // Reset stop flag
            
            // Start training in a separate thread
            trainingThread_ = std::thread([this]() {
                // Prepare data for training
                Eigen::MatrixXd X = dataHandler_.getSelectedFeatures(selectedFeatures_);
                Eigen::VectorXd y;
                
                // For simplicity, we'll use the first selected target
                if (!selectedTargetIndices_.empty()) {
                    y = dataHandler_.getSelectedTarget(selectedTargetIndices_[0]);
                }
                
                // Apply time series features if configured
                bool dataModified = false;
                std::map<std::string, int> bestLagValues;
                
                // Add lag features with variable-specific lags if enabled
                if (lagValues_ > 0) {
                    if (dataHandler_.addVariableLagFeatures(lagValues_, bestLagValues)) {
                        // Get updated feature set with optimal lag features
                        X = dataHandler_.getSelectedFeatures(selectedFeatures_);
                        
                        // Update target vector since rows were removed
                        if (!selectedTargetIndices_.empty()) {
                            y = dataHandler_.getSelectedTarget(selectedTargetIndices_[0]);
                        }
                        dataModified = true;
                        
                        // Store the discovered best lag values
                        std::stringstream ss;
                        ss << "Optimal lag values found for " << bestLagValues.size() << " features";
                        LOG_INFO(ss.str());
                        
                        for (const auto& [feature, lag] : bestLagValues) {
                            ss.str("");
                            ss << "Feature: " << feature << ", Optimal lag: " << lag;
                            LOG_INFO(ss.str());
                        }
                    } else {
                        LOG_WARNING("Unable to determine optimal lag values, falling back to standard method");
                        
                        // Fall back to standard method if the advanced one fails
                        if (dataHandler_.addLagFeatures(lagValues_)) {
                            // Get updated feature set with lag features
                            X = dataHandler_.getSelectedFeatures(selectedFeatures_);
                            
                            // Update target vector since rows were removed
                            if (!selectedTargetIndices_.empty()) {
                                y = dataHandler_.getSelectedTarget(selectedTargetIndices_[0]);
                            }
                            dataModified = true;
                        }
                    }
                }
                
                // Add seasonal lag features if specified
                if (seasonality_ > 0) {
                    // Try to find and use the best seasonal lags if the auto-detect option is enabled
                    bool useAutoDetect = true; // Set to true for auto-detection, or add a GUI option
                    
                    if (useAutoDetect) {
                        // Find and add variable-specific seasonal lags
                        std::map<std::string, int> bestSeasonalLags;
                        if (dataHandler_.addVariableSeasonalLags(seasonality_, bestSeasonalLags)) {
                            LOG_INFO("Added seasonal lag features with auto-detected periods");
                            
                            // Get updated feature set with seasonal lag features
                            X = dataHandler_.getSelectedFeatures(selectedFeatures_);
                            
                            // Update target vector since rows were removed
                            if (!selectedTargetIndices_.empty()) {
                                y = dataHandler_.getSelectedTarget(selectedTargetIndices_[0]);
                            }
                            dataModified = true;
                            
                            // Log the detected seasonal values
                            LOG_INFO("Detected optimal seasonal periods:");
                            for (const auto& [target, season] : bestSeasonalLags) {
                                std::stringstream ss;
                                ss << "Target: " << target << ", Optimal seasonality: " << season;
                                LOG_INFO(ss.str());
                                
                                // Also log the exact feature name that will be created
                                ss.str("");
                                ss << "Expected seasonal feature name: " << target << "_seasonal" << season;
                                LOG_INFO(ss.str());
                            }
                        } else {
                            LOG_WARNING("Unable to determine optimal seasonal values, falling back to standard method");
                            
                            // Fall back to standard method if the advanced one fails
                            if (dataHandler_.addSeasonalLags(seasonality_)) {
                                // Get updated feature set with seasonal lag features
                                X = dataHandler_.getSelectedFeatures(selectedFeatures_);
                                
                                // Update target vector since rows were removed
                                if (!selectedTargetIndices_.empty()) {
                                    y = dataHandler_.getSelectedTarget(selectedTargetIndices_[0]);
                                }
                                dataModified = true;
                            }
                        }
                    } else {
                        // Use the standard method with a fixed seasonality value
                        if (dataHandler_.addSeasonalLags(seasonality_)) {
                            // Get updated feature set with seasonal lag features
                            X = dataHandler_.getSelectedFeatures(selectedFeatures_);
                            
                            // Update target vector since rows were removed
                            if (!selectedTargetIndices_.empty()) {
                                y = dataHandler_.getSelectedTarget(selectedTargetIndices_[0]);
                            }
                            dataModified = true;
                        }
                    }
                }
                
                // Update feature names after all time series transformations are complete
                // and ensure the number matches the model input matrix
                {
                    std::lock_guard<std::mutex> lock(trainingMutex_);
                    
                    // Get feature names and log them
                    featureNames_ = dataHandler_.getFeatureNames();
                    
                    if (dataModified) {
                        LOG_INFO("Updated feature names after time series transformations");
                        std::stringstream ss;
                        ss << "Feature count before adjustment: " << featureNames_.size();
                        LOG_INFO(ss.str());
                    }
                    
                    // Check for date columns and remove them from feature list
                    std::vector<size_t> dateColumnIndices = dataHandler_.getDateColumnIndices();
                    if (!dateColumnIndices.empty()) {
                        LOG_INFO("Removing date columns from model features");
                        std::vector<size_t> dateIndices;
                        
                        // Find date columns in our feature names
                        for (size_t dateColIdx : dateColumnIndices) {
                            std::string dateColName = dataHandler_.getColumnNames()[dateColIdx];
                            for (size_t i = 0; i < featureNames_.size(); ++i) {
                                if (featureNames_[i] == dateColName) {
                                    dateIndices.push_back(i);
                                    break;
                                }
                            }
                        }
                        
                        // Remove date columns from feature names (in reverse order to preserve indices)
                        std::sort(dateIndices.begin(), dateIndices.end(), std::greater<size_t>());
                        for (size_t idx : dateIndices) {
                            if (idx < featureNames_.size()) {
                                LOG_INFO("Removing date column from features: " + featureNames_[idx]);
                                featureNames_.erase(featureNames_.begin() + idx);
                                
                                // Also remove from X matrix if needed
                                if (idx < static_cast<size_t>(X.cols())) {
                                    // Create new matrix without the date column
                                    Eigen::MatrixXd newX(X.rows(), X.cols() - 1);
                                    
                                    // Copy columns before the date column
                                    if (idx > 0) {
                                        newX.leftCols(idx) = X.leftCols(idx);
                                    }
                                    
                                    // Copy columns after the date column
                                    if (idx < static_cast<size_t>(X.cols()) - 1) {
                                        newX.rightCols(X.cols() - idx - 1) = X.rightCols(X.cols() - idx - 1);
                                    }
                                    
                                    X = newX;
                                }
                            }
                        }
                    }
                    
                    // Log the feature names after date column removal
                    if (dataModified) {
                        std::stringstream ss;
                        ss << "Feature count after date removal: " << featureNames_.size();
                        LOG_INFO(ss.str());
                    }
                    
                    // Ensure seasonal features are included - add explicit check for the existence of seasonal features
                    std::map<std::string, int> bestSeasonalLags = dataHandler_.getBestSeasonalLagValues();
                    if (!bestSeasonalLags.empty()) {
                        LOG_INFO("Checking for seasonal features in feature names:");
                        for (const auto& [target, period] : bestSeasonalLags) {
                            std::string expectedPattern = target + "_seasonal" + std::to_string(period);
                            
                            // Check if this feature exists in the feature names
                            bool found = false;
                            for (const auto& featureName : featureNames_) {
                                if (featureName == expectedPattern) {
                                    found = true;
                                    break;
                                }
                            }
                            
                            if (found) {
                                LOG_INFO("Found seasonal feature: " + expectedPattern);
                            } else {
                                LOG_WARNING("Missing seasonal feature: " + expectedPattern);
                            }
                        }
                    }
                    
                    // Adjust feature names list to match X dimensions
                    if (featureNames_.size() > static_cast<size_t>(X.cols())) {
                        // Special attention to preserve seasonal features when trimming
                        std::vector<std::string> newFeatureNames;
                        newFeatureNames.reserve(X.cols());
                        
                        // First, identify and prioritize important features (like seasonal features)
                        std::vector<std::string> priorityFeatures;
                        for (const auto& [target, period] : bestSeasonalLags) {
                            std::string expectedPattern = target + "_seasonal" + std::to_string(period);
                            priorityFeatures.push_back(expectedPattern);
                        }
                        
                        // Copy important features first
                        for (const auto& priorityFeature : priorityFeatures) {
                            auto it = std::find(featureNames_.begin(), featureNames_.end(), priorityFeature);
                            if (it != featureNames_.end()) {
                                newFeatureNames.push_back(priorityFeature);
                            }
                        }
                        
                        // Then add other features until we reach X.cols()
                        for (const auto& featureName : featureNames_) {
                            if (newFeatureNames.size() >= static_cast<size_t>(X.cols())) {
                                break;
                            }
                            
                            // Skip if already added as priority feature
                            if (std::find(newFeatureNames.begin(), newFeatureNames.end(), featureName) != newFeatureNames.end()) {
                                continue;
                            }
                            
                            newFeatureNames.push_back(featureName);
                        }
                        
                        LOG_WARNING("Adjusting feature names from " + std::to_string(featureNames_.size()) + 
                                  " to match actual input columns (" + std::to_string(X.cols()) + ")");
                        featureNames_ = newFeatureNames;
                    }
                    // Or pad with generic names if needed
                    else if (featureNames_.size() < static_cast<size_t>(X.cols())) {
                        LOG_WARNING("Adding generic feature names to match input columns (" + 
                                  std::to_string(X.cols()) + ")");
                        size_t originalSize = featureNames_.size();
                        featureNames_.resize(X.cols());
                        for (size_t i = originalSize; i < featureNames_.size(); i++) {
                            featureNames_[i] = "Feature_" + std::to_string(i);
                        }
                    }
                    
                    // Log final feature list
                    if (dataModified) {
                        std::stringstream ss;
                        ss << "Final feature count: " << featureNames_.size();
                        LOG_INFO(ss.str());
                        
                        LOG_INFO("Final feature list:");
                        for (size_t i = 0; i < std::min(featureNames_.size(), size_t(10)); ++i) {
                            LOG_INFO(std::to_string(i) + ": " + featureNames_[i]);
                        }
                        if (featureNames_.size() > 10) {
                            LOG_INFO("... and " + std::to_string(featureNames_.size() - 10) + " more");
                        }
                    }
                    
                    // Reset training results
                    modelStats_.clear();
                    modelHyperparams_.clear();
                    modelCoefficients_.resize(0);
                    modelIntercept_ = 0.0;
                }
                
                // Create model based on selection
                switch (selectedModelIndex_) {
                    case 0: // Linear Regression
                        model_ = std::make_unique<LinearRegression>();
                        break;
                        
                    case 1: // Elastic Net
                        if (autoHyperparameters_) {
                            // Define grid search values for ElasticNet
                            std::vector<double> alpha_values = {0.0, 0.2, 0.4, 0.6, 0.8, 1.0};
                            std::vector<double> lambda_values = {0.0001, 0.001, 0.01, 1.0, 10.0, 50.0, 100.0};
                            
                            // Create persistent model for grid search
                            model_ = std::make_unique<ElasticNet>();
                            // Initialize progress to 0
                            model_->getStats()["Grid Search Progress"] = 0.0;
                            
                            // Run grid search on the persistent model
                            auto [best_alpha, best_lambda] = 
                                static_cast<ElasticNet*>(model_.get())->gridSearch(X, y, alpha_values, lambda_values);
                            
                            // Create a new model with the best parameters
                            // This is important to ensure we're starting with a clean model
                            auto tempModel = std::make_unique<ElasticNet>(best_alpha, best_lambda);
                            if (tempModel->train(X, y)) {
                                // Only replace the model if training succeeded
                                model_ = std::move(tempModel);
                            }
                            
                            // Update UI values to reflect the best parameters
                            alpha_ = best_alpha;
                            lambda_ = best_lambda;
                        } else {
                            model_ = std::make_unique<ElasticNet>(alpha_, lambda_);
                        }
                        break;
                        
                    case 2: // XGBoost
                        if (autoHyperparameters_) {
                            // Define expanded grid search values for XGBoost
                            std::vector<int> n_estimators_values = {100, 200, 300};  // Reduced from 5 to 3 values
                            std::vector<double> learning_rate_values = {0.01, 0.05, 0.075, 0.1, 0.3};  // Adjusted range
                            std::vector<int> max_depth_values = {3, 4, 5, 6, 7};  // Reduced and centered range
                            std::vector<double> subsample_values = {0.8, 1.0};  // Most common values
                            
                            // Create persistent model for grid search
                            model_ = std::make_unique<XGBoost>();
                            // Initialize progress to 0
                            model_->getStats()["Grid Search Progress"] = 0.0;
                            
                            auto [best_n_estimators, best_learning_rate, best_max_depth, best_subsample] = 
                                static_cast<XGBoost*>(model_.get())->gridSearch(X, y, n_estimators_values, 
                                                                               learning_rate_values, max_depth_values, subsample_values);
                            
                            // Create temporary model with best hyperparameters
                            auto tempModel = std::make_unique<XGBoost>(best_n_estimators, best_learning_rate, best_max_depth, best_subsample);
                            if (tempModel->train(X, y)) {
                                // Only replace the model if training succeeded
                                model_ = std::move(tempModel);
                            }
                            
                            // Update UI values to reflect the best parameters
                            n_estimators_ = best_n_estimators;
                            learning_rate_ = best_learning_rate;
                            max_depth_ = best_max_depth;
                            subsample_ = best_subsample;
                        } else {
                            model_ = std::make_unique<XGBoost>(n_estimators_, learning_rate_, max_depth_, subsample_);
                        }
                        break;
                        
                    case 3: // Gradient Boosting
                        if (autoHyperparameters_) {
                            // Define grid search values for Gradient Boosting - adjusted for very small dataset
                            std::vector<int> n_estimators_values = {5, 10, 15, 20};  // Start with fewer trees
                            std::vector<double> learning_rate_values = {0.3, 0.5, 0.8, 1.0};  // Higher learning rates for faster convergence
                            std::vector<int> max_depth_values = {2, 3, 4};  // Shallow trees to prevent overfitting
                            std::vector<int> min_samples_split_values = {2, 3};  // Allow slightly larger minimum splits
                            
                            // Create persistent model for grid search
                            model_ = std::make_unique<GradientBoosting>();
                            // Initialize progress to 0
                            model_->getStats()["Grid Search Progress"] = 0.0;
                            
                            auto [best_n_estimators, best_learning_rate, best_max_depth, best_min_samples_split] = 
                                static_cast<GradientBoosting*>(model_.get())->gridSearch(X, y, n_estimators_values, 
                                                                                        learning_rate_values, max_depth_values, min_samples_split_values);
                            
                            // Create temporary model with best hyperparameters
                            auto tempModel = std::make_unique<GradientBoosting>(best_n_estimators, best_learning_rate, best_max_depth, best_min_samples_split);
                            if (tempModel->train(X, y)) {
                                // Only replace the model if training succeeded
                                model_ = std::move(tempModel);
                            }
                            
                            // Update UI values to reflect the best parameters
                            n_estimators_ = best_n_estimators;
                            learning_rate_ = best_learning_rate;
                            max_depth_ = best_max_depth;
                            min_samples_split_ = best_min_samples_split;
                        } else {
                            model_ = std::make_unique<GradientBoosting>(n_estimators_, learning_rate_, max_depth_, min_samples_split_);
                        }
                        break;
                        
                    case 4: // Neural Network
                        if (autoHyperparameters_) {
                            // Define expanded grid search values for Neural Network
                            std::vector<std::vector<int>> layer_configs = {
                                {10},           // Simple single layer
                                {20},           // Larger single layer
                                {10, 5},        // Two layers, decreasing
                                {10, 10}        // Two equal layers
                            };
                            std::vector<double> learning_rate_values = {0.01, 0.001, 0.0001};  // Common learning rates
                            std::vector<double> alpha_values = {0.01, 0.001};  // L2 regularization
                            std::vector<int> max_iterations_values = {200, 500, 1000};  // Reasonable iteration limits
                            
                            // Create persistent model for grid search
                            model_ = std::make_unique<NeuralNetwork>();
                            // Initialize progress to 0
                            model_->getStats()["Grid Search Progress"] = 0.0;
                            
                            auto [best_layers, best_learning_rate, best_alpha, best_iterations] = 
                                static_cast<NeuralNetwork*>(model_.get())->gridSearch(X, y, layer_configs, 
                                                                                     learning_rate_values, alpha_values, max_iterations_values);
                            
                            // Create temporary model with best hyperparameters
                            auto tempModel = std::make_unique<NeuralNetwork>(best_layers, best_learning_rate, best_iterations, best_alpha);
                            if (tempModel->train(X, y)) {
                                // Only replace the model if training succeeded
                                model_ = std::move(tempModel);
                            }
                            
                            // Update UI values to reflect the best parameters
                            hidden_layers_ = best_layers.size();
                            if (!best_layers.empty()) {
                                neurons_per_layer_ = best_layers[0]; // Use first layer size for UI
                            }
                            learning_rate_ = best_learning_rate;
                            max_iterations_ = best_iterations;
                        } else {
                            // Create vector for hidden layer sizes
                            std::vector<int> hidden_layer_sizes(hidden_layers_, neurons_per_layer_);
                            model_ = std::make_unique<NeuralNetwork>(hidden_layer_sizes, learning_rate_, max_iterations_);
                        }
                        break;
                }
                
                // Train the model with progress updates
                modelTrainingSuccess_ = model_->train(X, y);
                
                // Get final statistics and coefficients
                if (modelTrainingSuccess_) {
                    std::lock_guard<std::mutex> lock(trainingMutex_);
                    modelStats_ = model_->getStats();
                    modelCoefficients_ = model_->getCoefficients();
                    modelHyperparams_ = model_->getHyperparameters();
                    
                    // Ensure coefficient vector size matches feature names
                    if (modelCoefficients_.size() < featureNames_.size()) {
                        LOG_WARNING("Coefficient vector size (" + std::to_string(modelCoefficients_.size()) + 
                                  ") smaller than feature names (" + std::to_string(featureNames_.size()) + ")");
                    }
                    
                    // Get intercept from stats if available
                    if (modelStats_.find("Intercept") != modelStats_.end()) {
                        modelIntercept_ = modelStats_["Intercept"];
                    }
                    
                    // Get predictions
                    predictions_ = model_->predict(X);
                }
                
                // Reset training state
                isTraining_ = false;
                trainingProgress_ = modelTrainingSuccess_ ? 1.0 : 0.0;
            });
            
            // Detach the thread to let it run independently
            trainingThread_.detach();
        }
    }
    
    // Only show model results if training is complete and successful
    if (modelTrainingSuccess_ && model_) {
        ImGui::Separator();
        ImGui::Text("Model Results");
        ImGui::Separator();

        // Create a child window for scrollable results
        ImGui::BeginChild("ModelResults", ImVec2(0, ImGui::GetWindowHeight() * 0.4f), true);

        // Model Statistics Section
        if (ImGui::CollapsingHeader("Model Statistics", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Indent(20.0f);
            
            // Display R-squared
            if (modelStats_.find("R²") != modelStats_.end()) {
                ImGui::Text("R-squared (R²): %.4f", modelStats_["R²"]);
                ImGui::TextWrapped("Indicates how well the model fits the data (0-1, higher is better)");
            }
            
            // Display RMSE
            if (modelStats_.find("RMSE") != modelStats_.end()) {
                ImGui::Text("Root Mean Square Error: %.4f", modelStats_["RMSE"]);
                ImGui::TextWrapped("Average prediction error in the same units as the target variable");
            }

            // Display MAE
            if (modelStats_.find("MAE") != modelStats_.end()) {
                ImGui::Text("Mean Absolute Error: %.4f", modelStats_["MAE"]);
                ImGui::TextWrapped("Average absolute prediction error");
            }

            // Display MSE
            if (modelStats_.find("MSE") != modelStats_.end()) {
                ImGui::Text("Mean Squared Error: %.4f", modelStats_["MSE"]);
                ImGui::TextWrapped("Average squared prediction error");
            }

            // Tree-based model statistics (XGBoost and Gradient Boosting)
            if (selectedModelIndex_ == 2 || selectedModelIndex_ == 3) {
                ImGui::Spacing();
                ImGui::Separator();
                ImGui::Text("Tree Statistics:");
                
                // Number of trees
                if (modelStats_.find("Number of Trees") != modelStats_.end()) {
                    ImGui::Text("Number of Trees: %.0f", modelStats_["Number of Trees"]);
                }
                
                // Average tree depth
                if (modelStats_.find("Average Tree Depth") != modelStats_.end()) {
                    ImGui::Text("Average Tree Depth: %.2f", modelStats_["Average Tree Depth"]);
                }
                
                // Average leaf nodes
                if (modelStats_.find("Average Leaf Nodes") != modelStats_.end()) {
                    ImGui::Text("Average Leaf Nodes per Tree: %.2f", modelStats_["Average Leaf Nodes"]);
                }
                
                // Training loss
                if (modelStats_.find("Training Loss") != modelStats_.end()) {
                    ImGui::Text("Training Loss: %.4f", modelStats_["Training Loss"]);
                }
                
                // Learning rate
                if (modelStats_.find("Learning Rate") != modelStats_.end()) {
                    ImGui::Text("Learning Rate: %.4f", modelStats_["Learning Rate"]);
                }
                
                // Max tree depth
                if (modelStats_.find("Max Tree Depth") != modelStats_.end()) {
                    ImGui::Text("Maximum Tree Depth: %.0f", modelStats_["Max Tree Depth"]);
                }
                
                // XGBoost specific
                if (selectedModelIndex_ == 2 && modelStats_.find("Subsample Ratio") != modelStats_.end()) {
                    ImGui::Text("Subsample Ratio: %.2f", modelStats_["Subsample Ratio"]);
                }
                
                // Gradient Boosting specific
                if (selectedModelIndex_ == 3 && modelStats_.find("Min Samples Split") != modelStats_.end()) {
                    ImGui::Text("Minimum Samples Split: %.0f", modelStats_["Min Samples Split"]);
                }
            }

            ImGui::Unindent(20.0f);
        }

        // Model Coefficients Section (for applicable models)
        if (selectedModelIndex_ == 0 || selectedModelIndex_ == 1) { // Linear Regression or Elastic Net
            if (ImGui::CollapsingHeader("Model Coefficients", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::Indent(20.0f);

                // Display intercept with its standard error if available
                ImGui::Text("Intercept: %.4f", modelIntercept_);
                if (modelStats_.find("Intercept SE") != modelStats_.end()) {
                    ImGui::SameLine();
                    ImGui::Text("(SE: %.4f)", modelStats_["Intercept SE"]);
                }
                
                // Add simple time series information if applicable
                if (lagValues_ > 0 || seasonality_ > 0) {
                    ImGui::Spacing();
                    if (lagValues_ > 0) {
                        ImGui::Text("Maximum lag period: %d", lagValues_);
                        
                        // Show optimal lags if available
                        std::map<std::string, int> bestLags = dataHandler_.getBestLagValues();
                        if (!bestLags.empty()) {
                            ImGui::Text("Optimal lags detected: %zu", bestLags.size());
                        }
                    }
                    
                    if (seasonality_ > 0) {
                        ImGui::Text("Maximum seasonality period: %d", seasonality_);
                        
                        // Show optimal seasonal lags if available
                        std::map<std::string, int> bestSeasonalLags = dataHandler_.getBestSeasonalLagValues();
                        if (!bestSeasonalLags.empty()) {
                            ImGui::Text("Optimal seasonal patterns detected: %zu", bestSeasonalLags.size());
                            
                            // Display the first few best seasonal lags
                            int count = 0;
                            for (const auto& [name, period] : bestSeasonalLags) {
                                if (count < 3) { // Limit to first 3 to avoid cluttering the UI
                                    ImGui::Text("  %s: %d periods", name.c_str(), period);
                                    count++;
                                } else if (count == 3) {
                                    ImGui::Text("  ... and %zu more", bestSeasonalLags.size() - 3);
                                    break;
                                }
                            }
                        }
                    }
                    ImGui::Separator();
                }
                
                // Get best lag values for highlighting and reference
                std::map<std::string, int> bestLagValues = dataHandler_.getBestLagValues();
                
                // Get best seasonal lag values for highlighting and reference
                std::map<std::string, int> bestSeasonalLagValues = dataHandler_.getBestSeasonalLagValues();
                
                // Create a table for coefficients and their statistics
                if (ImGui::BeginTable("CoefficientsTable", 5, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
                    ImGui::TableSetupColumn("Feature");
                    ImGui::TableSetupColumn("Type");
                    ImGui::TableSetupColumn("Coefficient");
                    ImGui::TableSetupColumn("Std. Error");
                    ImGui::TableSetupColumn("t-value");
                    ImGui::TableHeadersRow();

                    // Instead of using the full featureNames_, we'll use only the coefficients we have
                    size_t rowCount = static_cast<size_t>(modelCoefficients_.size());
                    
                    for (size_t i = 0; i < rowCount; ++i) {
                        ImGui::TableNextRow();
                        
                        // Feature name
                        ImGui::TableSetColumnIndex(0);
                        
                        // Get the feature name safely, using index if it exists, or generating a name otherwise
                        std::string featureName = (i < featureNames_.size()) ? featureNames_[i] : "Feature_" + std::to_string(i);
                        
                        // Extract base feature name (without _lag or _seasonal suffix)
                        std::string baseFeatureName = featureName;
                        size_t lagPos = featureName.find("_lag");
                        size_t seasonalPos = featureName.find("_seasonal");
                        
                        if (lagPos != std::string::npos) {
                            baseFeatureName = featureName.substr(0, lagPos);
                        } else if (seasonalPos != std::string::npos) {
                            baseFeatureName = featureName.substr(0, seasonalPos);
                        }
                        
                        // Check if this is a lag or seasonal feature
                        bool isLagFeature = lagPos != std::string::npos;
                        bool isSeasonalFeature = seasonalPos != std::string::npos;
                        
                        // Apply color highlight for time series features
                        if (isLagFeature) {
                            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 0.7f, 1.0f, 1.0f)); // Blue for lag
                        } else if (isSeasonalFeature) {
                            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.5f, 0.0f, 1.0f)); // Orange for seasonal
                        }
                        
                        ImGui::Text("%s", featureName.c_str());
                        
                        if (isLagFeature || isSeasonalFeature) {
                            ImGui::PopStyleColor();
                        }
                        
                        // Feature type with enhanced information
                        ImGui::TableSetColumnIndex(1);
                        
                        // 1. Check if it's a lag feature
                        if (isLagFeature) {
                            // Try to extract lag value from feature name
                            int lagValue = 0;
                            if (lagPos != std::string::npos && lagPos + 4 < featureName.size()) {
                                std::string lagStr = featureName.substr(lagPos + 4);
                                try {
                                    lagValue = std::stoi(lagStr);
                                    ImGui::Text("Lag %d", lagValue);
                                } catch (...) {
                                    // If no lag value in name, try to get from bestLagValues
                                    auto it = bestLagValues.find(baseFeatureName);
                                    if (it != bestLagValues.end()) {
                                        ImGui::Text("Lag %d", it->second);
                                    } else {
                                        ImGui::Text("Lag");
                                    }
                                }
                            } else {
                                // If no lag suffix, try to get from bestLagValues
                                auto it = bestLagValues.find(baseFeatureName);
                                if (it != bestLagValues.end()) {
                                    ImGui::Text("Lag %d", it->second);
                                } else {
                                    ImGui::Text("Lag");
                                }
                            }
                        }
                        // 2. Check if it's a seasonal feature
                        else if (isSeasonalFeature) {
                            // First check if this is from an auto-detected best seasonal lag
                            bool isBestSeasonal = false;
                            int seasonValue = 0;
                            
                            // Extract seasonal value from feature name
                            if (seasonalPos != std::string::npos && seasonalPos + 9 < featureName.size()) {
                                std::string seasonalStr = featureName.substr(seasonalPos + 9);
                                try {
                                    seasonValue = std::stoi(seasonalStr);
                                } catch (...) {
                                    // If extraction fails, use the global seasonality value
                                    seasonValue = seasonality_;
                                }
                            } else {
                                seasonValue = seasonality_;
                            }
                            
                            // Extract original feature name (remove _seasonal suffix)
                            std::string baseName = featureName;
                            if (seasonalPos != std::string::npos) {
                                baseName = featureName.substr(0, seasonalPos);
                            }
                            
                            // Check against best seasonal lag map
                            for (const auto& [target, period] : bestSeasonalLagValues) {
                                if (baseName == target && seasonValue == period) {
                                    ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "Optimal Seasonal %d", period);
                                    isBestSeasonal = true;
                                    break;
                                }
                            }
                            
                            // If not identified as a best seasonal lag, display regular info
                            if (!isBestSeasonal) {
                                ImGui::Text("Seasonal %d", seasonValue);
                            }
                        }
                        // 3. Check if it's in bestLagValues (might be a lag feature without _lag suffix)
                        else if (bestLagValues.find(featureName) != bestLagValues.end()) {
                            ImGui::Text("Lag %d", bestLagValues[featureName]);
                        }
                        // 4. Default to Standard
                        else {
                            ImGui::Text("Standard");
                        }
                        
                        // Coefficient value
                        ImGui::TableSetColumnIndex(2);
                        ImGui::Text("%.4f", modelCoefficients_(i));
                        
                        // Standard Error
                        ImGui::TableSetColumnIndex(3);
                        std::string se_key = "SE_" + std::to_string(i);
                        if (modelStats_.find(se_key) != modelStats_.end()) {
                            ImGui::Text("%.4f", modelStats_[se_key]);
                        } else {
                            ImGui::Text("-");
                        }
                        
                        // t-value
                        ImGui::TableSetColumnIndex(4);
                        std::string t_key = "t_value_" + std::to_string(i);
                        if (modelStats_.find(t_key) != modelStats_.end()) {
                            ImGui::Text("%.4f", modelStats_[t_key]);
                        } else {
                            ImGui::Text("-");
                        }
                    }
                    ImGui::EndTable();
                }

                ImGui::Unindent(20.0f);
            }
        }

        // Feature Importance Section (for tree-based models)
        if (selectedModelIndex_ == 2 || selectedModelIndex_ == 3) { // XGBoost or Gradient Boosting
            if (ImGui::CollapsingHeader("Feature Importance", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::Indent(20.0f);
                
                if (ImGui::BeginTable("ImportanceTable", 2, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
                    ImGui::TableSetupColumn("Feature");
                    ImGui::TableSetupColumn("Importance");
                    ImGui::TableHeadersRow();

                    // Get feature importance scores
                    for (size_t i = 0; i < featureNames_.size(); ++i) {
                        std::string key = "Feature " + std::to_string(i) + " Importance";
                        if (modelStats_.find(key) != modelStats_.end()) {
                            ImGui::TableNextRow();
                            ImGui::TableSetColumnIndex(0);
                            ImGui::Text("%s", featureNames_[i].c_str());
                            ImGui::TableSetColumnIndex(1);
                            ImGui::Text("%.4f", modelStats_[key]);
                        }
                    }
                    ImGui::EndTable();
                }

                ImGui::Unindent(20.0f);
            }
        }

        // Neural Network Architecture (if applicable)
        if (selectedModelIndex_ == 4) { // Neural Network
            if (ImGui::CollapsingHeader("Network Architecture", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::Indent(20.0f);
                
                // Display layer information
                if (modelStats_.find("Layers") != modelStats_.end()) {
                    ImGui::Text("Number of Layers: %.0f", modelStats_["Layers"]);
                }
                if (modelStats_.find("Parameters") != modelStats_.end()) {
                    ImGui::Text("Total Parameters: %.0f", modelStats_["Parameters"]);
                }
                if (modelStats_.find("Training Loss") != modelStats_.end()) {
                    ImGui::Text("Final Training Loss: %.4f", modelStats_["Training Loss"]);
                }
                
                ImGui::Unindent(20.0f);
            }
        }

        // Show hyperparameter information if auto-tuning was used
        if (autoHyperparameters_) {
            if (ImGui::CollapsingHeader("Optimal Hyperparameters", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::Indent(20.0f);
                ImGui::TextColored(ImVec4(0.0f, 0.7f, 1.0f, 1.0f), "Optimal hyperparameters found via grid search:");
                
                if (ImGui::BeginTable("HyperparamsTable", 2, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
                    ImGui::TableSetupColumn("Parameter");
                    ImGui::TableSetupColumn("Value");
                    ImGui::TableHeadersRow();

                    for (const auto& [name, value] : modelHyperparams_) {
                        ImGui::TableNextRow();
                        ImGui::TableSetColumnIndex(0);
                        ImGui::Text("%s", name.c_str());
                        ImGui::TableSetColumnIndex(1);
                        ImGui::Text("%.4f", value);
                    }
                    ImGui::EndTable();
                }
                
                ImGui::Unindent(20.0f);
            }
        }

        ImGui::EndChild();
    }
    
    // Navigation buttons
    ImGui::Spacing();
    if (ImGui::Button("Back: Variable Selection", ImVec2(200, 0))) {
        setScreen(Screen::VARIABLE_SELECTION);
    }
    
    ImGui::SameLine();
    
    // Only enable Next button if model is trained
    if (modelTrainingSuccess_ && model_) {
        if (ImGui::Button("Next: Plotting", ImVec2(200, 0))) {
            setScreen(Screen::PLOTTING);
        }
    } else {
        ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
        ImGui::Button("Next: Plotting", ImVec2(200, 0));
        ImGui::PopStyleVar();
        ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "Please train the model first!");
    }
}

void GUI::renderPlotting() {
    if (!model_) {
        ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "No model trained yet. Please train a model first.");
        return;
    }

    // Display model results at the top
    ImGui::Text("Model Results");
    ImGui::Separator();
    
    // Get model statistics
    auto stats = model_->getStats();
    
    // Display R-squared
    if (stats.find("R-squared") != stats.end()) {
        ImGui::Text("R-squared: %.4f", stats["R-squared"]);
    }
    
    // Display coefficients for linear regression
    if (selectedModelIndex_ == 0) { // Linear Regression
        ImGui::Text("Model Coefficients:");
        ImGui::Indent(20.0f);
        
        // Get feature names
        const std::vector<std::string>& featureNames = dataHandler_.getFeatureNames();
        
        // Display intercept
        ImGui::Text("Intercept: %.4f", stats["Intercept"]);
        
        // Display feature coefficients
        for (size_t i = 0; i < featureNames.size(); ++i) {
            std::string coefKey = "Coefficient_" + std::to_string(i);
            if (stats.find(coefKey) != stats.end()) {
                ImGui::Text("%s: %.4f", featureNames[i].c_str(), stats[coefKey]);
            }
        }
        ImGui::Unindent(20.0f);
    }
    
    // Display RMSE if available
    if (stats.find("RMSE") != stats.end()) {
        ImGui::Text("Root Mean Square Error: %.4f", stats["RMSE"]);
    }
    
    ImGui::Separator();

    // Initialize plot manager if needed
    if (!plotManager_) {
        plotManager_ = std::make_shared<PlotManager>();
    }

    // Reset plot manager to clear previous plots
    plotManager_->reset();
    
    // Add time series plot
    if (model_ && !predictions_.isZero()) {
        auto timeSeriesPlot = std::make_shared<TimeSeries>(
            "Model Predictions Over Time",
            "Time",
            "Values"
        );
        timeSeriesPlot->setData(dataHandler_.getDates(), dataHandler_.getTargetValues(), predictions_);
        plotManager_->addPlot(timeSeriesPlot);
        
        // Add scatter plot
        auto scatterPlot = std::make_shared<ScatterPlot>(
            "Actual vs Predicted Values",
            "Actual Values",
            "Predicted Values"
        );
        scatterPlot->setData(dataHandler_.getTargetValues(), predictions_);
        plotManager_->addPlot(scatterPlot);
        
        // Add residual plot
        auto residualPlot = std::make_shared<ResidualPlot>(
            "Residual Analysis",
            "Predicted Values",
            "Residuals"
        );
        residualPlot->setData(predictions_, dataHandler_.getTargetValues() - predictions_);
        plotManager_->addPlot(residualPlot);
        
        // Add coefficient statistics plot for linear models
        if (selectedModelIndex_ == 0 || selectedModelIndex_ == 1) {
            const std::vector<std::string>& featureNames = dataHandler_.getFeatureNames();
            Eigen::VectorXd coefficients = model_->getCoefficients();
            Eigen::VectorXd standardErrors(coefficients.size());
            Eigen::VectorXd tValues(coefficients.size());
            
            for (int i = 0; i < coefficients.size(); ++i) {
                std::string se_key = "SE_" + std::to_string(i);
                std::string t_key = "t_value_" + std::to_string(i);
                standardErrors(i) = stats.find(se_key) != stats.end() ? stats[se_key] : 0.0;
                tValues(i) = stats.find(t_key) != stats.end() ? stats[t_key] : 0.0;
            }
            
            auto coeffStatsPlot = std::make_shared<CoefficientStatsPlot>(
                "Coefficient Statistics: Feature Importance and Significance",
                "Features",
                "Coefficient Values"
            );
            coeffStatsPlot->setData(featureNames, coefficients, standardErrors, tValues);
            plotManager_->addPlot(coeffStatsPlot);
        }
        
        // Add feature importance plot if applicable
        if (model_->hasFeatureImportance()) {
            auto featurePlot = std::make_shared<FeatureImportancePlot>(
                "Feature Importance Analysis",
                "Features",
                "Relative Importance"
            );
            featurePlot->setData(dataHandler_.getFeatureNames(), model_->getFeatureImportance());
            plotManager_->addPlot(featurePlot);
        }
    }

    // Add explanatory text before rendering the plot
    ImGui::TextWrapped("Use the navigation buttons below to cycle through different visualizations of the model results:");
    ImGui::BulletText("Time Series Plot: Shows how well the model predictions match actual values over time");
    ImGui::BulletText("Scatter Plot: Displays the correlation between actual and predicted values");
    ImGui::BulletText("Residual Analysis: Helps identify patterns in prediction errors");
    if (selectedModelIndex_ == 0 || selectedModelIndex_ == 1) {
        ImGui::BulletText("Coefficient Statistics: Shows the significance and impact of each feature");
    }
    if (model_->hasFeatureImportance()) {
        ImGui::BulletText("Feature Importance: Visualizes the relative importance of each input feature");
    }
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Render plots
    plotManager_->render();

    // Navigation buttons
    ImGui::Separator();
    if (ImGui::Button("Back to Hyperparameters")) {
        setScreen(Screen::HYPERPARAMETERS);
    }
    ImGui::SameLine();
    if (ImGui::MenuItem("New", "Ctrl+N")) {
        // Reset all state
        selectedFilePath_.clear();
        outliers_.clear();
        selectedModelIndex_ = 0;
        selectedFeatures_.clear();
        selectedTargetIndices_.clear();
        model_.reset();
        if (plotManager_) {
            plotManager_->reset();
        }
        predictions_ = Eigen::VectorXd();
        setScreen(Screen::FILE_BROWSER);
    }
}

} // namespace DataAnalyzer 