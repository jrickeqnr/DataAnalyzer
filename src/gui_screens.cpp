#include "../include/gui.h"
#include <imgui.h>
#include <implot.h>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <filesystem>

namespace fs = std::filesystem;

namespace DataAnalyzer {

// Helper to display a tooltip when the mouse hovers over a "(?)" text
static void HelpMarker(const char* desc) {
    ImGui::TextDisabled("(?)");
    if (ImGui::IsItemHovered()) {
        ImGui::BeginTooltip();
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
        ImGui::TextUnformatted(desc);
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }
}

void GUI::renderFileBrowser() {
    ImGui::Text("File Browser");
    ImGui::Separator();
    
    ImGui::TextWrapped("Select a CSV file to load data for analysis. The file should contain numeric columns and at least one date column.");
    ImGui::Spacing();
    
    static bool showFileDialog = false;
    
    if (ImGui::Button("Browse Files", ImVec2(120, 0))) {
        showFileDialog = true;
    }
    
    if (showFileDialog) {
        if (ImGuiFileDialog::fileDialog(selectedFilePath_, "Select CSV File", ".csv")) {
            showFileDialog = false;
            
            // Load the selected file
            if (!selectedFilePath_.empty()) {
                ImGui::Text("Loading file: %s", selectedFilePath_.c_str());
                
                // Load the CSV file
                if (dataHandler_.loadCSV(selectedFilePath_)) {
                    ImGui::Text("File loaded successfully!");
                    
                    // Automatically proceed to outlier detection
                    setScreen(Screen::OUTLIER_DETECTION);
                } else {
                    ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "Error loading file!");
                    selectedFilePath_.clear();
                }
            }
        }
    }
    
    if (!selectedFilePath_.empty()) {
        ImGui::Text("Selected File: %s", selectedFilePath_.c_str());
        
        // Display data summary
        ImGui::Spacing();
        ImGui::TextWrapped("%s", dataHandler_.getDataSummary().c_str());
        
        ImGui::Spacing();
        if (ImGui::Button("Next: Outlier Detection", ImVec2(200, 0))) {
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
    
    ImGui::TextWrapped("Detect outliers in the data using the IQR method. Outliers are defined as values outside Q1 - 1.5*IQR and Q3 + 1.5*IQR.");
    ImGui::Spacing();
    
    static bool showOutliers = false;
    static bool detectPressed = false;
    
    if (!detectPressed) {
        if (ImGui::Button("Detect Outliers", ImVec2(150, 0))) {
            // Detect outliers in all numeric columns
            outliers_ = dataHandler_.detectOutliers();
            showOutliers = true;
            detectPressed = true;
        }
    }
    
    if (showOutliers) {
        int totalOutliers = 0;
        for (const auto& [col, rows] : outliers_) {
            totalOutliers += rows.size();
        }
        
        ImGui::Spacing();
        ImGui::Text("Outlier Detection Results:");
        ImGui::Text("Total outliers found: %d", totalOutliers);
        
        // Display outliers by column
        std::vector<std::string> columnNames = dataHandler_.getColumnNames();
        std::vector<size_t> numericIndices = dataHandler_.getNumericColumnIndices();
        
        if (ImGui::BeginTable("OutliersTable", 3, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
            ImGui::TableSetupColumn("Column");
            ImGui::TableSetupColumn("# Outliers");
            ImGui::TableSetupColumn("Example Rows");
            ImGui::TableHeadersRow();
            
            for (const auto& [col, rows] : outliers_) {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                
                // Find the column name
                std::string colName = "Unknown";
                if (col < numericIndices.size()) {
                    size_t colIdx = numericIndices[col];
                    if (colIdx < columnNames.size()) {
                        colName = columnNames[colIdx];
                    }
                }
                
                ImGui::Text("%s", colName.c_str());
                
                ImGui::TableSetColumnIndex(1);
                ImGui::Text("%zu", rows.size());
                
                ImGui::TableSetColumnIndex(2);
                // Show a few example row indices
                std::string rowsStr;
                for (size_t i = 0; i < std::min(rows.size(), size_t(5)); ++i) {
                    rowsStr += std::to_string(rows[i]);
                    if (i < std::min(rows.size(), size_t(5)) - 1) {
                        rowsStr += ", ";
                    }
                }
                if (rows.size() > 5) {
                    rowsStr += "...";
                }
                ImGui::Text("%s", rowsStr.c_str());
            }
            
            ImGui::EndTable();
        }
        
        ImGui::Spacing();
        if (totalOutliers > 0) {
            if (ImGui::Button("Fix Outliers", ImVec2(120, 0))) {
                // Fix outliers by interpolating with previous/next values
                if (dataHandler_.fixOutliers(outliers_)) {
                    ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "Outliers fixed successfully!");
                } else {
                    ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "Error fixing outliers!");
                }
            }
            
            ImGui::SameLine();
            if (ImGui::Button("Skip (Keep Outliers)", ImVec2(200, 0))) {
                // Keep outliers and proceed to the next screen
                ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Keeping outliers in the data.");
            }
        } else {
            ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "No outliers detected!");
        }
        
        ImGui::Spacing();
        if (ImGui::Button("Next: Model Selection", ImVec2(200, 0))) {
            setScreen(Screen::MODEL_SELECTION);
        }
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
    
    ImGui::TextWrapped("Select a model for data analysis. Currently, only Elastic Net is available.");
    ImGui::Spacing();
    
    // Model selection (for now, just ElasticNet)
    const char* models[] = {"Elastic Net"};
    static int selectedModel = 0;
    
    ImGui::Combo("Model", &selectedModel, models, IM_ARRAYSIZE(models));
    selectedModelIndex_ = selectedModel;
    
    // Display model description
    ImGui::Spacing();
    ImGui::TextWrapped("Elastic Net Regression combines L1 and L2 regularization to balance sparsity and stability. It's effective for feature selection and handling correlated predictors. The alpha parameter controls the mix of L1 and L2 regularization, while lambda controls the overall regularization strength.");
    
    // Display model diagram (simplified)
    ImGui::Spacing();
    if (ImGui::CollapsingHeader("Model Diagram", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Indent(20.0f);
        ImGui::TextWrapped("Input Features → [Elastic Net] → Prediction");
        ImGui::TextWrapped("Loss = MSE + α·λ·L1 + (1-α)·λ·L2");
        ImGui::Unindent(20.0f);
    }
    
    // Buttons for navigation
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
    
    ImGui::TextWrapped("Select features (independent variables) and the target (dependent variable) for the model.");
    ImGui::Spacing();
    
    // Get column names and indices
    const std::vector<std::string>& columnNames = dataHandler_.getColumnNames();
    const std::vector<size_t>& numericIndices = dataHandler_.getNumericColumnIndices();
    
    // Initialize selected features if empty
    if (selectedFeatures_.empty()) {
        // Default: select all numeric columns except the last one
        for (size_t i = 0; i < numericIndices.size() - 1; ++i) {
            selectedFeatures_.push_back(numericIndices[i]);
        }
        
        // Default target: last numeric column
        if (!numericIndices.empty()) {
            selectedTargetIndex_ = numericIndices.back();
        }
    }
    
    // Feature selection
    if (ImGui::CollapsingHeader("Feature Selection", ImGuiTreeNodeFlags_DefaultOpen)) {
        // Select All / None buttons
        if (ImGui::Button("Select All")) {
            for (size_t idx : numericIndices) {
                if (idx != selectedTargetIndex_) {
                    if (std::find(selectedFeatures_.begin(), selectedFeatures_.end(), idx) == selectedFeatures_.end()) {
                        selectedFeatures_.push_back(idx);
                    }
                }
            }
        }
        
        ImGui::SameLine();
        if (ImGui::Button("Select None")) {
            selectedFeatures_.clear();
        }
        
        ImGui::Spacing();
        
        // Display checkboxes for each numeric column
        for (size_t idx : numericIndices) {
            if (idx != selectedTargetIndex_ && idx < columnNames.size()) {
                bool isSelected = std::find(selectedFeatures_.begin(), selectedFeatures_.end(), idx) != selectedFeatures_.end();
                if (ImGui::Checkbox(columnNames[idx].c_str(), &isSelected)) {
                    if (isSelected) {
                        selectedFeatures_.push_back(idx);
                    } else {
                        selectedFeatures_.erase(std::remove(selectedFeatures_.begin(), selectedFeatures_.end(), idx), selectedFeatures_.end());
                    }
                }
            }
        }
    }
    
    // Target selection
    if (ImGui::CollapsingHeader("Target Selection", ImGuiTreeNodeFlags_DefaultOpen)) {
        // Create a vector of target names
        std::vector<const char*> targetNames;
        for (size_t idx : numericIndices) {
            if (idx < columnNames.size()) {
                targetNames.push_back(columnNames[idx].c_str());
            }
        }
        
        // Display target dropdown
        if (!targetNames.empty()) {
            // Find the index in numericIndices for the selected target
            int selectedTargetIdx = 0;
            for (size_t i = 0; i < numericIndices.size(); ++i) {
                if (numericIndices[i] == selectedTargetIndex_) {
                    selectedTargetIdx = i;
                    break;
                }
            }
            
            if (ImGui::Combo("Target Variable", &selectedTargetIdx, targetNames.data(), targetNames.size())) {
                selectedTargetIndex_ = numericIndices[selectedTargetIdx];
                
                // Remove target from features if it was selected
                selectedFeatures_.erase(std::remove(selectedFeatures_.begin(), selectedFeatures_.end(), selectedTargetIndex_), selectedFeatures_.end());
            }
        } else {
            ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "No numeric columns available for target selection!");
        }
    }
    
    // Seasonality option
    if (ImGui::CollapsingHeader("Seasonality", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Include Seasonal Features", &includeSeasonality_);
        
        if (includeSeasonality_) {
            ImGui::Indent(20.0f);
            ImGui::TextWrapped("This will add sine and cosine features based on the detected frequency: %s",
                              dataHandler_.getFrequencyString().c_str());
            ImGui::Unindent(20.0f);
        }
    }
    
    // Buttons for navigation
    ImGui::Spacing();
    if (ImGui::Button("Back: Model Selection", ImVec2(200, 0))) {
        setScreen(Screen::MODEL_SELECTION);
    }
    
    ImGui::SameLine();
    
    // Validate that at least one feature is selected
    bool canProceed = !selectedFeatures_.empty();
    
    if (canProceed) {
        if (ImGui::Button("Next: Hyperparameters", ImVec2(200, 0))) {
            setScreen(Screen::HYPERPARAMETERS);
        }
    } else {
        ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
        ImGui::Button("Next: Hyperparameters", ImVec2(200, 0));
        ImGui::PopStyleVar();
        
        ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "Please select at least one feature!");
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
    
    ImGui::TextWrapped("Configure the hyperparameters for the Elastic Net model or use automatic tuning.");
    ImGui::Spacing();
    
    // Hyperparameter configuration
    static bool autoTune = false;
    ImGui::Checkbox("Automatic Hyperparameter Tuning", &autoTune);
    
    if (!autoTune) {
        ImGui::TextWrapped("Configure the alpha and lambda parameters manually:");
        ImGui::Spacing();
        
        static float alpha = 0.5f;  // Use float for ImGui SliderFloat
        alpha_ = alpha;  // Store in the class member
        ImGui::SliderFloat("Alpha", &alpha, 0.0f, 1.0f, "%.2f");
        ImGui::SameLine(); 
        HelpMarker("Alpha controls the balance between L1 and L2 regularization. 0 = Ridge, 1 = Lasso.");
        
        static float lambda = 0.1f;  // Use float for ImGui SliderFloat
        lambda_ = lambda;  // Store in the class member
        ImGui::SliderFloat("Lambda", &lambda, 0.001f, 10.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
        ImGui::SameLine(); 
        HelpMarker("Lambda controls the overall regularization strength. Higher values = more regularization.");
    } else {
        ImGui::TextWrapped("The model will automatically find the best hyperparameters using cross-validation.");
    }
    
    ImGui::Spacing();
    ImGui::Separator();
    
    // Train model button
    static bool modelTrained = false;
    static bool trainingFailed = false;
    
    if (!modelTrained) {
        if (ImGui::Button("Train Model", ImVec2(120, 0))) {
            // Prepare data for training
            Eigen::MatrixXd data = dataHandler_.getData();
            std::vector<size_t> indices = dataHandler_.getNumericColumnIndices();
            
            // Add seasonal features if requested
            if (includeSeasonality_) {
                dataHandler_.addSeasonalFeatures();
                data = dataHandler_.getData(); // Get updated data with seasonal features
                indices = dataHandler_.getNumericColumnIndices(); // Get updated indices
            }
            
            // Extract features and target
            std::vector<int> featureIndices;
            for (size_t idx : selectedFeatures_) {
                // Find the corresponding column in the data matrix
                auto it = std::find(indices.begin(), indices.end(), idx);
                if (it != indices.end()) {
                    featureIndices.push_back(it - indices.begin());
                }
            }
            
            // Find target index in the data matrix
            auto targetIt = std::find(indices.begin(), indices.end(), selectedTargetIndex_);
            if (targetIt == indices.end()) {
                trainingFailed = true;
                ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "Target column not found in numeric data!");
                return;
            }
            int targetIdx = targetIt - indices.begin();
            
            // Extract feature matrix and target vector
            Eigen::MatrixXd X(data.rows(), featureIndices.size());
            for (size_t i = 0; i < featureIndices.size(); ++i) {
                X.col(i) = data.col(featureIndices[i]);
            }
            Eigen::VectorXd y = data.col(targetIdx);
            
            // Create and train the model
            if (selectedModelIndex_ == 0) { // Elastic Net
                model_ = std::make_shared<ElasticNet>(alpha_, lambda_);
                
                if (autoHyperparameters_) {
                    // Grid search for best hyperparameters
                    std::vector<double> alphaValues = {0.0, 0.2, 0.5, 0.8, 1.0};
                    std::vector<double> lambdaValues = {0.001, 0.01, 0.1, 1.0, 10.0};
                    
                    auto elasticNet = std::dynamic_pointer_cast<ElasticNet>(model_);
                    auto bestParams = elasticNet->gridSearch(X, y, alphaValues, lambdaValues);
                    
                    alpha_ = bestParams.first;
                    lambda_ = bestParams.second;
                    
                    // Create a new model with the best hyperparameters
                    model_ = std::make_shared<ElasticNet>(alpha_, lambda_);
                }
                
                // Train the model
                if (model_->train(X, y)) {
                    // Generate predictions
                    predictions_ = model_->predict(X);
                    
                    // Create a time series plot
                    plot_ = std::make_shared<TimeSeries>("Actual vs Predicted", "Date", "Value");
                    plot_->setData(dataHandler_.getDates(), y, predictions_);
                    
                    modelTrained = true;
                } else {
                    trainingFailed = true;
                    ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "Model training failed!");
                }
            }
        }
    }
    
    if (modelTrained) {
        ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "Model trained successfully!");
        
        // Display model statistics
        ImGui::Spacing();
        ImGui::Text("Model Statistics:");
        
        for (const auto& [name, value] : model_->getStats()) {
            ImGui::Text("%s: %.4f", name.c_str(), value);
        }
        
        // Display hyperparameters
        ImGui::Spacing();
        ImGui::Text("Final Hyperparameters:");
        
        if (autoHyperparameters_) {
            ImGui::Text("(Automatically tuned)");
        }
        
        for (const auto& [name, value] : model_->getHyperparameters()) {
            ImGui::Text("%s: %.4f", name.c_str(), value);
        }
        
        // Coefficients preview
        ImGui::Spacing();
        if (ImGui::CollapsingHeader("Model Coefficients", ImGuiTreeNodeFlags_DefaultOpen)) {
            Eigen::VectorXd coefs = model_->getCoefficients();
            std::vector<std::string> colNames = dataHandler_.getColumnNames();
            
            if (ImGui::BeginTable("CoefficientsTable", 2, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
                ImGui::TableSetupColumn("Feature");
                ImGui::TableSetupColumn("Coefficient");
                ImGui::TableHeadersRow();
                
                for (size_t i = 0; i < coefs.size() && i < selectedFeatures_.size(); ++i) {
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);
                    
                    // Find the feature name
                    std::string featureName = "Feature " + std::to_string(i);
                    if (i < selectedFeatures_.size() && selectedFeatures_[i] < colNames.size()) {
                        featureName = colNames[selectedFeatures_[i]];
                    }
                    
                    ImGui::Text("%s", featureName.c_str());
                    
                    ImGui::TableSetColumnIndex(1);
                    ImGui::Text("%.4f", coefs(i));
                }
                
                ImGui::EndTable();
            }
        }
        
        ImGui::Spacing();
        if (ImGui::Button("View Plots", ImVec2(120, 0))) {
            setScreen(Screen::PLOTTING);
        }
    } else if (trainingFailed) {
        ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "Training failed! Check your data and parameters.");
    }
    
    // Buttons for navigation
    ImGui::Spacing();
    if (ImGui::Button("Back: Variable Selection", ImVec2(200, 0))) {
        setScreen(Screen::VARIABLE_SELECTION);
    }
}

void GUI::renderPlotting() {
    if (!model_ || !plot_) {
        ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "No model trained! Please train a model first.");
        if (ImGui::Button("Back to Hyperparameters", ImVec2(200, 0))) {
            setScreen(Screen::HYPERPARAMETERS);
        }
        return;
    }
    
    ImGui::Text("Results Visualization");
    ImGui::Separator();
    
    // Render the time series plot
    plot_->render();
    
    // Export section
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Text("Export Options:");
    
    static bool showExportDialog = false;
    
    // Export Button
    if (ImGui::Button("Export Results", ImVec2(150, 0))) {
        showExportDialog = true;
    }
    
    static std::string exportDir;
    
    // Show export dialog
    if (showExportDialog) {
        if (ImGuiFileDialog::fileDialog(exportDir, "Select Export Directory", "", true)) {
            showExportDialog = false;
            
            // Export data
            if (!exportDir.empty()) {
                exportResults(exportDir);
                ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "Results exported to %s", exportDir.c_str());
            }
        }
    }
    
    // Buttons for navigation
    ImGui::Spacing();
    if (ImGui::Button("Back: Hyperparameters", ImVec2(200, 0))) {
        setScreen(Screen::HYPERPARAMETERS);
    }
    
    ImGui::SameLine();
    if (ImGui::Button("New Analysis", ImVec2(120, 0))) {
        // Reset state
        selectedFilePath_.clear();
        outliers_.clear();
        selectedModelIndex_ = 0;
        selectedFeatures_.clear();
        selectedTargetIndex_ = 0;
        includeSeasonality_ = false;
        alpha_ = 0.5;
        lambda_ = 1.0;
        autoHyperparameters_ = false;
        predictions_.resize(0);
        model_.reset();
        plot_.reset();
        
        // Go back to file browser
        setScreen(Screen::FILE_BROWSER);
    }
}

} // namespace DataAnalyzer 