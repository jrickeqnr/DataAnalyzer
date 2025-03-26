#include "../include/gui.h"
#include <imgui.h>
#include <implot.h>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <filesystem>

// Model includes
#include "../include/model.h"

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
    
    ImGui::TextWrapped("Detect outliers in the data using the IQR method. Outliers are defined as values outside Q1 - 1.5*IQR and Q3 + 1.5*IQR.");
    ImGui::Spacing();
    
    static bool detectPressed = false;
    
    if (!detectPressed) {
        if (ImGui::Button("Detect Outliers", ImVec2(150, 0))) {
            // Detect outliers in all numeric columns
            outliers_ = dataHandler_.detectOutliers();
            detectPressed = true;
        }
    }
    
    // Show results if detection has been performed
    if (detectPressed) {
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Text("Outlier Detection Results:");
        
        int totalOutliers = 0;
        for (const auto& [col, rows] : outliers_) {
            totalOutliers += rows.size();
        }
        
        if (totalOutliers == 0) {
            ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "No outliers detected in the data!");
        } else {
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
            if (ImGui::Button("Fix Outliers", ImVec2(120, 0))) {
                // Fix outliers by interpolating with previous/next values
                if (dataHandler_.fixOutliers(outliers_)) {
                    ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "Outliers fixed successfully!");
                    detectPressed = false; // Reset detection state
                } else {
                    ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "Error fixing outliers!");
                }
            }
            
            ImGui::SameLine();
            if (ImGui::Button("Skip (Keep Outliers)", ImVec2(200, 0))) {
                // Keep outliers and proceed to the next screen
                ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Keeping outliers in the data.");
            }
        }
    }
    
    // Navigation buttons
    ImGui::Spacing();
    if (ImGui::Button("Back: File Browser", ImVec2(200, 0))) {
        detectPressed = false; // Reset detection state when navigating away
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
        for (size_t i = 0; i < modelVariables.size() - 1; ++i) {
            selectedFeatures_.push_back(modelVariables[i]);
        }
        
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

    // Target Variable Selection (Right Column)
    ImGui::BeginChild("TargetVariableFrame", ImVec2(columnWidth, ImGui::GetWindowHeight() * 0.6f), true);
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
    static bool autoTune = false;
    ImGui::Checkbox("Automatic Hyperparameter Tuning", &autoTune);
    
    autoHyperparameters_ = autoTune;
    
    // Different hyperparameters based on model type
    if (!autoTune) {
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
                    
                    if (ImGui::SliderInt("Number of Estimators", &n_estimators, 10, 500)) {
                        n_estimators_ = n_estimators;
                    }
                    if (ImGui::SliderFloat("Learning Rate", &learning_rate, 0.01f, 1.0f, "%.2f")) {
                        learning_rate_ = static_cast<double>(learning_rate);
                    }
                    if (ImGui::SliderInt("Max Depth", &max_depth, 1, 10)) {
                        max_depth_ = max_depth;
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
    
    // Display current model results if available
    if (modelTrainingSuccess_) {
        ImGui::Separator();
        ImGui::Text("Model Results");
        ImGui::Separator();
        
        // Display R-squared
        if (modelStats_.find("R²") != modelStats_.end()) {
            ImGui::Text("R-squared: %.4f", modelStats_["R²"]);
        }
        
        // Display coefficients for linear regression
        if (selectedModelIndex_ == 0) {
            ImGui::Text("Model Coefficients:");
            ImGui::Indent(20.0f);
            
            // Display intercept
            ImGui::Text("Intercept: %.4f", modelIntercept_);
            
            // Display feature coefficients
            for (size_t i = 0; i < featureNames_.size(); ++i) {
                if (static_cast<Eigen::Index>(i) < modelCoefficients_.size()) {
                    ImGui::Text("%s: %.4f", featureNames_[i].c_str(), modelCoefficients_(i));
                }
            }
            ImGui::Unindent(20.0f);
        }
        
        // Display RMSE if available
        if (modelStats_.find("RMSE") != modelStats_.end()) {
            ImGui::Text("Root Mean Square Error: %.4f", modelStats_["RMSE"]);
        }
        
        // Show hyperparameter information if auto-tuning was used
        if (autoHyperparameters_) {
            ImGui::TextColored(ImVec4(0.0f, 0.7f, 1.0f, 1.0f), "Optimal hyperparameters were found via grid search.");
            ImGui::Text("Best hyperparameters:");
            for (const auto& [name, value] : modelHyperparams_) {
                ImGui::BulletText("%s: %.4f", name.c_str(), value);
            }
        }
    }
    
    ImGui::Spacing();
    if (ImGui::Button("Train Model", ImVec2(200, 0))) {
        // Show a "Tuning hyperparameters..." message if using automatic tuning
        if (autoHyperparameters_) {
            ImGui::TextColored(ImVec4(0.0f, 0.7f, 1.0f, 1.0f), "Tuning hyperparameters... This may take a moment.");
            
            // Simple loading spinner
            float time = ImGui::GetTime();
            const float radius = 10.0f;
            const ImVec2 center = ImGui::GetCursorScreenPos();
            ImDrawList* draw_list = ImGui::GetWindowDrawList();
            
            // Draw spinner
            for (int i = 0; i < 8; i++) {
                float a = time * 8.0f + i * 0.5f;
                float alpha = 1.0f - 0.125f * i;
                draw_list->AddCircleFilled(
                    ImVec2(center.x + std::cos(a) * radius, center.y + std::sin(a) * radius),
                    2.5f,
                    ImGui::GetColorU32(ImVec4(1.0f, 1.0f, 1.0f, alpha)),
                    8
                );
            }
            
            ImGui::Dummy(ImVec2(2 * radius + 5, 2 * radius + 5));
        }
        
        // Prepare data for training
        Eigen::MatrixXd X = dataHandler_.getSelectedFeatures(selectedFeatures_);
        Eigen::VectorXd y;
        
        // For simplicity, we'll use the first selected target
        if (!selectedTargetIndices_.empty()) {
            y = dataHandler_.getSelectedTarget(selectedTargetIndices_[0]);
        }
        
        // Reset training results
        modelTrainingSuccess_ = false;
        modelStats_.clear();
        modelHyperparams_.clear();
        modelCoefficients_.resize(0);
        featureNames_ = dataHandler_.getFeatureNames();
        modelIntercept_ = 0.0;
        
        // Create model based on selection
        switch (selectedModelIndex_) {
            case 0: // Linear Regression
                model_ = std::make_unique<LinearRegression>();
                break;
                
            case 1: // Elastic Net
                if (autoHyperparameters_) {
                    // Define grid search values for ElasticNet
                    std::vector<double> alpha_values = {0.0, 0.2, 0.4, 0.6, 0.8, 1.0};
                    std::vector<double> lambda_values = {0.001, 0.01, 0.1, 1.0, 10.0};
                    
                    // Create temporary model for grid search
                    ElasticNet tempModel;
                    auto [best_alpha, best_lambda] = tempModel.gridSearch(X, y, alpha_values, lambda_values);
                    
                    // Create model with best hyperparameters
                    model_ = std::make_unique<ElasticNet>(best_alpha, best_lambda);
                    
                    // Update UI values to reflect the best parameters
                    alpha_ = best_alpha;
                    lambda_ = best_lambda;
                } else {
                    model_ = std::make_unique<ElasticNet>(alpha_, lambda_);
                }
                break;
                
            case 2: // XGBoost
                if (autoHyperparameters_) {
                    // Define grid search values for XGBoost
                    std::vector<int> n_estimators_values = {50, 100, 200};
                    std::vector<double> learning_rate_values = {0.01, 0.05, 0.1, 0.2};
                    std::vector<int> max_depth_values = {3, 5, 7};
                    
                    // Create temporary model for grid search
                    XGBoost tempModel;
                    auto [best_n_estimators, best_learning_rate, best_max_depth] = 
                        tempModel.gridSearch(X, y, n_estimators_values, learning_rate_values, max_depth_values);
                    
                    // Create model with best hyperparameters
                    model_ = std::make_unique<XGBoost>(best_n_estimators, best_learning_rate, best_max_depth, subsample_);
                    
                    // Update UI values to reflect the best parameters
                    n_estimators_ = best_n_estimators;
                    learning_rate_ = best_learning_rate;
                    max_depth_ = best_max_depth;
                } else {
                    model_ = std::make_unique<XGBoost>(n_estimators_, learning_rate_, max_depth_, subsample_);
                }
                break;
                
            case 3: // Gradient Boosting
                if (autoHyperparameters_) {
                    // Define grid search values for Gradient Boosting
                    std::vector<int> n_estimators_values = {50, 100, 200};
                    std::vector<double> learning_rate_values = {0.01, 0.05, 0.1, 0.2};
                    std::vector<int> max_depth_values = {3, 5, 7};
                    
                    // Create temporary model for grid search
                    GradientBoosting tempModel;
                    auto [best_n_estimators, best_learning_rate, best_max_depth] = 
                        tempModel.gridSearch(X, y, n_estimators_values, learning_rate_values, max_depth_values);
                    
                    // Create model with best hyperparameters
                    model_ = std::make_unique<GradientBoosting>(best_n_estimators, best_learning_rate, best_max_depth);
                    
                    // Update UI values to reflect the best parameters
                    n_estimators_ = best_n_estimators;
                    learning_rate_ = best_learning_rate;
                    max_depth_ = best_max_depth;
                } else {
                    model_ = std::make_unique<GradientBoosting>(n_estimators_, learning_rate_, max_depth_);
                }
                break;
                
            case 4: // Neural Network
                if (autoHyperparameters_) {
                    // Define grid search values for Neural Network
                    std::vector<std::vector<int>> layer_configs = {
                        {10}, {20}, {10, 10}, {20, 10}
                    };
                    std::vector<double> learning_rate_values = {0.001, 0.01, 0.05, 0.1};
                    std::vector<double> max_iterations_values = {500, 1000, 2000};
                    
                    // Create temporary model for grid search
                    NeuralNetwork tempModel;
                    auto [best_layers, best_learning_rate, best_iterations] = 
                        tempModel.gridSearch(X, y, layer_configs, learning_rate_values, max_iterations_values);
                    
                    // Create model with best hyperparameters
                    model_ = std::make_unique<NeuralNetwork>(best_layers, best_learning_rate, best_iterations);
                    
                    // Update UI values to reflect the best parameters
                    hidden_layers_ = best_layers.size();
                    if (!best_layers.empty()) {
                        neurons_per_layer_ = best_layers[0]; // Simplification: use first layer size
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
        
        // Train the model
        if (model_) {
            bool success = model_->train(X, y);
            
            if (success) {
                modelTrainingSuccess_ = true;
                
                // Store model results
                modelStats_ = model_->getStats();
                modelHyperparams_ = model_->getHyperparameters();
                modelCoefficients_ = model_->getCoefficients();
                
                // Get predictions for plotting
                predictions_ = model_->predict(X);
                
                // Create plot
                plotManager_ = std::make_shared<PlotManager>();
                auto timeSeriesPlot = std::make_shared<TimeSeries>("Model Predictions", "Date", "Value");
                timeSeriesPlot->setData(dataHandler_.getDates(), dataHandler_.getTargetValues(), predictions_);
                plotManager_->addPlot(timeSeriesPlot);
                
                // Show success message
                ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "Model trained successfully!");
            } else {
                ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "Error training model!");
            }
        }
    }
    
    // Navigation buttons
    ImGui::Spacing();
    if (ImGui::Button("Back: Variable Selection", ImVec2(200, 0))) {
        setScreen(Screen::VARIABLE_SELECTION);
    }
    
    ImGui::SameLine();
    
    // Only enable Next button if model is trained
    if (model_) {
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

    // Create plots if they don't exist
    if (!plotManager_) {
        plotManager_ = std::make_shared<PlotManager>();
        
        // Add time series plot
        auto timeSeriesPlot = std::make_shared<TimeSeries>("Model Predictions", "Date", "Value");
        std::vector<Date> dates = dataHandler_.getDates();
        Eigen::VectorXd actual = dataHandler_.getTargetValues();
        timeSeriesPlot->setData(dates, actual, predictions_);
        plotManager_->addPlot(timeSeriesPlot);
        
        // Add scatter plot
        auto scatterPlot = std::make_shared<ScatterPlot>("Actual vs Predicted", "Actual", "Predicted");
        scatterPlot->setData(actual, predictions_);
        plotManager_->addPlot(scatterPlot);
        
        // Add feature importance plot if the model supports it
        if (model_->hasFeatureImportance()) {
            auto featurePlot = std::make_shared<FeatureImportancePlot>("Feature Importance", "Feature", "Importance");
            featurePlot->setData(dataHandler_.getFeatureNames(), model_->getFeatureImportance());
            plotManager_->addPlot(featurePlot);
        }
    }

    // Render plots
    plotManager_->render();

    // Export options
    ImGui::Separator();
    ImGui::Text("Export Results:");
    
    if (ImGui::Button("Export Data")) {
        ImGuiFileDialog::fileDialog(exportDir_, "Select Export Directory", ".*", true);
    }
    
    if (!exportDir_.empty()) {
        exportResults(exportDir_);
        exportDir_.clear();
    }
    
    // Navigation buttons
    ImGui::Separator();
    if (ImGui::Button("Back to Hyperparameters")) {
        setScreen(Screen::HYPERPARAMETERS);
    }
    ImGui::SameLine();
    if (ImGui::Button("Start New Analysis")) {
        // Reset state
        selectedFilePath_.clear();
        outliers_.clear();
        selectedModelIndex_ = 0;
        selectedFeatures_.clear();
        selectedTargetIndices_.clear();
        includeSeasonality_ = false;
        model_.reset();
        plotManager_.reset();
        predictions_ = Eigen::VectorXd();
        setScreen(Screen::FILE_BROWSER);
    }
}

} // namespace DataAnalyzer 