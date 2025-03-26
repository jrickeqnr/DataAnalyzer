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
        // Set the file dialog window position to be further right
        ImGui::SetNextWindowPos(ImVec2(ImGui::GetIO().DisplaySize.x * 0.3f, 0.0f));
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
                ImGui::TextWrapped("Model: y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ");
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
    
    // Training section
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Text("Model Training");
    
    // Check if we have valid selections
    bool canTrain = !selectedFeatures_.empty() && !selectedTargetIndices_.empty();
    
    if (canTrain) {
        // Train button
        if (ImGui::Button("Train Model", ImVec2(120, 0))) {
            // Prepare data for training
            Eigen::MatrixXd X = dataHandler_.getSelectedFeatures(selectedFeatures_);
            Eigen::VectorXd y;
            
            // For simplicity, we'll use the first selected target
            if (!selectedTargetIndices_.empty()) {
                y = dataHandler_.getSelectedTarget(selectedTargetIndices_[0]);
            }
            
            // Create model based on selection
            switch (selectedModelIndex_) {
                case 0: // Linear Regression
                    model_ = std::make_unique<LinearRegression>();
                    break;
                    
                case 1: // Elastic Net
                    model_ = std::make_unique<ElasticNet>(alpha_, lambda_);
                    break;
                    
                case 2: // XGBoost
                    model_ = std::make_unique<XGBoost>(n_estimators_, learning_rate_, max_depth_, subsample_);
                    break;
                    
                case 3: // Gradient Boosting
                    model_ = std::make_unique<GradientBoosting>(n_estimators_, learning_rate_, max_depth_);
                    break;
                    
                case 4: // Neural Network
                    // Create vector for hidden layer sizes
                    std::vector<int> hidden_layer_sizes(hidden_layers_, neurons_per_layer_);
                    model_ = std::make_unique<NeuralNetwork>(hidden_layer_sizes, learning_rate_, max_iterations_);
                    break;
            }
            
            // Train the model
            if (model_) {
                bool success = model_->train(X, y);
                
                if (success) {
                    // Get predictions
                    predictions_ = model_->predict(X);
                    
                    // Create plot
                    plot_ = std::make_unique<TimeSeries>("Model Predictions", 
                                                        "Time", 
                                                        dataHandler_.getColumnNames()[selectedTargetIndices_[0]]);
                    
                    // Get dates and actual values for plotting
                    std::vector<Date> dates = dataHandler_.getDates();
                    Eigen::VectorXd actual = y;
                    
                    // Set plot data
                    plot_->setData(dates, actual, predictions_);
                    
                    // Show success message
                    ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "Model trained successfully!");
                } else {
                    ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "Error training model!");
                }
            }
        }
        
        // Display model statistics if trained
        if (model_) {
            ImGui::Spacing();
            ImGui::Text("Model Statistics:");
            
            auto stats = model_->getStats();
            ImGui::Text("RMSE: %.4f", stats["RMSE"]);
            ImGui::Text("R²: %.4f", stats["R²"]);
            
            // Show model coefficients (if available)
            Eigen::VectorXd coefficients = model_->getCoefficients();
            if (coefficients.size() > 0) {
                ImGui::Spacing();
                ImGui::Text("Model Coefficients:");
                
                if (ImGui::BeginTable("CoefficientsTable", 2, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
                    ImGui::TableSetupColumn("Feature");
                    ImGui::TableSetupColumn("Coefficient");
                    ImGui::TableHeadersRow();
                    
                    // Show intercept
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);
                    ImGui::Text("Intercept");
                    ImGui::TableSetColumnIndex(1);
                    ImGui::Text("%.4f", coefficients(0));
                    
                    // Show coefficients for each feature
                    for (size_t i = 0; i < selectedFeatures_.size() && i + 1 < static_cast<size_t>(coefficients.size()); ++i) {
                        ImGui::TableNextRow();
                        ImGui::TableSetColumnIndex(0);
                        ImGui::Text("%s", dataHandler_.getColumnNames()[selectedFeatures_[i]].c_str());
                        ImGui::TableSetColumnIndex(1);
                        ImGui::Text("%.4f", coefficients(i + 1));
                    }
                    
                    ImGui::EndTable();
                }
            }
        }
    } else {
        ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "Cannot train: Please select input features and target variable first.");
    }
    
    // Navigation buttons
    ImGui::Spacing();
    if (ImGui::Button("Back: Variable Selection", ImVec2(200, 0))) {
        setScreen(Screen::VARIABLE_SELECTION);
    }
    
    ImGui::SameLine();
    if (ImGui::Button("Next: Plotting", ImVec2(200, 0))) {
        setScreen(Screen::PLOTTING);
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
    
    // Display model name and brief stats
    const char* modelTypes[] = {"Linear Regression", "Elastic Net", "XGBoost", "Gradient Boosting", "Neural Network"};
    ImGui::Text("Model: %s", modelTypes[selectedModelIndex_]);
    
    auto stats = model_->getStats();
    ImGui::Text("RMSE: %.4f", stats["RMSE"]);
    ImGui::Text("R²: %.4f", stats["R²"]);
    ImGui::Spacing();
    
    // Render the time series plot
    ImGui::BeginChild("PlotArea", ImVec2(0, ImGui::GetWindowHeight() * 0.6f), true);
    plot_->render();
    ImGui::EndChild();
    
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
    
    // Navigation buttons
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
        selectedTargetIndices_.clear();
        includeSeasonality_ = false;
        alpha_ = 0.5;
        lambda_ = 1.0;
        n_estimators_ = 100;
        learning_rate_ = 0.1;
        max_depth_ = 3;
        subsample_ = 0.8;
        hidden_layers_ = 1;
        neurons_per_layer_ = 10;
        max_iterations_ = 1000;
        autoHyperparameters_ = false;
        predictions_.resize(0);
        model_.reset();
        plot_.reset();
        
        // Go back to file browser
        setScreen(Screen::FILE_BROWSER);
    }
}

} // namespace DataAnalyzer 