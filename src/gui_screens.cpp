#include "../include/gui.h"
#include <imgui.h>
#include <implot.h>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <filesystem>
#include <thread>
#include <mutex>

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
    
    ImGui::TextWrapped("Detect outliers in the data using the IQR method. Outliers are defined as values outside Q1 - 1.5*IQR and Q3 + 1.5*IQR.");
    ImGui::Spacing();
    
    static bool detectPressed = false;
    static bool showOutlierPopup = false;
    static std::string popupMessage;
    static int totalOutliers = 0;
    
    if (!detectPressed) {
        if (ImGui::Button("Detect Outliers", ImVec2(150, 0))) {
            // Detect outliers in all numeric columns
            outliers_ = dataHandler_.detectOutliers();
            detectPressed = true;
            
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
                    if (ImGui::BeginTable(("OutliersTable##" + std::to_string(col)).c_str(), outlierFixed ? 3 : 2, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
                        ImGui::TableSetupColumn("Date");
                        ImGui::TableSetupColumn(outlierFixed ? "Original Value" : "Value");
                        if (outlierFixed) {
                            ImGui::TableSetupColumn("Fixed Value");
                        }
                        ImGui::TableHeadersRow();
                        
                        for (size_t row : rows) {
                            ImGui::TableNextRow();
                            
                            // Date column
                            ImGui::TableSetColumnIndex(0);
                            if (row < dates.size()) {
                                ImGui::Text("%s", dates[row].toString().c_str());
                            } else {
                                ImGui::Text("Unknown Date");
                            }
                            
                            // Value column
                            ImGui::TableSetColumnIndex(1);
                            if (row < static_cast<size_t>(originalData.rows()) && col < static_cast<size_t>(originalData.cols())) {
                                ImGui::Text("%.2f", originalData(row, col));
                            } else {
                                ImGui::Text("Invalid Value");
                            }
                            
                            // Fixed value column (only shown after fixing)
                            if (outlierFixed) {
                                ImGui::TableSetColumnIndex(2);
                                if (row < static_cast<size_t>(dataHandler_.getData().rows()) && col < static_cast<size_t>(dataHandler_.getData().cols())) {
                                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 1.0f, 0.0f, 1.0f));
                                    ImGui::Text("%.2f", dataHandler_.getData()(row, col));
                                    ImGui::PopStyleColor();
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
            if (ImGui::Button("Fix Outliers", ImVec2(150, 0))) {
                if (dataHandler_.fixOutliers(outliers_)) {
                    showSuccessPopup("Outliers fixed successfully!");
                    outlierFixed = true;
                } else {
                    showErrorPopup("Failed to fix outliers!");
                }
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
    static bool isTraining = false;  // Add state variable for training
    ImGui::Checkbox("Automatic Hyperparameter Tuning", &autoTune);
    
    autoHyperparameters_ = autoTune;

    // Show status during training
    if (isTraining) {
        ImGui::TextColored(ImVec4(0.0f, 0.7f, 1.0f, 1.0f), "Tuning hyperparameters... This may take a moment.");
    }

    // Show best parameters if available during auto-tuning
    if (autoTune && model_) {
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
    
    // Show training progress section (always visible)
    ImGui::Spacing();
    ImGui::Separator();
    
    // Status message
    if (isTraining_) {
        if (autoHyperparameters_) {
            ImGui::TextColored(ImVec4(0.0f, 0.7f, 1.0f, 1.0f), "Training model with automatic hyperparameter tuning...");
        } else {
            ImGui::TextColored(ImVec4(0.0f, 0.7f, 1.0f, 1.0f), "Training model...");
        }
    } else if (trainingProgress_ >= 1.0) {
        ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "Training completed!");
    } else {
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Ready to train model");
    }
    
    // Progress bar
    ImGui::ProgressBar(trainingProgress_, ImVec2(-1, 0), "Training Progress");
    ImGui::SameLine();
    ImGui::Text("%.0f%%", trainingProgress_ * 100.0);
    
    // Update progress during training
    if (isTraining_ && model_) {
        auto stats = model_->getStats();
        if (stats.find("Training Progress") != stats.end()) {
            trainingProgress_ = stats["Training Progress"];
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
                
                // Reset training results
                {
                    std::lock_guard<std::mutex> lock(trainingMutex_);
                    modelStats_.clear();
                    modelHyperparams_.clear();
                    modelCoefficients_.resize(0);
                    featureNames_ = dataHandler_.getFeatureNames();
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
                            // Define expanded grid search values for XGBoost
                            std::vector<int> n_estimators_values = {50, 100, 200};  // Reduced from 5 to 3 values
                            std::vector<double> learning_rate_values = {0.01, 0.1, 0.3};  // Adjusted range
                            std::vector<int> max_depth_values = {3, 4, 5};  // Reduced and centered range
                            std::vector<double> subsample_values = {0.8, 1.0};  // Most common values
                            
                            // Create temporary model for grid search
                            XGBoost tempModel;
                            auto [best_n_estimators, best_learning_rate, best_max_depth, best_subsample] = 
                                tempModel.gridSearch(X, y, n_estimators_values, learning_rate_values, max_depth_values, subsample_values);
                            
                            // Create model with best hyperparameters
                            model_ = std::make_unique<XGBoost>(best_n_estimators, best_learning_rate, best_max_depth, best_subsample);
                            
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
                            
                            // Create temporary model for grid search
                            GradientBoosting tempModel;
                            auto [best_n_estimators, best_learning_rate, best_max_depth, best_min_samples_split] = 
                                tempModel.gridSearch(X, y, n_estimators_values, learning_rate_values, max_depth_values, min_samples_split_values);
                            
                            // Create model with best hyperparameters
                            model_ = std::make_unique<GradientBoosting>(best_n_estimators, best_learning_rate, best_max_depth, best_min_samples_split);
                            
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
                            
                            // Create temporary model for grid search
                            NeuralNetwork tempModel;
                            auto [best_layers, best_learning_rate, best_alpha, best_iterations] = 
                                tempModel.gridSearch(X, y, layer_configs, learning_rate_values, alpha_values, max_iterations_values);
                            
                            // Create model with best hyperparameters
                            model_ = std::make_unique<NeuralNetwork>(best_layers, best_learning_rate, best_iterations, best_alpha);
                            
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

                // Create a table for coefficients and their statistics
                if (ImGui::BeginTable("CoefficientsTable", 4, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
                    ImGui::TableSetupColumn("Feature");
                    ImGui::TableSetupColumn("Coefficient");
                    ImGui::TableSetupColumn("Std. Error");
                    ImGui::TableSetupColumn("t-value");
                    ImGui::TableHeadersRow();

                    for (size_t i = 0; i < featureNames_.size(); ++i) {
                        if (static_cast<Eigen::Index>(i) < modelCoefficients_.size()) {
                            ImGui::TableNextRow();
                            
                            // Feature name
                            ImGui::TableSetColumnIndex(0);
                            ImGui::Text("%s", featureNames_[i].c_str());
                            
                            // Coefficient value
                            ImGui::TableSetColumnIndex(1);
                            ImGui::Text("%.4f", modelCoefficients_(i));
                            
                            // Standard Error
                            ImGui::TableSetColumnIndex(2);
                            std::string se_key = "SE_" + std::to_string(i);
                            if (modelStats_.find(se_key) != modelStats_.end()) {
                                ImGui::Text("%.4f", modelStats_[se_key]);
                            } else {
                                ImGui::Text("-");
                            }
                            
                            // t-value
                            ImGui::TableSetColumnIndex(3);
                            std::string t_key = "t_value_" + std::to_string(i);
                            if (modelStats_.find(t_key) != modelStats_.end()) {
                                ImGui::Text("%.4f", modelStats_[t_key]);
                            } else {
                                ImGui::Text("-");
                            }
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