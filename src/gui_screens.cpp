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
    
    static bool detectPressed = false;
    
    if (!detectPressed) {
        if (ImGui::Button("Detect Outliers", ImVec2(150, 0))) {
            // Detect outliers in all numeric columns
            outliers_ = dataHandler_.detectOutliers();
            detectPressed = true;
        }
    }
    
    // Navigation buttons
    ImGui::Spacing();
    if (ImGui::Button("Back: File Browser", ImVec2(200, 0))) {
        setScreen(Screen::FILE_BROWSER);
    }
    
    ImGui::SameLine();
    if (ImGui::Button("Next: Model Selection", ImVec2(200, 0))) {
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
    ImGui::Text("Select Features:");
    ImGui::Indent(20.0f);
    
    for (size_t idx : numericIndices) {
        bool isSelected = std::find(selectedFeatures_.begin(), selectedFeatures_.end(), idx) != selectedFeatures_.end();
        bool isTarget = idx == selectedTargetIndex_;
        
        if (ImGui::Checkbox(columnNames[idx].c_str(), &isSelected)) {
            if (isSelected) {
                if (std::find(selectedFeatures_.begin(), selectedFeatures_.end(), idx) == selectedFeatures_.end()) {
                    selectedFeatures_.push_back(idx);
                }
            } else {
                selectedFeatures_.erase(
                    std::remove(selectedFeatures_.begin(), selectedFeatures_.end(), idx),
                    selectedFeatures_.end()
                );
            }
        }
        
        ImGui::SameLine();
        int targetIdx = static_cast<int>(idx);
        if (ImGui::RadioButton(("Target##" + std::to_string(idx)).c_str(), &targetIdx, static_cast<int>(selectedTargetIndex_))) {
            selectedTargetIndex_ = static_cast<size_t>(targetIdx);
            // If the target is selected as a feature, remove it from features
            if (std::find(selectedFeatures_.begin(), selectedFeatures_.end(), idx) != selectedFeatures_.end()) {
                selectedFeatures_.erase(
                    std::remove(selectedFeatures_.begin(), selectedFeatures_.end(), idx),
                    selectedFeatures_.end()
                );
            }
        }
    }
    
    ImGui::Unindent(20.0f);
    
    // Navigation buttons
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
        float alpha = static_cast<float>(alpha_);
        float lambda = static_cast<float>(lambda_);
        
        if (ImGui::SliderFloat("Alpha (L1/L2 Mix)", &alpha, 0.0f, 1.0f, "%.2f")) {
            alpha_ = static_cast<double>(alpha);
        }
        if (ImGui::SliderFloat("Lambda (Regularization)", &lambda, 0.0f, 10.0f, "%.2f")) {
            lambda_ = static_cast<double>(lambda);
        }
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