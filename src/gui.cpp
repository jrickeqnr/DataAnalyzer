#include "../include/gui.h"
#include <imgui.h>
#include "../lib/imgui/backends/imgui_impl_glfw.h"
#include "../lib/imgui/backends/imgui_impl_opengl3.h"
#include <implot.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include <map>

namespace fs = std::filesystem;

namespace DataAnalyzer {

// GLFW error callback
static void glfw_error_callback(int error, const char* description) {
    std::cerr << "GLFW Error " << error << ": " << description << std::endl;
}

// ImGui file dialog helper functions
namespace ImGuiFileDialog {
    bool fileDialog(std::string& selectedPath, const std::string& title, const std::string& filter, bool onlyDirectories) {
        static std::string currentPath = fs::current_path().string();
        static std::string selectedFile;
        
        bool selected = false;
        
        if (ImGui::Begin(title.c_str(), nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            // Current path
            ImGui::Text("Current Directory: %s", currentPath.c_str());
            ImGui::Separator();
            
            // Parent directory button
            if (ImGui::Button("..")) {
                fs::path parentPath = fs::path(currentPath).parent_path();
                if (parentPath != currentPath) {
                    currentPath = parentPath.string();
                    selectedFile.clear();
                }
            }
            
            ImGui::SameLine();
            
            // Home directory button
            if (ImGui::Button("Home")) {
                currentPath = fs::path(getenv("HOME")).string();
                selectedFile.clear();
            }
            
            ImGui::SameLine();
            
            // Refresh button
            if (ImGui::Button("Refresh")) {
                selectedFile.clear();
            }
            
            ImGui::Separator();
            
            // Display directories and files
            std::error_code ec;
            std::vector<fs::directory_entry> directories;
            std::vector<fs::directory_entry> files;
            
            try {
                // Collect directories and files
                for (const auto& entry : fs::directory_iterator(currentPath, ec)) {
                    if (entry.is_directory(ec)) {
                        directories.push_back(entry);
                    } else if (entry.is_regular_file(ec) && !onlyDirectories) {
                        std::string ext = entry.path().extension().string();
                        if (filter == ".*" || ext == filter) {
                            files.push_back(entry);
                        }
                    }
                }
                
                // Sort alphabetically
                std::sort(directories.begin(), directories.end(), 
                    [](const fs::directory_entry& a, const fs::directory_entry& b) {
                        return a.path().filename().string() < b.path().filename().string();
                    });
                
                std::sort(files.begin(), files.end(), 
                    [](const fs::directory_entry& a, const fs::directory_entry& b) {
                        return a.path().filename().string() < b.path().filename().string();
                    });
                
                // Display directories
                if (ImGui::BeginChild("Directories", ImVec2(400, 200), true)) {
                    for (const auto& dir : directories) {
                        std::string dirName = dir.path().filename().string();
                        
                        if (ImGui::Selectable(dirName.c_str(), selectedFile == dirName)) {
                            if (onlyDirectories) {
                                selectedFile = dirName;
                                selectedPath = (fs::path(currentPath) / dirName).string();
                            } else {
                                selectedFile.clear();
                                currentPath = (fs::path(currentPath) / dirName).string();
                            }
                        }
                    }
                    
                    // Display files if not only showing directories
                    if (!onlyDirectories) {
                        for (const auto& file : files) {
                            std::string fileName = file.path().filename().string();
                            
                            if (ImGui::Selectable(fileName.c_str(), selectedFile == fileName)) {
                                selectedFile = fileName;
                                selectedPath = (fs::path(currentPath) / fileName).string();
                            }
                        }
                    }
                }
                ImGui::EndChild();
            } catch (const std::exception& e) {
                ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "Error: %s", e.what());
            }
            
            // Selected file display
            ImGui::Text("Selected: %s", selectedFile.c_str());
            
            // Cancel and OK buttons
            ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 10);
            
            if (ImGui::Button("Cancel", ImVec2(120, 0))) {
                ImGui::End();
                return false;
            }
            
            ImGui::SameLine();
            
            bool okEnabled = !selectedFile.empty() || (onlyDirectories && fs::is_directory(currentPath, ec));
            
            if (okEnabled) {
                if (ImGui::Button("OK", ImVec2(120, 0))) {
                    if (onlyDirectories && selectedFile.empty()) {
                        selectedPath = currentPath;
                    }
                    selected = true;
                }
            } else {
                ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
                ImGui::Button("OK", ImVec2(120, 0));
                ImGui::PopStyleVar();
            }
        }
        ImGui::End();
        
        return selected;
    }
}

// GUI Implementation
GUI::GUI(const std::string& title, int width, int height)
    : title_(title), width_(width), height_(height), window_(nullptr),
      currentScreen_(Screen::FILE_BROWSER) {
}

GUI::~GUI() {
    // Cleanup
    ImPlot::DestroyContext();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    
    if (window_) {
        glfwDestroyWindow(window_);
    }
    
    glfwTerminate();
}

bool GUI::initialize() {
    // Setup GLFW
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit()) {
        return false;
    }
    
    // GL 3.3 + GLSL 330
    const char* glsl_version = "#version 330";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
    
    // Create window with graphics context
    window_ = glfwCreateWindow(width_, height_, title_.c_str(), nullptr, nullptr);
    if (window_ == nullptr) {
        return false;
    }
    
    glfwMakeContextCurrent(window_);
    glfwSwapInterval(1); // Enable vsync
    
    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();
    
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;  // Enable Keyboard Controls
    
    // Setup UTF-8 encoding
    io.Fonts->AddFontFromFileTTF("lib/imgui/misc/fonts/DroidSansFallback.ttf", 16.0f, nullptr, io.Fonts->GetGlyphRangesChineseFull());
    
    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window_, true);
    ImGui_ImplOpenGL3_Init(glsl_version);
    
    // Setup ImGui style
    setupImGuiStyle();
    
    return true;
}

void GUI::run() {
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
    
    // Main loop
    while (!glfwWindowShouldClose(window_)) {
        // Poll and handle events
        glfwPollEvents();
        
        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        
        // Create a full window
        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
        ImGui::Begin("DataAnalyzer", nullptr, 
            ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | 
            ImGuiWindowFlags_NoMove | ImGuiWindowFlags_MenuBar);
        
        // Render main menu
        renderMainMenu();
        
        // Render the current screen
        switch (currentScreen_) {
            case Screen::FILE_BROWSER:
                renderFileBrowser();
                break;
            case Screen::OUTLIER_DETECTION:
                renderOutlierDetection();
                break;
            case Screen::MODEL_SELECTION:
                renderModelSelection();
                break;
            case Screen::VARIABLE_SELECTION:
                renderVariableSelection();
                break;
            case Screen::HYPERPARAMETERS:
                renderHyperparameters();
                break;
            case Screen::PLOTTING:
                renderPlotting();
                break;
        }
        
        ImGui::End();
        
        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window_, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(clear_color.x, clear_color.y, clear_color.z, clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        
        glfwSwapBuffers(window_);
    }
}

void GUI::setScreen(Screen screen) {
    currentScreen_ = screen;
}

Screen GUI::getCurrentScreen() const {
    return currentScreen_;
}

void GUI::setupImGuiStyle() {
    ImGuiStyle& style = ImGui::GetStyle();
    
    // Set a blue theme
    ImVec4* colors = style.Colors;
    colors[ImGuiCol_Text]                   = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);
    colors[ImGuiCol_TextDisabled]           = ImVec4(0.50f, 0.50f, 0.50f, 1.00f);
    colors[ImGuiCol_WindowBg]               = ImVec4(0.06f, 0.06f, 0.06f, 0.94f);
    colors[ImGuiCol_ChildBg]                = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_PopupBg]                = ImVec4(0.08f, 0.08f, 0.08f, 0.94f);
    colors[ImGuiCol_Border]                 = ImVec4(0.43f, 0.43f, 0.50f, 0.50f);
    colors[ImGuiCol_BorderShadow]           = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_FrameBg]                = ImVec4(0.16f, 0.29f, 0.48f, 0.54f);
    colors[ImGuiCol_FrameBgHovered]         = ImVec4(0.26f, 0.59f, 0.98f, 0.40f);
    colors[ImGuiCol_FrameBgActive]          = ImVec4(0.26f, 0.59f, 0.98f, 0.67f);
    colors[ImGuiCol_TitleBg]                = ImVec4(0.04f, 0.04f, 0.04f, 1.00f);
    colors[ImGuiCol_TitleBgActive]          = ImVec4(0.16f, 0.29f, 0.48f, 1.00f);
    colors[ImGuiCol_TitleBgCollapsed]       = ImVec4(0.00f, 0.00f, 0.00f, 0.51f);
    colors[ImGuiCol_MenuBarBg]              = ImVec4(0.14f, 0.14f, 0.14f, 1.00f);
    colors[ImGuiCol_ScrollbarBg]            = ImVec4(0.02f, 0.02f, 0.02f, 0.53f);
    colors[ImGuiCol_ScrollbarGrab]          = ImVec4(0.31f, 0.31f, 0.31f, 1.00f);
    colors[ImGuiCol_ScrollbarGrabHovered]   = ImVec4(0.41f, 0.41f, 0.41f, 1.00f);
    colors[ImGuiCol_ScrollbarGrabActive]    = ImVec4(0.51f, 0.51f, 0.51f, 1.00f);
    colors[ImGuiCol_CheckMark]              = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
    colors[ImGuiCol_SliderGrab]             = ImVec4(0.24f, 0.52f, 0.88f, 1.00f);
    colors[ImGuiCol_SliderGrabActive]       = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
    colors[ImGuiCol_Button]                 = ImVec4(0.26f, 0.59f, 0.98f, 0.40f);
    colors[ImGuiCol_ButtonHovered]          = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
    colors[ImGuiCol_ButtonActive]           = ImVec4(0.06f, 0.53f, 0.98f, 1.00f);
    colors[ImGuiCol_Header]                 = ImVec4(0.26f, 0.59f, 0.98f, 0.31f);
    colors[ImGuiCol_HeaderHovered]          = ImVec4(0.26f, 0.59f, 0.98f, 0.80f);
    colors[ImGuiCol_HeaderActive]           = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
    colors[ImGuiCol_Separator]              = ImVec4(0.43f, 0.43f, 0.50f, 0.50f);
    colors[ImGuiCol_SeparatorHovered]       = ImVec4(0.10f, 0.40f, 0.75f, 0.78f);
    colors[ImGuiCol_SeparatorActive]        = ImVec4(0.10f, 0.40f, 0.75f, 1.00f);
    colors[ImGuiCol_ResizeGrip]             = ImVec4(0.26f, 0.59f, 0.98f, 0.25f);
    colors[ImGuiCol_ResizeGripHovered]      = ImVec4(0.26f, 0.59f, 0.98f, 0.67f);
    colors[ImGuiCol_ResizeGripActive]       = ImVec4(0.26f, 0.59f, 0.98f, 0.95f);
    colors[ImGuiCol_Tab]                    = ImVec4(0.18f, 0.35f, 0.58f, 0.86f);
    colors[ImGuiCol_TabHovered]             = ImVec4(0.26f, 0.59f, 0.98f, 0.80f);
    colors[ImGuiCol_TabActive]              = ImVec4(0.20f, 0.41f, 0.68f, 1.00f);
    colors[ImGuiCol_TabUnfocused]           = ImVec4(0.07f, 0.10f, 0.15f, 0.97f);
    colors[ImGuiCol_TabUnfocusedActive]     = ImVec4(0.14f, 0.26f, 0.42f, 1.00f);
    colors[ImGuiCol_PlotLines]              = ImVec4(0.61f, 0.61f, 0.61f, 1.00f);
    colors[ImGuiCol_PlotLinesHovered]       = ImVec4(1.00f, 0.43f, 0.35f, 1.00f);
    colors[ImGuiCol_PlotHistogram]          = ImVec4(0.90f, 0.70f, 0.00f, 1.00f);
    colors[ImGuiCol_PlotHistogramHovered]   = ImVec4(1.00f, 0.60f, 0.00f, 1.00f);
    colors[ImGuiCol_TextSelectedBg]         = ImVec4(0.26f, 0.59f, 0.98f, 0.35f);
    colors[ImGuiCol_DragDropTarget]         = ImVec4(1.00f, 1.00f, 0.00f, 0.90f);
    colors[ImGuiCol_NavHighlight]           = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
    colors[ImGuiCol_NavWindowingHighlight]  = ImVec4(1.00f, 1.00f, 1.00f, 0.70f);
    colors[ImGuiCol_NavWindowingDimBg]      = ImVec4(0.80f, 0.80f, 0.80f, 0.20f);
    colors[ImGuiCol_ModalWindowDimBg]       = ImVec4(0.80f, 0.80f, 0.80f, 0.35f);
    
    // Adjust style
    style.FrameRounding = 4.0f;
    style.GrabRounding = 4.0f;
    style.WindowRounding = 4.0f;
    style.ScrollbarRounding = 4.0f;
    style.FramePadding = ImVec2(8, 4);
    style.ItemSpacing = ImVec2(10, 8);
    style.WindowPadding = ImVec2(10, 10);
}

void GUI::renderMainMenu() {
    if (ImGui::BeginMenuBar()) {
        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("Home", "Ctrl+H")) {
                setScreen(Screen::FILE_BROWSER);
            }
            
            if (ImGui::MenuItem("New", "Ctrl+N")) {
                // Reset all state
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
            
            if (ImGui::MenuItem("Export Results", "Ctrl+S", false, model_ != nullptr)) {
                // Show a file dialog to select a directory
                static std::string exportDir;
                static bool showExportDialog = false;
                
                showExportDialog = true;
                
                if (showExportDialog) {
                    if (ImGuiFileDialog::fileDialog(exportDir, "Select Export Directory", "", true)) {
                        showExportDialog = false;
                        exportResults(exportDir);
                    }
                }
            }
            
            ImGui::Separator();
            
            if (ImGui::MenuItem("Exit", "Ctrl+Q")) {
                glfwSetWindowShouldClose(window_, GLFW_TRUE);
            }
            
            ImGui::EndMenu();
        }
        
        if (ImGui::BeginMenu("View")) {
            if (ImGui::MenuItem("File Browser", nullptr, currentScreen_ == Screen::FILE_BROWSER)) {
                setScreen(Screen::FILE_BROWSER);
            }
            
            if (ImGui::MenuItem("Outlier Detection", nullptr, currentScreen_ == Screen::OUTLIER_DETECTION, 
                                !selectedFilePath_.empty())) {
                setScreen(Screen::OUTLIER_DETECTION);
            }
            
            if (ImGui::MenuItem("Model Selection", nullptr, currentScreen_ == Screen::MODEL_SELECTION,
                                !selectedFilePath_.empty())) {
                setScreen(Screen::MODEL_SELECTION);
            }
            
            if (ImGui::MenuItem("Variable Selection", nullptr, currentScreen_ == Screen::VARIABLE_SELECTION,
                                !selectedFilePath_.empty())) {
                setScreen(Screen::VARIABLE_SELECTION);
            }
            
            if (ImGui::MenuItem("Hyperparameters", nullptr, currentScreen_ == Screen::HYPERPARAMETERS,
                                !selectedFilePath_.empty() && !selectedFeatures_.empty())) {
                setScreen(Screen::HYPERPARAMETERS);
            }
            
            if (ImGui::MenuItem("Plotting", nullptr, currentScreen_ == Screen::PLOTTING,
                                model_ != nullptr)) {
                setScreen(Screen::PLOTTING);
            } else if (!model_) {
                ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "Please train a model first!");
            }
            
            ImGui::EndMenu();
        }
        
        if (ImGui::BeginMenu("Help")) {
            if (ImGui::MenuItem("About")) {
                // Show about dialog
                ImGui::OpenPopup("About DataAnalyzer");
            }
            
            ImGui::EndMenu();
        }
        
        // Status information on the right
        ImGui::SetCursorPosX(ImGui::GetWindowWidth() - 150);
        ImGui::Text("Screen: %d/6", static_cast<int>(currentScreen_) + 1);
        
        ImGui::EndMenuBar();
    }
    
    // About dialog
    if (ImGui::BeginPopupModal("About DataAnalyzer", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::Text("DataAnalyzer v1.0");
        ImGui::Separator();
        ImGui::Text("A portable C++ data analysis program.");
        ImGui::Text("Libraries:");
        ImGui::BulletText("Dear ImGui & ImPlot for GUI and plotting");
        ImGui::BulletText("Eigen for matrix operations");
        ImGui::BulletText("GLFW for window management");
        ImGui::BulletText("fast-cpp-csv-parser for CSV parsing");
        
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 10);
        if (ImGui::Button("OK", ImVec2(120, 0))) {
            ImGui::CloseCurrentPopup();
        }
        
        ImGui::EndPopup();
    }
}

void GUI::exportResults(const std::string& dirPath) const {
    if (!model_ || dirPath.empty()) {
        return;
    }
    
    // Create directory
    fs::create_directories(dirPath);
    
    // Export predictions as CSV
    if (plotManager_) {
        // Export each plot's data
        for (size_t i = 0; i < plotManager_->getPlotCount(); ++i) {
            auto plot = plotManager_->getPlot(i);
            if (plot) {
                std::string filename = dirPath + "/plot_" + std::to_string(i) + ".csv";
                plot->saveImage(filename);
                std::cout << "Exported plot " << i << " to " << filename << std::endl;
            }
        }
    }
    
    // Export model stats as TXT
    std::ofstream statsFile(dirPath + "/model_stats.txt");
    if (statsFile.is_open()) {
        statsFile << "Model Statistics\n";
        statsFile << "================\n\n";
        
        // Write model type
        const char* modelTypes[] = {"Linear Regression", "Elastic Net", "XGBoost", "Gradient Boosting", "Neural Network"};
        statsFile << "Model Type: " << modelTypes[selectedModelIndex_] << "\n\n";
        
        // Write hyperparameters
        statsFile << "Hyperparameters:\n";
        statsFile << "----------------\n";
        auto hyperparams = model_->getHyperparameters();
        for (const auto& [name, value] : hyperparams) {
            statsFile << name << ": " << value << "\n";
        }
        statsFile << "\n";
        
        // Write model statistics
        statsFile << "Model Statistics:\n";
        statsFile << "----------------\n";
        auto stats = model_->getStats();
        for (const auto& [name, value] : stats) {
            statsFile << name << ": " << value << "\n";
        }
        
        statsFile.close();
        std::cout << "Exported model statistics to " << dirPath << "/model_stats.txt" << std::endl;
    }
}
} 