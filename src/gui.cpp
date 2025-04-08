#include "../include/gui.h"
#include "../include/logger.h"
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
#ifdef __APPLE__
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif

// Windows-specific includes for dark mode title bar
#ifdef _WIN32
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#include <Windows.h>
#include <dwmapi.h>
#include <uxtheme.h>  // For SetWindowTheme
#pragma comment(lib, "dwmapi.lib")
#pragma comment(lib, "uxtheme.lib")  // Link uxtheme.lib
// Define DWMWA_CAPTION_COLOR if not defined
#ifndef DWMWA_CAPTION_COLOR
#define DWMWA_CAPTION_COLOR 35
#endif
#endif

// For icon loading with stb_image
#define STB_IMAGE_IMPLEMENTATION
#include "../lib/stb/stb_image.h"

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
        static std::string newFolderName;
        static bool showNewFolderInput = false;
        
        bool selected = false;
        
        if (ImGui::Begin(title.c_str(), nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            // Current path
            ImGui::Text("Current Directory: %s", currentPath.c_str());
            ImGui::Separator();
            
            // Navigation buttons
            if (ImGui::Button("..")) {
                fs::path parentPath = fs::path(currentPath).parent_path();
                if (parentPath != currentPath) {
                    currentPath = parentPath.string();
                    selectedFile.clear();
                }
            }
            
            ImGui::SameLine();
            
            if (ImGui::Button("Home")) {
                currentPath = fs::path(getenv("HOME")).string();
                selectedFile.clear();
            }
            
            ImGui::SameLine();
            
            if (ImGui::Button("Refresh")) {
                selectedFile.clear();
            }
            
            ImGui::SameLine();
            
            if (ImGui::Button("New Folder")) {
                showNewFolderInput = true;
                newFolderName.clear();
            }
            
            ImGui::Separator();
            
            // New folder input
            if (showNewFolderInput) {
                ImGui::Text("New Folder Name:");
                static char buffer[256] = {0};
                bool createFolder = false;
                
                // Update newFolderName whenever text changes, not just on Enter
                if (ImGui::InputText("##newFolder", buffer, sizeof(buffer), ImGuiInputTextFlags_EnterReturnsTrue)) {
                    createFolder = true;
                }
                // Always update newFolderName with current buffer contents
                newFolderName = buffer;
                
                ImGui::SameLine();
                if (ImGui::Button("Create")) {
                    createFolder = true;
                }
                
                if (createFolder && !newFolderName.empty()) {
                    try {
                        fs::path newFolderPath = fs::path(currentPath) / newFolderName;
                        if (fs::create_directory(newFolderPath)) {
                            showNewFolderInput = false;
                            newFolderName.clear();
                            memset(buffer, 0, sizeof(buffer));
                        }
                    } catch (const std::exception& e) {
                        ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "Error creating folder: %s", e.what());
                    }
                }
                
                if (ImGui::Button("Cancel")) {
                    showNewFolderInput = false;
                    newFolderName.clear();
                    memset(buffer, 0, sizeof(buffer));
                }
                ImGui::Separator();
            }
            
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
                                currentPath = selectedPath;  // Update current path when directory is selected
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
            
            // Selected file/directory display
            ImGui::Text("Selected: %s", selectedFile.c_str());
            
            // OK button
            ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 10);
            
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
} // namespace ImGuiFileDialog

// GUI Implementation
GUI::GUI(const std::string& title, int width, int height)
    : title_(title), width_(width), height_(height), window_(nullptr),
      currentScreen_(Screen::FILE_BROWSER), plotManager_(std::make_shared<PlotManager>()) {
}

GUI::~GUI() {
    // Stop any ongoing training
    shouldStopTraining_ = true;
    if (trainingThread_.joinable()) {
        trainingThread_.join();
    }
    
    // Cleanup ImGui
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    
    // Cleanup GLFW
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

#ifdef _WIN32
    // Tell Windows we want dark mode before creating the window
    // This helps ensure consistent appearance from the start
    SetEnvironmentVariableA("ENABLE_DARK_MODE", "1");
    
    // Add these hints to indicate we want a dark-themed window
    glfwWindowHint(GLFW_DECORATED, GLFW_TRUE);
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
    // io.Fonts->AddFontFromFileTTF("lib/imgui/misc/fonts/DroidSansFallback.ttf", 16.0f, nullptr, io.Fonts->GetGlyphRangesChineseFull());
    
    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window_, true);
    ImGui_ImplOpenGL3_Init(glsl_version);
    
    // Setup ImGui style
    setupImGuiStyle();
    
#ifdef _WIN32
    // Apply dark mode to the window (after ImGui setup)
    HWND hwnd = glfwGetWin32Window(window_);
    if (hwnd) {
        // Apply dark mode with multiple approaches for better compatibility
        
        // Approach 1: Use the modern dark mode API (Win10 19041+)
        BOOL dark_mode = TRUE;
        DWORD attr_new = 20; // DWMWA_USE_IMMERSIVE_DARK_MODE (newer builds)
        HRESULT hr = DwmSetWindowAttribute(hwnd, attr_new, &dark_mode, sizeof(dark_mode));
        
        // Approach 2: Use the older dark mode API (Pre-19041)
        if (FAILED(hr)) {
            DWORD attr_old = 19; // DWMWA_USE_IMMERSIVE_DARK_MODE (older builds)
            hr = DwmSetWindowAttribute(hwnd, attr_old, &dark_mode, sizeof(dark_mode));
        }
        
        // Approach 3: Try to set the caption color directly
        if (FAILED(hr)) {
            COLORREF darkColor = RGB(32, 32, 32);
            DwmSetWindowAttribute(hwnd, DWMWA_CAPTION_COLOR, &darkColor, sizeof(darkColor));
        }
        
        // Approach 4: Set window theme for additional elements like the top bar controls
        SetWindowTheme(hwnd, L"DarkMode_Explorer", NULL);
        
        // Ensure window is fully updated
        InvalidateRect(hwnd, NULL, TRUE);
        UpdateWindow(hwnd);
        
        // Approach 5: Hack to force Windows to refresh the title bar
        // This simulates a window activation/deactivation cycle to refresh the appearance
        RECT windowRect;
        GetWindowRect(hwnd, &windowRect);
        
        // Force a redraw by moving the window slightly and then back
        SetWindowPos(hwnd, NULL, 
                     windowRect.left + 1, windowRect.top,
                     0, 0, SWP_NOSIZE | SWP_NOZORDER | SWP_FRAMECHANGED);
        Sleep(10); // Brief delay to let Windows process the change
        
        SetWindowPos(hwnd, NULL, 
                     windowRect.left, windowRect.top,
                     0, 0, SWP_NOSIZE | SWP_NOZORDER | SWP_FRAMECHANGED);
        
        // Approach 6: Simulate focus events
        SendMessage(hwnd, WM_ACTIVATE, WA_INACTIVE, 0);
        SendMessage(hwnd, WM_ACTIVATE, WA_ACTIVE, 0);
        
        // Ensure window gets focus
        SetForegroundWindow(hwnd);
        SetActiveWindow(hwnd);
        
        // Force a brief delay to allow Windows to apply the theme
        Sleep(50);
        
        // Do an initial render cycle to propagate the dark theme
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window_, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window_);
    }
#endif

    // Set window icon using both the resource and manual method for redundancy
    // The resource method (from app_icon.rc) should work for taskbar icons
    // The manual method ensures the window icon shows correctly
    try {
        GLFWimage icons[1];
        std::vector<std::string> possibleIconPaths = {
            "assets/EQNR.png",                                  // Relative to current working directory
            "../assets/EQNR.png",                               // One level up
            fs::current_path().string() + "/assets/EQNR.png",   // Absolute path to current directory
            fs::current_path().string() + "/../assets/EQNR.png" // One level up from current directory
        };
        
        // Log current working directory to help debugging
        std::cout << "Current working directory: " << fs::current_path().string() << std::endl;
        
        bool iconLoaded = false;
        
        // Try each possible path
        for (const auto& iconPath : possibleIconPaths) {
            LOG_INFO("GUI","Trying icon path:" + iconPath);

            
            if (fs::exists(iconPath)) {
                LOG_INFO("GUI","Found icon at: " + iconPath);
                
                int channels;
                icons[0].pixels = stbi_load(iconPath.c_str(), &icons[0].width, &icons[0].height, &channels, 4);
                
                if (icons[0].pixels) {
                    // Use GLFW to set the window icon - this works for the window decoration
                    glfwSetWindowIcon(window_, 1, icons);
                    
                    // Free the image data
                    stbi_image_free(icons[0].pixels);
                    
                    iconLoaded = true;
                    LOG_CLASS_INFO("GUI","Successfully loaded icon from " + iconPath);
                    break;
                } else {
                    LOG_CLASS_WARNING("GUI","Failed to load icon image from " + iconPath);
                }
            } else {
                LOG_CLASS_WARNING("GUI","Icon file not found at: " + iconPath);
            }
        }
        
        if (!iconLoaded) {
            LOG_CLASS_WARNING("GUI", "Failed to load icon from any location. Icon will not be diusplayed.");
        }
    } catch (const std::exception& e) {
        LOG_CLASS_WARNING("GUI","Error loading window icon");
        // Not critical, continue without icon
    }
    
    return true;
}

void GUI::run() {
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
    
#ifdef _WIN32
    // Schedule a delayed dark mode reapplication
    // This helps ensure the title bar is properly styled after the window is fully shown
    static bool needsDarkModeReapply = true;
    static int frameCount = 0;
#endif
    
    // Main loop
    while (!glfwWindowShouldClose(window_)) {
        // Poll and handle events
        glfwPollEvents();
        
#ifdef _WIN32
        // Reapply dark mode after a few frames
        if (needsDarkModeReapply && frameCount++ > 5) {
            needsDarkModeReapply = false;
            
            HWND hwnd = glfwGetWin32Window(window_);
            if (hwnd) {
                // Apply dark mode settings
                BOOL dark_mode = TRUE;
                DWORD attr_new = 20; // DWMWA_USE_IMMERSIVE_DARK_MODE
                DwmSetWindowAttribute(hwnd, attr_new, &dark_mode, sizeof(dark_mode));
                
                // Force a window update
                SendMessage(hwnd, WM_NCACTIVATE, FALSE, 0);
                SendMessage(hwnd, WM_NCACTIVATE, TRUE, 0);
                
                // Ensure title bar is redrawn
                SetWindowPos(hwnd, NULL, 0, 0, 0, 0, 
                             SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_FRAMECHANGED);
            }
        }
#endif
        
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
    if (screen == Screen::FILE_BROWSER) {
        // Reset all state when going back to file browser
        selectedFilePath_.clear();
        outliers_.clear();
        selectedModelIndex_ = 0;
        selectedFeatures_.clear();
        selectedTargetIndices_.clear();
        model_.reset();
        if (plotManager_) {
            plotManager_->reset();
            plotManager_.reset();  // Reset the shared_ptr itself
        }
        predictions_ = Eigen::VectorXd();
    }
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
    colors[ImGuiCol_MenuBarBg]              = ImVec4(0.06f, 0.06f, 0.06f, 1.00f); // Darker menu bar to match Windows dark mode
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
    
#ifdef _WIN32
    // On Windows, ensure the menu bar has the correct color
    // This is important to prevent flickering when the app starts
    style.Colors[ImGuiCol_MenuBarBg] = ImVec4(0.06f, 0.06f, 0.06f, 1.00f);
#endif
}

void GUI::renderMainMenu() {
    // Static variables for export dialog
    static std::string exportDir;
    static ExportType exportType;
    static bool showFileDialog = false;
    static std::string dialogTitle;

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
                model_.reset();
                if (plotManager_) {
                    plotManager_->reset();
                    plotManager_.reset();  // Reset the shared_ptr itself
                }
                predictions_ = Eigen::VectorXd();
                setScreen(Screen::FILE_BROWSER);
            }
            
            ImGui::Separator();
            
            if (ImGui::BeginMenu("Export", model_ != nullptr)) {
                if (ImGui::MenuItem("Predicted values (csv)")) {
                    dialogTitle = "Select Export Directory for Predictions";
                    exportType = ExportType::PREDICTIONS;
                    showFileDialog = true;
                }
                if (ImGui::MenuItem("Model structure (txt)")) {
                    dialogTitle = "Select Export Directory for Model Structure";
                    exportType = ExportType::MODEL_STRUCTURE;
                    showFileDialog = true;
                }
                if (ImGui::MenuItem("Model results (txt)")) {
                    dialogTitle = "Select Export Directory for Results";
                    exportType = ExportType::MODEL_RESULTS;
                    showFileDialog = true;
                }
                if (ImGui::MenuItem("Plot data (csv)")) {
                    dialogTitle = "Select Export Directory for Plots";
                    exportType = ExportType::PLOT_DATA;
                    showFileDialog = true;
                }
                if (ImGui::MenuItem("All")) {
                    dialogTitle = "Select Export Directory for All Data";
                    exportType = ExportType::ALL;
                    showFileDialog = true;
                }
                
                ImGui::EndMenu();
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
            
            if (ImGui::MenuItem("Model Training", nullptr, currentScreen_ == Screen::HYPERPARAMETERS,
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
        ImGui::Text("Equinor Data Analysis");
        
        ImGui::EndMenuBar();
    }
    
    // Show file dialog if requested (outside of menu)
    if (showFileDialog) {
        if (ImGuiFileDialog::fileDialog(exportDir, dialogTitle, ".", true)) {
            // Dialog was closed with OK
            if (!exportDir.empty()) {
                exportResults(exportDir, exportType);
                exportDir.clear();
            }
            showFileDialog = false;
        }
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

void GUI::exportResults(const std::string& directory, ExportType type) {
    if (!model_) {
        showErrorPopup("No model available to export results.");
        return;
    }

    if (directory.empty()) {
        showErrorPopup("Invalid export directory.");
        return;
    }

    // Create the export directory if it doesn't exist
    std::filesystem::create_directories(directory);

    // Helper function to create subdirectory
    auto createSubDir = [](const std::string& baseDir, const std::string& subDir) {
        std::string path = baseDir + "/" + subDir;
        std::filesystem::create_directories(path);
        return path;
    };

    try {
        // Export predicted values
        if (type == ExportType::PREDICTIONS || type == ExportType::ALL) {
            std::string predDir = createSubDir(directory, "predictions");
            std::ofstream predFile(predDir + "/predictions.csv");
            
            // Get date column name and values
            std::string dateColumnName = "Date";
            std::vector<Date> dates = dataHandler_.getDates();
            std::vector<size_t> dateIndices = dataHandler_.getDateColumnIndices();
            if (!dateIndices.empty()) {
                dateColumnName = dataHandler_.getColumnNames()[dateIndices[0]];
            }
            
            // Write header
            predFile << dateColumnName << ",Predicted Value\n";
            
            // Write data
            for (Eigen::Index i = 0; i < predictions_.size(); ++i) {
                std::string dateStr = i < static_cast<Eigen::Index>(dates.size()) ? dates[i].toString() : "";
                predFile << dateStr << "," << predictions_[i] << "\n";
            }
        }

        // Export model structure
        if (type == ExportType::MODEL_STRUCTURE || type == ExportType::ALL) {
            std::string modelDir = createSubDir(directory, "model");
            std::ofstream structureFile(modelDir + "/structure.txt");
            structureFile << "Model Structure:\n";
            structureFile << model_->getDescription() << "\n\n";
            structureFile << "Hyperparameters:\n";
            auto hyperparams = model_->getHyperparameters();
            for (const auto& [name, value] : hyperparams) {
                structureFile << name << ": " << value << "\n";
            }

            // Add coefficients for linear regression models
            if (auto* linearModel = dynamic_cast<LinearRegression*>(model_.get())) {
                structureFile << "\nModel Coefficients:\n";
                Eigen::VectorXd coefficients = linearModel->getCoefficients();
                auto stats = linearModel->getStats();
                
                // Write intercept
                structureFile << "Intercept: " << stats["Intercept"] << "\n";
                if (stats.find("Intercept SE") != stats.end()) {
                    structureFile << "Intercept SE: " << stats["Intercept SE"] << "\n";
                }
                if (stats.find("Intercept t-value") != stats.end()) {
                    structureFile << "Intercept t-value: " << stats["Intercept t-value"] << "\n";
                }
                
                // Write feature coefficients
                const std::vector<std::string>& featureNames = dataHandler_.getFeatureNames();
                for (Eigen::Index i = 0; i < coefficients.size(); ++i) {
                    std::string featureName = i < static_cast<Eigen::Index>(featureNames.size()) ? 
                                            featureNames[i] : "Feature_" + std::to_string(i);
                    structureFile << featureName << ": " << coefficients(i) << "\n";
                    
                    // Write standard error and t-value if available
                    std::string se_key = "SE_" + std::to_string(i);
                    std::string t_key = "t_value_" + std::to_string(i);
                    if (stats.find(se_key) != stats.end()) {
                        structureFile << featureName << " SE: " << stats[se_key] << "\n";
                    }
                    if (stats.find(t_key) != stats.end()) {
                        structureFile << featureName << " t-value: " << stats[t_key] << "\n";
                    }
                }
            }
        }

        // Export model results
        if (type == ExportType::MODEL_RESULTS || type == ExportType::ALL) {
            std::string resultsDir = createSubDir(directory, "results");
            std::ofstream resultsFile(resultsDir + "/metrics.txt");
            resultsFile << "Model Performance Metrics:\n";
            auto stats = model_->getStats();
            for (const auto& [name, value] : stats) {
                resultsFile << name << ": " << value << "\n";
            }
        }

        // Export plot data
        if (type == ExportType::PLOT_DATA || type == ExportType::ALL) {
            std::string plotDataDir = createSubDir(directory, "plot_data");
            if (!plotManager_->exportPlotData(plotDataDir)) {
                throw std::runtime_error("Failed to export plot data");
            }
        }

        // Show success dialog
        std::string message = "Successfully exported ";
        switch (type) {
            case ExportType::PREDICTIONS:
                message += "predictions";
                break;
            case ExportType::MODEL_STRUCTURE:
                message += "model structure";
                break;
            case ExportType::MODEL_RESULTS:
                message += "model results";
                break;
            case ExportType::PLOT_DATA:
                message += "plot data";
                break;
            case ExportType::ALL:
                message += "all data";
                break;
        }
        message += " to " + directory;
        showSuccessPopup(message);
    } catch (const std::exception& e) {
        showErrorPopup("Error exporting results: " + std::string(e.what()));
    }
}

void GUI::showErrorPopup(const std::string& message) {
    ImGui::OpenPopup("Error");
    if (ImGui::BeginPopupModal("Error", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::Text("%s", message.c_str());
        if (ImGui::Button("OK")) {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
}

void GUI::showSuccessPopup(const std::string& message) {
    ImGui::OpenPopup("Success");
    if (ImGui::BeginPopupModal("Success", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::Text("%s", message.c_str());
        if (ImGui::Button("OK")) {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
}

} // namespace DataAnalyzer 