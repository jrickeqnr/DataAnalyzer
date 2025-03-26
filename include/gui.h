#pragma once

#include <memory>
#include <string>
#include <vector>
#include <functional>
#include "data_handler.h"
#include "model.h"
#include "plotting.h"

struct GLFWwindow;

namespace DataAnalyzer {

// Forward declaration of file dialog
namespace ImGuiFileDialog {
    bool fileDialog(std::string& selectedPath, const std::string& title, 
                   const std::string& filter = ".*", bool onlyDirectories = false);
}

/**
 * @brief Enumeration for application screens
 */
enum class Screen {
    FILE_BROWSER,      // Screen 1: File browser to load CSV
    OUTLIER_DETECTION, // Screen 2: Detect outliers
    MODEL_SELECTION,   // Screen 3: Model selection
    VARIABLE_SELECTION, // Screen 4: Variable selection
    HYPERPARAMETERS,   // Screen 5: Hyperparameter input
    PLOTTING           // Screen 6: Plot window
};

/**
 * @brief Main GUI class for the application
 */
class GUI {
public:
    GUI(const std::string& title = "DataAnalyzer", int width = 1280, int height = 720);
    ~GUI();
    
    /**
     * @brief Initialize the GUI
     * @return bool True if initialization was successful
     */
    bool initialize();
    
    /**
     * @brief Run the main loop
     */
    void run();
    
    /**
     * @brief Set the current screen
     * @param screen The screen to set
     */
    void setScreen(Screen screen);
    
    /**
     * @brief Get the current screen
     * @return Screen The current screen
     */
    Screen getCurrentScreen() const;

private:
    std::string title_;
    int width_;
    int height_;
    GLFWwindow* window_ = nullptr;
    Screen currentScreen_ = Screen::FILE_BROWSER;
    
    // Data and model
    DataHandler dataHandler_;
    std::shared_ptr<Model> model_;
    std::shared_ptr<TimeSeries> plot_;
    
    // State variables for each screen
    std::string selectedFilePath_;
    std::map<size_t, std::vector<size_t>> outliers_;
    size_t selectedModelIndex_ = 0;
    std::vector<size_t> selectedFeatures_;
    size_t selectedTargetIndex_ = 0;
    bool includeSeasonality_ = false;
    double alpha_ = 0.5;
    double lambda_ = 1.0;
    bool autoHyperparameters_ = false;
    Eigen::VectorXd predictions_;
    std::string exportDir_;
    
    // Render methods for each screen
    void renderFileBrowser();
    void renderOutlierDetection();
    void renderModelSelection();
    void renderVariableSelection();
    void renderHyperparameters();
    void renderPlotting();
    
    // Helper methods
    void renderMainMenu();
    void exportResults(const std::string& dirPath) const;
    
    // ImGui style setup
    void setupImGuiStyle();
};

} // namespace DataAnalyzer 