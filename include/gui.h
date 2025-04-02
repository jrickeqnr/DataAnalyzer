#pragma once

#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <Eigen/Dense>
#include <GLFW/glfw3.h>
#include "data_handler.h"
#include "model.h"
#include "plotting.h"
#include <thread>
#include <mutex>

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
    DATA_VIEWER,       // Screen 2: Data viewer
    MODEL_TRAINING,    // Screen 3: Model training
    PREDICTIONS,       // Screen 4: Predictions
    OUTLIER_DETECTION, // Screen 5: Detect outliers
    MODEL_SELECTION,   // Screen 6: Model selection
    VARIABLE_SELECTION,// Screen 7: Variable selection
    HYPERPARAMETERS,   // Screen 8: Hyperparameter input
    PLOTTING          // Screen 9: Plot window
};

enum class ExportType {
    PREDICTIONS,
    MODEL_STRUCTURE,
    MODEL_RESULTS,
    PLOT_DATA,
    ALL
};

/**
 * @brief Main GUI class for the application
 */
class GUI {
public:
    GUI() : currentScreen_(Screen::FILE_BROWSER),
            selectedModelIndex_(0),
            modelTrainingSuccess_(false),
            modelIntercept_(0.0),
            alpha_(0.5),
            lambda_(1.0),
            n_estimators_(100),
            learning_rate_(0.1),
            max_depth_(3),
            min_samples_split_(2),
            subsample_(1.0),
            hidden_layers_(2),
            neurons_per_layer_(10),
            max_iterations_(1000),
            lagValues_(0),
            seasonality_(0),
            autoHyperparameters_(false) {}
    
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

    void render();

private:
    std::string title_;
    int width_;
    int height_;
    GLFWwindow* window_ = nullptr;
    Screen currentScreen_ = Screen::FILE_BROWSER;
    
    // Data and model
    DataHandler dataHandler_;
    std::unique_ptr<Model> model_;
    std::shared_ptr<PlotManager> plotManager_;
    
    // State variables for each screen
    std::string selectedFilePath_;
    std::map<size_t, std::vector<size_t>> outliers_;
    std::map<size_t, std::vector<size_t>> selectedOutliers_; // To store selected outliers
    float outlierSensitivity_ = 1.5f; // Default sensitivity
    int selectedModelIndex_ = 0;
    std::vector<size_t> selectedFeatures_;
    std::vector<size_t> selectedTargetIndices_;
    
    // Model training results
    bool modelTrainingSuccess_ = false;
    std::map<std::string, double> modelStats_;
    std::map<std::string, double> modelHyperparams_;
    Eigen::VectorXd modelCoefficients_;
    std::vector<std::string> featureNames_;
    double modelIntercept_ = 0.0;
    
    // Model hyperparameters
    // Elastic Net
    double alpha_ = 0.5;
    double lambda_ = 1.0;
    
    // XGBoost & Gradient Boosting
    int n_estimators_ = 100;
    double learning_rate_ = 0.1;
    int max_depth_ = 3;
    int min_samples_split_ = 2;
    double subsample_ = 1.0;
    
    // Neural Network
    int hidden_layers_ = 2;
    int neurons_per_layer_ = 10;
    int max_iterations_ = 1000;
    
    // Time series options
    int lagValues_ = 0;
    int seasonality_ = 0;
    
    bool autoHyperparameters_ = false;
    Eigen::VectorXd predictions_;
    std::string exportDir_;
    
    // Training progress state
    double trainingProgress_ = 0.0;
    bool isTraining_ = false;
    std::thread trainingThread_;
    std::mutex trainingMutex_;
    bool shouldStopTraining_ = false;
    
    // Render methods for each screen
    void renderFileBrowser();
    void renderOutlierDetection();
    void renderModelSelection();
    void renderVariableSelection();
    void renderHyperparameters();
    void renderPlotting();
    
    // Helper methods
    void renderMainMenu();
    void exportResults(const std::string& directory, ExportType type);
    
    // ImGui style setup
    void setupImGuiStyle();

    void showErrorPopup(const std::string& message);
    void showSuccessPopup(const std::string& message);
};

} // namespace DataAnalyzer 