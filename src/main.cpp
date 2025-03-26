#include "../include/gui.h"
#include <iostream>

int main([[maybe_unused]] int argc, [[maybe_unused]] char** argv) {
    try {
        // Create the application
        DataAnalyzer::GUI app("DataAnalyzer");
        
        // Initialize the application
        if (!app.initialize()) {
            std::cerr << "Failed to initialize DataAnalyzer" << std::endl;
            return -1;
        }
        
        // Run the application
        app.run();
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "Unknown error occurred" << std::endl;
        return -1;
    }
} 