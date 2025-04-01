#pragma once

#include <string>
#include <fstream>
#include <mutex>
#include <memory>
#include <sstream>

namespace DataAnalyzer {

enum class LogLevel {
    DEBUG,
    INFO,
    WARNING,
    ERROR,
    CRITICAL
};

class Logger {
public:
    static Logger& getInstance();
    
    void setLogFile(const std::string& filepath);
    void setLogLevel(LogLevel level);
    
    void debug(const std::string& message);
    void info(const std::string& message);
    void warning(const std::string& message);
    void error(const std::string& message);
    void critical(const std::string& message);
    
    // Delete copy constructor and assignment operator
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

private:
    Logger(); // Private constructor for singleton
    ~Logger();
    
    void log(LogLevel level, const std::string& message);
    std::string getTimestamp() const;
    std::string levelToString(LogLevel level) const;
    
    std::ofstream logFile_;
    LogLevel currentLevel_;
    std::mutex mutex_;
    bool consoleOutput_;
    std::string logFilePath_;
};

// Convenience macro for logging
#define LOG_DEBUG(msg) DataAnalyzer::Logger::getInstance().debug(msg)
#define LOG_INFO(msg) DataAnalyzer::Logger::getInstance().info(msg)
#define LOG_WARNING(msg) DataAnalyzer::Logger::getInstance().warning(msg)
#define LOG_ERROR(msg) DataAnalyzer::Logger::getInstance().error(msg)
#define LOG_CRITICAL(msg) DataAnalyzer::Logger::getInstance().critical(msg)

} // namespace DataAnalyzer 