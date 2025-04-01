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
    
    // Overloaded methods with source info
    void debug(const std::string& message, const std::string& source);
    void info(const std::string& message, const std::string& source);
    void warning(const std::string& message, const std::string& source);
    void error(const std::string& message, const std::string& source);
    void critical(const std::string& message, const std::string& source);
    
    // Delete copy constructor and assignment operator
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

private:
    Logger(); // Private constructor for singleton
    ~Logger();
    
    void log(LogLevel level, const std::string& message);
    void log(LogLevel level, const std::string& message, const std::string& source);
    std::string getTimestamp() const;
    std::string levelToString(LogLevel level) const;
    
    std::ofstream logFile_;
    LogLevel currentLevel_;
    std::mutex mutex_;
    bool consoleOutput_;
    std::string logFilePath_;
};

// Basic convenience macros for logging
#define LOG_DEBUG(msg) DataAnalyzer::Logger::getInstance().debug(msg)
#define LOG_INFO(msg) DataAnalyzer::Logger::getInstance().info(msg)
#define LOG_WARNING(msg) DataAnalyzer::Logger::getInstance().warning(msg)
#define LOG_ERROR(msg) DataAnalyzer::Logger::getInstance().error(msg)
#define LOG_CRITICAL(msg) DataAnalyzer::Logger::getInstance().critical(msg)

// Enhanced macros with source information
#define LOG_DEBUG_SRC(msg, src) DataAnalyzer::Logger::getInstance().debug(msg, src)
#define LOG_INFO_SRC(msg, src) DataAnalyzer::Logger::getInstance().info(msg, src)
#define LOG_WARNING_SRC(msg, src) DataAnalyzer::Logger::getInstance().warning(msg, src)
#define LOG_ERROR_SRC(msg, src) DataAnalyzer::Logger::getInstance().error(msg, src)
#define LOG_CRITICAL_SRC(msg, src) DataAnalyzer::Logger::getInstance().critical(msg, src)

// Class-specific logging macros (usage: LOG_CLASS_INFO("MyClass", "message"))
#define LOG_CLASS_DEBUG(cls, msg) DataAnalyzer::Logger::getInstance().debug(msg, cls)
#define LOG_CLASS_INFO(cls, msg) DataAnalyzer::Logger::getInstance().info(msg, cls)
#define LOG_CLASS_WARNING(cls, msg) DataAnalyzer::Logger::getInstance().warning(msg, cls)
#define LOG_CLASS_ERROR(cls, msg) DataAnalyzer::Logger::getInstance().error(msg, cls)
#define LOG_CLASS_CRITICAL(cls, msg) DataAnalyzer::Logger::getInstance().critical(msg, cls)

// File-specific logging macros (using __FILE__ macro)
#define LOG_FILE_DEBUG(msg) DataAnalyzer::Logger::getInstance().debug(msg, __FILE__)
#define LOG_FILE_INFO(msg) DataAnalyzer::Logger::getInstance().info(msg, __FILE__)
#define LOG_FILE_WARNING(msg) DataAnalyzer::Logger::getInstance().warning(msg, __FILE__)
#define LOG_FILE_ERROR(msg) DataAnalyzer::Logger::getInstance().error(msg, __FILE__)
#define LOG_FILE_CRITICAL(msg) DataAnalyzer::Logger::getInstance().critical(msg, __FILE__)

} // namespace DataAnalyzer 