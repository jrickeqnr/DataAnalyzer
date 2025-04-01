#include "../include/logger.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <filesystem>

namespace DataAnalyzer {

Logger::Logger() 
    : currentLevel_(LogLevel::INFO)
    , consoleOutput_(true) {
}

Logger::~Logger() {
    if (logFile_.is_open()) {
        logFile_.close();
    }
}

Logger& Logger::getInstance() {
    static Logger instance;
    return instance;
}

void Logger::setLogFile(const std::string& filepath) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (logFile_.is_open()) {
        logFile_.close();
    }
    
    // Create directories if they don't exist
    std::filesystem::path path(filepath);
    if (auto dir = path.parent_path(); !dir.empty()) {
        std::filesystem::create_directories(dir);
    }
    
    logFile_.open(filepath, std::ios::app);
    logFilePath_ = filepath;
    
    if (!logFile_.is_open()) {
        std::cerr << "Failed to open log file: " << filepath << std::endl;
    }
}

void Logger::setLogLevel(LogLevel level) {
    std::lock_guard<std::mutex> lock(mutex_);
    currentLevel_ = level;
}

void Logger::debug(const std::string& message) {
    log(LogLevel::DEBUG, message);
}

void Logger::info(const std::string& message) {
    log(LogLevel::INFO, message);
}

void Logger::warning(const std::string& message) {
    log(LogLevel::WARNING, message);
}

void Logger::error(const std::string& message) {
    log(LogLevel::ERROR, message);
}

void Logger::critical(const std::string& message) {
    log(LogLevel::CRITICAL, message);
}

void Logger::log(LogLevel level, const std::string& message) {
    if (level < currentLevel_) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    std::stringstream ss;
    ss << getTimestamp() << " [" << levelToString(level) << "] " << message << std::endl;
    
    if (consoleOutput_) {
        std::cout << ss.str();
    }
    
    if (logFile_.is_open()) {
        logFile_ << ss.str();
        logFile_.flush();
    }
}

std::string Logger::getTimestamp() const {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S");
    ss << '.' << std::setfill('0') << std::setw(3) << ms.count();
    return ss.str();
}

std::string Logger::levelToString(LogLevel level) const {
    switch (level) {
        case LogLevel::DEBUG:    return "DEBUG";
        case LogLevel::INFO:     return "INFO";
        case LogLevel::WARNING:  return "WARNING";
        case LogLevel::ERROR:    return "ERROR";
        case LogLevel::CRITICAL: return "CRITICAL";
        default:                 return "UNKNOWN";
    }
}

} // namespace DataAnalyzer 