#pragma once

#include <Eigen/Dense>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <chrono>
#include <optional>

namespace DataAnalyzer {

/**
 * @brief Enumeration for data frequency
 */
enum class Frequency {
    DAILY,
    WEEKLY,
    MONTHLY,
    QUARTERLY,
    YEARLY,
    UNKNOWN
};

/**
 * @brief Class to represent a date
 */
class Date {
public:
    Date() = default;
    Date(int year, int month, int day);
    
    int getYear() const { return year_; }
    int getMonth() const { return month_; }
    int getDay() const { return day_; }
    
    static std::optional<Date> parse(const std::string& dateStr);
    std::string toString() const;
    
    // Convert to UNIX timestamp (seconds since epoch)
    double toTimestamp() const {
        std::tm timeinfo = {};
        timeinfo.tm_year = year_ - 1900;  // Years since 1900
        timeinfo.tm_mon = month_ - 1;     // Months since January (0-11)
        timeinfo.tm_mday = day_;          // Day of the month (1-31)
        timeinfo.tm_hour = 0;             // Hours since midnight (0-23)
        timeinfo.tm_min = 0;              // Minutes after the hour (0-59)
        timeinfo.tm_sec = 0;              // Seconds after the minute (0-59)
        
        #ifdef _WIN32
            return static_cast<double>(_mkgmtime(&timeinfo));
        #else
            return static_cast<double>(timegm(&timeinfo));
        #endif
    }
    
    // Compare dates
    bool operator<(const Date& other) const;
    bool operator==(const Date& other) const;
    bool operator!=(const Date& other) const { return !(*this == other); }

private:
    int year_ = 0;
    int month_ = 0;
    int day_ = 0;
};

/**
 * @brief Class to handle CSV data loading and preprocessing
 */
class DataHandler {
public:
    DataHandler() = default;
    ~DataHandler() = default;
    
    /**
     * @brief Load data from a CSV file
     * @param filepath Path to the CSV file
     * @return bool True if loading was successful
     */
    bool loadCSV(const std::string& filepath);
    
    /**
     * @brief Get the data as an Eigen matrix
     * @return Eigen::MatrixXd Data matrix (numeric columns only)
     */
    Eigen::MatrixXd getData() const;
    
    /**
     * @brief Get the column names
     * @return std::vector<std::string> Column names
     */
    std::vector<std::string> getColumnNames() const;
    
    /**
     * @brief Get the numeric column indices
     * @return std::vector<size_t> Indices of numeric columns
     */
    std::vector<size_t> getNumericColumnIndices() const;
    
    /**
     * @brief Get the date column indices
     * @return std::vector<size_t> Indices of date columns
     */
    std::vector<size_t> getDateColumnIndices() const;
    
    /**
     * @brief Detect the frequency of the data
     * @return Frequency The detected frequency
     */
    Frequency detectFrequency() const;
    
    /**
     * @brief Get a string representation of the frequency
     * @return std::string Frequency as a string
     */
    std::string getFrequencyString() const;
    
    /**
     * @brief Detect outliers in the data using the IQR method
     * @param columnIndices Indices of columns to check for outliers (default: all numeric columns)
     * @param sensitivity Multiplier for IQR to determine outlier threshold (default: 1.5)
     * @return std::map<size_t, std::vector<size_t>> Map of column indices to row indices of outliers
     */
    std::map<size_t, std::vector<size_t>> detectOutliers(
        const std::vector<size_t>& columnIndices = {},
        double sensitivity = 1.5) const;
    
    /**
     * @brief Fix outliers by interpolating with previous/next values
     * @param outliers Map of column indices to row indices of outliers
     * @param selectedOutliers Map of column indices to list of selected row indices to fix (if empty, fix all outliers)
     * @return bool True if fixing was successful
     */
    bool fixOutliers(const std::map<size_t, std::vector<size_t>>& outliers,
                     const std::map<size_t, std::vector<size_t>>& selectedOutliers = {});
    
    /**
     * @brief Add seasonal features based on the detected frequency
     * @param dateColumnIndex Index of the date column to use
     * @return bool True if features were added successfully
     */
    bool addSeasonalFeatures(size_t dateColumnIndex = 0);
    
    /**
     * @brief Get the data summary
     * @return std::string Summary of the data
     */
    std::string getDataSummary() const;
    
    /**
     * @brief Get the dates
     * @return std::vector<Date> Vector of dates
     */
    std::vector<Date> getDates() const;
    
    /**
     * @brief Get selected features as an Eigen matrix
     * @param featureIndices Indices of columns to use as features
     * @return Eigen::MatrixXd Matrix with selected features
     */
    Eigen::MatrixXd getSelectedFeatures(const std::vector<size_t>& featureIndices) const;
    
    /**
     * @brief Get selected target as an Eigen vector
     * @param targetIndex Index of column to use as target
     * @return Eigen::VectorXd Vector with selected target
     */
    Eigen::VectorXd getSelectedTarget(size_t targetIndex) const;
    
    /**
     * @brief Export the data to a CSV file
     * @param filepath Path to the output CSV file
     * @return bool True if export was successful
     */
    bool exportToCSV(const std::string& filepath) const;

    /**
     * @brief Set the selected features
     * @param features Vector of feature indices
     */
    void setSelectedFeatures(const std::vector<size_t>& features) {
        selectedFeatures_ = features;
    }

    /**
     * @brief Set the selected target indices
     * @param targets Vector of target indices
     */
    void setSelectedTargets(const std::vector<size_t>& targets) {
        selectedTargetIndices_ = targets;
    }

    /**
     * @brief Get the target values (first selected target)
     * @return Eigen::VectorXd Target values
     */
    Eigen::VectorXd getTargetValues() const {
        if (selectedTargetIndices_.empty()) {
            return Eigen::VectorXd();
        }
        return getSelectedTarget(selectedTargetIndices_[0]);
    }

    /**
     * @brief Get the feature names for the selected features
     * @return std::vector<std::string> Feature names
     */
    std::vector<std::string> getFeatureNames() const {
        std::vector<std::string> names;
        for (size_t idx : selectedFeatures_) {
            if (idx < columnNames_.size()) {
                names.push_back(columnNames_[idx]);
            }
        }
        return names;
    }

private:
    std::vector<std::string> columnNames_;
    std::vector<std::vector<std::string>> rawData_;
    Eigen::MatrixXd numericData_;
    std::vector<Date> dates_;
    std::vector<size_t> numericColumnIndices_;
    std::vector<size_t> dateColumnIndices_;
    Frequency frequency_ = Frequency::UNKNOWN;
    std::vector<size_t> selectedFeatures_;
    std::vector<size_t> selectedTargetIndices_;
    
    // Helper methods
    bool parseNumericData();
    bool parseDateColumns();
    Frequency computeFrequency() const;
};

} // namespace DataAnalyzer 