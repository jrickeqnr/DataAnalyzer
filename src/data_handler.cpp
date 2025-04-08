#include "../include/data_handler.h"
#include "../lib/fast-cpp-csv-parser/csv.h"
#include "../include/logger.h"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <cmath>
#include <ctime>
#include <numeric>
#include <regex>
#include <filesystem>
#include <set>

namespace DataAnalyzer {

// Date Implementation
Date::Date(int year, int month, int day)
    : year_(year), month_(month), day_(day) {
}

constexpr double PI = 3.14159265358979323846;

std::optional<Date> Date::parse(const std::string& dateStr) {
    // Try different date formats
    std::vector<std::regex> dateFormats = {
        std::regex(R"(\d{4}-\d{1,2}-\d{1,2})"),              // YYYY-MM-DD
        std::regex(R"(\d{4}/\d{1,2}/\d{1,2})"),              // YYYY/MM/DD
        std::regex(R"(\d{1,2}-\d{1,2}-\d{4})"),              // MM-DD-YYYY or DD-MM-YYYY
        std::regex(R"(\d{1,2}/\d{1,2}/\d{4})"),              // MM/DD/YYYY or DD/MM/YYYY
        std::regex(R"(\d{1,2}\.\d{1,2}\.\d{4})"),            // DD.MM.YYYY
        std::regex(R"(\d{4}-\d{1,2}-\d{1,2}T\d{1,2}:\d{2})") // ISO format partial
    };
    
    // Match date string against patterns
    for (const auto& regex : dateFormats) {
        std::smatch match;
        if (std::regex_search(dateStr, match, regex)) {
            std::string matchedDate = match[0];
            
            int year = 0, month = 0, day = 0;
            
            // Parse based on format
            if (matchedDate.find('-') != std::string::npos) {
                char delimiter = '-';
                // Check if it's YYYY-MM-DD format
                if (matchedDate.find_first_of(delimiter) > 2) { // YYYY-MM-DD
                    std::stringstream ss(matchedDate);
                    ss >> year;
                    ss.ignore(); // Skip delimiter
                    ss >> month;
                    ss.ignore(); // Skip delimiter
                    ss >> day;
                } else { // MM-DD-YYYY or DD-MM-YYYY
                    std::stringstream ss(matchedDate);
                    ss >> month;
                    ss.ignore(); // Skip delimiter
                    ss >> day;
                    ss.ignore(); // Skip delimiter
                    ss >> year;
                    
                    // Assume MM-DD-YYYY for simplicity (could be DD-MM-YYYY in some regions)
                    // For a more robust solution, we'd need additional context
                }
            } else if (matchedDate.find('/') != std::string::npos) {
                char delimiter = '/';
                // Check if it's YYYY/MM/DD format
                if (matchedDate.find_first_of(delimiter) > 2) { // YYYY/MM/DD
                    std::stringstream ss(matchedDate);
                    ss >> year;
                    ss.ignore(); // Skip delimiter
                    ss >> month;
                    ss.ignore(); // Skip delimiter
                    ss >> day;
                } else { // MM/DD/YYYY or DD/MM/YYYY
                    std::stringstream ss(matchedDate);
                    ss >> month;
                    ss.ignore(); // Skip delimiter
                    ss >> day;
                    ss.ignore(); // Skip delimiter
                    ss >> year;
                    
                    // Assume MM/DD/YYYY for simplicity
                }
            } else if (matchedDate.find('.') != std::string::npos) {
                // DD.MM.YYYY format
                std::stringstream ss(matchedDate);
                ss >> day;
                ss.ignore(); // Skip delimiter
                ss >> month;
                ss.ignore(); // Skip delimiter
                ss >> year;
            } else if (matchedDate.find('T') != std::string::npos) {
                // ISO format
                std::string datePart = matchedDate.substr(0, matchedDate.find('T'));
                std::stringstream ss(datePart);
                ss >> year;
                ss.ignore(); // Skip delimiter
                ss >> month;
                ss.ignore(); // Skip delimiter
                ss >> day;
            }
            
            // Validate date components
            if (year >= 1900 && year <= 2100 && month >= 1 && month <= 12 && day >= 1 && day <= 31) {
                return Date(year, month, day);
            }
        }
    }
    
    return std::nullopt; // Failed to parse
}

std::string Date::toString() const {
    std::stringstream ss;
    ss << year_ << '-' 
       << std::setw(2) << std::setfill('0') << month_ << '-'
       << std::setw(2) << std::setfill('0') << day_;
    return ss.str();
}

bool Date::operator<(const Date& other) const {
    if (year_ != other.year_) return year_ < other.year_;
    if (month_ != other.month_) return month_ < other.month_;
    return day_ < other.day_;
}

bool Date::operator==(const Date& other) const {
    return year_ == other.year_ && month_ == other.month_ && day_ == other.day_;
}

// DataHandler Implementation
bool DataHandler::loadCSV(const std::string& filepath) {
    try {
        // Clear any existing data
        columnNames_.clear();
        rawData_.clear();
        numericData_.resize(0, 0);
        dates_.clear();
        numericColumnIndices_.clear();
        dateColumnIndices_.clear();
        
        // Read the file line by line
        std::ifstream file(filepath);
        if (!file.is_open()) {
            return false;
        }
        
        // Read header line
        std::string headerLine;
        if (std::getline(file, headerLine)) {
            std::stringstream ss(headerLine);
            std::string colName;
            
            // Split by comma
            while (std::getline(ss, colName, ',')) {
                columnNames_.push_back(colName);
            }
        } else {
            return false;
        }
        
        if (columnNames_.empty()) {
            return false;
        }
        
        // Read data rows
        std::string line;
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string value;
            std::vector<std::string> row;
            
            while (std::getline(ss, value, ',')) {
                row.push_back(value);
            }
            
            if (row.size() == columnNames_.size()) {
                rawData_.push_back(row);
            }
        }
        
        // Parse numeric and date columns
        if (!parseNumericData() || !parseDateColumns()) {
            return false;
        }
        
        // Detect frequency if we have date columns
        if (!dateColumnIndices_.empty()) {
            frequency_ = computeFrequency();
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error loading CSV: " << e.what() << std::endl;
        return false;
    }
}

Eigen::MatrixXd DataHandler::getData() const {
    return numericData_;
}

std::vector<std::string> DataHandler::getColumnNames() const {
    return columnNames_;
}

std::vector<size_t> DataHandler::getNumericColumnIndices() const {
    return numericColumnIndices_;
}

std::vector<size_t> DataHandler::getDateColumnIndices() const {
    return dateColumnIndices_;
}

Frequency DataHandler::detectFrequency() const {
    return frequency_;
}

std::string DataHandler::getFrequencyString() const {
    switch (frequency_) {
        case Frequency::DAILY:     return "Daily";
        case Frequency::WEEKLY:    return "Weekly";
        case Frequency::MONTHLY:   return "Monthly";
        case Frequency::QUARTERLY: return "Quarterly";
        case Frequency::YEARLY:    return "Yearly";
        default:                   return "Unknown";
    }
}

std::map<size_t, std::vector<size_t>> DataHandler::detectOutliers(
    const std::vector<size_t>& columnIndices,
    double sensitivity) const {
    
    std::map<size_t, std::vector<size_t>> outliers;
    
    // Use all numeric columns if none specified
    std::vector<size_t> columns = columnIndices.empty() ? numericColumnIndices_ : columnIndices;
    
    for (size_t col : columns) {
        if (col >= static_cast<size_t>(numericData_.cols())) {
            continue;
        }
        
        // Extract the column data and filter out NaN values
        Eigen::VectorXd column = numericData_.col(col);
        std::vector<double> valid_values;
        valid_values.reserve(column.size());
        for (int i = 0; i < column.size(); ++i) {
            if (!std::isnan(column(i))) {
                valid_values.push_back(column(i));
            }
        }
        
        // Skip if not enough valid values
        if (valid_values.size() < 4) { // Need at least 4 points for meaningful quartiles
            continue;
        }
        
        // Calculate quartiles using valid values
        std::sort(valid_values.begin(), valid_values.end());
        
        // Calculate quartile indices with interpolation
        double q1_idx = (valid_values.size() - 1) * 0.25;
        double q3_idx = (valid_values.size() - 1) * 0.75;
        
        // Get Q1 (25th percentile) with interpolation
        double q1;
        if (q1_idx == floor(q1_idx)) {
            q1 = valid_values[static_cast<size_t>(q1_idx)];
        } else {
            size_t idx_low = static_cast<size_t>(floor(q1_idx));
            size_t idx_high = static_cast<size_t>(ceil(q1_idx));
            q1 = valid_values[idx_low] + (q1_idx - floor(q1_idx)) * (valid_values[idx_high] - valid_values[idx_low]);
        }
        
        // Get Q3 (75th percentile) with interpolation
        double q3;
        if (q3_idx == floor(q3_idx)) {
            q3 = valid_values[static_cast<size_t>(q3_idx)];
        } else {
            size_t idx_low = static_cast<size_t>(floor(q3_idx));
            size_t idx_high = static_cast<size_t>(ceil(q3_idx));
            q3 = valid_values[idx_low] + (q3_idx - floor(q3_idx)) * (valid_values[idx_high] - valid_values[idx_low]);
        }
        
        // Calculate IQR (Interquartile Range)
        double iqr = q3 - q1;
        
        // Define outlier thresholds (sensitivity * IQR)
        double lowerBound = q1 - sensitivity * iqr;
        double upperBound = q3 + sensitivity * iqr;
        
        // Find outliers (excluding NaN values)
        std::vector<size_t> colOutliers;
        for (int i = 0; i < column.size(); ++i) {
            if (!std::isnan(column(i)) && (column(i) < lowerBound || column(i) > upperBound)) {
                colOutliers.push_back(i);
            }
        }
        
        if (!colOutliers.empty()) {
            outliers[col] = colOutliers;
        }
    }
    
    return outliers;
}

bool DataHandler::fixOutliers(const std::map<size_t, std::vector<size_t>>& outliers,
                             const std::map<size_t, std::vector<size_t>>& selectedOutliers) {
    if (numericData_.rows() == 0 || numericData_.cols() == 0) {
        return false;
    }
    
    // Use all outliers if selectedOutliers is empty
    const std::map<size_t, std::vector<size_t>>& outliersToFix = 
        selectedOutliers.empty() ? outliers : selectedOutliers;
    
    for (const auto& [col, rows] : outliersToFix) {
        if (static_cast<Eigen::Index>(col) >= numericData_.cols()) {
            continue;
        }
        
        for (size_t row : rows) {
            // Skip if this row is not actually an outlier
            if (outliers.find(col) == outliers.end() || 
                std::find(outliers.at(col).begin(), outliers.at(col).end(), row) == outliers.at(col).end()) {
                continue;
            }
            
            if (static_cast<Eigen::Index>(row) >= numericData_.rows()) {
                continue;
            }
            
            // Get previous and next non-outlier values
            double prevValue = 0.0;
            double nextValue = 0.0;
            int prevCount = 0;
            int nextCount = 0;
            
            // Look backward for non-outlier values
            for (int i = static_cast<int>(row) - 1; i >= 0; --i) {
                // Check if this row is an outlier
                bool isOutlier = false;
                auto it = outliers.find(col);
                if (it != outliers.end()) {
                    isOutlier = std::find(it->second.begin(), it->second.end(), i) != it->second.end();
                }
                
                if (!isOutlier) {
                    prevValue = numericData_(i, col);
                    prevCount = 1;
                    break;
                }
            }
            
            // Look forward for non-outlier values
            for (int i = static_cast<int>(row) + 1; i < numericData_.rows(); ++i) {
                // Check if this row is an outlier
                bool isOutlier = false;
                auto it = outliers.find(col);
                if (it != outliers.end()) {
                    isOutlier = std::find(it->second.begin(), it->second.end(), i) != it->second.end();
                }
                
                if (!isOutlier) {
                    nextValue = numericData_(i, col);
                    nextCount = 1;
                    break;
                }
            }
            
            // Interpolate based on available values
            if (prevCount > 0 && nextCount > 0) {
                // Average of previous and next
                numericData_(row, col) = (prevValue + nextValue) / 2.0;
            } else if (prevCount > 0) {
                // Only previous available
                numericData_(row, col) = prevValue;
            } else if (nextCount > 0) {
                // Only next available
                numericData_(row, col) = nextValue;
            } else {
                // No valid neighbors, use column mean
                numericData_(row, col) = numericData_.col(col).mean();
            }
            
            // Update raw data to reflect the change
            if (col < columnNames_.size()) {
                size_t rawCol = 0;
                for (size_t i = 0; i < numericColumnIndices_.size(); ++i) {
                    if (numericColumnIndices_[i] == col) {
                        rawCol = i;
                        break;
                    }
                }
                
                if (row < rawData_.size() && rawCol < rawData_[row].size()) {
                    rawData_[row][rawCol] = std::to_string(numericData_(row, col));
                }
            }
        }
    }
    
    return true;
}

bool DataHandler::addSeasonalFeatures([[maybe_unused]] size_t dateColumnIndex) {
    if (dates_.empty() || frequency_ == Frequency::UNKNOWN) {
        return false;
    }
    
    // Create sin/cos features based on the detected frequency
    int numRows = numericData_.rows();
    int numCols = numericData_.cols();
    
    // Determine seasonality period
    int period = 0;
    switch (frequency_) {
        case Frequency::DAILY:     period = 7;    break; // Days in a week
        case Frequency::WEEKLY:    period = 52;   break; // Weeks in a year
        case Frequency::MONTHLY:   period = 12;   break; // Months in a year
        case Frequency::QUARTERLY: period = 4;    break; // Quarters in a year
        case Frequency::YEARLY:    period = 1;    break; // 1 year period
        default: return false;
    }
    
    if (period <= 1) {
        return true; // No seasonal features needed
    }
    
    // Resize the numeric data matrix to add sin/cos columns
    Eigen::MatrixXd newData(numRows, numCols + 2);
    newData.leftCols(numCols) = numericData_;
    
    // Extract the time component based on frequency
    std::vector<int> timeComponents(numRows);
    for (int i = 0; i < numRows; ++i) {
        const Date& date = dates_[i];
        switch (frequency_) {
            case Frequency::DAILY:
                // Day of week (0-6)
                {
                    std::tm tm = {};
                    tm.tm_year = date.getYear() - 1900;
                    tm.tm_mon = date.getMonth() - 1;
                    tm.tm_mday = date.getDay();
                    std::mktime(&tm);
                    timeComponents[i] = tm.tm_wday;
                }
                break;
            case Frequency::WEEKLY:
                // Week of year (1-52)
                {
                    std::tm tm = {};
                    tm.tm_year = date.getYear() - 1900;
                    tm.tm_mon = date.getMonth() - 1;
                    tm.tm_mday = date.getDay();
                    std::mktime(&tm);
                    timeComponents[i] = (tm.tm_yday / 7) + 1;
                }
                break;
            case Frequency::MONTHLY:
                // Month of year (1-12)
                timeComponents[i] = date.getMonth();
                break;
            case Frequency::QUARTERLY:
                // Quarter of year (1-4)
                timeComponents[i] = (date.getMonth() - 1) / 3 + 1;
                break;
            default:
                timeComponents[i] = 0;
        }
    }
    
    // Create sin/cos features
    for (int i = 0; i < numRows; ++i) {
        double angle = 2 * PI * timeComponents[i] / period;
        newData(i, numCols) = std::sin(angle);     // Sin feature
        newData(i, numCols + 1) = std::cos(angle); // Cos feature
    }
    
    // Update column names and numeric data
    columnNames_.push_back("Seasonal_Sin");
    columnNames_.push_back("Seasonal_Cos");
    numericData_ = newData;
    
    // Update numeric column indices
    numericColumnIndices_.push_back(numCols);
    numericColumnIndices_.push_back(numCols + 1);
    
    return true;
}

std::string DataHandler::getDataSummary() const {
    std::stringstream ss;
    
    ss << "Data Summary:\n";
    ss << "-------------\n";
    ss << "Total Rows: " << rawData_.size() << "\n";
    ss << "Total Columns: " << columnNames_.size() << "\n";
    ss << "Numeric Columns: " << numericColumnIndices_.size() << "\n";
    ss << "Date Columns: " << dateColumnIndices_.size() << "\n";
    ss << "Detected Frequency: " << getFrequencyString() << "\n\n";
    
    ss << "Column Statistics:\n";
    
    // Calculate statistics for numeric columns
    for (size_t i = 0; i < numericColumnIndices_.size(); ++i) {
        size_t colIndex = numericColumnIndices_[i];
        if (colIndex < columnNames_.size()) {
            ss << "Column [" << colIndex << "]: " << columnNames_[colIndex] << "\n";
            
            if (numericData_.rows() > 0 && colIndex < static_cast<size_t>(numericData_.cols())) {
                Eigen::VectorXd col = numericData_.col(colIndex);
                
                double min = col.minCoeff();
                double max = col.maxCoeff();
                double mean = col.mean();
                double sum = col.sum();
                
                // Calculate standard deviation
                double variance = 0.0;
                for (int j = 0; j < col.size(); ++j) {
                    variance += std::pow(col(j) - mean, 2);
                }
                variance /= col.size();
                double std_dev = std::sqrt(variance);
                
                ss << "  Min: " << min << "\n";
                ss << "  Max: " << max << "\n";
                ss << "  Mean: " << mean << "\n";
                ss << "  Sum: " << sum << "\n";
                ss << "  Std Dev: " << std_dev << "\n";
            }
            
            ss << "\n";
        }
    }
    
    return ss.str();
}

std::vector<Date> DataHandler::getDates() const {
    return dates_;
}

Eigen::MatrixXd DataHandler::getSelectedFeatures(const std::vector<size_t>& featureIndices) const {
    if (featureIndices.empty() || numericData_.rows() == 0) {
        return Eigen::MatrixXd();
    }
    
    Eigen::MatrixXd features(numericData_.rows(), featureIndices.size());
    for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(featureIndices.size()); ++i) {
        if (static_cast<Eigen::Index>(featureIndices[i]) < numericData_.cols()) {
            features.col(i) = numericData_.col(featureIndices[i]);
        }
    }
    return features;
}

Eigen::VectorXd DataHandler::getSelectedTarget(size_t targetIndex) const {
    if (static_cast<Eigen::Index>(targetIndex) < numericData_.cols()) {
        return numericData_.col(targetIndex);
    }
    return Eigen::VectorXd();
}

bool DataHandler::exportToCSV(const std::string& filepath) const {
    try {
        std::ofstream file(filepath);
        if (!file.is_open()) {
            return false;
        }
        
        // Write header
        for (size_t i = 0; i < columnNames_.size(); ++i) {
            file << columnNames_[i];
            if (i < columnNames_.size() - 1) {
                file << ",";
            }
        }
        file << "\n";
        
        // Write data
        for (const auto& row : rawData_) {
            for (size_t i = 0; i < row.size(); ++i) {
                file << row[i];
                if (i < row.size() - 1) {
                    file << ",";
                }
            }
            file << "\n";
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error exporting CSV: " << e.what() << std::endl;
        return false;
    }
}

bool DataHandler::parseNumericData() {
    if (rawData_.empty() || columnNames_.empty()) {
        return false;
    }
    
    int numRows = rawData_.size();
    int numCols = columnNames_.size();
    
    // Identify numeric columns
    std::vector<bool> isNumeric(numCols, true);
    
    // Check first few rows to determine if columns are numeric
    int checkRows = std::min(numRows, 10);
    for (int i = 0; i < checkRows; ++i) {
        const auto& row = rawData_[i];
        for (int j = 0; j < numCols && j < static_cast<int>(row.size()); ++j) {
            if (isNumeric[j]) {
                try {
                    // Try to convert to double
                    std::stod(row[j]);
                } catch (...) {
                    isNumeric[j] = false;
                }
            }
        }
    }
    
    // Count numeric columns
    int numNumericCols = 0;
    for (bool b : isNumeric) {
        if (b) numNumericCols++;
    }
    
    // Create numeric data matrix
    numericData_ = Eigen::MatrixXd::Zero(numRows, numNumericCols);
    
    // Fill numeric indices
    numericColumnIndices_.clear();
    for (int j = 0; j < numCols; ++j) {
        if (isNumeric[j]) {
            numericColumnIndices_.push_back(j);
        }
    }
    
    // Fill numeric data matrix
    for (int i = 0; i < numRows; ++i) {
        const auto& row = rawData_[i];
        for (size_t j = 0; j < numericColumnIndices_.size(); ++j) {
            int colIndex = numericColumnIndices_[j];
            if (colIndex < static_cast<int>(row.size())) {
                const std::string& value = row[colIndex];
                if (value.empty() || value == "NA" || value == "N/A" || value == "null" || value == "NULL") {
                    numericData_(i, j) = std::numeric_limits<double>::quiet_NaN();
                } else {
                    try {
                        numericData_(i, j) = std::stod(value);
                    } catch (...) {
                        numericData_(i, j) = std::numeric_limits<double>::quiet_NaN();
                    }
                }
            } else {
                numericData_(i, j) = std::numeric_limits<double>::quiet_NaN();
            }
        }
    }
    
    return true;
}

bool DataHandler::parseDateColumns() {
    if (rawData_.empty() || columnNames_.empty()) {
        return false;
    }
    
    int numRows = rawData_.size();
    int numCols = columnNames_.size();
    
    // Identify date columns
    std::vector<bool> isDate(numCols, false);
    
    // Check for date columns
    for (int j = 0; j < numCols; ++j) {
        // Look for date-related keywords in column name
        std::string colName = columnNames_[j];
        std::transform(colName.begin(), colName.end(), colName.begin(), 
                       [](unsigned char c){ return std::tolower(c); });
        
        bool nameIndicatesDate = (colName.find("date") != std::string::npos ||
                               colName.find("time") != std::string::npos ||
                               colName.find("day") != std::string::npos ||
                               colName.find("month") != std::string::npos ||
                               colName.find("year") != std::string::npos);
        
        // Check first few rows to see if they can be parsed as dates
        int checkRows = std::min(numRows, 10);
        int dateParseCount = 0;
        
        for (int i = 0; i < checkRows; ++i) {
            if (j < static_cast<int>(rawData_[i].size())) {
                const std::string& cellValue = rawData_[i][j];
                if (Date::parse(cellValue).has_value()) {
                    dateParseCount++;
                }
            }
        }
        
        // Consider it a date column if most values can be parsed as dates
        if (dateParseCount >= checkRows * 0.7 || 
            (nameIndicatesDate && dateParseCount > 0)) {
            isDate[j] = true;
        }
    }
    
    // Fill date column indices
    dateColumnIndices_.clear();
    for (int j = 0; j < numCols; ++j) {
        if (isDate[j]) {
            dateColumnIndices_.push_back(j);
        }
    }
    
    // Parse dates
    if (!dateColumnIndices_.empty()) {
        // Use the first date column
        size_t dateColIndex = dateColumnIndices_[0];
        
        dates_.resize(numRows);
        for (int i = 0; i < numRows; ++i) {
            if (dateColIndex < rawData_[i].size()) {
                auto dateOpt = Date::parse(rawData_[i][dateColIndex]);
                if (dateOpt.has_value()) {
                    dates_[i] = dateOpt.value();
                } else {
                    // Create a dummy date for rows with unparseable dates
                    dates_[i] = Date(1900, 1, 1);
                }
            } else {
                dates_[i] = Date(1900, 1, 1);
            }
        }
        
        // Sort by date (no need to reorder data yet, just check if it's sorted)
        bool isSorted = std::is_sorted(dates_.begin(), dates_.end());
        if (!isSorted) {
            // For now, we'll just assume the data is in chronological order
            // A more robust solution would reorder everything
            
            // Just sort the dates for frequency detection
            std::sort(dates_.begin(), dates_.end());
        }
    }
    
    return true;
}

Frequency DataHandler::computeFrequency() const {
    if (dates_.size() < 2) {
        return Frequency::UNKNOWN;
    }
    
    // Sort dates (should already be sorted)
    std::vector<Date> sortedDates = dates_;
    std::sort(sortedDates.begin(), sortedDates.end());
    
    // Remove duplicates
    sortedDates.erase(std::unique(sortedDates.begin(), sortedDates.end()), sortedDates.end());
    
    if (sortedDates.size() < 2) {
        return Frequency::UNKNOWN;
    }
    
    // Calculate time differences (in days) between consecutive dates
    std::vector<int> diffs;
    for (size_t i = 1; i < sortedDates.size(); ++i) {
        const Date& d1 = sortedDates[i-1];
        const Date& d2 = sortedDates[i];
        
        // Convert dates to tm structs for difftime
        std::tm tm1 = {};
        tm1.tm_year = d1.getYear() - 1900;
        tm1.tm_mon = d1.getMonth() - 1;
        tm1.tm_mday = d1.getDay();
        
        std::tm tm2 = {};
        tm2.tm_year = d2.getYear() - 1900;
        tm2.tm_mon = d2.getMonth() - 1;
        tm2.tm_mday = d2.getDay();
        
        std::time_t time1 = std::mktime(&tm1);
        std::time_t time2 = std::mktime(&tm2);
        
        // Calculate difference in seconds, convert to days
        double diffSeconds = std::difftime(time2, time1);
        int diffDays = static_cast<int>(diffSeconds / (60 * 60 * 24));
        
        diffs.push_back(diffDays);
    }
    
    // Calculate the median difference
    std::vector<int> diffsCopy = diffs;
    std::sort(diffsCopy.begin(), diffsCopy.end());
    int median = diffsCopy[diffsCopy.size() / 2];
    
    // Determine frequency based on median difference
    if (median <= 1) {
        return Frequency::DAILY;
    } else if (median <= 7) {
        return Frequency::WEEKLY;
    } else if (median <= 31) {
        return Frequency::MONTHLY;
    } else if (median <= 92) {
        return Frequency::QUARTERLY;
    } else {
        return Frequency::YEARLY;
    }
}

bool DataHandler::addLagFeatures(int lagValues) {
    if (lagValues <= 0 || numericData_.rows() <= lagValues) {
        return false;
    }
    
    // Store original dimensions
    int originalRows = numericData_.rows();
    int originalCols = numericData_.cols();
    
    // Create new matrix with space for lag features
    // We lose 'lagValues' rows at the beginning because we need previous values
    int newRows = originalRows - lagValues;
    int newCols = originalCols + selectedFeatures_.size() * lagValues;
    
    Eigen::MatrixXd newData(newRows, newCols);
    
    // Copy non-lagged data (excluding the first 'lagValues' rows)
    for (int r = 0; r < newRows; ++r) {
        for (int c = 0; c < originalCols; ++c) {
            newData(r, c) = numericData_(r + lagValues, c);
        }
    }
    
    // Calculate and add lag features
    int lagColOffset = originalCols;
    for (size_t featureIdx = 0; featureIdx < selectedFeatures_.size(); ++featureIdx) {
        size_t origColIdx = selectedFeatures_[featureIdx];
        
        // Skip if not a valid column
        if (origColIdx >= static_cast<size_t>(originalCols)) {
            continue;
        }
        
        // Generate lags for this feature
        for (int lag = 1; lag <= lagValues; ++lag) {
            int lagCol = lagColOffset + (featureIdx * lagValues) + (lag - 1);
            
            // For each row in the new matrix, get the lagged value
            for (int r = 0; r < newRows; ++r) {
                // The lag value comes from 'lag' steps back from the current row + lagValues
                newData(r, lagCol) = numericData_(r + lagValues - lag, origColIdx);
            }
            
            // Add a new column name for this lag
            std::string colName = columnNames_[origColIdx] + "_lag" + std::to_string(lag);
            columnNames_.push_back(colName);
            numericColumnIndices_.push_back(columnNames_.size() - 1);
        }
    }
    
    // Update numeric data and adjust raw data accordingly
    numericData_ = newData;
    
    // Adjust raw data to match the new numeric data size
    // We remove the first 'lagValues' rows
    if (rawData_.size() > static_cast<size_t>(lagValues)) {
        std::vector<std::vector<std::string>> newRawData;
        for (size_t i = lagValues; i < rawData_.size(); ++i) {
            newRawData.push_back(rawData_[i]);
        }
        rawData_ = newRawData;
    }
    
    // If we have dates, also adjust them
    if (!dates_.empty() && dates_.size() > static_cast<size_t>(lagValues)) {
        std::vector<Date> newDates;
        for (size_t i = lagValues; i < dates_.size(); ++i) {
            newDates.push_back(dates_[i]);
        }
        dates_ = newDates;
    }
    
    return true;
}

bool DataHandler::addSeasonalLags(int seasonality) {
    if (seasonality <= 0 || numericData_.rows() <= seasonality || selectedTargetIndices_.empty()) {
        return false;
    }
    
    // Store original dimensions
    int originalRows = numericData_.rows();
    int originalCols = numericData_.cols();
    
    // Create new matrix with space for seasonal lag features
    // We lose 'seasonality' rows at the beginning
    int newRows = originalRows - seasonality;
    int newCols = originalCols + selectedTargetIndices_.size();
    
    Eigen::MatrixXd newData(newRows, newCols);
    
    // Copy original data (excluding the first 'seasonality' rows)
    for (int r = 0; r < newRows; ++r) {
        for (int c = 0; c < originalCols; ++c) {
            newData(r, c) = numericData_(r + seasonality, c);
        }
    }
    
    // Add seasonal lag features for each target variable
    int seasonalColOffset = originalCols;
    for (size_t targetIdx = 0; targetIdx < selectedTargetIndices_.size(); ++targetIdx) {
        size_t origColIdx = selectedTargetIndices_[targetIdx];
        
        // Skip if not a valid column
        if (origColIdx >= static_cast<size_t>(originalCols)) {
            continue;
        }
        
        // Generate seasonal lag for this target
        int seasonalCol = seasonalColOffset + targetIdx;
        
        // For each row in the new matrix, get the seasonal lagged value
        for (int r = 0; r < newRows; ++r) {
            // The seasonal lag value comes from 'seasonality' steps back
            newData(r, seasonalCol) = numericData_(r, origColIdx);
        }
        
        // Add a new column name for this seasonal lag
        std::string colName = columnNames_[origColIdx] + "_seasonal" + std::to_string(seasonality);
        columnNames_.push_back(colName);
        numericColumnIndices_.push_back(columnNames_.size() - 1);
    }
    
    // Update numeric data and adjust raw data accordingly
    numericData_ = newData;
    
    // Adjust raw data to match the new numeric data size
    // We remove the first 'seasonality' rows
    if (rawData_.size() > static_cast<size_t>(seasonality)) {
        std::vector<std::vector<std::string>> newRawData;
        for (size_t i = seasonality; i < rawData_.size(); ++i) {
            newRawData.push_back(rawData_[i]);
        }
        rawData_ = newRawData;
    }
    
    // If we have dates, also adjust them
    if (!dates_.empty() && dates_.size() > static_cast<size_t>(seasonality)) {
        std::vector<Date> newDates;
        for (size_t i = seasonality; i < dates_.size(); ++i) {
            newDates.push_back(dates_[i]);
        }
        dates_ = newDates;
    }
    
    return true;
}

std::map<std::string, int> DataHandler::findBestSeasonalLags(int maxSeasonality) const {
    std::map<std::string, int> bestSeasonalLags;
    
    // Ensure we have data and a target
    if (numericData_.rows() <= maxSeasonality || selectedTargetIndices_.empty()) {
        return bestSeasonalLags;
    }
    
    // For each target variable, find the best seasonality
    for (size_t targetIdx : selectedTargetIndices_) {
        if (targetIdx >= static_cast<size_t>(numericData_.cols())) {
            continue;
        }
        
        std::string targetName = columnNames_[targetIdx];
        Eigen::VectorXd targetColumn = numericData_.col(targetIdx);
        
        // Find the best seasonality by highest absolute autocorrelation
        int bestSeasonality = 0;
        double bestCorr = 0.0;
        
        // Try different seasonality values
        for (int s = 1; s <= maxSeasonality; ++s) {
            // We need enough data points for the comparison
            int validRows = numericData_.rows() - s;
            if (validRows < 10) {  // Need reasonable sample size
                continue;
            }
            
            // Create vectors for autocorrelation calculation
            Eigen::VectorXd currentValues(validRows);
            Eigen::VectorXd seasonalValues(validRows);
            
            // Fill vectors with appropriate values
            for (int r = 0; r < validRows; ++r) {
                currentValues(r) = targetColumn(r + s);
                seasonalValues(r) = targetColumn(r);
            }
            
            // Calculate autocorrelation (Pearson's)
            double meanX = currentValues.mean();
            double meanY = seasonalValues.mean();
            double normX = (currentValues.array() - meanX).matrix().norm();
            double normY = (seasonalValues.array() - meanY).matrix().norm();
            
            if (normX < 1e-10 || normY < 1e-10) {
                continue;  // Skip if standard deviation is too small
            }
            
            double corr = ((currentValues.array() - meanX) * (seasonalValues.array() - meanY)).sum() / (normX * normY);
            double absCorr = std::abs(corr);
            
            if (absCorr > bestCorr) {
                bestCorr = absCorr;
                bestSeasonality = s;
            }
        }
        
        // Only add to map if we found a significant seasonality
        if (bestSeasonality > 0 && bestCorr > 0.2) {  // 0.2 correlation threshold is arbitrary, adjust as needed
            bestSeasonalLags[targetName] = bestSeasonality;
        }
    }
    
    return bestSeasonalLags;
}

bool DataHandler::addVariableSeasonalLags(int maxSeasonality, std::map<std::string, int>& bestSeasonalLagsMap) {
    if (maxSeasonality <= 0 || numericData_.rows() <= maxSeasonality || selectedTargetIndices_.empty()) {
        std::cerr << "Failed to add variable seasonal lags: invalid parameters" << std::endl;
        return false;
    }
    
    // First, find best seasonal lag values if map is empty
    if (bestSeasonalLagsMap.empty()) {
        std::cerr << "Finding best seasonal lags..." << std::endl;
        bestSeasonalLagsMap = findBestSeasonalLags(maxSeasonality);
    }
    
    // If we still don't have any best seasonal lags, return false
    if (bestSeasonalLagsMap.empty()) {
        std::cerr << "No significant seasonal patterns found" << std::endl;
        return false;
    } else {
        std::cerr << "Found " << bestSeasonalLagsMap.size() << " best seasonal patterns" << std::endl;
        for (const auto& [target, period] : bestSeasonalLagsMap) {
            std::cerr << "Target: " << target << ", Period: " << period << std::endl;
        }
    }
    
    // Store original dimensions
    int originalRows = numericData_.rows();
    int originalCols = numericData_.cols();
    std::cerr << "Original data dimensions: " << originalRows << " x " << originalCols << std::endl;
    
    // Determine the maximum seasonality across all targets
    int maxSeason = 0;
    for (const auto& [_, season] : bestSeasonalLagsMap) {
        maxSeason = std::max(maxSeason, season);
    }
    std::cerr << "Maximum seasonality found: " << maxSeason << std::endl;
    
    // Create new matrix with space for variable-specific seasonal lag features
    int newRows = originalRows - maxSeason;
    int newCols = originalCols + bestSeasonalLagsMap.size();  // One column per target with best seasonality
    std::cerr << "New data dimensions: " << newRows << " x " << newCols << std::endl;
    
    Eigen::MatrixXd newData(newRows, newCols);
    
    // Copy non-lagged data (excluding the first 'maxSeason' rows)
    for (int r = 0; r < newRows; ++r) {
        for (int c = 0; c < originalCols; ++c) {
            newData(r, c) = numericData_(r + maxSeason, c);
        }
    }
    
    // Add seasonal lag features with target-specific seasonality
    int seasonalColOffset = originalCols;
    int seasonalColIdx = 0;
    
    for (size_t targetIdx = 0; targetIdx < selectedTargetIndices_.size(); ++targetIdx) {
        size_t origColIdx = selectedTargetIndices_[targetIdx];
        
        // Skip if not a valid column
        if (origColIdx >= static_cast<size_t>(originalCols)) {
            std::cerr << "Skipping target index " << origColIdx << " (out of range)" << std::endl;
            continue;
        }
        
        std::string targetName = columnNames_[origColIdx];
        std::cerr << "Processing target: " << targetName << std::endl;
        
        // Check if this target has a best seasonality value
        if (bestSeasonalLagsMap.find(targetName) != bestSeasonalLagsMap.end()) {
            int season = bestSeasonalLagsMap[targetName];
            std::cerr << "  Found optimal seasonality: " << season << std::endl;
            
            // For each row in the new matrix, get the seasonal lag value
            for (int r = 0; r < newRows; ++r) {
                // The seasonal value comes from 'season' steps back from the current row + maxSeason
                newData(r, seasonalColOffset + seasonalColIdx) = numericData_(r + maxSeason - season, origColIdx);
            }
            
            // Add a new column name for this seasonal lag
            std::string colName = targetName + "_seasonal" + std::to_string(season);
            columnNames_.push_back(colName);
            numericColumnIndices_.push_back(columnNames_.size() - 1);
            std::cerr << "  Added seasonal feature: " << colName << std::endl;
            
            seasonalColIdx++;
        } else {
            std::cerr << "  No optimal seasonality found for this target" << std::endl;
        }
    }
    
    // Update numeric data and adjust raw data accordingly
    numericData_ = newData;
    std::cerr << "Updated numeric data matrix" << std::endl;
    
    // Adjust raw data to match the new numeric data size
    if (rawData_.size() > static_cast<size_t>(maxSeason)) {
        std::vector<std::vector<std::string>> newRawData;
        for (size_t i = maxSeason; i < rawData_.size(); ++i) {
            newRawData.push_back(rawData_[i]);
        }
        rawData_ = newRawData;
        std::cerr << "Adjusted raw data, new size: " << rawData_.size() << std::endl;
    }
    
    // If we have dates, also adjust them
    if (!dates_.empty() && dates_.size() > static_cast<size_t>(maxSeason)) {
        std::vector<Date> newDates;
        for (size_t i = maxSeason; i < dates_.size(); ++i) {
            newDates.push_back(dates_[i]);
        }
        dates_ = newDates;
        std::cerr << "Adjusted dates, new size: " << dates_.size() << std::endl;
    }
    
    // Store the best seasonal lag values
    bestSeasonalLagValues_ = bestSeasonalLagsMap;
    std::cerr << "Stored best seasonal lag values" << std::endl;
    
    return true;
}

std::map<std::string, int> DataHandler::findBestLagValues(int maxLag) const {
    std::map<std::string, int> bestLags;
    
    // Ensure we have data and a target
    if (numericData_.rows() <= maxLag || selectedTargetIndices_.empty() || selectedFeatures_.empty()) {
        return bestLags;
    }
    
    // Get the target column
    size_t targetIdx = selectedTargetIndices_[0];
    Eigen::VectorXd targetColumn = numericData_.col(targetIdx);
    
    // Calculate correlations for each feature
    for (size_t featureIdx : selectedFeatures_) {
        if (featureIdx >= static_cast<size_t>(numericData_.cols()) || featureIdx == targetIdx) {
            continue;
        }
        
        std::string featureName = columnNames_[featureIdx];
        Eigen::VectorXd featureColumn = numericData_.col(featureIdx);
        
        // Find the best lag by highest absolute correlation
        int bestLag = 0;
        double bestCorr = 0.0;
        
        // Try different lag values
        for (int lag = 1; lag <= maxLag; ++lag) {
            // We can only compare rows with valid lag values
            int validRows = numericData_.rows() - lag;
            if (validRows < 10) {  // Need reasonable sample size
                continue;
            }
            
            // Create vectors for correlation calculation
            Eigen::VectorXd laggedFeature(validRows);
            Eigen::VectorXd targetSubset(validRows);
            
            // Fill vectors with appropriate values
            for (int r = 0; r < validRows; ++r) {
                laggedFeature(r) = featureColumn(r);
                targetSubset(r) = targetColumn(r + lag);
            }
            
            // Calculate correlation (Pearson's)
            double meanX = laggedFeature.mean();
            double meanY = targetSubset.mean();
            double normX = (laggedFeature.array() - meanX).matrix().norm();
            double normY = (targetSubset.array() - meanY).matrix().norm();
            
            if (normX < 1e-10 || normY < 1e-10) {
                continue;  // Skip if standard deviation is too small
            }
            
            double corr = ((laggedFeature.array() - meanX) * (targetSubset.array() - meanY)).sum() / (normX * normY);
            double absCorr = std::abs(corr);
            
            if (absCorr > bestCorr) {
                bestCorr = absCorr;
                bestLag = lag;
            }
        }
        
        // Only add to map if we found a significant lag
        if (bestLag > 0 && bestCorr > 0.2) {  // 0.2 correlation threshold is arbitrary, adjust as needed
            bestLags[featureName] = bestLag;
        }
    }
    
    return bestLags;
}

bool DataHandler::addVariableLagFeatures(int maxLagValues, std::map<std::string, int>& bestLagsMap) {
    if (maxLagValues <= 0 || numericData_.rows() <= maxLagValues || selectedFeatures_.empty()) {
        return false;
    }
    
    // First, find best lag values if map is empty
    if (bestLagsMap.empty()) {
        bestLagsMap = findBestLagValues(maxLagValues);
    }
    
    // If we still don't have any best lags, return false
    if (bestLagsMap.empty()) {
        return false;
    }
    
    // Store original dimensions
    int originalRows = numericData_.rows();
    int originalCols = numericData_.cols();
    
    // Determine the maximum lag across all features
    int maxLag = 0;
    for (const auto& [_, lag] : bestLagsMap) {
        maxLag = std::max(maxLag, lag);
    }
    
    // Create new matrix with space for variable-specific lag features
    int newRows = originalRows - maxLag;
    int newCols = originalCols + bestLagsMap.size();  // One column per feature with best lag
    
    Eigen::MatrixXd newData(newRows, newCols);
    
    // Copy non-lagged data (excluding the first 'maxLag' rows)
    for (int r = 0; r < newRows; ++r) {
        for (int c = 0; c < originalCols; ++c) {
            newData(r, c) = numericData_(r + maxLag, c);
        }
    }
    
    // Add lag features with feature-specific lags
    int lagColOffset = originalCols;
    int lagColIdx = 0;
    
    for (size_t featureIdx = 0; featureIdx < selectedFeatures_.size(); ++featureIdx) {
        size_t origColIdx = selectedFeatures_[featureIdx];
        
        // Skip if not a valid column
        if (origColIdx >= static_cast<size_t>(originalCols)) {
            continue;
        }
        
        std::string featureName = columnNames_[origColIdx];
        
        // Check if this feature has a best lag value
        if (bestLagsMap.find(featureName) != bestLagsMap.end()) {
            int lag = bestLagsMap[featureName];
            
            // For each row in the new matrix, get the lagged value
            for (int r = 0; r < newRows; ++r) {
                // The lag value comes from 'lag' steps back from the current row + maxLag
                newData(r, lagColOffset + lagColIdx) = numericData_(r + maxLag - lag, origColIdx);
            }
            
            // Add a new column name for this lag
            std::string colName = featureName + "_lag" + std::to_string(lag);
            columnNames_.push_back(colName);
            numericColumnIndices_.push_back(columnNames_.size() - 1);
            
            lagColIdx++;
        }
    }
    
    // Update numeric data and adjust raw data accordingly
    numericData_ = newData;
    
    // Adjust raw data to match the new numeric data size
    if (rawData_.size() > static_cast<size_t>(maxLag)) {
        std::vector<std::vector<std::string>> newRawData;
        for (size_t i = maxLag; i < rawData_.size(); ++i) {
            newRawData.push_back(rawData_[i]);
        }
        rawData_ = newRawData;
    }
    
    // If we have dates, also adjust them
    if (!dates_.empty() && dates_.size() > static_cast<size_t>(maxLag)) {
        std::vector<Date> newDates;
        for (size_t i = maxLag; i < dates_.size(); ++i) {
            newDates.push_back(dates_[i]);
        }
        dates_ = newDates;
    }
    
    // Store the best lag values
    bestLagValues_ = bestLagsMap;
    
    return true;
}

std::vector<std::string> DataHandler::getFeatureNames() const {
    std::vector<std::string> featureNames = columnNames_;
    
    // Remove target variable from the feature list if it exists
    if (!selectedTargetIndices_.empty()) {
        for (size_t targetIdx : selectedTargetIndices_) {
            if (targetIdx < columnNames_.size()) {
                std::string targetName = columnNames_[targetIdx];
                featureNames.erase(
                    std::remove(featureNames.begin(), featureNames.end(), targetName),
                    featureNames.end()
                );
            }
        }
    }
    
    // Add original feature names first
    std::vector<std::string> result = featureNames;
    
    // Add best seasonal lag features (highest priority)
    if (!bestSeasonalLagValues_.empty()) {
        for (const auto& [target, seasonality] : bestSeasonalLagValues_) {
            // Check if the target exists in the column names
            bool targetExists = std::find(columnNames_.begin(), columnNames_.end(), target) != columnNames_.end();
            
            if (targetExists) {
                std::string seasonalFeature = target + "_seasonal" + std::to_string(seasonality);
                
                // Only add if it's not already in the result
                if (std::find(result.begin(), result.end(), seasonalFeature) == result.end()) {
                    result.push_back(seasonalFeature);
                }
            }
        }
    }
    
    // Add variable-specific best lag features (second priority)
    if (!bestLagValues_.empty()) {
        for (const auto& [feature, lag] : bestLagValues_) {
            // Check if the feature exists in the column names
            bool featureExists = std::find(columnNames_.begin(), columnNames_.end(), feature) != columnNames_.end();
            
            if (featureExists) {
                std::string lagFeature = feature + "_lag" + std::to_string(lag);
                
                // Only add if it's not already in the result
                if (std::find(result.begin(), result.end(), lagFeature) == result.end()) {
                    result.push_back(lagFeature);
                }
            }
        }
    }
    
    // Add regular lag feature names if lag > 0
    if (lagValues_ > 0) {
        // Create a copy of original names to prevent contamination
        std::vector<std::string> originalNames = featureNames;
        for (const std::string& name : originalNames) {
            for (int lag = 1; lag <= lagValues_; ++lag) {
                std::string lagFeature = name + "_lag" + std::to_string(lag);
                
                // Only add if it's not already in the result (not a best lag)
                if (std::find(result.begin(), result.end(), lagFeature) == result.end()) {
                    result.push_back(lagFeature);
                }
            }
        }
    }
    
    // Add regular seasonal features if seasonality > 0
    if (seasonality_ > 0) {
        // Create a copy of original names to prevent contamination
        std::vector<std::string> originalNames = featureNames;
        for (const std::string& name : originalNames) {
            for (int s = 1; s <= seasonality_; ++s) {
                std::string seasonalFeature = name + "_seasonal" + std::to_string(s);
                
                // Only add if it's not already in the result (not a best seasonal)
                if (std::find(result.begin(), result.end(), seasonalFeature) == result.end()) {
                    result.push_back(seasonalFeature);
                }
            }
        }
    }
    
    return result;
}

} // namespace DataAnalyzer 