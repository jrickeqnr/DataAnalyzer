#include "../include/data_handler.h"
#include "../lib/fast-cpp-csv-parser/csv.h"
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
            
            int year, month, day;
            
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
    const std::vector<size_t>& columnIndices) const {
    
    std::map<size_t, std::vector<size_t>> outliers;
    
    // Use all numeric columns if none specified
    std::vector<size_t> columns = columnIndices.empty() ? numericColumnIndices_ : columnIndices;
    
    for (size_t col : columns) {
        if (col >= numericData_.cols()) {
            continue;
        }
        
        // Extract the column data
        Eigen::VectorXd column = numericData_.col(col);
        
        // Calculate quartiles using a sorted copy of the data
        std::vector<double> sorted(column.data(), column.data() + column.size());
        std::sort(sorted.begin(), sorted.end());
        
        // Get Q1 (25th percentile)
        double q1 = sorted[sorted.size() * 0.25];
        
        // Get Q3 (75th percentile)
        double q3 = sorted[sorted.size() * 0.75];
        
        // Calculate IQR (Interquartile Range)
        double iqr = q3 - q1;
        
        // Define outlier thresholds (1.5 * IQR)
        double lowerBound = q1 - 1.5 * iqr;
        double upperBound = q3 + 1.5 * iqr;
        
        // Find outliers
        std::vector<size_t> colOutliers;
        for (int i = 0; i < column.size(); ++i) {
            if (column(i) < lowerBound || column(i) > upperBound) {
                colOutliers.push_back(i);
            }
        }
        
        if (!colOutliers.empty()) {
            outliers[col] = colOutliers;
        }
    }
    
    return outliers;
}

bool DataHandler::fixOutliers(const std::map<size_t, std::vector<size_t>>& outliers) {
    if (numericData_.rows() == 0 || numericData_.cols() == 0) {
        return false;
    }
    
    for (const auto& [col, rows] : outliers) {
        if (static_cast<Eigen::Index>(col) >= numericData_.cols()) {
            continue;
        }
        
        for (size_t row : rows) {
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
        double angle = 2 * M_PI * timeComponents[i] / period;
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
            
            if (numericData_.rows() > 0 && colIndex < numericData_.cols()) {
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
        for (int j = 0; j < numCols && j < row.size(); ++j) {
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
            if (colIndex < row.size()) {
                try {
                    numericData_(i, j) = std::stod(row[colIndex]);
                } catch (...) {
                    numericData_(i, j) = 0.0; // Default for non-numeric values
                }
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
            if (j < rawData_[i].size()) {
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

} // namespace DataAnalyzer 