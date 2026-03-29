#ifndef LOGGING_H
#define LOGGING_H

#include <cstddef>
#include <string>

namespace utci {

void logInfo(const std::string& message);
void logWarn(const std::string& message);
void logError(const std::string& message);
void logSection(const std::string& title);
void logDetail(const std::string& message);
void logSummary(const std::string& message);
void logProgress(const std::string& tag, size_t done, size_t total, double elapsedSeconds, double etaSeconds);

}

#endif
