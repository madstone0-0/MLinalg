/**
 * @file Logging.hpp
 * @brief This file contains utility functions for logging
 */
#pragma once

#include <chrono>
#include <format>
#include <iostream>
#include <string_view>

using std::cout, std::endl, std::format, std::string_view;

constexpr const char* BLK = "\e[0;30m";
constexpr const char* RED = "\e[0;31m";
constexpr const char* GRN = "\e[0;32m";
constexpr const char* YEL = "\e[0;33m";
constexpr const char* BLU = "\e[0;34m";
constexpr const char* MAG = "\e[0;35m";
constexpr const char* CYN = "\e[0;36m";
constexpr const char* WHT = "\e[0;37m";

constexpr const char* reset = "\e[0m";
constexpr const char* CRESET = "\e[0m";
constexpr const char* COLOR_RESET = "\e[0m";

namespace logging {

    enum class Level : std::uint8_t { DEB, INF, WARN, ERR };

    inline void log(string_view message, string_view function, Level level = Level::DEB) {
        auto fmt = [&]() {
            return std::format("[{:%F T %T}] ({}) : {}\n", std::chrono::system_clock::now(), function, message);
        };
        switch (level) {
            case Level::DEB:
                cout << BLU << fmt() << reset;
                break;
            case Level::INF:
                cout << GRN << fmt() << reset;
                break;
            case Level::WARN:
                cout << YEL << fmt() << reset;
                break;
            case Level::ERR:
                cout << RED << fmt() << reset;
                break;
        }
    }
}  // namespace logging
