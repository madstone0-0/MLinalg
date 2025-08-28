/**
 * @file Stacktrace.hpp
 * @brief Header file for StackError exception class with optional stack trace support.
 */

#pragma once

#include <concepts>
#include <exception>
#include <stdexcept>
#include <string>

namespace mlinalg::stacktrace {

    /**
     * @brief Concept to ensure the type is an exception derived from
     * std::exception and constructible from std::string.
     *
     * @tparam E The type to be checked.
     */
    template <typename E>
    concept Exception = requires {
        std::derived_from<E, std::exception>;
        std::constructible_from<E, std::string>;
    };

    /**
     * @brief Exception class that optionally includes a stack trace in its message.
     *
     * @param msg The error message.
     * @param st  The stack trace (optional, included if STACKTRACE is defined).
     * @tparam E The base exception type (default is std::runtime_error).
     */
    template <Exception E = std::runtime_error>
    class StackError : public E {
       public:
#if defined(STACKTRACE) || (defined(DEBUG) && defined(STACKTRACE))
        StackError(const char* msg, const boost::stacktrace::stacktrace& st = boost::stacktrace::stacktrace())
            : E(buffer_.data()) {
            int written = std::snprintf(buffer_.data(), buffer_.size(), "%s\n%s", msg, to_string(st).c_str());
            if (written < 0) buffer_[0] = '\0';  // fallback on error
        }

       private:
        std::array<char, 1024> buffer_;  // static storage
#else
        StackError(const char* msg) : E(msg) {}
#endif
    };
}  // namespace mlinalg::stacktrace
