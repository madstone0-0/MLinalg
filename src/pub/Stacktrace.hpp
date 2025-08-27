#pragma once

#include <concepts>
#include <exception>
#include <stdexcept>
#include <string>

namespace mlinalg::stacktrace {

    template <typename E>
    concept Exception = requires {
        std::derived_from<E, std::exception>;
        std::constructible_from<E, std::string>;
    };

    template <Exception E = std::runtime_error>
    class StackError : public E {
       public:
#if defined(STACKTRACE) || (defined(DEBUG) && defined(STACKTRACE))
        StackError(const std::string_view& msg,
                   const boost::stacktrace::stacktrace& st = boost::stacktrace::stacktrace())
            : E(makeMsg(msg, st)) {}

       private:
        std::string makeMsg(const std::string& msg, const boost::stacktrace::stacktrace& st) {
            std::ostringstream oss;
            oss << msg << "\n" << st;
            return oss.str();
        }
#else
        StackError(const std::string& msg) : E(msg) {}
#endif
    };
}  // namespace mlinalg::stacktrace
