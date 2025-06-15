/**
 * @file Numeric.hpp
 * @brief This file contains utility functions for numeric operations
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <limits>
#include <type_traits>

#include "Concepts.hpp"

using std::abs, std::isinf;
namespace mlinalg {
    /**
     * @brief Epsilon value for floating point comparison
     */
    static constexpr double EPSILON = std::numeric_limits<double>::epsilon();
    static constexpr double EPSILON_FIXED = 1.0e-08;

    template <typename T>
    struct is_floating_point_or_complex
        : std::disjunction<std::is_floating_point<T>, std::is_same<T, std::complex<float>>,
                           std::is_same<T, std::complex<double>>, std::is_same<T, std::complex<long double>>> {};

    /**
     * @brief Compare two numbers with a tolerance of EPSILON
     *
     * Adapted from https://embeddeduse.com/2019/08/26/qt-compare-two-floats/
     * @param a
     * @param b
     * @return True if the numbers are equal within EPSILON, false otherwise
     */
    template <Number number>
    constexpr bool fuzzyCompare(const number& a, const number& b) {
        // Check if either value is infinity.
        const auto& absA{abs(a)};
        const auto& absB{abs(b)};
        if (isinf(absA) || isinf(absB)) {
            // They are equal only if both are infinite and have the same sign.
            return isinf(absA) && isinf(absB) && (std::signbit(absA) == std::signbit(absB));
        }

        if constexpr (!is_floating_point_or_complex<number>::value) {
            auto diff{static_cast<double>(a - b)};
            if (abs(diff) <= EPSILON * std::max(abs(static_cast<double>(a)), abs(static_cast<double>(b))))
                return true;
            else
                return abs(diff) <= EPSILON_FIXED;
        } else {
            auto diff{a - b};
            if (abs(diff) <= EPSILON * std::max(abs(a), abs(b)))
                return true;
            else
                return abs(diff) <= EPSILON_FIXED;
        }
    }
}  // namespace mlinalg
