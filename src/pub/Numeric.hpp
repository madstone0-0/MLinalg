/**
 * @file Numeric.hpp
 * @brief This file contains utility functions for numeric operations
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>

#include "Concepts.hpp"

namespace mlinalg {
    /**
     * @brief Epsilon value for floating point comparison
     */
    static constexpr double EPSILON = std::numeric_limits<double>::epsilon();
    static constexpr double EPSILON_FIXED = 1.0e-05;

    /**
     * @brief Compare two numbers with a tolerance of EPSILON
     *
     * Adapted from https://embeddeduse.com/2019/08/26/qt-compare-two-floats/
     * @param a
     * @param b
     * @return True if the numbers are equal within EPSILON, false otherwise
     */
    template <Number number>
    inline bool fuzzyCompare(const number& a, const number& b) {
        // Check if either value is infinity.
        if (std::isinf(a) || std::isinf(b)) {
            // They are equal only if both are infinite and have the same sign.
            return std::isinf(a) && std::isinf(b) && (std::signbit(a) == std::signbit(b));
        }
        auto diff{static_cast<double>(a - b)};
        if (std::abs(diff) <= EPSILON_FIXED) return true;
        return std::abs(diff) <= EPSILON * std::max(std::abs(static_cast<double>(a)), std::abs(static_cast<double>(b)));
    }
}  // namespace mlinalg
