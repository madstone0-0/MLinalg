/**
 * @file Aliases.hpp
 * @brief Header file for common type aliases for operations
 */

#pragma once

#include "../structures/Matrix.hpp"

namespace mlinalg {
    using namespace structures;

    /**
     * @brief Linear System type alias.
     */
    template <Number number, Dim m, Dim n>
    using LinearSystem = Matrix<number, m, n>;
}  // namespace mlinalg
