#pragma once

#include "../structures/Matrix.hpp"

namespace mlinalg {
    using namespace structures;

    /**
     * @brief Linear System type alias.
     */
    template <Number number, int m, int n>
    using LinearSystem = Matrix<number, m, n>;
}  // namespace mlinalg
