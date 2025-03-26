/**
 * @file Aliases.hpp
 * @brief Header file for type aliases
 */

#pragma once
#include <array>
#include <cstddef>
#include <memory>
#include <utility>
#include <variant>
#include <vector>

#include "../Concepts.hpp"

namespace mlinalg::structures {
    /**
     * @brief Dynamic size constant
     */
    static constexpr int Dynamic{-1};
    // =============================
    // Vector Aliases
    // =============================
    template <Number number, int n>
    class Vector;

    // Type alias for the backing array of a Vector
    template <Number number, int n>
    using VectorRow = std::array<number, n>;

    // Type alias for the backing array of a dynamic Vector
    template <Number number>
    using VectorRowDynamic = std::vector<number>;

    // Type alias for a unique pointer to a VectorRow
    template <Number number, int n>
    using VectorRowPtr = std::unique_ptr<VectorRow<number, n>>;

    // Type alias for a unique pointer to a dynamic VectorRow
    template <Number number>
    using VectorRowDynamicPtr = std::unique_ptr<VectorRowDynamic<number>>;

    // =============================
    // Matrix Aliases
    // =============================
    template <Number number, int m, int n>
    class Matrix;

    using SizePair = std::pair<int, int>;

    using SizeTPair = std::pair<size_t, size_t>;

    /**
     * @brief Type alias for a variant of a Vector and a Matrix
     *
     * This is used to represent the result of a matrix or vector transposition. As the transpose of a vector is a 1xM
     * matrix and the transpose of a matrix is an NxM matrix, this variant is used to represent either of these
     */
    template <Number number, int m, int n>
    using TransposeVariant = std::variant<Vector<number, m>, Matrix<number, n, m>>;

    /**
     * @brief Type alias for a Vector as a row in a Matrix
     */
    template <Number number, int n>
    using Row = Vector<number, n>;

    template <Number number>
    using RowDynamic = Vector<number, -1>;

    template <Number number, int m, int n>
    using TDArray = std::array<Row<number, n>, m>;

    template <Number number>
    using TDArrayDynamic = std::vector<RowDynamic<number>>;

}  // namespace mlinalg::structures
