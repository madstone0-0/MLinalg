/**
 * @file Aliases.hpp
 * @brief Header file for type aliases
 */

#pragma once
#include <array>
#include <cstddef>
#include <memory>
#include <optional>
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

    /**
     * @brief  Type alias for the backing array of a Vector
     */
    template <Number number, int n>
    using VectorRow = std::array<number, n>;

    /**
     * @brief Type alias for the backing array of a dynamic Vector
     */
    template <Number number>
    using VectorRowDynamic = std::vector<number>;

    /**
     * @brief Type alias for a unique pointer to a VectorRow
     */
    template <Number number, int n>
    using VectorRowPtr = std::unique_ptr<VectorRow<number, n>>;

    /**
     * @brief Type alias for a unique pointer to a dynamic VectorRow
     */
    template <Number number>
    using VectorRowDynamicPtr = std::unique_ptr<VectorRowDynamic<number>>;

    /**
     * @brief Type alias for a 2D Vector
     */
    template <Number number>
    using V2 = Vector<number, 2>;

    /**
     * @brief Type alias for a 3D Vector
     */
    template <Number number>
    using V3 = Vector<number, 3>;

    /**
     * @brief Type alias for a dynamic vector
     */
    template <Number number>
    using VD = Vector<number, Dynamic>;

    // Convenience vector aliases
    using V2f = V2<float>;
    using V2d = V2<double>;
    using V2i = V2<int>;
    using V2ui = V2<unsigned int>;

    using V3f = V3<float>;
    using V3d = V3<double>;
    using V3i = V3<int>;
    using V3ui = V3<unsigned int>;

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
    using TransposeVariant = std::variant<Vector<number, n>, Matrix<number, n, m>>;

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

    constexpr SizePair DynamicPair{Dynamic, Dynamic};

    /**
     * @brief Type alias for a 2x2 Matrix
     */
    template <Number number>
    using M2x2 = Matrix<number, 2, 2>;

    /**
     * @brief Type alias for a 3x3 Matrix
     */
    template <Number number>
    using M3x3 = Matrix<number, 3, 3>;

    /**
     * @brief Type alias for a dynamic matrix
     */
    template <Number number>
    using MD = Matrix<number, Dynamic, Dynamic>;

    // Convenience matrix aliases
    using M2x2f = M2x2<float>;
    using M2x2d = M2x2<double>;
    using M2x2i = M2x2<int>;
    using M2x2ui = M2x2<unsigned int>;

    using M3x3f = M3x3<float>;
    using M3x3d = M3x3<double>;
    using M3x3i = M3x3<int>;
    using M3x3ui = M3x3<unsigned int>;

    // ==================
    // Misc Aliases
    // ==================

    /**
     * @brief Row of optional numbers type alias.
     */
    template <Number number, int m>
    using RowOptional = std::conditional_t<m == -1, RowDynamic<std::optional<number>>, Row<std::optional<number>, m>>;

    template <Number number, int m, int n>
    using ConditionalOptionalRowOptional =
        std::conditional_t<m == Dynamic || n == Dynamic, std::optional<RowOptional<number, Dynamic>>,
                           std::optional<RowOptional<number, n - 1>>>;

    template <Number number, int m, int n>
    using ConditionalRowOptional =
        std::conditional_t<m == Dynamic || n == Dynamic, RowOptional<number, Dynamic>, RowOptional<number, n - 1>>;

    template <Number number, int m, int n>
    using ConditionalRowOptionalN =
        std::conditional_t<m == Dynamic || n == Dynamic, RowOptional<number, Dynamic>, RowOptional<number, n>>;

    using Seed = std::optional<size_t>;
}  // namespace mlinalg::structures
