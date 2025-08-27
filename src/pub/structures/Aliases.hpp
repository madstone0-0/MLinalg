/**
 * @file Aliases.hpp
 * @brief Header file for type aliases
 */

#pragma once
#include <boost/container/small_vector.hpp>
#include <cstddef>
#include <optional>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "../Concepts.hpp"
#include "Allocator.hpp"
#include "Container.hpp"

namespace mlinalg::structures {
    using Dim = int;
    using SizeType = size_t;

    /**
     * @brief Dynamic size constant
     */
    static constexpr Dim Dynamic{-1};
    // =============================
    // Vector Aliases
    // =============================
    template <Number number, Dim n>
    class Vector;

    template <Number number>
    using DefaultAllocator = mlinalg::allocator::BootlegAllocator<number>;
    // using DefaultAllocator = std::allocator<number>;

    template <Number number>
    using VectorRowType = std::vector<number, DefaultAllocator<number>>;

    template <Number number, SizeType n>
    // using SmallVectorType = boost::container::small_vector<number, n, DefaultAllocator<number>>;
    using SmallVectorType = container::StaticContainer<number, n>;

    /**
     * @brief Threshold for small vector optimization
     *
     * If the size of the vector is less than this threshold, it will be stored inline on the stack
     *
     * @tparam T The type of the elements in the vector
     */
    template <typename T>
    constexpr std::size_t inlineThreshold = (256 + sizeof(T) - 1) / sizeof(T);

    /**
     * @brief  Type alias for the backing array of a Vector
     *
     * Supports small vector optimization for vectors with size less than 256 bytes.
     */
    template <Number number, SizeType n>
    using VectorRow =
        std::conditional_t<n >= inlineThreshold<number>, VectorRowType<number>, SmallVectorType<number, n>>;

    /**
     * @brief Type alias for the backing array of a dynamic Vector
     */
    template <Number number>
    using VectorRowDynamic = VectorRowType<number>;

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
    template <Dim n>
    using Vf = Vector<float, n>;
    template <Dim n>
    using Vd = Vector<double, n>;
    template <Dim n>
    using Vi = Vector<int, n>;
    template <Dim n>
    using Vui = Vector<unsigned int, n>;

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
    template <Number number, Dim m, Dim n>
    class Matrix;

    using SizePair = std::pair<Dim, Dim>;

    using SizeTPair = std::pair<SizeType, SizeType>;

    constexpr SizePair DynamicPair{Dynamic, Dynamic};

    template <Number number, Dim m, Dim n>
    using VectorVariant = std::conditional_t<n != 1, Vector<number, n>, Vector<number, m>>;

    /**
     * @brief Type alias for a variant of a Vector and a Matrix
     *
     * This is used to represent the result of a matrix or vector transposition. As the transpose of a vector is a 1xM
     * matrix and the transpose of a matrix is an NxM matrix, this variant is used to represent either of these
     */
    template <Number number, Dim m, Dim n>
    using TransposeVariant = std::variant<VectorVariant<number, m, n>, Matrix<number, n, m>>;

    template <Number number, Dim m, Dim n>
    using VectorTransposeVariant = std::variant_alternative_t<0, TransposeVariant<number, m, n>>;

    template <Number number, Dim m, Dim n>
    using MatrixTransposeVariant = std::variant_alternative_t<1, TransposeVariant<number, m, n>>;

    /**
     * @brief Type alias for a Vector as a row in a Matrix
     */
    template <Number number, Dim n>
    using Row = Vector<number, n>;

    template <Number number>
    using RowDynamic = Vector<number, Dynamic>;

    template <Number number, Dim m, Dim n>
    using TDArray = std::vector<Row<number, n>>;

    template <Number number>
    using TDArrayDynamic = std::vector<RowDynamic<number>>;

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
    template <Dim m, Dim n>
    using Mf = Matrix<float, m, n>;
    template <Dim m, Dim n>
    using Md = Matrix<double, m, n>;
    template <Dim m, Dim n>
    using Mi = Matrix<int, m, n>;
    template <Dim m, Dim n>
    using Mui = Matrix<unsigned int, m, n>;

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
    template <Number number, Dim m>
    using RowOptional =
        std::conditional_t<m == Dynamic, RowDynamic<std::optional<number>>, Row<std::optional<number>, m>>;

    template <Number number, Dim m, Dim n>
    using ConditionalOptionalRowOptional =
        std::conditional_t<m == Dynamic || n == Dynamic, std::optional<RowOptional<number, Dynamic>>,
                           std::optional<RowOptional<number, n - 1>>>;

    template <Number number, Dim m, Dim n>
    using ConditionalRow =
        std::conditional_t<m == Dynamic || n == Dynamic, Vector<number, Dynamic>, Vector<number, n - 1>>;

    template <Number number, Dim m, Dim n>
    using ConditionalRowOptional =
        std::conditional_t<m == Dynamic || n == Dynamic, RowOptional<number, Dynamic>, RowOptional<number, n - 1>>;

    template <Number number, Dim m, Dim n>
    using ConditionalRowOptionalN =
        std::conditional_t<m == Dynamic || n == Dynamic, RowOptional<number, Dynamic>, RowOptional<number, n>>;

    using Seed = std::optional<SizeType>;
}  // namespace mlinalg::structures
