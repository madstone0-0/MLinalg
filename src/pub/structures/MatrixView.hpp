/**
 * @file MatrixView.hpp
 * @brief Header file for the MatrixView class
 */

#pragma once

#include "../Concepts.hpp"
#include "../Helpers.hpp"
#include "Aliases.hpp"

namespace mlinalg::structures {
    template <Number number, int m, int n>
    using MatrixViewBase = helpers::IsDynamicT<m, n, TDArrayDynamic<number>, TDArray<number, m, n>>;

    template <Number number, int m, int n>
    struct MatrixView {
       public:
        MatrixView(MatrixViewBase<number, m, n>* matrix, size_t rowOffset = 0, size_t colOffset = 0,
                   size_t rowStride = 1, size_t colStride = 1)
            : matrix(matrix), rowOffset(rowOffset), colOffset(colOffset), rowStride(rowStride), colStride(colStride) {}

        /**
         * @brief  Read-write access to the element at the ith row and jth column
         *
         * @param i  The index of the row
         * @param j  The index of the column
         * @return A reference to the element at the ith row and jth column
         */
        number& operator()(size_t i, size_t j) {
            return (*matrix)[rowOffset + (i * rowStride)][colOffset + (j * colStride)];
        }

        /**
         * @brief Read-only access to the element at the ith row and jth column
         *
         * @param i The index of the row
         * @param j  The index of the column
         * @return A const reference to the element at the ith row and jth column
         */
        const number& operator()(size_t i, size_t j) const {
            return (*matrix)[rowOffset + (i * rowStride)][colOffset + (j * colStride)];
        }

        friend class Matrix<number, m, n>;

       protected:
        MatrixViewBase<number, m, n>* matrix;
        size_t rowOffset{};   // Starting row index
        size_t colOffset{};   // Starting column index
        size_t rowStride{1};  // Stride between rows
        size_t colStride{1};  // Stride between columns
    };

    template <Number number, int m, int n, size_t i, size_t j, Container T>
    inline MatrixView<number, m, n> View(T& matrix, size_t rowOffset = 0, size_t colOffset = 0, size_t rowStride = 1,
                                         size_t colStride = 1) {
        const auto rows = matrix.size();
        const auto cols = matrix.at(0).size();

        if (i > rows || j > cols)
            throw mlinalg::stacktrace::StackError<std::out_of_range>("View dimensions exceed matrix size");
        if (rowOffset >= rows || colOffset >= cols)
            throw mlinalg::stacktrace::StackError<std::out_of_range>("Offset out of range");
        return MatrixView<number, m, n>{&matrix, rowOffset, colOffset, rowStride, colStride};
    }

}  // namespace mlinalg::structures
