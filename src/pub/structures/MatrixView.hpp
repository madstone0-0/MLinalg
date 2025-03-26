/**
 * @file MatrixView.hpp
 * @brief Header file for the MatrixView class
 */

#pragma once

#include "../Concepts.hpp"
#include "Aliases.hpp"

namespace mlinalg::structures {
    /**
     * @brief MatrixView class for representing a memory view of a matrix
     *
     * @param i Row index
     * @param j Column index
     */
    template <Number number, int m, int n>
    struct MatrixView {
        TDArray<number, m, n>* matrix;
        size_t rowOffset{};   // Starting row index
        size_t colOffset{};   // Starting column index
        size_t rowStride{1};  // Stride between rows
        size_t colStride{1};  // Stride between columns

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
    };

    template <Number number>
    struct MatrixView<number, Dynamic, Dynamic> {
        TDArrayDynamic<number>* matrix;
        size_t rowOffset{};   // Starting row index
        size_t colOffset{};   // Starting column index
        size_t rowStride{1};  // Stride between rows
        size_t colStride{1};  // Stride between columns

        // Read-write access
        number& operator()(size_t i, size_t j) {
            return (*matrix)[rowOffset + (i * rowStride)][colOffset + (j * colStride)];
        }

        // Read-only access
        const number& operator()(size_t i, size_t j) const {
            return (*matrix)[rowOffset + (i * rowStride)][colOffset + (j * colStride)];
        }

        friend class Matrix<number, Dynamic, Dynamic>;
    };

}  // namespace mlinalg::structures
