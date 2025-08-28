/**
 * @file MatrixView.hpp
 * @brief Header file for the MatrixView class
 */

#pragma once

#include <functional>
#include <type_traits>

#include "../Concepts.hpp"
#include "../Helpers.hpp"
#include "Aliases.hpp"
#include "Container.hpp"
#include "VectorView.hpp"

namespace mlinalg::structures {
    template <typename D, Number number>
    class MatrixBase;

    template <Number number, Dim m, Dim n>
    class Matrix;

    template <Number number, Dim n, Dim newSize>
    using ViewArray = std::vector<VectorView<number, n, newSize>>;

    struct OffsetPair {
        long start;
        long end;
    };

    struct StridePair {
        long row;
        long col;
    };

    /**
     * @brief A container providing access to the columns of a matrix
     *
     *  Each column accessed through this container is a ColumnView, which provides
     *  efficient access to the elements of the column without copying the data.
     */
    template <Number number, Dim m, Dim n, Container T>
    class Columns {
        using MatrixType = T;
        using MatrixRef = MatrixType&;
        using Base = Columns<number, m, n, T>;
        using MatrixPtr = MatrixType*;

       public:
        using difference_type = std::ptrdiff_t;
        using value_type = number;
        using size_type = size_t;

        explicit Columns(MatrixPtr matrix) : matrix{matrix} {}

        Columns(Columns&&) = default;
        Columns& operator=(Columns&&) = default;
        Columns(const Columns& other) = default;
        Columns& operator=(const Columns& other) = default;
        ~Columns() = default;

        auto operator[](size_t j) { return ColumnView<number, m, n, MatrixType>{matrix, j}; }

        auto operator[](size_t j) const { return ColumnView<number, m, n, MatrixType>{matrix, j}; }

       private:
        MatrixPtr matrix;
    };

    /**
     * @brief A memory view of a matrix, allowing for efficient access to a subset of the matrix
     *
     * It supports all the operations of a matrix, but does not own the data. As such each
     * out-of-place operation (like addition, subtraction, multiplication, division) will return a new matrix
     * that is a copy of the original matrix with the operation applied, however, in-place operations
     * (like +=, -=, *=, /=) will modify the original matrix.
     *
     * A MatrixView is built directly upon VectorViews of the rows of the matrix, a design choice that allows
     * the matrix view to build upon existing VectorView functionality on a row level.
     *
     * @tparam number The type of the elements in the matrix
     * @tparam m The number of rows in the matrix
     * @tparam n The number of columns in the matrix
     * @tparam nM The number of rows in the view (default is m), used for converting to a concrete matrix
     * @tparam nN The number of columns in the view (default is n), used for converting to a concrete matrix
     * @tparam colOffsetT The offset for the columns in the view, used for constructing the VectorViews of the
     * MatrixView
     * @tparam colStrideT The stride for the columns in the view, used for constructing the VectorViews of the
     * MatrixView
     * @tparam T The type of the container that holds the matrix data
     * @param matrix A reference to the matrix data, which can be anything that satisfies the  Container concept
     * @param rowOffset The offset for the rows in the view, used to determine the starting and ending row indices
     * @param colOffset The offset for the columns in the view, used to determine the starting and ending column indices
     * @param stride The stride for the rows and columns in the view, used to determine how to access the elements
     * @return A MatrixView object that provides a view of the matrix data
     */
    template <Number number, Dim m, Dim n, Dim nM = m, Dim nN = n, OffsetPair colOffsetT = {0, n}, long colStrideT = 1>
    struct MatrixView : public MatrixBase<MatrixView<number, m, n, nM, nN, colOffsetT, colStrideT>, number> {
        using View = ViewArray<number, n, nN>;
        using ViewRef = View&;
        using MatrixType = helpers::IsDynamicT<m, n, MatrixArrayDynamic<number>, MatrixArray<number, m, n>>;
        using MatrixRef = MatrixType&;
        using Base = MatrixView<number, m, n, nM, nN, colOffsetT, colStrideT>;
        using iterator = View::iterator;
        using value_type = number;
        using size_type = View::size_type;

       private:
        /**
         * @brief Construct a view of the matrix based on the provided offsets and strides.
         *
         * @param A The matrix reference to construct the view from.
         * @param offset  The column offset used to create VectorViews of the rows of A.
         * @param stride The stride used to access the elements of the rows in A.
         * @return A View object containing VectorViews of the rows of A.
         */
        View constructView(MatrixRef A, OffsetPair offset, long stride) {
            View view;
            for (auto& row : A) {
                view.emplace_back(
                    row.template view<colOffsetT.start, colOffsetT.end, colStrideT>(offset.start, offset.end, stride));
            }
            return view;
        }

       public:
        static constexpr auto rows = nM;
        static constexpr auto cols = nN;

        MatrixView(MatrixRef matrix,                                //
                   OffsetPair rowOffset = {.start = 0, .end = -1},  //
                   OffsetPair colOffset = {.start = 0, .end = -1},  //
                   StridePair stride = {.row = 1, .col = 1}         //
                   )
            : viewArray{constructView(matrix, rowOffset, stride.row)},
              matrix{viewArray, rowOffset.start, rowOffset.end, stride.row},
              rowOffset{rowOffset},
              colOffset{colOffset},
              stride{stride} {}

        // ======================
        // Arithmetic Operations
        // ======================

        template <typename OtherD>
        friend auto operator+(const Base& lhs, const MatrixBase<OtherD, number>& other) {
            auto res = lhs.toMatrix();
            res += other;
            return res;
        }

        template <typename OtherD>
        friend auto operator-(const Base& lhs, const MatrixBase<OtherD, number>& other) {
            auto res = lhs.toMatrix();
            res -= other;
            return res;
        }

        friend auto operator*(const Base& lhs, const number& scalar) {
            auto res = lhs.toMatrix();
            res *= scalar;
            return res;
        }

        friend auto operator*(const number& scalar, const Base& rhs) { return rhs * scalar; }

        template <typename OtherD>
        friend auto operator*(const Base& lhs, const MatrixBase<OtherD, number>& rhs) {
            return lhs.toMatrix() * rhs;
        }

        template <typename OtherD>
        friend auto operator*(const MatrixBase<OtherD, number>& lhs, const Base& rhs) {
            return lhs * rhs.toMatrix();
        }

        friend auto operator/(const Base& lhs, const number& scalar) {
            auto res = lhs.toMatrix();
            res /= scalar;
            return res;
        }

        // ======================
        // Miscellaneous Operations
        // ======================

        /**
         * @brief Convert the MatrixView to a concrete Matrix type.
         *
         * @return A Matrix object of type Matrix<number, nM, nN> containing the data from the view.
         */
        auto toMatrix() const {
            Matrix<number, nM, nN> res(numRows(), numCols());
            for (size_t i = 0; i < numRows(); ++i) {
                for (size_t j = 0; j < numCols(); ++j) {
                    res(i, j) = matrix[i][j];
                }
            }
            return res;
        }

        /**
         * @brief Get the number of rows in the matrix view.
         *
         * @return The number of rows in the matrix view.
         */
        [[nodiscard]] size_t numRows() const { return static_cast<size_t>(matrix.size()); }

        [[nodiscard]] size_t size() const { return static_cast<size_t>(matrix.size()); }

        /**
         * @brief Get the number of columns in the matrix view.
         *
         * @return The number of columns in the matrix view.
         */
        [[nodiscard]] size_t numCols() const { return static_cast<size_t>(matrix[0].size()); }

        friend class Matrix<number, m, n>;

        friend class MatrixBase<Base, number>;

       protected:
        View viewArray;
        Columns<number, m, n, Base> columns{this};  // Columns of the matrix view
        container::StrideContainer<ViewRef> matrix;
        OffsetPair rowOffset{.start = 0, .end = -1};  // Start and end row index respectively
        OffsetPair colOffset{.start = 0, .end = -1};  // Start and end column index respectively
        StridePair stride{.row = 1, .col = 1};        // Row and column stride respectively
    };

    template <Number number, Dim m, Dim n, Dim nM = m, Dim nN = n, OffsetPair colOffsetT = {0, n}, long colStrideT = 1,
              Container T>
    inline auto View(T& matrix,                                       //
                     OffsetPair rowOffset = {.start = 0, .end = -1},  //
                     OffsetPair colOffset = {.start = 0, .end = -1},  //
                     StridePair stride = {.row = 1, .col = 1}         //
    ) {
        const long rows = matrix.size();
        const long cols = matrix.at(0).size();

        if (rowOffset.end < 0) rowOffset.end = rows;
        if (colOffset.end < 0) colOffset.end = cols;

        if (rowOffset.start < 0 || rowOffset.end > rows)
            throw mlinalg::stacktrace::StackError<std::out_of_range>{"Row offset out of range"};
        if (colOffset.start < 0 || colOffset.end > cols)
            throw mlinalg::stacktrace::StackError<std::out_of_range>{"Column offset out of range"};

        if (stride.row == 0 || stride.col == 0)
            throw mlinalg::stacktrace::StackError<std::invalid_argument>("Stride cannot be zero");
        return MatrixView<number, m, n, nM, nN, colOffsetT, colStrideT>{matrix, rowOffset, colOffset, stride};
    }

}  // namespace mlinalg::structures
