/**
 * @file Matrix.hpp
 * @brief Header file for the Matrix class
 */

// FIX: Fix matrix-vector and vector-matrix multiplication order

#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "../Concepts.hpp"
#include "../Helpers.hpp"
#include "Aliases.hpp"
#include "MatrixOps.hpp"
#include "MatrixView.hpp"
#include "Vector.hpp"

using std::vector, std::array, std::optional, std::unique_ptr, std::shared_ptr, std::pair, std::invalid_argument,
    std::string, std::is_same_v, std::runtime_error, std::get;

namespace mlinalg::structures {
    /**
     * @class Shape
     * @brief A struct to represent the shape of a matrix
     *
     */
    struct Shape {
        size_t rows;
        size_t cols;
    };

    template <Number number, int m, int n>
    class Matrix;

    /**
     * @brief Matrix class for representing NxM matrices
     *
     * @param m Number of rows
     * @param n Number of columns
     */
    template <Number number, int m, int n>
    class Matrix {
       public:
        Matrix() = default;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

        // Constructor to keep consistency with the Dynamic Matrix specialization to allow them to be used
        // interchangeably
        Matrix(int nRows, int nCols) {}  // NOLINT

#pragma GCC diagnostic pop

        /**
         * @brief Construct a new Matrix object from an initializer list of row vectors
         *
         * @param rows  Initializer list of row vectors
         */
        constexpr Matrix(const std::initializer_list<std::initializer_list<number>>& rows) {
            static_assert(m > 0, "Number of rows cannot be 0");
            static_assert(n > 0, "Number of colmns cannot be 0");
            for (int i{}; i < m; i++) matrix.at(i) = Row<number, n>{*(rows.begin() + i)};
        }

        /**
         * @brief Copy construct a new Matrix object
         *
         * @param other Matrix to copy
         */
        Matrix(const Matrix& other) = default;

        /**
         * @brief Move construct a new Matrix object
         *
         * @param other  Matrix to move
         */
        Matrix(Matrix&& other) noexcept : matrix{std::move(other.matrix)} {}

        /**
         * @brief Copy assignment operator
         *
         * @param other Matrix to copy
         * @return A reference to the current matrix
         */
        Matrix& operator=(const Matrix& other) {
            if (this == &other) return *this;
            matrix = other.matrix;
            return *this;
        }

        /**
         * @brief Move assignment operator
         *
         * @param other The matrix to move
         * @return A reference to the current matrix
         */
        Matrix& operator=(Matrix&& other) noexcept {
            matrix = std::move(other.matrix);
            return *this;
        }

        static constexpr int rows = m;
        static constexpr int cols = n;

        /**
         * @brief Access the ith row of the matrix
         *
         * @param i The index of the row to access
         * @return A reference to the ith row
         */
        Row<number, n>& at(size_t i) { return matrixRowAt(matrix, i); }

        /**
         * @brief Const access the ith row of the matrix
         *
         * @param i The index of the row to access
         * @return A const reference to the ith row
         */
        Row<number, n> at(size_t i) const { return matrixRowAtConst(matrix, i); }

        /**
         * @brief Access the ith row of the matrix
         *
         * @param i The index of the row to access
         * @return A reference to the ith row
         */
        Row<number, n>& operator[](size_t i) { return matrix[i]; }

        /**
         * @brief Const access the ith row of the matrix
         *
         * @param i The index of the row to access
         * @return The ith row
         */
        Row<number, n> operator[](size_t i) const { return matrix[i]; }

        /**
         * @brief Access the element at the ith row and jth column
         *
         * @param i The index of the row
         * @param j The index of the column
         * @return A reference to the element at the ith row and jth column
         */
        number& at(size_t i, size_t j) { return matrixAt<number>(matrix, i, j); }

        /**
         * @brief Const cccess the element at the ith row and jth column
         *
         * @param i The index of the row
         * @param j The index of the column
         * @return A const reference to the element at the ith row and jth column
         */
        number at(size_t i, size_t j) const { return matrixAtConst<number>(matrix, i, j); }

        /**
         * @brief Access the element at the ith row and jth column
         *
         * @param i The index of the row
         * @param j The index of the column
         * @return A reference to the element at the ith row and jth column
         */
        number& operator()(size_t i, size_t j) { return matrix[i][j]; }

        /**
         * @brief Const access the element at the ith row and jth column
         *
         * NOTE: When using this operator in an expression it should be encased in parathese to avoid being
         * intepreted as a comma operator. For example, instead of writing:
         * `matrix(i, j) + 1`, you should write `(matrix(i, j)) + 1`. (2025-06-23 09:53)
         *
         * @param i The index of the row
         * @param j The index of the column
         * @return The element at the ith row and jth column
         */
        number& operator()(size_t i, size_t j) const { return matrix[i][j]; }

        /**
         * @brief Convert the columns of the matrix to a vector of column vectors
         *
         * @return A vector of column vectors
         */
        vector<Vector<number, m>> colToVectorSet() const { return matrixColsToVectorSet<number, m, n>(matrix); }

        /**
         * @brief Convert the rows of the matrix to a vector of row vectors
         *
         * @return A vector of row vectors
         */
        vector<Vector<number, n>> rowToVectorSet() const { return matrixRowsToVectorSet<number, m, n>(matrix); }

        /**
         * @brief Matrix multiplication by a scalar
         *
         * @param scalar  A scalar of the same type as the matrix
         * @return The matrix resulting from the multiplication
         */
        Matrix operator*(const number& scalar) const { return matrixScalarMult<number, m, n>(matrix, scalar); }

        /**
         * @brief Inplace matrix multiplication by a scalar
         *
         * @param scalar A scalar of the same type as the matrix
         * @return A reference to the current matrix
         */
        Matrix& operator*=(const number& scalar) {
            matrixScalarMultI<number, m, n>(matrix, scalar);
            return *this;
        }

        /**
         * @brief Matrix division by a scalar
         *
         * @param scalar A scalar of the same type as the matrix
         * @return The matrix resulting from the division
         */
        Matrix operator/(const number& scalar) const { return matrixScalarDiv<number, m, n>(matrix, scalar); }

        /**
         * @brief Inplace matrix division by a scalar
         *
         * @param scalar A scalar of the same type as the matrix
         * @return A reference to the current matrix
         */
        Matrix operator/=(const number& scalar) {
            matrixScalarDivI<number, m, n>(matrix, scalar);
            return *this;
        }

        /**
         * @brief Matrix addition
         *
         * @param other The matrix to add
         * @return The matrix resulting from the addition
         */
        Matrix operator+(const Matrix<number, m, n>& other) const {
            return matrixAdd<number, m, n>(matrix, other.matrix);
        }

        /**
         * @brief Inplace matrix addition
         *
         * @param other The matrix to add
         * @return reference to the current matrix
         */
        Matrix& operator+=(const Matrix<number, m, n>& other) {
            matrixAddI<number, m, n>(matrix, other.matrix);
            return *this;
        }

        /**
         * @brief Matrix subtraction
         *
         * @param other The matrix to subtract
         * @return The matrix resulting from the subtraction
         */
        Matrix operator-(const Matrix<number, m, n>& other) const {
            return matrixSub<number, m, n>(matrix, other.matrix);
        }

        /**
         * @brief Inplace matrix subtraction
         *
         * @param other The matrix to subtract
         * @return reference to the current matrix
         */
        Matrix& operator-=(const Matrix<number, m, n>& other) {
            matrixSubI<number, m, n>(matrix, other.matrix);
            return *this;
        }

        /**
         * @brief Equality operator for the matrix
         *
         * @param other The matrix to compare
         * @return True if the matrices are equal, i.e. are of the same dimensions and have the same elements
         */

        template <int otherM, int otherN>
        bool operator==(const Matrix<number, otherM, otherN>& other) const {
            return matrixEqual<number>(this->matrix, other.matrix);
        }

        /**
         * @brief Matrix multiplication by a vector
         *
         * @param vec The vector to multiply by
         * @return The vector resulting from the multiplication of size m
         */
        template <int nOther>
        Vector<number, m> operator*(const Vector<number, nOther>& vec) const {
            if (nOther != n) throw runtime_error("The columns of the matrix must be equal to the size of the vector");
            return multMatByVec<number, m, n>(matrix, vec);
        }

        template <int nOther>
        Vector<number, Dynamic> operator*(const Vector<number, nOther>& vec) const
            requires(nOther == Dynamic)
        {
            if (this->numCols() != vec.size())
                throw runtime_error("The columns of the matrix must be equal to the size of the vector");
            return multMatByVec<number, Dynamic, Dynamic>(matrix, vec);
        }

        /**
         * @brief Matrix multiplication by a matrix
         *
         * @param other
         * @return
         */
        template <int mOther, int nOther>
        Matrix<number, m, nOther> operator*(const Matrix<number, mOther, nOther>& other) const
            requires(m != Dynamic && n != Dynamic && mOther != Dynamic && nOther != Dynamic)
        {
            return MatrixMultiplication<number, m, n, mOther, nOther>(matrix, other.matrix);
        }

        template <int mOther, int nOther>
        Matrix<number, Dynamic, Dynamic> operator*(const Matrix<number, mOther, nOther>& other) const
            requires((n == Dynamic && m == Dynamic) || (mOther == Dynamic && nOther == Dynamic))
        {
            return MatrixMultiplication<number, Dynamic, Dynamic, Dynamic, Dynamic>(matrix, other.matrix);
        }

        /**
         * @brief Matrix multiplication by a transposed matrix
         *
         * @param other The transposed matrix to multiply by
         * @return The matrix resulting from the multiplication
         */
        template <int nOther>
        Matrix<number, m, nOther> operator*(const TransposeVariant<number, n, nOther>& other) const {
            return MatrixMultiplication<number, m, n, n, nOther>(matrix,
                                                                 helpers::extractMatrixFromTranspose(other).matrix);
        }

        /**
         * @brief Default destructor
         */
        ~Matrix() = default;

        /**
         * @brief Number of rows in the matrix
         *
         * @return The number of rows in the matrix
         */
        [[nodiscard]] constexpr size_t numRows() const { return static_cast<size_t>(m); }

        /**
         * @brief Number of columns in the matrix
         *
         * @return The number of columns in the matrix
         */
        [[nodiscard]] constexpr size_t numCols() const { return static_cast<size_t>(n); }

        /**
         * @brief Shape of the matrix
         *
         * @return The shape of the matrix
         */
        [[nodiscard]] constexpr Shape shape() const { return {numRows(), numCols()}; }

        explicit operator string() const { return matrixStringRepr(matrix); }

        friend std::ostream& operator<<(std::ostream& os, const Matrix<number, m, n>& system) {
            os << string(system);
            return os;
        }

        /**
         * @brief Transpose a mxn matrix to a nxm matrix
         *
         * @return The transposed matrix of size nxm
         */
        TransposeVariant<number, m, n> T() const { return TransposeMatrix<number, m, n>(matrix); }

        friend std::ostream& operator<<(std::ostream& os, const TransposeVariant<number, n, m>& system) {
            if (std::holds_alternative<Vector<number, m>>(system)) {
                os << get<Vector<number, m>>(system);
            } else if (std::holds_alternative<Matrix<number, m, n>>(system)) {
                os << get<Matrix<number, m, n>>(system);
            }
            return os;
        }

        /**
         * @brief Begin iterator for the matrix
         *
         * @return the beginning iterator for the matrix
         */
        constexpr auto begin() const { return matrix.begin(); }

        /**
         * @brief End iterator for the matrix
         *
         * @return The end iterator for the matrix
         */
        constexpr auto end() const { return matrix.end(); }

        /**
         * @brief Const begin iterator for the matrix
         *
         * @return The const begin iterator for the matrix
         */
        constexpr auto cbegin() const { return matrix.cbegin(); }

        /**
         * @brief The const end iterator for the matrix
         *
         * @return The const end iterator for the matrix
         */
        constexpr auto cend() const { return matrix.cend(); }

        /**
         * @brief Return the last row of the matrix
         *
         * @return The last row of the matrix
         */
        constexpr auto back() { return matrix.back(); }

        /**
         * @brief Const reverse begin iterator for the matrix
         *
         * @return The const reverse begin iterator for the matrix
         */
        constexpr auto rbegin() { return matrix.rbegin(); }

        /**
         * @brief Const reverse end iterator for the matrix
         *
         * @return The const reverse end iterator for the matrix
         */
        constexpr auto rend() { return matrix.rend(); }

        /**
         * @brief Augment the matrix with another matrix
         *
         * @param other The matrix to augment with
         * @return The augmented matrix of size mx(n + nN)
         */
        template <int nN>
        Matrix<number, m, nN + n> augment(const Matrix<number, m, nN>& other) const {
            return MatrixAugmentMatrix<number, m, n, nN>(matrix, other.matrix);
        }

        /**
         * @brief Augment the matrix with a vector
         *
         * @param other The vector to augment with
         * @return The augmented matrix of size mx(n + 1)
         */
        Matrix<number, m, n + 1> augment(const Vector<number, m>& other) const {
            return MatrixAugmentVector<number, m, n>(matrix, *other.row);
        }

        /**
         * @brief Determinant of the matrix
         *
         * @return The determinant of the matrix
         */
        number det() const {
            if (m != n) throw runtime_error("Finding determinant of rectangular matrices is not defined");
            if constexpr (m == 2 && n == 2)
                return MatrixDet2x2<number>(matrix);
            else
                return MatrixCofactor<number, m, n>(matrix);
        }

        /**
         * @brief Subset the matrix by removing a row and a column
         *
         * @param i Row index to remove
         * @param j Column index to remove
         * @return The subsetted matrix of size (m - 1)x(n - 1)
         */
        Matrix<number, m - 1, n - 1> subset(optional<int> i, optional<int> j) const {
            return MatrixSubset<number, m, n>(matrix, i, j);
        }

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
        /**
         * @brief Slice the matrix
         *
         * @return The sliced matrix of size (m - (i1 - i0))x(n - (j1 - j0))
         */
        template <int i0, int i1, int j0, int j1>
        Matrix<number, (i1 - i0), (j1 - j0)> slice(SizeTPair i = {0, 0}, SizeTPair j = {0, 0}) const {
            return MatrixSlice<i0, i1, j0, j1, number, m, n>(matrix);
        }
#pragma GCC diagnostic pop

        /**
         * @brief Create a view of the matrix
         *
         * @param rowOffset The starting row index
         * @param colOffset The starting column index
         * @param rowStride The stride between rows
         * @param colStride The stride between columns
         * @return A MatrixView object of the matrix
         */
        template <size_t i, size_t j>
        MatrixView<number, m, n> view(size_t rowOffset = 0, size_t colOffset = 0, size_t rowStride = 1,
                                      size_t colStride = 1) {
            return View<number, m, n, i, j>(matrix, rowOffset, colOffset, rowStride, colStride);
        }

        /**
         * @brief Calculate the Frobenius norm of the matrix
         *
         * This is the same as the L2 norm of the matrix
         *
         * \f[
         * ||A||_F = \sqrt{\sum_{i=1}^{m} \sum_{j=1}^{n} |a_{ij}|^2}
         * \f]
         *
         * @return The Frobenius norm of the matrix
         */
        double frob() { return FrobenisNorm<number, m, n>(matrix); }

        /**
         * @brief Calculate the L1 norm of the matrix
         *
         * This is the maximum absolute column sum of the matrix
         *
         * \f[
         * ||A||_1 = \max(\sum_{i=1}^n(|a_j|))
         * \f]
         *
         * @return The L1 norm of the matrix
         */
        double l1() { return L1Norm<number, m, n>(*this); }

        /**
         * @brief Calculate the L-inf norm of the matrix
         *
         * This is the maximum absolute row sum of the matrix
         *
         * \f[
         * ||A||_\infty = \max(\sum_{j=1}^m(|a_i|))
         * \f]
         *
         * @return The L-inf norm of the matrix
         */
        double lInf() { return LInfNorm<number, m, n>(*this); }

        /**
         * @brief Calculate the trace of the matrix, i.e. the sum of the diagonal elements
         *
         * @return The trace of the matrix
         */
        number trace() const { return MatrixTrace<number, m>(matrix); }

        auto getMatrix() const { return matrix; }

       private:
        template <Number num, int mM, int nN>
        friend class Matrix;

        template <Number num, int nN>
        friend class Vector;

        template <Number num, int mM, int nN, int nOther, Container T, Container U>
        friend Matrix<number, m, nOther> strassen(const T& A, const U& B);

        /**
         * @brief Swap the contents of two matrices for copy swap idiom
         *
         * @param first  The first matrix
         * @param second  The second matrix
         */
        friend void swap(Matrix& first, Matrix& second) noexcept {
            using std::swap;
            swap(first.matrix, second.matrix);
        }

        void checkDimensions() {
            if constexpr (n <= 0) throw invalid_argument("Matrix n must be greater than zero");
            if constexpr (m <= 0) throw invalid_argument("Matrix m must be greater than zero");
        }

        /**
         * @brief Backing array for the matrix
         */
        TDArray<number, m, n> matrix{};
    };

    template <Number number, int m, int n, int nOther>
    TransposeVariant<number, nOther, m> operator*(TransposeVariant<number, m, n> lhs, Matrix<number, n, nOther> rhs) {
        if (std::holds_alternative<Vector<number, n>>(lhs)) {
            auto vec = get<Vector<number, n>>(lhs);
            return vec * rhs;
        } else {
            auto mat = get<Matrix<number, n, m>>(lhs);
            return mat * rhs;
        }
    }

    template <Number number, int m, int n>
    TransposeVariant<number, n, n> operator*(TransposeVariant<number, m, n> lhs, Matrix<number, m, n> rhs)
        requires(m != n)
    {
        if (std::holds_alternative<Vector<number, n>>(lhs)) {
            auto vec = get<Vector<number, n>>(lhs);
            const auto& res = vec.T() * rhs;
            return helpers::extractVectorFromTranspose(res.T());
        } else {
            auto mat = get<Matrix<number, n, m>>(lhs);
            return mat * rhs;
        }
    }

    template <Number number, int m, int n>
    TransposeVariant<number, m, n> operator*(const number& lhs, const TransposeVariant<number, m, n>& rhs) {
        if (std::holds_alternative<Vector<number, m>>(rhs)) {
            auto vec = get<Vector<number, m>>(rhs);
            return lhs * vec;
        } else {
            auto mat = get<Matrix<number, m, n>>(rhs);
            return lhs * mat;
        }
    }

    template <Number number, int m, int n>
    Matrix<number, m, n> operator*(const number& scalar, Matrix<number, m, n> rhs) {
        return rhs * scalar;
    }

}  // namespace mlinalg::structures

namespace mlinalg::structures {
    /**
     * @brief Dynamic Matrix class for representing NxM matrices
     *
     * @param m  Number of rows
     * @param n  Number of columns
     */
    template <Number number>
    class Matrix<number, Dynamic, Dynamic> {
       public:
        Matrix(int m, int n) : m(m), n(n) {
            if (m <= 0) throw invalid_argument("Matrix must have at least one row");
            if (n <= 0) throw invalid_argument("Matrix must have at least one column");
            matrix.reserve(m);
            matrix.resize(m, Vector<number, Dynamic>(n));
        }

        Matrix(const std::initializer_list<std::initializer_list<number>>& rows)
            : m{rows.size()}, n{rows.begin()->size()} {
            if (m <= 0) throw invalid_argument("Matrix must have at least one row");
            if (n <= 0) throw invalid_argument("Matrix must have at least one column");

            for (const auto& row : rows) {
                if (row.size() != n) throw invalid_argument("All rows must have the same number of columns");
            }

            matrix.reserve(m);
            for (const auto& row : rows) {
                matrix.emplace_back(row);
            }
        }

        template <int m, int n>
        explicit Matrix(const Matrix<number, m, n>& other)
            requires(m != -1 && n != -1)
            : m{m}, n{n} {
            matrix.reserve(m);
            for (const auto& row : other.matrix) matrix.emplace_back(row);
        }

        static constexpr int rows = Dynamic;
        static constexpr int cols = Dynamic;

        /**
         * @brief Access the ith row of the matrix
         *
         * @param i The index of the row to access
         * @return A reference to the ith row
         */
        RowDynamic<number>& at(int i) { return matrixRowAt(matrix, i); }

        /**
         * @brief Const access the ith row of the matrix
         *
         * @param i The index of the row to access
         * @return A const reference to the ith row
         */
        RowDynamic<number> at(int i) const { return matrixRowAtConst(matrix, i); }

        /**
         * @brief Access the ith row of the matrix
         *
         * @param i The index of the row to access
         * @return A reference to the ith row
         */
        RowDynamic<number>& operator[](size_t i) { return matrixRowAt(matrix, i); }

        /**
         * @brief Const access the ith row of the matrix
         *
         * @param i The index of the row to access
         * @return The ith row
         */
        RowDynamic<number> operator[](size_t i) const { return matrixRowAtConst(matrix, i); }

        /**
         * @brief Access the element at the ith row and jth column
         *
         * @param i The index of the row
         * @param j The index of the column
         * @return A reference to the element at the ith row and jth column
         */
        number& at(size_t i, size_t j) { return matrixAt<number>(matrix, i, j); }

        /**
         * @brief Const access the element at the ith row and jth column
         *
         * @param i The index of the row
         * @param j The index of the column
         * @return The element at the ith row and jth column
         */
        number at(size_t i, size_t j) const { return matrixAtConst<number>(matrix, i, j); }

        /**
         * @brief Access the element at the ith row and jth column
         *
         * @param i The index of the row
         * @param j The index of the column
         * @return A reference to the element at the ith row and jth column
         */
        number& operator()(size_t i, size_t j) { return matrix[i][j]; }

        /**
         * @brief Const access the element at the ith row and jth column
         *
         * @param i The index of the row
         * @param j The index of the column
         * @return The element at the ith row and jth column
         */
        number operator()(size_t i, size_t j) const { return matrix[i][j]; }

        /**
         * @brief Copy construct a new Matrix object
         *
         * @param other Matrix to copy
         */
        Matrix(const Matrix& other) = default;

        /**
         * @brief Move construct a new Matrix object
         *
         * @param other  Matrix to move
         */
        Matrix(Matrix&& other) noexcept : m{other.m}, n{other.n}, matrix{std::move(other.matrix)} {
            other.m = 0;
            other.n = 0;
        }

        /**
         * @brief Copy assignment operator
         *
         * @param other Matrix to copy
         * @return A reference to the current matrix
         */
        Matrix& operator=(const Matrix& other) {
            if (this == &other) return *this;
            matrix = other.matrix;
            n = other.n;
            m = other.m;
            return *this;
        }

        /**
         * @brief Move assignment operator
         *
         * @param other The matrix to move
         * @return A reference to the current matrix
         */
        Matrix& operator=(Matrix&& other) noexcept {
            matrix = std::move(other.matrix);
            m = other.m;
            n = other.n;
            other.m = 0;
            other.n = 0;
            return *this;
        }

        /**
         * @brief Convert the columns of the matrix to a vector of column vectors
         *
         * @return A vector of column vectors
         */
        vector<Vector<number, Dynamic>> colToVectorSet() const {
            return matrixColsToVectorSet<number, Dynamic, Dynamic>(matrix);
        }

        /**
         * @brief Convert the rows of the matrix to a vector of row vectors
         *
         * @return A vector of row vectors
         */
        vector<Vector<number, Dynamic>> rowToVectorSet() const {
            return matrixRowsToVectorSet<number, Dynamic, Dynamic>(matrix);
        }

        /**
         * @brief Matrix multiplication by a scalar
         *
         * @param scalar  A scalar of the same type as the matrix
         * @return The matrix resulting from the multiplication
         */
        Matrix<number, Dynamic, Dynamic> operator*(const number& scalar) const {
            return matrixScalarMult<number, Dynamic, Dynamic>(matrix, scalar);
        }

        /**
         * @brief Inplace matrix multiplication by a scalar
         *
         * @param scalar A scalar of the same type as the matrix
         * @return A reference to the current matrix
         */
        Matrix<number, Dynamic, Dynamic> operator*=(const number& scalar) {
            matrixScalarMultI<number, Dynamic, Dynamic>(matrix, scalar);
            return *this;
        }

        /**
         * @brief Matrix division by a scalar
         *
         * @param scalar A scalar of the same type as the matrix
         * @return The matrix resulting from the division
         */
        Matrix<number, Dynamic, Dynamic> operator/(const number& scalar) const {
            return matrixScalarDiv<number, Dynamic, Dynamic>(matrix, scalar);
        }

        /**
         * @brief Inplace matrix division by a scalar
         *
         * @param scalar A scalar of the same type as the matrix
         * @return A reference to the current matrix
         */
        Matrix<number, Dynamic, Dynamic> operator/=(const number& scalar) {
            matrixScalarDivI<number, Dynamic, Dynamic>(matrix, scalar);
            return *this;
        }

        /**
         * @brief Matrix addition
         *
         * @param other The matrix to add
         * @return The matrix resulting from the addition
         */
        template <int otherM, int otherN>
        Matrix<number, Dynamic, Dynamic> operator+(const Matrix<number, otherM, otherN>& other) const {
            return matrixAdd<number, Dynamic, Dynamic>(matrix, other.matrix);
        }

        /**
         * @brief Inplace matrix addition
         *
         * @param other The matrix to add
         * @return reference to the current matrix
         */
        template <int otherM, int otherN>
        Matrix& operator+=(const Matrix<number, otherM, otherN>& other) {
            matrixAddI<number, Dynamic, Dynamic>(matrix, other.matrix);
            return *this;
        }

        template <int m, int n, int otherM, int otherN>
        friend Matrix<number, Dynamic, Dynamic> operator+(const Matrix<number, otherM, otherN>& lhs,
                                                          const Matrix<number, m, n> rhs)
            requires((n == Dynamic && m == Dynamic) && (otherN != Dynamic && otherM != Dynamic))
        {
            return matrixAdd<number, Dynamic, Dynamic>(lhs.matrix, rhs.matrix);
        }

        /**
         * @brief Matrix subtraction
         *
         * @param other The matrix to subtract
         * @return The matrix resulting from the subtraction
         */
        template <int otherM, int otherN>
        Matrix<number, Dynamic, Dynamic> operator-(const Matrix<number, otherM, otherN>& other) const {
            return matrixSub<number, Dynamic, Dynamic>(matrix, other.matrix);
        }

        /**
         * @brief Matrix subtraction
         *
         * @param other The matrix to subtract
         * @return The matrix resulting from the subtraction
         */
        template <int otherM, int otherN>
        Matrix<number, Dynamic, Dynamic> operator-(const TransposeVariant<number, otherM, otherN>& otherT) const {
            if (std::holds_alternative<Matrix<number, otherM, otherN>>(otherT)) {
                auto other = get<Matrix<number, otherM, otherN>>(otherT);
                return matrixSub<number, Dynamic, Dynamic>(matrix, other.matrix);
            }
            throw std::logic_error("Cannot subtract a vector from a matrix");
        }

        /**
         * @brief Inplace matrix subtraction
         *
         * @param other The matrix to subtract
         * @return reference to the current matrix
         */
        template <int otherM, int otherN>
        Matrix& operator-=(const Matrix<number, otherM, otherN>& other) {
            matrixSubI<number, Dynamic, Dynamic>(matrix, other.matrix);
            return *this;
        }

        template <int m, int n, int otherM, int otherN>
        friend Matrix<number, Dynamic, Dynamic> operator-(const Matrix<number, otherM, otherN>& lhs,
                                                          const Matrix<number, m, n> rhs)
            requires((n == Dynamic && m == Dynamic) && (otherN != Dynamic && otherM != Dynamic))
        {
            return matrixSub<number, Dynamic, Dynamic>(lhs.matrix, rhs.matrix);
        }

        /**
         * @brief Equality operator for the matrix
         *
         * @param other The matrix to compare
         * @return True if the matrices are equal, i.e. are of the same dimensions and have the same elements
         */
        template <int otherM, int otherN>
        bool operator==(const Matrix<number, otherM, otherN>& other) const {
            return matrixEqual<number>(this->matrix, other.matrix);
        }

        /**
         * @brief Matrix multiplication by a vector
         *
         * @param vec The vector to multiply by
         * @return The vector resulting from the multiplication of size m
         */

        template <int nOther>
        Vector<number, Dynamic> operator*(const Vector<number, nOther>& vec) const {
            return multMatByVec<number, Dynamic, Dynamic, Dynamic>(matrix, vec);
        }

        /**
         * @brief Matrix multiplication by a matrix
         *
         * @param other
         * @return
         */
        template <int otherM, int otherN>
        Matrix<number, Dynamic, Dynamic> operator*(const Matrix<number, otherM, otherN>& other) const {
            return MatrixMultiplication<number, Dynamic, Dynamic, Dynamic, Dynamic>(matrix, other.matrix);
        }

        template <int m, int n, int otherM, int otherN>
        friend Matrix<number, Dynamic, Dynamic> operator*(const Matrix<number, otherM, otherN>& lhs,
                                                          const Matrix<number, m, n> rhs)
            requires((n == Dynamic && m == Dynamic) && (otherN != Dynamic && otherM != Dynamic))
        {
            return MatrixMultiplication<number, Dynamic, Dynamic, Dynamic>(lhs.matrix, rhs.matrix);
        }

        /**
         * @brief Matrix multiplication by a transposed matrix
         *
         * @param other The transposed matrix to multiply by
         * @return The matrix resulting from the multiplication
         */
        template <int otherM, int otherN>
        Matrix<number, Dynamic, Dynamic> operator*(const TransposeVariant<number, otherM, Dynamic>& other) const {
            return MatrixMultiplication<number, Dynamic, Dynamic, Dynamic>(helpers::extractMatrixFromTranspose(other));
        }

        template <int m, int n, int otherM, int otherN>
        friend Matrix<number, Dynamic, Dynamic> operator*(const TransposeVariant<number, otherM, otherN>& lhs,
                                                          const Matrix<number, m, n> rhs)
            requires((n == Dynamic && m == Dynamic) && (otherN != Dynamic && otherM != Dynamic))
        {
            return MatrixMultiplication<number, Dynamic, Dynamic, Dynamic>(helpers::extractMatrixFromTranspose(lhs),
                                                                           rhs.matrix);
        }

        /**
         * @brief Transpose a mxn matrix to a nxm matrix
         *
         * @return The transposed matrix of size nxm
         */
        TransposeVariant<number, Dynamic, Dynamic> T() const {
            return TransposeMatrix<number, Dynamic, Dynamic>(matrix);
        }

        /**
         * @brief Default destructor
         */
        ~Matrix() = default;

        /**
         * @brief Number of rows in the matrix
         *
         * @return
         */
        [[nodiscard]] size_t numRows() const { return m; }

        /**
         * @brief Number of columns in the matrix
         *
         * @return
         */
        [[nodiscard]] size_t numCols() const { return n; }

        /**
         * @brief Shape of the matrix
         *
         * @return The shape of the matrix
         */
        [[nodiscard]] Shape shape() const { return {numRows(), numCols()}; }

        explicit operator string() const { return matrixStringRepr(matrix); }

        friend std::ostream& operator<<(std::ostream& os, const Matrix<number, Dynamic, Dynamic>& system) {
            os << string(system);
            return os;
        }

        /**
         * @brief Begin iterator for the matrix
         *
         * @return the beginning iterator for the matrix
         */
        constexpr auto begin() const { return matrix.begin(); }

        /**
         * @brief End iterator for the matrix
         *
         * @return The end iterator for the matrix
         */
        constexpr auto end() const { return matrix.end(); }

        /**
         * @brief Const begin iterator for the matrix
         *
         * @return The const begin iterator for the matrix
         */
        constexpr auto cbegin() const { return matrix.cbegin(); }

        /**
         * @brief The const end iterator for the matrix
         *
         * @return The const end iterator for the matrix
         */
        constexpr auto cend() const { return matrix.cend(); }

        /**
         * @brief Return the last row of the matrix
         *
         * @return The last row of the matrix
         */
        constexpr auto back() { return matrix.back(); }

        /**
         * @brief Const reverse begin iterator for the matrix
         *
         * @return The const reverse begin iterator for the matrix
         */
        constexpr auto rbegin() { return matrix.rbegin(); }

        /**
         * @brief Const reverse end iterator for the matrix
         *
         * @return The const reverse end iterator for the matrix
         */
        constexpr auto rend() { return matrix.rend(); }

        /**
         * @brief Augment the matrix with another matrix
         *
         * @param other The matrix to augment with
         * @return The augmented matrix of size mx(n + nN)
         */
        Matrix<number, Dynamic, Dynamic> augment(const Matrix<number, Dynamic, Dynamic>& other) const {
            return MatrixAugmentMatrix<number, Dynamic, 0, Dynamic>(matrix, other.matrix);
        }

        /**
         * @brief Augment the matrix with a vector
         *
         * @param other The vector to augment with
         * @return The augmented matrix of size mx(n + 1)
         */
        Matrix<number, Dynamic, Dynamic> augment(const Vector<number, Dynamic>& other) const {
            return MatrixAugmentVector<number, Dynamic, -2>(matrix, *other.row);
        }

        /**
         * @brief Subset the matrix by removing a row and a column
         *
         * @param i Row index to remove
         * @param j Column index to remove
         * @return The subsetted matrix of size (m - 1)x(n - 1)
         */
        Matrix<number, Dynamic, Dynamic> subset(optional<int> i, optional<int> j) const {
            return MatrixSubset<number, 0, 0>(matrix, i, j);
        }

        /**
         * @brief Slice the matrix
         *
         * @param i SlicePair for the rows in the form {i0, i1}
         * @param j SlicePair for the columns in the form {j0, j1}
         * @return The sliced matrix of size (m - (i1 - i0))x(n - (j1 - j0))
         */
        template <int i0 = 0, int i1 = 0, int j0 = 0, int j1 = 0>
        Matrix<number, Dynamic, Dynamic> slice(const SizeTPair& i, const SizeTPair& j) const {
            return MatrixSlice<number>(matrix, i, j);
        }

        /**
         * @brief Determinant of the matrix
         *
         * @return The determinant of the matrix
         */
        number det() const {
            if (m != n) throw runtime_error("Finding determinant of rectangular matrices is not defined");
            if (m == 2 && n == 2)
                return MatrixDet2x2<number>(matrix);
            else
                return MatrixCofactor<number, Dynamic, Dynamic>(matrix);
        }

        /**
         * @brief Create a view of the matrix
         *
         * @param rowOffset The starting row index
         * @param colOffset The starting column index
         * @param rowStride The stride between rows
         * @param colStride The stride between columns
         * @return A MatrixView object of the matrix
         */
        template <size_t i, size_t j>
        MatrixView<number, Dynamic, Dynamic> view(size_t rowOffset = 0, size_t colOffset = 0, size_t rowStride = 1,
                                                  size_t colStride = 1) {
            return View<number, Dynamic, Dynamic, i, j>(matrix, rowOffset, colOffset, rowStride, colStride);
        }

        /**
         * @brief Calculate the Frobenius norm of the matrix
         *
         * This is the same as the L2 norm of the matrix
         *
         * \f[
         * ||A||_F = \sqrt{\sum_{i=1}^{m} \sum_{j=1}^{n} |a_{ij}|^2}
         * \f]
         *
         * @return The Frobenius norm of the matrix
         */
        double frob() { return FrobenisNorm<number, Dynamic, Dynamic>(matrix); }

        /**
         * @brief Calculate the L1 norm of the matrix
         *
         * This is the maximum absolute column sum of the matrix
         *
         * \f[
         * ||A||_1 = \max(\sum_{i=1}^n(|a_j|))
         * \f]
         *
         * @return The L1 norm of the matrix
         */
        double l1() { return L1Norm<number, Dynamic, Dynamic>(*this); }

        /**
         * @brief Calculate the L-inf norm of the matrix
         *
         * This is the maximum absolute row sum of the matrix
         *
         * \f[
         * ||A||_\infty = \max(\sum_{j=1}^m(|a_i|))
         * \f]
         *
         * @return The L-inf norm of the matrix
         */
        double lInf() { return LInfNorm<number, Dynamic, Dynamic>(*this); }

        /**
         * @brief Calculate the trace of the matrix, i.e. the sum of the diagonal elements
         *
         * @return The trace of the matrix
         */
        number trace() const { return MatrixTrace<number, Dynamic>(matrix); }

       private:
        template <Number num, int mM, int nN>
        friend class Matrix;

        template <Number num, int nN>
        friend class Vector;

        size_t m;
        size_t n;
        TDArrayDynamic<number> matrix;
    };

}  // namespace mlinalg::structures
