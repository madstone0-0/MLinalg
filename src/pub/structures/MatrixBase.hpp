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

    /**
     * @brief Matrix class for representing MxN matrices
     *
     * @param m Number of rows
     * @param n Number of columns
     */
    template <typename D, Number number>
    class MatrixBase {
       protected:
        // CRTP helpers
        D& d() { return static_cast<D&>(*this); }
        const D& d() const { return static_cast<const D&>(*this); }

        // Protected constructor - only derived classes can construct
        MatrixBase() = default;
        ~MatrixBase() = default;

       public:
        using value_type = number;
        using size_type = size_t;
        using ref = number&;
        using const_ref = const number&;

        // ======================
        // Indexing and Accessors
        // ======================

        /**
         * @brief Access the ith row of the matrix
         *
         * @param i The index of the row to access
         * @return A reference to the ith row
         */
        auto& at(size_t i) { return matrixRowAt(d().matrix, i); }

        /**
         * @brief Const access the ith row of the matrix
         *
         * @param i The index of the row to access
         * @return A const reference to the ith row
         */
        auto at(size_t i) const { return matrixRowAtConst(d().matrix, i); }

        /**
         * @brief Access the ith row of the matrix
         *
         * @param i The index of the row to access
         * @return A reference to the ith row
         */
        auto& operator[](size_t i) { return d().matrix[i]; }

        /**
         * @brief Const access the ith row of the matrix
         *
         * @param i The index of the row to access
         * @return The ith row
         */
        auto operator[](size_t i) const { return d().matrix[i]; }

        /**
         * @brief Access the element at the ith row and jth column
         *
         * @param i The index of the row
         * @param j The index of the column
         * @return A reference to the element at the ith row and jth column
         */
        number& at(size_t i, size_t j) { return matrixAt<number>(d().matrix, i, j); }

        /**
         * @brief Const cccess the element at the ith row and jth column
         *
         * @param i The index of the row
         * @param j The index of the column
         * @return A const reference to the element at the ith row and jth column
         */
        number at(size_t i, size_t j) const { return matrixAtConst<number>(d().matrix, i, j); }

        /**
         * @brief Access the element at the ith row and jth column
         *
         * @param i The index of the row
         * @param j The index of the column
         * @return A reference to the element at the ith row and jth column
         */
        number& operator()(size_t i, size_t j) { return d().matrix[i][j]; }

        /**
         * @brief Const access the element at the ith row and jth column
         *
         * NOTE: When using this operator in an expression it should be encased in parantheses to avoid being
         * intepreted as a comma operator. For example, instead of writing:
         * `matrix(i, j) + 1`, you should write `(matrix(i, j)) + 1`. (2025-06-23 09:53)
         *
         * @param i The index of the row
         * @param j The index of the column
         * @return The element at the ith row and jth column
         */
        number& operator()(size_t i, size_t j) const { return d().matrix[i][j]; }

        // ============
        // Iteration
        // ============

        /**
         * @brief Begin iterator for the matrix
         *
         * @return the beginning iterator for the matrix
         */
        constexpr auto begin() const { return d().matrix.begin(); }

        /**
         * @brief End iterator for the matrix
         *
         * @return The end iterator for the matrix
         */
        constexpr auto end() const { return d().matrix.end(); }

        /**
         * @brief Const begin iterator for the matrix
         *
         * @return The const begin iterator for the matrix
         */
        constexpr auto cbegin() const { return d().matrix.cbegin(); }

        /**
         * @brief The const end iterator for the matrix
         *
         * @return The const end iterator for the matrix
         */
        constexpr auto cend() const { return d().matrix.cend(); }

        /**
         * @brief Return the last row of the matrix
         *
         * @return The last row of the matrix
         */
        constexpr auto back() { return d().matrix.back(); }

        /**
         * @brief Const reverse begin iterator for the matrix
         *
         * @return The const reverse begin iterator for the matrix
         */
        constexpr auto rbegin() { return d().matrix.rbegin(); }

        /**
         * @brief Const reverse end iterator for the matrix
         *
         * @return The const reverse end iterator for the matrix
         */
        constexpr auto rend() { return d().matrix.rend(); }

        // ============
        // Comparision
        // ============

        /**
         * @brief Equality operator for the matrix
         *
         * @param other The matrix to compare
         * @return True if the matrices are equal, i.e. are of the same dimensions and have the same elements
         */
        template <typename OtherD>
        bool operator==(const MatrixBase<OtherD, number>& other) const {
            return matrixEqual<number>(d().matrix, static_cast<const OtherD&>(other).matrix);
        }

        // ======================
        // Arithmetic Operations
        // ======================

        /**
         * @brief Matrix multiplication by a scalar
         *
         * @param scalar  A scalar of the same type as the matrix
         * @return The matrix resulting from the multiplication
         */
        friend auto operator*(const MatrixBase<D, number>& lhs, const number& scalar) {
            auto res = lhs.d();
            res *= scalar;
            return static_cast<D&>(res);
        }

        /**
         * @brief Inplace matrix multiplication by a scalar
         *
         * @param scalar A scalar of the same type as the matrix
         * @return A reference to the current matrix
         */
        auto& operator*=(const number& scalar) {
            matrixScalarMultI<number, D::rows, D::cols>(d().matrix, scalar);
            return d();
        }

        /**
         * @brief Matrix division by a scalar
         *
         * @param scalar A scalar of the same type as the matrix
         * @return The matrix resulting from the division
         */
        auto operator/(const number& scalar) const {
            auto res = d();
            res /= scalar;
            return static_cast<D&>(res);
        }

        /**
         * @brief Inplace matrix division by a scalar
         *
         * @param scalar A scalar of the same type as the matrix
         * @return A reference to the current matrix
         */
        auto operator/=(const number& scalar) {
            matrixScalarDivI<number, D::rows, D::cols>(d().matrix, scalar);
            return d();
        }

        /**
         * @brief Matrix addition
         *
         * @param other The matrix to add
         * @return The matrix resulting from the addition
         */
        template <typename OtherD>
        auto operator+(const MatrixBase<OtherD, number>& other) const {
            auto res = d();
            res += other;
            return static_cast<D&>(res);
        }

        /**
         * @brief Inplace matrix addition
         *
         * @param other The matrix to add
         * @return reference to the current matrix
         */
        template <typename OtherD>
        auto& operator+=(const MatrixBase<OtherD, number>& other) {
            matrixAddI<number, D::rows, D::cols>(d().matrix, static_cast<const OtherD&>(other).matrix);
            return d();
        }

        /**
         * @brief Matrix subtraction
         *
         * @param other The matrix to subtract
         * @return The matrix resulting from the subtraction
         */
        template <typename OtherD>
        auto operator-(const MatrixBase<OtherD, number>& other) const {
            auto res = d();
            res -= other;
            return static_cast<D&>(res);
        }

        /**
         * @brief Inplace matrix subtraction
         *
         * @param other The matrix to subtract
         * @return reference to the current matrix
         */
        template <typename OtherD>
        auto& operator-=(const MatrixBase<OtherD, number>& other) {
            matrixSubI<number, D::rows, D::cols>(d().matrix, static_cast<const OtherD&>(other).matrix);
            return d();
        }

        /**
         * @brief Matrix multiplication by a matrix
         *
         * @param other
         * @return
         */
        template <typename OtherD>
        friend auto operator*(const MatrixBase<D, number>& lhs, const MatrixBase<OtherD, number>& rhs) {
            return MatrixMultiplication<number, D::rows, D::cols, OtherD::rows, OtherD::cols>(
                lhs.d().matrix, static_cast<const OtherD&>(rhs).matrix);
        }

        /**
         * @brief Matrix subtraction
         *
         * @param other The matrix to subtract
         * @return The matrix resulting from the subtraction
         */
        template <int m, int n>
        friend auto operator-(const MatrixBase& lhs, const TransposeVariant<number, m, n>& rhs) {
            if (std::holds_alternative<Matrix<number, m, n>>(rhs)) {
                auto res = lhs.d();
                auto other = get<Matrix<number, m, n>>(rhs);
                matrixSubI<number, Dynamic, Dynamic>(res.matrix, other.matrix);
                return res;
            }
            throw std::logic_error("Cannot subtract a vector from a matrix");
        }

        friend auto operator*(const number& scalar, const MatrixBase& rhs) { return rhs * scalar; }

        // =================
        // Matrix Operations
        // =================

        /**
         * @brief Transpose a mxn matrix to a nxm matrix
         *
         * @return The transposed matrix of size nxm
         */
        auto T() const { return TransposeMatrix<number, D::rows, D::cols>(d().matrix); }

        /**
         * @brief Augment the matrix with another matrix
         *
         * @param other The matrix to augment with
         * @return The augmented matrix of size mx(n + nN)
         */
        template <typename OtherD>
        auto augment(const MatrixBase<OtherD, number>& other) const {
            return MatrixAugmentMatrix<number, D::rows, D::cols, OtherD::cols>(
                d().matrix, static_cast<const OtherD&>(other).matrix);
        }

        /**
         * @brief Augment the matrix with a vector
         *
         * @param other The vector to augment with
         * @return The augmented matrix of size mx(n + 1)
         */
        template <int n>
        auto augment(const Vector<number, n>& other) const {
            if constexpr (D::rows != Dynamic || D::cols != Dynamic)
                return MatrixAugmentVector<number, D::rows, D::cols>(d().matrix, other.row);
            else
                return MatrixAugmentVector<number, Dynamic, -2>(d().matrix, other.row);
        }

        /**
         * @brief Determinant of the matrix
         *
         * @return The determinant of the matrix
         */
        number det() const {
            if constexpr (D::rows != Dynamic || D::cols != Dynamic) {
                if constexpr (D::rows != D::cols)
                    throw runtime_error("Finding determinant of rectangular matrices is not defined");
                if constexpr (D::rows == 2 && D::cols == 2)
                    return MatrixDet2x2<number>(d().matrix);
                else
                    return MatrixCofactor<number, D::rows, D::cols>(d().matrix);
            } else {
                const auto& [nR, nC] = d().shape();

                if (nR != nC) throw runtime_error("Finding determinant of rectangular matrices is not defined");

                if (nR == 2 && nC == 2)
                    return MatrixDet2x2<number>(d().matrix);
                else
                    return MatrixCofactor<number, D::rows, D::cols>(d().matrix);
            }
        }

        /**
         * @brief Subset the matrix by removing a row and a column
         *
         * @param i Row index to remove
         * @param j Column index to remove
         * @return The subsetted matrix of size (m - 1)x(n - 1)
         */
        auto subset(optional<int> i, optional<int> j) const {
            return MatrixSubset<number, D::rows, D::cols>(d().matrix, i, j);
        }

        /**
         * @brief Calculate the trace of the matrix, i.e. the sum of the diagonal elements
         *
         * @return The trace of the matrix
         */
        number trace() const { return MatrixTrace<number>(d().matrix); }

        /**
         * @brief Apply a function to each element of the matrix
         *
         * @tparam F Function type that takes a number and returns void
         * @param f Function to apply to each element of the matrix
         * @return A reference to the same matrix
         */
        template <typename F>
        D& apply(F f) {
            matrixApply(d().matrix, f);
            return d();
        }

        /**
         * @brief Apply a function to each element of the matrix with another matrix
         *
         * @tparam F Function type that takes two numbers and returns void
         * @param other The other matrix to apply the function with
         * @param f Function to apply to each element of the matrix
         * @return A reference to the same matrix
         */
        template <typename OtherD, typename F>
        D& apply(const MatrixBase<OtherD, number>& other, F f) {
            const auto& otherMatrix = static_cast<const OtherD&>(other).matrix;
            matrixApply<decltype(f), decltype(d().matrix), decltype(otherMatrix), true>(d().matrix, otherMatrix, f);
            return d();
        }

        // ======================
        // Norms and Determinant
        // ======================

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
        double frob() { return FrobenisNorm(d().matrix); }

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
        double l1() { return L1Norm<number, D::rows, D::cols>(d()); }

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
        double lInf() { return LInfNorm<number, D::rows, D::cols>(d()); }

        // ======================
        // Miscellaneous Operations
        // ======================

        /**
         * @brief Convert the columns of the matrix to a vector of column vectors
         *
         * @return A vector of column vectors
         */
        auto colToVectorSet() const { return matrixColsToVectorSet<number, D::rows, D::cols>(d().matrix); }

        /**
         * @brief Convert the rows of the matrix to a vector of row vectors
         *
         * @return A vector of row vectors
         */
        auto rowToVectorSet() const { return matrixRowsToVectorSet<number, D::rows, D::cols>(d().matrix); }

        explicit operator string() const { return matrixStringRepr(d().matrix); }

        friend std::ostream& operator<<(std::ostream& os, const MatrixBase& system) {
            os << string(system.d());
            return os;
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
        auto view(size_t rowOffset = 0, size_t colOffset = 0, size_t rowStride = 1, size_t colStride = 1) {
            return View<number, D::rows, D::cols, i, j>(d().matrix, rowOffset, colOffset, rowStride, colStride);
        }

        /**
         * @brief Clear the matrix, i.e. set all elements to zero
         */
        void clear() { return matrixClear(d()); }

        const auto* getMatrix() const noexcept { return d().matrix; }

        auto& getMatrix() noexcept { return d().matrix; }

        const auto* data() const noexcept { return d().matrix.data(); }

        auto* data() noexcept { return d().matrix.data(); }
    };

    template <int n, int m, Number number>
    std::ostream& operator<<(std::ostream& os, const TransposeVariant<number, n, m>& system) {
        if (std::holds_alternative<VectorTransposeVariant<number, n, m>>(system)) {
            os << get<VectorTransposeVariant<number, n, m>>(system);
        } else if (std::holds_alternative<MatrixTransposeVariant<number, n, m>>(system)) {
            os << get<MatrixTransposeVariant<number, n, m>>(system);
        }
        return os;
    }

    template <Number number, int m, int n, int nOther>
    TransposeVariant<number, nOther, m> operator*(TransposeVariant<number, m, n> lhs, Matrix<number, n, nOther> rhs) {
        if (std::holds_alternative<VectorTransposeVariant<number, n, m>>(lhs)) {
            auto vec = get<VectorTransposeVariant<number, n, m>>(lhs);
            return vec * rhs;
        } else {
            auto mat = get<MatrixTransposeVariant<number, n, m>>(lhs);
            return mat * rhs;
        }
    }

    template <Number number, int m, int n>
    TransposeVariant<number, n, n> operator*(TransposeVariant<number, m, n> lhs, Matrix<number, m, n> rhs)
        requires(m != n)
    {
        if (std::holds_alternative<VectorTransposeVariant<number, m, n>>(lhs)) {
            auto vec = get<VectorTransposeVariant<number, m, n>>(lhs);
            const auto& res = vec.T() * rhs;
            return helpers::extractVectorFromTranspose(res.T());
        } else {
            auto mat = get<MatrixTransposeVariant<number, m, n>>(lhs);
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

}  // namespace mlinalg::structures
