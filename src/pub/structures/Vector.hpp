/**
 * @file Vector.hpp
 * @brief Header file for the Vector class
 */

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <iterator>
#include <memory>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../Concepts.hpp"
#include "Aliases.hpp"
#include "Matrix.hpp"
#include "VectorOps.hpp"

using std::vector, std::array, std::optional, std::unique_ptr, std::shared_ptr, mlinalg::structures::helpers::unwrap;

namespace mlinalg::structures {

    /**
     * @brief Vector class for represeting both row and column vectors in n-dimensional space
     *
     * @param n The number of elements in the vector
     */
    template <Number number, int n>
    class Vector {
       public:
        Vector() { checkDimensions(); }

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

        // Constructor to keep consistency with the Dynamic Vector specialization to allow them to be used
        // interchangeably
        explicit Vector(size_t size) {}  // NOLINT

#pragma GCC diagnostic pop

        /**
         * @brief Construct a new Vector object from an initializer list
         *
         * @param list Initializer list of numbers
         */
        Vector(const std::initializer_list<number>& list) {
            checkDimensions();
            if (list.size() != n)
                throw std::invalid_argument("Initializer list must be the same size as the defined vector size");
            for (int i{}; i < n; i++) row.at(i) = *(list.begin() + i);
        }

        /**
         * @brief Copy construct a new Vector object
         *
         * @param other Vector to copy
         */
        Vector(const Vector& other) : row{other.row} {}

        /**
         * @brief Move construct a new Vector object
         *
         * @param other Vector to move
         */
        Vector(Vector&& other) noexcept : row{std::move(other.row)} {}

        /**
         * @brief Copy assignment operator
         *
         * @param other Vector to copy
         */
        Vector& operator=(const Vector& other) {
            if (this == &other) return *this;
            row = VectorRow<number, n>{other.row};
            return *this;
        }

        /**
         * @brief Copy assign a Vector from a 1xN matrix
         *
         * @param other A 1xN matrix
         */
        Vector& operator=(const Matrix<number, 1, n>& other) {
            for (int i{}; i < n; i++) row.at(i) = other.at(0).at(i);
            return *this;
        }

        /**
         * @brief Move assignment operator
         *
         * @param other Vector to move
         */
        Vector& operator=(Vector&& other) noexcept {
            row = std::move(other.row);
            return *this;
        }

        /**
         * @brief Equality operator
         *
         * @param other Vector to compare
         * @return true if all the entires in the vector are equal to all the entires in the other vector else false
         */
        bool operator==(const Vector& other) const { return vectorEqual(row, other.row); }

        ~Vector() = default;

        /**
         * @brief Access the ith element of the vector
         *
         * @param i the index of the element to access
         * @return a reference to the ith element
         */
        number& at(size_t i) { return vectorAt<number>(row, i); }

        /**
         * @brief Access the ith element of the vector
         *
         * @param i  the index of the element to access
         * @return a reference to the ith element
         */
        number& operator[](size_t i) { return row[i]; }

        /**
         * @brief Const access the ith element of the vector
         *
         * @param i the index of the element to access
         * @return  the ith element
         */
        const number& at(size_t i) const { return vectorConstAt<number>(row, i); }

        /**
         * @brief Const access the ith element of the vector
         *
         * @param i  the index of the element to access
         * @return The ith element
         */
        number& operator[](size_t i) const { return const_cast<number&>(row[i]); }

        /**
         * @brief Find the dot product of this vector and another vector
         *
         * @param other The other vector
         * @return the dot product of the two vectors
         */
        double dot(const Vector<number, n>& other) const { return vectorDot(*this, other); }

        /**
         * @brief Find the length of the vector
         *
         * @return the length of the vector
         */
        [[nodiscard]] double length() const { return EuclideanNorm(*this); }

        /**
         * @brief Find the euclidean norm of the vector
         *
         * This is the same as the length of the vector
         *
         * \f[
         * ||x||_2 = \sqrt{x^T \cdot x}
         * \f]
         *
         * @return the euclidean norm of the vector
         */
        [[nodiscard]] double euclid() const { return length(); }

        /**
         * @brief Find the L1 norm of the vector
         *
         * This is the same as the sum of the absolute values of the elements in the vector
         *
         * \f[
         * ||x||_1 = \sum{|x_i|}
         * \f]
         *
         * @return the L1 norm of the vector
         */
        [[nodiscard]] double l1() const { return L1Norm<number, n>(row); }

        /**
         * @brief Find the weighted L2 norm of the vector
         *
         * Each of the coordinates of a vector space is given a weight
         *
         * \f[
         * ||x||_W = \sqrt{\sum{w_i * x_i^2}}
         * \f]
         *
         * @param otherVec The other vector
         * @return the weighted L2 norm of the vector
         */
        template <int otherN>
        double weightedL2(const Vector<number, otherN>& otherVec) const {
            return WeightedL2Norm<number, n>(row, otherVec.row);
        }

        /**
         * @brief Find the distance between this vector and another vector
         *
         * @param other The other vector
         * @return the distance between the two vectors
         */
        [[nodiscard]] double dist(const Vector<number, n>& other) const { return vectorDist(*this, other); }

        /**
         * @brief Normalize a vector into a unit vector
         *
         * @return the normalized vector
         */
        Vector<double, n> normalize() const { return vectorNormalize(*this); }

        /**
         * @brief Vector subtraction
         *
         * @param other the vector to subtract
         * @return the vector resulting from the subtraction
         */
        Vector<number, n> operator-(const Vector<number, n>& other) const {
            return vectorSub<number, n>(row, other.row);
        }

        /**
         * @brief Vector Negation
         *
         * @return the negeated vector
         */
        Vector<number, n> operator-() const { return vectorNeg<number, n>(row); }

        /**
         * @brief In-place vector subtraction
         *
         * @param other  the vector to add
         * @return A reference to the same vector
         */
        Vector<number, n>& operator-=(const Vector<number, n>& other) {
            vectorSubI<number>(row, other.row);
            return *this;
        }

        /**
         * @brief Vector addition
         *
         * @param other the vector to add
         * @return the vector resulting from the addition
         */
        Vector<number, n> operator+(const Vector<number, n>& other) const {
            return vectorAdd<number, n>(row, other.row);
        }

        /**
         * @brief In-place vector addition
         *
         * @param other  the vector to add
         * @return A reference to the same vector
         */
        Vector<number, n>& operator+=(const Vector<number, n>& other) {
            vectorAddI<number>(row, other.row);
            return *this;
        }

        /**
         * @brief Size of the vector, i.e the number of elements in the vector
         *
         * @return the size of the vector
         */
        [[nodiscard]] size_t size() const { return static_cast<size_t>(n); }

        /**
         * @brief Vector multiplication by a scalar
         *
         * @param scalar A scalar of the same type as the vector
         * @return the vector resulting from the multiplication
         */
        Vector<number, n> operator*(const number& scalar) const { return vectorScalarMult<number, n>(row, scalar); }

        /**
         * @brief Vector division by a scalar
         *
         * @param scalar A scalar of the same type as the vector
         * @return  The vector resulting from the division
         */
        Vector<number, n> operator/(const number& scalar) const { return vectorScalarDiv<number, n>(row, scalar); }

        /**
         * @brief In-place vector multiplication by a scalar.
         *
         * @param scalar A scalar of the same type as the vector.
         * @return
         */
        Vector<number, n>& operator*=(const number& scalar) {
            vectorScalarMultI<number>(row, scalar);
            return *this;
        }

        /**
         * @brief In-place vector division by a scalar
         *
         * @param scalar A scalar of the same type as the vector
         * @return
         */
        Vector<number, n>& operator/=(const number& scalar) {
            vectorScalarDivI(row, scalar);
            return *this;
        }

        /**
         * @brief Vector multiplication by a vector
         *
         * @param vec Another vector of the same size as the vector
         * @return  A 1x1 vector containing the dot product of the two vectors
         */
        auto operator*(const Vector<number, n>& vec) const { return vectorVectorMult(*this, vec); }

        /**
         * @brief Vector multiplication by a matrix
         *
         * @param mat A matrix of size MxN
         * @return The vector resulting from the multiplication
         */
        template <int m>
        Vector<number, n> operator*(const Matrix<number, m, n>& mat) const {
            Vector<number, n> res;
            auto asCols{std::move(mat.colToVectorSet())};
            for (int i{}; i < n; i++) {
                res.at(i) = *this * asCols.at(i);
            }
            return res;
        }

        /**
         * @brief Vector multiplication by a matrix of size 1xn.
         * This results in a matrix of size nxn
         *
         * @param mat A matrix of size 1xn
         * @return The matrix resulting from the multiplication
         */
        Matrix<number, n, n> operator*(const Matrix<number, 1, n>& mat) const
            requires(n != 1)
        {
            Matrix<number, n, n> res;
            auto asCols{std::move(mat.colToVectorSet())};
            for (int i{}; i < n; i++) {
                res.at(i) = *this * asCols.at(i).at(0);
            }
            return res;
        }

        /**
         * @brief Transpose a row vector to a column vector
         *
         * @return ColumnVector<number, n>
         */
        [[nodiscard]] Matrix<number, 1, n> T() const { return vectorTranspose<number, 1, n>(row); }

        /**
         * @brief Begin iterator for the vector
         *
         * @return An iterator to the beginning of the vector
         */
        constexpr auto begin() const { return row.begin(); }

        /**
         * @brief End iterator for the vector
         *
         * @return An iterator to the end of the vector
         */
        constexpr auto end() const { return row.end(); }

        /**
         * @brief Const begin iterator for the vector
         *
         * @return A const iterator to the beginning of the vector
         */
        constexpr auto cbegin() const { return row.cbegin(); }

        /**
         * @brief Const end iterator for the vector
         *
         * @return A c onst iterator to the end of the vector
         */
        constexpr auto cend() const { return row.cend(); }

        /**
         * @brief Last element of the vector
         *
         * @return A reference to the last element of the vector
         */
        constexpr auto& back() { return row.back(); }

        /**
         * @brief Const last element of the vector
         *
         * @return A const reference to the last element of the vector
         */
        constexpr auto& back() const { return row.back(); }

        /**
         * @brief Reverse begin iterator for the vector
         *
         * @return An iterator to the beginning of the vector in reverse
         */
        constexpr auto rbegin() { return row.rbegin(); }

        /**
         * @brief Reverse end iterator for the vector
         *
         * @return An iterator to the end of the vector in reverse
         */
        constexpr auto rend() { return row.rend(); }

        explicit operator std::string() const { return vectorStringRepr(row); }

        friend std::ostream& operator<<(std::ostream& os, const Vector<number, n>& row) {
            os << std::string(row);
            return os;
        }

        friend std::ostream& operator<<(std::ostream& os, const optional<Vector<number, n>>& rowPot) {
            if (!rowPot.has_value()) {
                os << "Empty Vector";
                return os;
            }

            return vectorOptionalRepr(os, rowPot.value().row);
        }

       private:
        template <Number num, int mM, int nN>
        friend class Matrix;

        template <Number num, int nN>
        friend class Vector;

        /**
         * @brief Swap the contents of two vectors
         *
         * @param first
         * @param second
         */
        friend void swap(Vector& first, Vector& second) noexcept {
            using std::swap;
            swap(first.row, second.row);
        }

        void checkDimensions() {
            if constexpr (n <= 0) throw std::invalid_argument("Vector size must be greater than 0");
        }

        // VectorRowPtr<number, n> row{new array<number, n>{}};
        VectorRow<number, n> row{VectorRow<number, n>(n, number{})};
    };

    template <Number number, int n>
    Vector<number, n> operator*(const number& scalar, const Vector<number, n>& vec) {
        return vec * scalar;
    }

}  // namespace mlinalg::structures

namespace mlinalg::structures {

    /**
     * @brief Dynamic Vector class for representing both row and column vectors in n-dimensional space
     */
    template <Number number>
    class Vector<number, Dynamic> {
       public:
        Vector() = delete;
        explicit Vector(size_t size) : n{size}, row(size) {}

        Vector(const std::initializer_list<number>& list) : n{list.size()}, row{list} {}

        template <typename Iterator>
        Vector(Iterator begin, Iterator end) : n(std::distance(begin, end)), row{begin, end} {}

        template <int nN>
        Vector(const Vector<number, nN>& other)
            requires(nN != Dynamic)
            : n{nN}, row(nN) {
            for (int i{}; i < nN; i++) {
                this->at(i) = other.at(i);
            }
        }

        /**
         * @brief Copy construct a new Vector object
         *
         * @param other Vector to copy
         */
        Vector(const Vector<number, Dynamic>& other) : n{other.n}, row{other.row} {}

        /**
         * @brief Move construct a new Vector object
         *
         * @param other Vector to move
         */
        Vector(Vector<number, Dynamic>&& other) noexcept : n{other.n}, row{std::move(other.row)} { other.n = 0; }

        /**
         * @brief Copy assignment operator
         *
         * @param other Vector to copy
         */
        Vector& operator=(const Vector<number, Dynamic>& other) {
            if (this == &other) return *this;
            row = VectorRowDynamic<number>{other.row};
            n = other.n;
            return *this;
        }

        /**
         * @brief Copy assign a Vector from a 1xN matrix
         *
         * @param other A 1xN matrix
         */
        Vector& operator=(const Matrix<number, Dynamic, Dynamic>& other) {
            const auto& nRows{other.numRows()};
            const auto& nCols{other.numCols()};
            if (nRows != 1) throw std::invalid_argument("Matrix must have 1 row to be converted to a vector");
            for (int i{}; i < nCols; i++) row.at(i) = other.at(0).at(i);
            return *this;
        }

        /**
         * @brief Move assignment operator
         *
         * @param other Vector to move
         */
        Vector& operator=(Vector<number, Dynamic>&& other) noexcept {
            row = std::move(other.row);
            n = other.n;
            return *this;
        }

        /**
         * @brief Equality operator
         *
         * @param other Vector to compare
         * @return true if all the entires in the vector are equal to all the entires in the other vector else false
         */
        template <int otherN>
        bool operator==(const Vector<number, otherN>& other) const {
            return vectorEqual(row, other.row);
        }

        template <int n, int otherN>
        friend bool operator==(const Vector<number, otherN>& lhs, const Vector<number, n> rhs)
            requires(n == Dynamic && otherN != Dynamic)
        {
            return vectorEqual(lhs.row, rhs.row);
        }

        ~Vector() = default;

        /**
         * @brief Access the ith element of the vector
         *
         * @param i the index of the element to access
         * @return a reference to the ith element
         */
        number& at(size_t i) { return vectorAt<number>(row, i); }

        /**
         * @brief Access the ith element of the vector
         *
         * @param i  the index of the element to access
         * @return a reference to the ith element
         */
        number& operator[](size_t i) { return row[i]; }

        /**
         * @brief Const access the ith element of the vector
         *
         * @param i the index of the element to access
         * @return  the ith element
         */
        number at(size_t i) const { return vectorConstAt<number>(row, i); }

        /**
         * @brief Const access the ith element of the vector
         *
         * @param i  the index of the element to access
         * @return The ith element
         */
        number& operator[](size_t i) const { return const_cast<number&>(row[i]); }

        /**
         * @brief Find the dot product of this vector and another vector
         *
         * @param other The other vector
         * @return the dot product of the two vectors
         */
        template <int otherN>
        double dot(const Vector<number, otherN>& other) const {
            return vectorDot(*this, other);
        }

        /**
         * @brief Find the length of the vector
         *
         * @return the length of the vector
         */
        [[nodiscard]] double length() const { return EuclideanNorm(*this); }

        /**
         * @brief Find the euclidean norm of the vector
         *
         * This is the same as the length of the vector
         *
         * \f[
         * ||x||_2 = \sqrt{x^T \cdot x}
         * \f]
         *
         * @return the euclidean norm of the vector
         */
        [[nodiscard]] double euclid() const { return length(); }

        /**
         * @brief Find the L1 norm of the vector
         *
         * This is the same as the sum of the absolute values of the elements in the vector
         *
         * \f[
         * ||x||_1 = \sum{|x_i|}
         * \f]
         *
         * @return the L1 norm of the vector
         */
        [[nodiscard]] double l1() const { return L1Norm<number, Dynamic>(row); }

        /**
         * @brief Find the weighted L2 norm of the vector
         *
         * Each of the coordinates of a vector space is given a weight
         *
         * \f[
         * ||x||_W = \sqrt{\sum{|w_i * x_i|^2}}
         * \f]
         *
         * @param otherVec The other vector
         * @return the weighted L2 norm of the vector
         */
        template <int otherN>
        double weightedL2(const Vector<number, otherN>& otherVec) const {
            return WeightedL2Norm<number, Dynamic>(row, otherVec.row);
        }

        /**
         * @brief Find the distance between this vector and another vector
         *
         * @param other The other vector
         * @return the distance between the two vectors
         */

        template <int otherN>
        [[nodiscard]] double dist(const Vector<number, otherN>& other) const {
            return vectorDist(*this, other);
        }

        /**
         * @brief Normalize a vector into a unit vector
         *
         * @return the normalized vector
         */
        Vector<double, Dynamic> normalize() { return vectorNormalize(*this); }

        /**
         * @brief Vector subtraction
         *
         * @param other the vector to subtract
         * @return the vector resulting from the subtraction
         */
        template <int otherN>
        Vector<number, Dynamic> operator-(const Vector<number, otherN>& other) const {
            return vectorSub<number, Dynamic>(row, other.row);
        }

        template <int n, int otherN>
        friend Vector<number, Dynamic> operator-(const Vector<number, otherN>& lhs, const Vector<number, n> rhs)
            requires(n == Dynamic && otherN != Dynamic)
        {
            return vectorSub<number, Dynamic>(lhs.row, rhs.row);
        }

        /**
         * @brief Vector addition
         *
         * @param other the vector to add
         * @return the vector resulting from the addition
         */
        template <int otherN>
        Vector<number, Dynamic> operator+(const Vector<number, otherN>& other) const {
            return vectorAdd<number, Dynamic>(row, other.row);
        }

        template <int n, int otherN>
        friend Vector<number, Dynamic> operator+(const Vector<number, otherN>& lhs, const Vector<number, n> rhs)
            requires(n == Dynamic && otherN != Dynamic)
        {
            return vectorAdd<number, Dynamic>(lhs.row, rhs.row);
        }

        /**
         * @brief In-place vector addition
         *
         * @param other  the vector to add
         * @return A reference to the same vector
         */
        template <int otherN>
        Vector<number, Dynamic>& operator+=(const Vector<number, otherN>& other) {
            vectorAddI<number>(row, other.row);
            return *this;
        }

        /**
         * @brief In-place vector subtraction
         *
         * @param other  the vector to add
         * @return A reference to the same vector
         */
        template <int otherN>
        Vector<number, Dynamic>& operator-=(const Vector<number, otherN>& other) {
            vectorSubI<number>(row, other.row);
            return *this;
        }

        /**
         * @brief Size of the vector, i.e the number of elements in the vector
         *
         * @return the size of the vector
         */
        [[nodiscard]] size_t size() const { return n; }

        /**
         * @brief Vector multiplication by a scalar
         *
         * @param scalar A scalar of the same type as the vector
         * @return the vector resulting from the multiplication
         */
        Vector<number, Dynamic> operator*(const number& scalar) const {
            return vectorScalarMult<number, Dynamic>(row, scalar);
        }

        /**
         * @brief Vector division by a scalar
         *
         * @param scalar A scalar of the same type as the vector
         * @return  The vector resulting from the division
         */
        Vector<number, Dynamic> operator/(const number& scalar) const {
            return vectorScalarDiv<number, Dynamic>(row, scalar);
        }

        /**
         * @brief In-place vector multiplication by a scalar.
         *
         * @param scalar A scalar of the same type as the vector.
         * @return
         */
        Vector<number, Dynamic>& operator*=(const number& scalar) {
            vectorScalarMultI<number>(row, scalar);
            return *this;
        }

        /**
         * @brief In-place vector division by a scalar
         *
         * @param scalar A scalar of the same type as the vector
         * @return
         */
        Vector<number, Dynamic>& operator/=(const number& scalar) {
            vectorScalarDivI(row, scalar);
            return *this;
        }

        /**
         * @brief Vector multiplication by a vector
         *
         * @param vec Another vector of the same size as the vector
         * @return  A 1x1 vector containing the dot product of the two vectors
         */
        template <int otherN>
        auto operator*(const Vector<number, otherN>& vec) const {
            checkOperandSize(row, vec.row);
            return vectorVectorMult(*this, vec);
        }

        /**
         * @brief Vector multiplication by a matrix
         * A 1xn matrix left multiplied by a vector of size n results in an nxn matrix
         *
         * @param mat A matrix of size MxN
         * @return The vector resulting from the multiplication
         */
        TransposeVariant<number, Dynamic, Dynamic> operator*(const Matrix<number, Dynamic, Dynamic>& mat) const {
            const auto& numRows = mat.numRows();
            const auto numCols = mat.numCols();
            if (numCols != this->n)
                throw std::invalid_argument("Matrix must have the same number of columns as the vector size");

            auto asCols{std::move(mat.colToVectorSet())};
            // If the matrix has only one row it is equivalent to a transposed vector
            // And thus the result is a matrix with the cols of this matrix being the result of the dot product of the
            // vector and the cols of the matrix
            if (numRows == 1) {
                Matrix<number, Dynamic, Dynamic> resMat(this->n, this->n);
                for (size_t i{}; i < n; i++) {
                    resMat.at(i) = *this * asCols.at(i).at(0);
                }
                return resMat;
            } else {
                Vector<number, Dynamic> resVec(this->n);
                for (size_t i{}; i < n; i++) {
                    resVec.at(i) = *this * asCols.at(i);
                }
                return resVec;
            }
        }

        /**
         * @brief Transpose a row vector to a column vector
         *
         * @return Matrix<number, 1, n>
         */
        [[nodiscard]] Matrix<number, Dynamic, Dynamic> T() const {
            return vectorTranspose<number, Dynamic, Dynamic>(row);
        }

        /**
         * @brief Begin iterator for the vector
         *
         * @return An iterator to the beginning of the vector
         */
        constexpr auto begin() const { return row.begin(); }

        /**
         * @brief End iterator for the vector
         *
         * @return An iterator to the end of the vector
         */
        constexpr auto end() const { return row.end(); }

        /**
         * @brief Const begin iterator for the vector
         *
         * @return A const iterator to the beginning of the vector
         */
        constexpr auto cbegin() const { return row.cbegin(); }

        /**
         * @brief Const end iterator for the vector
         *
         * @return A c onst iterator to the end of the vector
         */
        constexpr auto cend() const { return row.cend(); }

        /**
         * @brief Last element of the vector
         *
         * @return A reference to the last element of the vector
         */
        constexpr auto& back() { return row.back(); }

        /**
         * @brief Const last element of the vector
         *
         * @return A const reference to the last element of the vector
         */
        constexpr auto& back() const { return row.back(); }

        /**
         * @brief Reverse begin iterator for the vector
         *
         * @return An iterator to the beginning of the vector in reverse
         */
        constexpr auto rbegin() { return row.rbegin(); }

        /**
         * @brief Reverse end iterator for the vector
         *
         * @return An iterator to the end of the vector in reverse
         */
        constexpr auto rend() { return row.rend(); }

        explicit operator std::string() const { return vectorStringRepr(row); }

        friend std::ostream& operator<<(std::ostream& os, const Vector<number, Dynamic>& row) {
            os << std::string(row);
            return os;
        }

        friend std::ostream& operator<<(std::ostream& os, const optional<Vector<number, Dynamic>>& rowPot) {
            if (!rowPot.has_value()) {
                os << "Empty Vector";
                return os;
            }

            return vectorOptionalRepr(os, rowPot.value().row);
        }

       private:
        template <Number num, int mM, int nN>
        friend class Matrix;

        template <Number num, int nN>
        friend class Vector;

        /**
         * @brief Swap the contents of two vectors
         *
         * @param first
         * @param second
         */
        friend void swap(Vector& first, Vector& second) noexcept {
            using std::swap;
            swap(first.row, second.row);
        }

        size_t n;
        // VectorRowDynamicPtr<number> row{std::make_unique<VectorRowDynamic<number>>()};
        VectorRowDynamic<number> row{std::make_unique<VectorRowDynamic<number>>()};
    };
}  // namespace mlinalg::structures
