/**
 * @file Structures.hpp
 * @brief Declaration and implementation of templated Vector and Matrix classes.
 */

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <memory>
#include <numeric>
#include <optional>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

/**
 * @brief Concept for a number type
 *
 * @tparam T Type to check
 */
template <typename T>
concept Number = requires {
    std::is_integral_v<T> || std::is_floating_point_v<T>;
    std::is_convertible_v<T, std::string>;
};

using std::vector, std::array, std::optional, std::unique_ptr, std::shared_ptr;
namespace rg = std::ranges;

namespace mlinalg::structures {
    template <Number number, size_t m, size_t n>
    class Matrix;

    template <Number number, size_t n>
    class Vector;

    /**
     * @brief Type alias for a variant of a Vector and a Matrix
     *
     * This is used to represent the result of a matrix or vector transposition. As the transpose of a vector is a 1xM
     * matrix and the transpose of a matrix is an NxM matrix, this variant is used to represent either of these
     */
    template <Number number, size_t m, size_t n>
    using TransposeVariant = std::variant<Vector<number, m>, Matrix<number, n, m>>;

    /* @breif Helper functions for the Vector and Matrix classes
     *
     */
    namespace helpers {

        /**
         * @brief Generate a matrix from a vector of column vectors
         *
         * @param vecSet Vector of column vectors
         * @return  Matrix<number, m, n>
         */
        template <Number num, size_t m, size_t n>
        Matrix<num, m, n> fromColVectorSet(const vector<Vector<num, m>>& vecSet) {
            Matrix<num, m, n> res;
            for (size_t i{}; i < n; i++) {
                const auto& vec{vecSet.at(i)};
                for (size_t j{}; j < m; j++) {
                    res.at(j).at(i) = vec.at(j);
                }
            }
            return res;
        }

        /**
         * @brief Generate a matrix from a vector of row vectors
         *
         * @param vecSet  Vector of row vectors
         * @return  Matrix<number, m, n>
         */
        template <Number num, size_t m, size_t n>
        Matrix<num, m, n> fromRowVectorSet(const vector<Vector<num, n>>& vecSet) {
            Matrix<num, m, n> res;
            for (size_t i{}; i < m; i++) {
                res.at(i) = vecSet.at(i);
            }
            return res;
        }

        /**
         * @brief Extract a matrix from a TransposeVariant
         *
         * @param T  TransposeVariant
         * @return Matrix<number, n, m>
         */
        template <Number num, size_t m, size_t n>
        Matrix<num, n, m> extractMatrixFromTranspose(const TransposeVariant<num, m, n> T) {
            return std::get<Matrix<num, n, m>>(T);
        }

        /**
         * @brief Extract a vector from a TransposeVariant
         *
         * @param T TransposeVariant
         * @return Vector<number, m>
         */
        template <Number num, size_t m, size_t n>
        Vector<num, m> extractVectorFromTranspose(const TransposeVariant<num, m, n> T) {
            return std::get<Vector<num, m>>(T);
        }

    }  // namespace helpers

    // Type alias for the backing array of a Vector
    template <Number number, size_t n>
    using VectorRow = std::array<number, n>;

    // Type alias for a unique pointer to a VectorRow
    template <Number number, size_t n>
    using VectorRowPtr = unique_ptr<VectorRow<number, n>>;

    /**
     * @brief Vector class for represeting both row and column vectors in n-dimensional space
     *
     * @param n The number of elements in the vector
     */
    template <Number number, size_t n>
    class Vector {
       public:
        Vector() = default;

        /**
         * @brief Construct a new Vector object from an initializer list
         *
         * @param list Initializer list of numbers
         */
        Vector(const std::initializer_list<number>& list) {
            for (size_t i{}; i < n; i++) row->at(i) = *(list.begin() + i);
        }

        /**
         * @brief Copy construct a new Vector object
         *
         * @param other Vector to copy
         */
        Vector(const Vector& other) : row{new VectorRow<number, n>{*other.row}} {}

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
            row = VectorRowPtr<number, n>{new VectorRow<number, n>{*other.row}};
            return *this;
        }

        /**
         * @brief Copy assign a Vector from a 1xN matrix
         *
         * @param other A 1xN matrix
         */
        Vector& operator=(const Matrix<number, 1, n>& other) {
            for (size_t i{}; i < n; i++) row->at(i) = other.at(0).at(i);
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

        // auto operator<=>(const Vector& other) const = default;

        /**
         * @brief Equality operator
         *
         * @param other Vector to compare
         * @return true if all the entires in the vector are equal to all the entires in the other vector else false
         */
        bool operator==(const Vector& other) const { return *row == *(other.row); }

        /**
         * @brief Greater than operator
         *
         * @param other Vector to compare
         * @return true if all the entries in the vector are greater than all the entries in the other vector else false
         */
        bool operator>(const Vector& other) const { return *row > *(other.row); }

        /**
         * @brief Less than operator
         *
         * @param other Vector to compare
         * @return true if all the entries in the other vector are less than all the entries in this vector else false
         */
        bool operator<(const Vector& other) const { return *row < *(other.row); }

        /**
         * @brief Greater than or equal to operator
         *
         * @param other Vector to compare
         * @return true if all the entries in the vector are greater than or equal to the entries in the other vector
         * else false
         */
        bool operator>=(const Vector& other) const { return *row >= *(other.row); }

        /**
         * @brief Less than or equal to operator
         *
         * @param other Vector to compare
         * @return true if all the entries in the vector are less than or equal to the entries in the other vector else
         * false
         */
        bool operator<=(const Vector& other) const { return *row <= *(other.row); }

        /**
         * @brief Inequalty Operator
         *
         * @param other Vector to compare
         * @return true if all the entires in the vector are not equal to all the entries in the other vector else false
         */
        bool operator!=(const Vector& other) const { return *row != *(other.row); }

        /*bool operator==(const Vector<number, n>& other) const { return row == other.row; }*/

        ~Vector() { row.reset(); }

        /**
         * @brief Access the ith element of the vector
         *
         * @param i the index of the element to access
         * @return a reference to the ith element
         */
        number& at(size_t i) { return row->at(i); }

        /**
         * @brief Const access the ith element of the vector
         *
         * @param i the index of the element to access
         * @return  the ith element
         */
        number at(size_t i) const { return row->at(i); }

        /**
         * @brief Find the dot product of this vector and another vector
         *
         * @param other The other vector
         * @return the dot product of the two vectors
         */
        double dot(const Vector<number, n>& other) const { return (this->T() * other).at(0); }

        /**
         * @brief Find the length of the vector
         *
         * @return the length of the vector
         */
        [[nodiscard]] double length() const { return std::sqrt(this->dot(*this)); }

        /**
         * @brief Find the distance between this vector and another vector
         *
         * @param other The other vector
         * @return the distance between the two vectors
         */
        [[nodiscard]] double dist(const Vector<number, n>& other) const {
            auto diff = *this - other;
            return std::sqrt(diff.dot(diff));
        }

        /**
         * @brief Vector subtraction
         *
         * @param other the vector to subtract
         * @return the vector resulting from the subtraction
         */
        Vector<number, n> operator-(const Vector<number, n>& other) const {
            Vector<number, n> res;
            for (size_t i{}; i < n; i++) res.at(i) = row->at(i) - other.at(i);
            return res;
        }

        /**
         * @brief Vector addition
         *
         * @param other the vector to add
         * @return the vector resulting from the addition
         */
        Vector<number, n> operator+(const Vector<number, n>& other) const {
            Vector<number, n> res;
            for (size_t i{}; i < n; i++) res.at(i) = row->at(i) + other.at(i);
            return res;
        }

        /**
         * @brief In-place vector addition
         *
         * @param other  the vector to add
         * @return A reference to the same vector
         */
        Vector<number, n>& operator+=(const Vector<number, n>& other) {
            for (size_t i{}; i < n; i++) this->at(i) += other.at(i);
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
        Vector<number, n> operator*(const number& scalar) const {
            Vector<number, n> res;
            for (size_t i{}; i < n; i++) res.at(i) = scalar * this->at(i);
            return res;
        }

        /**
         * @brief Vector division by a scalar
         *
         * @param scalar A scalar of the same type as the vector
         * @return  The vector resulting from the division
         */
        Vector<number, n> operator/(const number& scalar) const {
            if (scalar == 0) throw std::domain_error("Division by zero");
            Vector<number, n> res;
            for (size_t i{}; i < n; i++) res.at(i) = this->at(i) / scalar;
            return res;
        }

        /**
         * @brief In-place vector division by a scalar.
         *
         * @param scalar A scalar of the same type as the vector.
         * @return
         */
        Vector<number, n>& operator*=(const number& scalar) {
            for (size_t i{}; i < n; i++) this->at(i) *= scalar;
            return *this;
        }

        /**
         * @brief Vector multiplication by a vector
         *
         * @param vec Another vector of the same size as the vector
         * @return  A 1x1 vector containing the dot product of the two vectors
         */
        auto operator*(const Vector<number, n>& vec) const { return this->T() * vec; }

        /**
         * @brief Vector multiplication by a matrix
         *
         * @param mat A matrix of size MxN
         * @return The vector resulting from the multiplication
         */
        template <size_t m>
        Vector<number, n> operator*(const Matrix<number, m, n>& mat) {
            Vector<number, n> res;
            auto asCols{std::move(mat.colToVectorSet())};
            for (size_t i{}; i < n; i++) {
                auto multRes = *this * asCols.at(i);
                res.at(i) = std::accumulate(multRes.begin(), multRes.end(), 0);
            }
            return res;
        }

        /**
         * @brief Transpose a row vector to a column vector
         *
         * @return ColumnVector<number, n>
         */
        [[nodiscard]] Matrix<number, 1, n> T() const {
            Matrix<number, 1, n> res;
            for (size_t i{}; i < n; i++) res.at(0).at(i) = this->at(i);
            return res;
        }

        /**
         * @brief Begin itertor for the vector
         *
         * @return An iterator to the beginning of the vector
         */
        constexpr auto begin() const { return row->begin(); }

        /**
         * @brief End iterator for the vector
         *
         * @return An iterator to the end of the vector
         */
        constexpr auto end() const { return row->end(); }

        /**
         * @brief Const begin iterator for the vector
         *
         * @return A const iterator to the beginning of the vector
         */
        constexpr auto cbegin() const { return row->cbegin(); }

        /**
         * @brief Const end iterator for the vector
         *
         * @return A c onst iterator to the end of the vector
         */
        constexpr auto cend() const { return row->cend(); }

        /**
         * @brief Last element of the vector
         *
         * @return A reference to the last element of the vector
         */
        constexpr auto& back() { return row->back(); }

        /**
         * @brief Const last element of the vector
         *
         * @return A const reference to the last element of the vector
         */
        constexpr auto& back() const { return row->back(); }

        /**
         * @brief Reverse begin iterator for the vector
         *
         * @return An iterator to the beginning of the vector in reverse
         */
        constexpr auto rbegin() { return row->rbegin(); }

        /**
         * @brief Reverse end iterator for the vector
         *
         * @return An iterator to the end of the vector in reverse
         */
        constexpr auto rend() { return row->rend(); }

        explicit operator std::string() const {
            std::stringstream ss{};

            size_t maxWidth = 0;
            for (const auto& elem : *row) {
                std::stringstream temp_ss;
                temp_ss << elem;
                maxWidth = std::max(maxWidth, temp_ss.str().length());
            }

            if (row->size() == 1)
                ss << "[ " << std::setw(maxWidth) << row->at(0) << " ]\n";
            else
                for (size_t i{}; i < row->size(); i++)
                    if (i == 0) {
                        ss << "⎡ " << std::setw(maxWidth) << row->at(i) << " ⎤\n";
                    } else if (i == row->size() - 1) {
                        ss << "⎣ " << std::setw(maxWidth) << row->at(i) << " ⎦\n";
                    } else {
                        ss << "| " << std::setw(maxWidth) << row->at(i) << " |\n";
                    }
            return ss.str();
        }

        friend std::ostream& operator<<(std::ostream& os, const Vector<number, n>& row) {
            os << std::string(row);
            return os;
        }

        friend std::ostream& operator<<(std::ostream& os, const optional<Vector<number, n>>& rowPot) {
            if (!rowPot.has_value()) {
                os << "Empty Vector";
                return os;
            }
            const auto& row = rowPot.value();
            const auto& size = row->size();

            auto hasVal = [](auto rowVal) {
                std::stringstream val;
                if (rowVal.has_value())
                    val << rowVal.value();
                else
                    val << "None";
                return val.str();
            };

            size_t maxWidth = 0;
            for (const auto& elem : *row) {
                std::stringstream temp_ss;
                temp_ss << hasVal(elem);
                maxWidth = std::max(maxWidth, temp_ss.str().length());
            }

            if (size == 1) {
                os << "[ ";
                if (row->at(0).has_value())
                    os << row->at(0).value();
                else
                    os << "None";
                os << " ]\n";
            } else
                for (size_t i{}; i < row->size(); i++) {
                    if (i == 0) {
                        os << "⎡ " << std::setw(maxWidth) << hasVal(row->at(i)) << " ⎤\n";
                    } else if (i == row->size() - 1) {
                        os << "⎣ " << std::setw(maxWidth) << hasVal(row->at(i)) << " ⎦\n";

                    } else {
                        os << "| " << std::setw(maxWidth) << hasVal(row->at(i)) << " |\n";
                    }
                }
            return os;
        }

       private:
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

        // array<number, n> row{};
        VectorRowPtr<number, n> row{new array<number, n>{}};
    };

    template <Number number, size_t n>
    Vector<number, n> operator*(const number& scalar, const Vector<number, n>& vec) {
        return vec * scalar;
    }

    /**
     * @brief Type alias for a 2D Vector
     */
    template <Number number>
    using Vector2 = mlinalg::structures::Vector<number, 2>;

    /**
     * @brief Type alias for a 3D Vector
     */
    template <Number number>
    using Vector3 = mlinalg::structures::Vector<number, 3>;

    /**
     * @brief Type alias for a Vector as a row in a Matrix
     */
    template <Number number, size_t n>
    using Row = Vector<number, n>;

    /**
     * @brief Matrix class for representing NxM matrices
     *
     * @param m Number of rows
     * @param n Number of columns
     */
    template <Number number, size_t m, size_t n>
    class Matrix {
       public:
        /**
         * @brief Access the ith row of the matrix
         *
         * @param i The index of the row to access
         * @return A reference to the ith row
         */
        Row<number, n>& at(size_t i) { return matrix.at(i); }

        /**
         * @brief Const access the ith row of the matrix
         *
         * @param i The index of the row to access
         * @return A const reference to the ith row
         */
        Row<number, n> at(size_t i) const { return matrix.at(i); }

        /**
         * @brief Access the element at the ith row and jth column
         *
         * @param i The index of the row
         * @param j The index of the column
         * @return A reference to the element at the ith row and jth column
         */
        number& at(size_t i, size_t j) { return matrix.at(i).at(j); }

        Matrix() = default;

        /**
         * @brief Construct a new Matrix object from an initializer list of row vectors
         *
         * @param rows  Initializer list of row vectors
         */
        constexpr Matrix(const std::initializer_list<std::initializer_list<number>>& rows) {
            for (size_t i{}; i < m; i++) matrix.at(i) = Row<number, n>{*(rows.begin() + i)};
        }

        /**
         * @brief Copy construct a new Matrix object
         *
         * @param other Matrix to copy
         */
        Matrix(const Matrix& other) : matrix{other.matrix} {}

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

        /**
         * @brief Convert the columns of the matrix to a vector of column vectors
         *
         * @return A vector of column vectors
         */
        vector<Vector<number, m>> colToVectorSet() const {
            vector<Vector<number, m>> res;
            res.reserve(n);
            for (size_t i{}; i < n; i++) {
                Vector<number, m> vec;
                for (size_t j{}; j < m; j++) {
                    vec.at(j) = this->matrix.at(j).at(i);
                }
                res.emplace_back(std::move(vec));
            }
            return res;
        }

        /**
         * @brief Convert the rows of the matrix to a vector of row vectors
         *
         * @return A vector of row vectors
         */
        vector<Vector<number, n>> rowToVectorSet() const {
            vector<Vector<number, n>> res;
            res.reserve(m);
            for (const auto& row : matrix) res.emplace_back(row);
            return res;
        }

        /**
         * @brief Matrix multiplication by a scalar
         *
         * @param scalar  A scalar of the same type as the matrix
         * @return The matrix resulting from the multiplication
         */
        Matrix operator*(const number& scalar) const {
            Matrix<number, m, n> res;
            auto asRowVectorSet{std::move(rowToVectorSet())};
            for (size_t i{}; i < m; i++) {
                res.at(i) = asRowVectorSet.at(i) * scalar;
            }
            return res;
        }

        /**
         * @brief Matrix division by a scalar
         *
         * @param scalar A scalar of the same type as the matrix
         * @return The matrix resulting from the division
         */
        Matrix operator/(const number& scalar) const {
            Matrix<number, m, n> res;
            auto asRowVectorSet{std::move(rowToVectorSet())};
            for (size_t i{}; i < m; i++) res.at(i) = asRowVectorSet.at(i) / scalar;
            return res;
        }

        /**
         * @brief Matrix addition
         *
         * @param other The matrix to add
         * @return The matrix resulting from the addition
         */
        Matrix operator+(const Matrix<number, m, n>& other) const {
            Matrix<number, m, n> res;
            for (size_t i{}; i < m; i++) res.at(i) = this->at(i) + other.at(i);
            return res;
        }

        /**
         * @brief Matrix subtraction
         *
         * @param other The matrix to subtract
         * @return The matrix resulting from the subtraction
         */
        Matrix operator-(const Matrix<number, m, n>& other) const {
            Matrix<number, m, n> res;
            for (size_t i{}; i < m; i++) res.at(i) = this->at(i) - other.at(i);
            return res;
        }

        // friend number operator*(Matrix<number, 1, n> columnVector, Vector<number, n> vector) {
        //     return columnVector.multMatByVec(vector).at(0);
        // }

        /**
         * @brief Spaceship operator implementing the comparison operators
         *
         * @param other The matrix to compare
         */
        auto operator<=>(const Matrix& other) const = default;

        /**
         * @brief Matrix multiplication by a vector
         *
         * @param vec The vector to multiply by
         * @return The vector resulting from the multiplication of size m
         */
        Vector<number, m> operator*(const Vector<number, n>& vec) const { return multMatByVec(vec); }

        /**
         * @brief Matrix multiplication by a matrix
         *
         * @param other
         * @return
         */
        template <size_t nOther>
        Matrix<number, m, nOther> operator*(const Matrix<number, n, nOther>& other) const {
            return multMatByDef(other);
        }

        /**
         * @brief Matrix multiplication by a transposed matrix
         *
         * @param other The transposed matrix to multiply by
         * @return The matrix resulting from the multiplication
         */
        template <size_t nOther>
        Matrix<number, m, nOther> operator*(const TransposeVariant<number, n, nOther>& other) const {
            return multMatByDef(helpers::extractMatrixFromTranspose(other));
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
        [[nodiscard]] size_t numCols() const { return m; }

        explicit operator std::string() const {
            std::stringstream ss{};
            const auto& nRows = this->numRows();

            size_t maxWidth = 0;
            for (const auto& row : matrix) {
                for (const auto& elem : row) {
                    std::stringstream temp_ss;
                    temp_ss << elem;
                    maxWidth = std::max(maxWidth, temp_ss.str().length());
                }
            }

            if (this->numRows() == 1) {
                ss << "[ ";
                for (const auto& elem : this->at(0)) ss << " " << std::setw(maxWidth) << elem << " ";
                ss << "]\n";
            } else {
                int i{};
                for (const auto& row : this->matrix) {
                    if (i == 0) {
                        ss << "⎡";
                        for (const auto& elem : row) ss << " " << std::setw(maxWidth) << elem << " ";
                        ss << "⎤\n";
                    } else if (i == nRows - 1) {
                        ss << "⎣";
                        for (const auto& elem : row) ss << " " << std::setw(maxWidth) << elem << " ";
                        ss << "⎦\n";
                    } else {
                        ss << "|";
                        for (const auto& elem : row) ss << " " << std::setw(maxWidth) << elem << " ";
                        ss << "|\n";
                    }
                    i++;
                }
            }
            return ss.str();
        }

        friend std::ostream& operator<<(std::ostream& os, const Matrix<number, m, n>& system) {
            os << std::string(system);
            return os;
        }

        /**
         * @brief Transpose a mxn matrix to a nxm matrix
         *
         * @return The transposed matrix of size nxm
         */
        TransposeVariant<number, m, n> T() const {
            TransposeVariant<number, m, n> res;

            auto mutateMatrix = [this](auto& variant) {
                if constexpr (std::is_same_v<std::decay_t<decltype(variant)>, Matrix<number, n, m>>) {
                    for (size_t i{}; i < m; i++)
                        for (size_t j{}; j < n; j++) variant.at(j).at(i) = this->matrix.at(i).at(j);
                }
            };

            auto mutateVector = [this](auto& variant) {
                if constexpr (std::is_same_v<std::decay_t<decltype(variant)>, Vector<number, m>>) {
                    for (size_t i{}; i < m; i++) variant.at(i) = this->matrix.at(i).at(0);
                }
            };

            if (n != 1) {
                res = Matrix<number, n, m>{};
                std::visit(mutateMatrix, res);
                return res;
            }

            res = Vector<number, m>{};
            std::visit(mutateVector, res);
            return res;
        }

        friend std::ostream& operator<<(std::ostream& os, const TransposeVariant<number, n, m>& system) {
            if (std::holds_alternative<Vector<number, n>>(system)) {
                os << std::get<Vector<number, n>>(system);
            } else if (std::holds_alternative<Matrix<number, m, n>>(system)) {
                os << std::get<Matrix<number, m, n>>(system);
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
        template <size_t nN>
        Matrix<number, m, nN + n> augment(const Matrix<number, m, nN>& other) const {
            Matrix<number, m, nN + n> res;
            for (int i{}; i < m; i++) {
                auto& row{res.at(i)};
                const auto& thisRow{this->at(i)};
                const auto& otherRow{other.at(i)};
                for (int j{}; j < (nN + n); j++) {
                    if (j < n)
                        row.at(j) = thisRow.at(j);
                    else
                        row.at(j) = otherRow.at((j - static_cast<int>(nN)));
                }
            }
            return res;
        }

        /**
         * @brief Augment the matrix with a vector
         *
         * @param other The vector to augment with
         * @return The augmented matrix of size mx(n + 1)
         */
        Matrix<number, m, n + 1> augment(const Vector<number, m>& other) const {
            Matrix<number, m, n + 1> res;
            for (size_t i{}; i < m; i++) {
                auto& row{res.at(i)};
                const auto& thisRow{this->at(i)};
                size_t j{};
                for (; j < n; j++) {
                    row.at(j) = thisRow.at(j);
                }
                row.at(j) = other.at(i);
            }
            return res;
        }

        /**
         * @brief Determinant of the matrix
         *
         * @return The determinant of the matrix
         */
        number det() const {
            if (m != n) throw std::runtime_error("Finding determinant of rectangular matrices not implemented");
            if constexpr (m == 2 && n == 2)
                return det2x2();
            else
                return cofactor();
        }

        /**
         * @brief Subset the matrix by removing a row and a column
         *
         * @param i Row index to remove
         * @param j Column index to remove
         * @return The subsetted matrix of size (m - 1)x(n - 1)
         */
        Matrix<number, m - 1, n - 1> subset(std::optional<size_t> i, std::optional<size_t> j) const {
            static_assert(m == n, "Not a square matrix");
            static_assert(m > 1 && n > 1, "Cannot subset");

            Matrix<number, m - 1, n - 1> res;
            size_t resRow = 0;
            for (size_t k = 0; k < m; ++k) {
                if (i.has_value() && i.value() == k) continue;  // Skip the row to be removed

                size_t resCol = 0;
                for (size_t z = 0; z < n; ++z) {
                    if (j.has_value() && j.value() == z) continue;  // Skip the column to be removed

                    res.at(resRow).at(resCol) = this->at(k).at(z);
                    ++resCol;
                }
                ++resRow;
            }
            return res;
        }

       private:
        /**
         * @brief Calculate the determinant of a 2x2 matrix
         *
         * @return The determinant of the matrix
         */
        number det2x2() const {
            return (this->at(0).at(0) * this->at(1).at(1)) - (this->at(0).at(1) * this->at(1).at(0));
        }

        enum By { ROW = 0, COL };

        /**
         * @brief Pick the row or column with the most zeros as a cofactor row or column
         */
        std::pair<By, size_t> pickCofactorRowOrCol() const {
            size_t maxRowZeros{};
            size_t rowPos{};

            size_t maxColZeros{};
            size_t colPos{};

            int pos{};
            for (const auto& row : *this) {
                auto count = rg::count_if(row, [](auto x) { return x == 0; });
                if (count > maxRowZeros) {
                    maxRowZeros = count;
                    rowPos = pos;
                }
                pos++;
            }

            pos = 0;
            for (const auto& col : colToVectorSet()) {
                auto count = rg::count_if(col, [](auto x) { return x == 0; });
                if (count > maxColZeros) {
                    maxColZeros = count;
                    colPos = pos;
                }
                pos++;
            }

            if (maxColZeros >= maxRowZeros)
                return {COL, colPos};
            else
                return {ROW, rowPos};
        }

        /**
         * @brief Common cofactor calculation for a row or column
         *
         * @param i The row index
         * @param j The column index
         * @return The cofactor for the row or column
         */
        constexpr number cofactorCommon(size_t i, size_t j) const {
            number res{};
            auto a = this->at(i).at(j);
            if (a == 0) return 0;
            auto A_ij = this->subset(i, j);
            auto C = ((int)std::pow(-1, ((i + 1) + (j + 1)))) * A_ij.det();
            res += a * C;
            return res;
        }

        /**
         * @brief Calculate the cofactor for a row
         *
         * @param row The row index
         * @return The cofactor for the row
         */
        constexpr number cofactorRow(size_t row) const {
            number res{};
            size_t i{row};
            for (size_t j{0}; j < n; j++) {
                res += cofactorCommon(i, j);
            }
            return res;
        }

        /**
         * @brief Calculate the cofactor for a column
         *
         * @param col The column index
         * @return The cofactor for the column
         */
        constexpr number cofactorCol(size_t col) const {
            number res{};
            size_t j{col};
            for (size_t i{0}; i < n; i++) {
                res += cofactorCommon(i, j);
            }
            return res;
        }

        /**
         * @brief Main cofactor calculation function
         *
         * @return The determinant of the matrix
         */
        constexpr number cofactor() const {
            auto [by, val] = pickCofactorRowOrCol();
            if (by == ROW)
                return cofactorRow(val);
            else
                return cofactorCol(val);
        }

        /**
         * @brief Matrix multiplication by a vector implementation function
         *
         * @param vec The vector to multiply by
         * @return The vector resulting from the multiplication
         */
        Vector<number, m> multMatByVec(const Vector<number, n>& vec) const {
            Vector<number, m> res;
            auto asCols{std::move(colToVectorSet())};
            int i{};
            for (auto& col : asCols) {
                const auto& mult = vec.at(i);
                col *= mult;
                i++;
            }

            for (size_t i{}; i < m; i++) {
                number sumRes{};
                for (const auto& col : asCols) sumRes += col.at(i);
                res.at(i) = sumRes;
            }

            return res;
        }

        /**
         * @brief Matrix multiplication by a matrix by the definition of matrix multiplication
         *
         * @param other The matrix to multiply by
         * @return The matrix resulting from the multiplication
         */
        template <size_t nOther>
        Matrix<number, m, nOther> multMatByDef(const Matrix<number, n, nOther>& other) const {
            auto otherColVecSet{std::move(other.colToVectorSet())};
            vector<Vector<number, m>> res;
            res.reserve(nOther);
            for (size_t i{}; i < nOther; i++) {
                const auto& col{otherColVecSet.at(i)};
                auto multRes = *this * col;
                res.emplace_back(multRes);
            }
            return helpers::fromColVectorSet<number, m, nOther>(res);
        }

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

        /**
         * @brief Backing array for the matrix
         */
        array<Row<number, n>, m> matrix{};
    };

    // Vector(std::array<number, n>) -> Vector<number, n>;

    template <Number number, size_t m, size_t n, size_t nOther>
    Matrix<number, m, nOther> operator*(Matrix<number, m, n> lhs, Matrix<number, n, nOther> rhs) {
        return lhs * rhs;
    }

    template <Number number, size_t m, size_t n, size_t nOther>
    TransposeVariant<number, m, nOther> operator*(TransposeVariant<number, m, n> lhs, Matrix<number, n, nOther> rhs) {
        if (std::holds_alternative<Vector<number, m>>(lhs)) {
            auto vec = std::get<Vector<number, m>>(lhs);
            return vec * rhs;
        } else {
            auto mat = std::get<Matrix<number, m, n>>(lhs);
            return mat * rhs;
        }
    }

    template <Number number, size_t m, size_t n>
    Matrix<number, m, n> operator*(const number& scalar, Matrix<number, m, n> rhs) {
        return rhs * scalar;
    }

    template <Number number, size_t m, size_t n>
    Matrix<number, m, n> operator*(Vector<number, n> vec, Matrix<number, m, n> rhs) {
        Matrix<number, m, n> res;
        for (size_t i{}; i < n; i++) {
            const auto& mult = vec.at(i);
            for (size_t j{}; j < m; j++) res.at(j).at(i) = mult * rhs.matrix.at(j).at(i);
        }
        return res;
    }

}  // namespace mlinalg::structures
