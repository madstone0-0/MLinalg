/**
 * @file Vector.hpp
 * @brief Header file for the Vector class
 */

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iterator>
#include <memory>
#include <numeric>
#include <optional>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../Concepts.hpp"
#include "../Numeric.hpp"

using std::vector, std::array, std::optional, std::unique_ptr, std::shared_ptr;

namespace mlinalg::structures {
    template <Number number, int m, int n>
    class Matrix;

    template <Number number, int n>
    class Vector;

    namespace {
        template <Container T, Container U>
        void checkOperandSize(const T& row, const U& otherRow) {
            if (row.size() != otherRow.size()) throw std::invalid_argument("Vectors must be of the same size");
        }

        template <Container T, Container U>
        bool vectorEqual(const T& row, const U& otherRow) {
            checkOperandSize(row, otherRow);
            auto n = row.size();
            for (size_t i{}; i < n; i++)
                if (!fuzzyCompare(row.at(i), otherRow.at(i))) return false;
            return true;
        }

        template <Number number, Container T>
        number& vectorAt(T& row, size_t i) {
            return row.at(i);
        }

        template <Number number, Container T>
        number vectorConstAt(const T& row, size_t i) {
            return row.at(i);
        }

        template <Number number, int n, Container T, Container U>
        Vector<number, n> vectorSub(const T& row, const U& otherRow) {
            checkOperandSize(row, otherRow);
            constexpr int vSize = (n != -1) ? n : -1;
            auto size = row.size();
            Vector<number, vSize> res(size);
            for (size_t i{}; i < size; i++) res.at(i) = row.at(i) - otherRow.at(i);
            return res;
        }

        template <Number number, int n, Container T>
        Vector<number, n> vectorNeg(const T& row) {
            constexpr int vSize = (n != -1) ? n : -1;
            auto size = row.size();
            Vector<number, vSize> res(size);
            for (size_t i{}; i < size; i++) res.at(i) = -row.at(i);
            return res;
        }

        template <Number number, Container T, Container U>
        void vectorSubI(T& row, const U& otherRow) {
            checkOperandSize(row, otherRow);
            for (size_t i{}; i < row.size(); i++) row.at(i) -= otherRow.at(i);
        }

        template <Number number, int n, Container T, Container U>
        Vector<number, n> vectorAdd(const T& row, const U& otherRow) {
            checkOperandSize(row, otherRow);
            constexpr int vSize = (n != -1) ? n : -1;
            auto size = row.size();
            Vector<number, vSize> res(size);
            for (size_t i{}; i < size; i++) res.at(i) = row.at(i) + otherRow.at(i);
            return res;
        }

        template <Number number, Container T, Container U>
        void vectorAddI(T& row, const U& otherRow) {
            checkOperandSize(row, otherRow);
            for (size_t i{}; i < row.size(); i++) row.at(i) += otherRow.at(i);
        }

        template <Number number, int n, Container T>
        Vector<number, n> vectorScalarMult(const T& row, const number& scalar) {
            constexpr int vSize = (n != -1) ? n : -1;
            auto size = row.size();
            Vector<number, vSize> res(size);
            for (size_t i{}; i < size; i++) res.at(i) = scalar * row.at(i);
            return res;
        }

        template <Number number, int n, Container T>
        Vector<number, n> vectorScalarDiv(const T& row, const number& scalar) {
            if (fuzzyCompare(scalar, number(0))) throw std::domain_error("Division by zero");
            constexpr int vSize = (n != -1) ? n : -1;
            auto size = row.size();
            Vector<number, vSize> res(size);
            for (size_t i{}; i < size; i++) res.at(i) = row.at(i) / scalar;
            return res;
        }

        template <Number number, Container T>
        void vectorScalarMultI(T& row, const number& scalar) {
            for (size_t i{}; i < row.size(); i++) row.at(i) *= scalar;
        }

        template <Number number, Container T>
        void vectorScalarDivI(T& row, const number& scalar) {
            for (size_t i{}; i < row.size(); i++) row.at(i) /= scalar;
        }

        template <Number number, int n, int otherN>
        number vectorVectorMult(const Vector<number, n>& vec, const Vector<number, otherN>& otherVec) {
            if (vec.size() != otherVec.size()) throw std::invalid_argument("Vectors must be of the same size");

            const auto res = vec.T() * otherVec;
            return res.at(0);
        }

        template <Container T>
        std::string vectorStringRepr(const T& row) {
            std::stringstream ss{};

            size_t maxWidth = 0;
            for (const auto& elem : row) {
                std::stringstream temp_ss;
                temp_ss << elem;
                maxWidth = std::max(maxWidth, temp_ss.str().length());
            }

            if (row.size() == 1)
                ss << "[ " << std::setw(static_cast<int>(maxWidth)) << row.at(0) << " ]\n";
            else
                for (size_t i{}; i < row.size(); i++)
                    if (i == 0) {
                        ss << "⎡ " << std::setw(static_cast<int>(maxWidth)) << row.at(i) << " ⎤\n";
                    } else if (i == row.size() - 1) {
                        ss << "⎣ " << std::setw(static_cast<int>(maxWidth)) << row.at(i) << " ⎦\n";
                    } else {
                        ss << "| " << std::setw(static_cast<int>(maxWidth)) << row.at(i) << " |\n";
                    }
            return ss.str();
        }

        template <Container T>
        std::ostream& vectorOptionalRepr(std::ostream& os, const T& row) {
            const auto& size = row.size();

            auto hasVal = [](auto rowVal) {
                std::stringstream val;
                if (rowVal.has_value())
                    val << rowVal.value();
                else
                    val << "None";
                return val.str();
            };

            size_t maxWidth = 0;
            for (const auto& elem : row) {
                std::stringstream temp_ss;
                temp_ss << hasVal(elem);
                maxWidth = std::max(maxWidth, temp_ss.str().length());
            }

            if (size == 1) {
                os << "[ ";
                if (row.at(0).has_value())
                    os << row.at(0).value();
                else
                    os << "None";
                os << " ]\n";
            } else
                for (size_t i{}; i < row.size(); i++) {
                    if (i == 0) {
                        os << "⎡ " << std::setw(static_cast<int>(maxWidth)) << hasVal(row.at(i)) << " ⎤\n";
                    } else if (i == row.size() - 1) {
                        os << "⎣ " << std::setw(static_cast<int>(maxWidth)) << hasVal(row.at(i)) << " ⎦\n";

                    } else {
                        os << "| " << std::setw(static_cast<int>(maxWidth)) << hasVal(row.at(i)) << " |\n";
                    }
                }
            return os;
        }

        template <Number number, int m, int n, Container T>
        Matrix<number, m, n> vectorTranspose(const T& row) {
            constexpr auto sizeP = (n == -1) ? std::pair<int, int>{-1, -1} : std::pair<int, int>{1, n};
            const auto size = row.size();
            Matrix<number, sizeP.first, sizeP.second> res(1, size);
            for (size_t i{}; i < size; i++) res.at(0, i) = row.at(i);
            return res;
        }

    }  // namespace

    // Type alias for the backing array of a Vector
    template <Number number, int n>
    using VectorRow = std::array<number, n>;

    // Type alias for the backing array of a dynamic Vector
    template <Number number>
    using VectorRowDynamic = std::vector<number>;

    // Type alias for a unique pointer to a VectorRow
    template <Number number, int n>
    using VectorRowPtr = unique_ptr<VectorRow<number, n>>;

    // Type alias for a unique pointer to a dynamic VectorRow
    template <Number number>
    using VectorRowDynamicPtr = unique_ptr<VectorRowDynamic<number>>;

    /**
     * @brief Vector class for represeting both row and column vectors in n-dimensional space
     *
     * @param n The number of elements in the vector
     */
    template <Number number, int n>
    class Vector {
       public:
        Vector() { checkDimensions(); }

        // Constructor to keep consistency with the Dynamic Vector specialization to allow them to be used
        // interchangeably
        explicit Vector(size_t size) {}  // NOLINT

        /**
         * @brief Construct a new Vector object from an initializer list
         *
         * @param list Initializer list of numbers
         */
        Vector(const std::initializer_list<number>& list) {
            checkDimensions();
            if (list.size() != n)
                throw std::invalid_argument("Initializer list must be the same size as the defined vector size");
            for (int i{}; i < n; i++) row->at(i) = *(list.begin() + i);
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
            for (int i{}; i < n; i++) row->at(i) = other.at(0).at(i);
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
        bool operator==(const Vector& other) const { return vectorEqual(*row, *other.row); }

        ~Vector() { row.reset(); }

        /**
         * @brief Access the ith element of the vector
         *
         * @param i the index of the element to access
         * @return a reference to the ith element
         */
        number& at(size_t i) { return vectorAt<number>(*row, i); }

        /**
         * @brief Const access the ith element of the vector
         *
         * @param i the index of the element to access
         * @return  the ith element
         */
        number at(size_t i) const { return vectorConstAt<number>(*row, i); }

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
            return vectorSub<number, n>(*row, *other.row);
        }

        /**
         * @brief Vector Negation
         *
         * @return the negeated vector
         */
        Vector<number, n> operator-() const { return vectorNeg<number, n>(*row); }

        /**
         * @brief In-place vector subtraction
         *
         * @param other  the vector to add
         * @return A reference to the same vector
         */
        Vector<number, n>& operator-=(const Vector<number, n>& other) {
            vectorSubI<number>(*row, *other.row);
            return *this;
        }

        /**
         * @brief Vector addition
         *
         * @param other the vector to add
         * @return the vector resulting from the addition
         */
        Vector<number, n> operator+(const Vector<number, n>& other) const {
            return vectorAdd<number, n>(*row, *other.row);
        }

        /**
         * @brief In-place vector addition
         *
         * @param other  the vector to add
         * @return A reference to the same vector
         */
        Vector<number, n>& operator+=(const Vector<number, n>& other) {
            vectorAddI<number>(*row, *other.row);
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
        Vector<number, n> operator*(const number& scalar) const { return vectorScalarMult<number, n>(*row, scalar); }

        /**
         * @brief Vector division by a scalar
         *
         * @param scalar A scalar of the same type as the vector
         * @return  The vector resulting from the division
         */
        Vector<number, n> operator/(const number& scalar) const { return vectorScalarDiv<number, n>(*row, scalar); }

        /**
         * @brief In-place vector multiplication by a scalar.
         *
         * @param scalar A scalar of the same type as the vector.
         * @return
         */
        Vector<number, n>& operator*=(const number& scalar) {
            vectorScalarMultI<number>(*row, scalar);
            return *this;
        }

        /**
         * @brief In-place vector division by a scalar
         *
         * @param scalar A scalar of the same type as the vector
         * @return
         */
        Vector<number, n>& operator/=(const number& scalar) {
            vectorScalarDivI(*row, scalar);
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
        Vector<number, n> operator*(const Matrix<number, m, n>& mat) {
            Vector<number, n> res;
            auto asCols{std::move(mat.colToVectorSet())};
            for (int i{}; i < n; i++) {
                res.at(i) = *this * asCols.at(i);
            }
            return res;
        }

        /**
         * @brief Transpose a row vector to a column vector
         *
         * @return ColumnVector<number, n>
         */
        [[nodiscard]] Matrix<number, 1, n> T() const { return vectorTranspose<number, 1, n>(*row); }

        /**
         * @brief Begin iterator for the vector
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

        explicit operator std::string() const { return vectorStringRepr(*row); }

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

        // array<number, n> row{};
        VectorRowPtr<number, n> row{new array<number, n>{}};
    };

    template <Number number, int n>
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
}  // namespace mlinalg::structures

namespace mlinalg::structures {
    static const int Dynamic{-1};

    /**
     * @brief Dynamic Vector class for representing both row and column vectors in n-dimensional space
     */
    template <Number number>
    class Vector<number, Dynamic> {
       public:
        Vector() = delete;
        explicit Vector(size_t size) : n{size}, row{new VectorRowDynamic<number>(size)} {}

        Vector(const std::initializer_list<number>& list)
            : n{list.size()}, row{std::make_unique<VectorRowDynamic<number>>(list)} {}

        template <typename Iterator>
        Vector(Iterator begin, Iterator end)
            : n(std::distance(begin, end)), row{std::make_unique<VectorRowDynamic<number>>(begin, end)} {}

        template <int nN>
        Vector(const Vector<number, nN>& other)
            requires(nN != Dynamic)
            : n{nN}, row{new VectorRowDynamic<number>(nN)} {
            for (int i{}; i < nN; i++) {
                this->at(i) = other.at(i);
            }
        }

        /**
         * @brief Copy construct a new Vector object
         *
         * @param other Vector to copy
         */
        Vector(const Vector<number, Dynamic>& other) : n{other.n}, row{new VectorRowDynamic<number>{*other.row}} {}

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
            row = VectorRowDynamicPtr<number>{new VectorRowDynamic<number>{*other.row}};
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
            for (int i{}; i < nCols; i++) row->at(i) = other.at(0).at(i);
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
            return vectorEqual(*row, *other.row);
        }

        template <int n, int otherN>
        friend bool operator==(const Vector<number, otherN>& lhs, const Vector<number, n> rhs)
            requires(n == Dynamic && otherN != Dynamic)
        {
            return vectorEqual(*lhs.row, *rhs.row);
        }

        ~Vector() { row.reset(); }

        /**
         * @brief Access the ith element of the vector
         *
         * @param i the index of the element to access
         * @return a reference to the ith element
         */
        number& at(size_t i) { return vectorAt<number>(*row, i); }

        /**
         * @brief Const access the ith element of the vector
         *
         * @param i the index of the element to access
         * @return  the ith element
         */
        number at(size_t i) const { return vectorConstAt<number>(*row, i); }

        /**
         * @brief Find the dot product of this vector and another vector
         *
         * @param other The other vector
         * @return the dot product of the two vectors
         */
        template <int otherN>
        double dot(const Vector<number, otherN>& other) const {
            checkOperandSize(*row, *(other.row));
            return (this->T() * other).at(0);
        }

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

        template <int otherN>
        [[nodiscard]] double dist(const Vector<number, otherN>& other) const {
            checkOperandSize(*row, *(other.row));
            auto diff = *this - other;
            return std::sqrt(diff.dot(diff));
        }

        /**
         * @brief Vector subtraction
         *
         * @param other the vector to subtract
         * @return the vector resulting from the subtraction
         */
        template <int otherN>
        Vector<number, Dynamic> operator-(const Vector<number, otherN>& other) const {
            return vectorSub<number, Dynamic>(*row, *other.row);
        }

        template <int n, int otherN>
        friend Vector<number, Dynamic> operator-(const Vector<number, otherN>& lhs, const Vector<number, n> rhs)
            requires(n == Dynamic && otherN != Dynamic)
        {
            return vectorSub<number, Dynamic>(*lhs.row, *rhs.row);
        }

        /**
         * @brief Vector addition
         *
         * @param other the vector to add
         * @return the vector resulting from the addition
         */
        template <int otherN>
        Vector<number, Dynamic> operator+(const Vector<number, otherN>& other) const {
            return vectorAdd<number, Dynamic>(*row, *other.row);
        }

        template <int n, int otherN>
        friend Vector<number, Dynamic> operator+(const Vector<number, otherN>& lhs, const Vector<number, n> rhs)
            requires(n == Dynamic && otherN != Dynamic)
        {
            return vectorAdd<number, Dynamic>(*lhs.row, *rhs.row);
        }

        /**
         * @brief In-place vector addition
         *
         * @param other  the vector to add
         * @return A reference to the same vector
         */
        template <int otherN>
        Vector<number, Dynamic>& operator+=(const Vector<number, otherN>& other) {
            vectorAddI<number, Dynamic>(*row, *other.row);
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
            vectorSubI<number>(*row, *other.row);
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
            return vectorScalarMult<number, Dynamic>(*row, scalar);
        }

        /**
         * @brief Vector division by a scalar
         *
         * @param scalar A scalar of the same type as the vector
         * @return  The vector resulting from the division
         */
        Vector<number, Dynamic> operator/(const number& scalar) const {
            return vectorScalarDiv<number, Dynamic>(*row, scalar);
        }

        /**
         * @brief In-place vector multiplication by a scalar.
         *
         * @param scalar A scalar of the same type as the vector.
         * @return
         */
        Vector<number, Dynamic>& operator*=(const number& scalar) {
            vectorScalarMultI<number>(*row, scalar);
            return *this;
        }

        /**
         * @brief In-place vector division by a scalar
         *
         * @param scalar A scalar of the same type as the vector
         * @return
         */
        Vector<number, Dynamic>& operator/=(const number& scalar) {
            vectorScalarDivI(*row, scalar);
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
            checkOperandSize(*row, *(vec.row));
            return vectorVectorMult(*this, vec);
        }

        /**
         * @brief Vector multiplication by a matrix
         *
         * @param mat A matrix of size MxN
         * @return The vector resulting from the multiplication
         */
        Vector<number, Dynamic> operator*(const Matrix<number, Dynamic, Dynamic>& mat) {
            const auto& numRows = mat.numRows();
            if (numRows != n)
                throw std::invalid_argument("Matrix must have the same number of rows as the vector size");

            Vector<number, Dynamic> res(n);
            auto asCols{std::move(mat.colToVectorSet())};
            for (int i{}; i < n; i++) {
                res.at(i) = *this * asCols.at(i);
            }
            return res;
        }

        /**
         * @brief Transpose a row vector to a column vector
         *
         * @return Matrix<number, 1, n>
         */
        [[nodiscard]] Matrix<number, Dynamic, Dynamic> T() const {
            return vectorTranspose<number, Dynamic, Dynamic>(*row);
        }

        /**
         * @brief Begin iterator for the vector
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

        explicit operator std::string() const { return vectorStringRepr(*row); }

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
        VectorRowDynamicPtr<number> row{std::make_unique<VectorRowDynamic<number>>()};
    };
}  // namespace mlinalg::structures
