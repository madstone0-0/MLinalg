/**
 * @file Vector.hpp
 * @brief Header file for the Vector class
 */

#pragma once

#include "VectorBase.hpp"

using std::vector, std::array, std::optional, std::unique_ptr, std::shared_ptr, std::optional;

namespace mlinalg::structures {

    /**
     * @brief Vector class for represeting both row and column vectors in n-dimensional space
     *
     * @tparam n The number of elements in the vector
     */
    template <Number number, int n>
    class Vector : public VectorBase<Vector<number, n>, number> {
       public:
        using Base = VectorBase<Vector<number, n>, number>;
        static constexpr auto elements = n;

        // ===========================
        // Constructors and Destructor
        // ===========================

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

        ~Vector() = default;

        // =====================
        // Arithmetic Operators
        // ====================

        /**
         * @brief Vector multiplication by a matrix
         *
         * @param mat A matrix of size MxN
         * @return The vector resulting from the multiplication
         */
        template <int m>
        Vector<number, n> operator*(const Matrix<number, m, n>& mat) const {
            Vector<number, n> res;
            auto asCols{mat.colToVectorSet()};
            for (int i{}; i < n; i++) {
                res.at(i) = *this * asCols[i];
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
            for (int i{}; i < n; i++) {
                res[i] = *this;
                res[i] *= mat.col(i)[0];
            }
            return res;
        }

        // =================
        // Vector Operations
        // =================

        /**
         * @brief Transpose a row vector to a column vector
         *
         * @return ColumnVector<number, n>
         */
        [[nodiscard]] Matrix<number, 1, n> T() const { return vectorTranspose<number, 1, n>(row); }

        // ======================
        // Miscellaneous Operations
        // ======================

        template <long start = 0, long end = n, long stride = 1, int newSize = (end - start + stride - 1) / stride>
        auto view(long /*offsetArg*/ = 0, long /*endArg*/ = 0, long /*strideArg*/ = 1) {
            return View<number, n, newSize>(row, start, end, stride);
        }

        /**
         * @brief Size of the vector, i.e the number of elements in the vector
         *
         * @return the size of the vector
         */
        [[nodiscard]] size_t size() const override { return static_cast<size_t>(n); }

       private:
        template <Number num, int mM, int nN>
        friend class Matrix;

        template <typename D, Number num>
        friend class MatrixBase;

        template <typename D, Number num>
        friend class VectorBase;

        void checkDimensions() {
            if constexpr (n <= 0) throw std::invalid_argument("Vector size must be greater than 0");
        }

        VectorRow<number, n> row{VectorRow<number, n>(n, number{})};
    };

}  // namespace mlinalg::structures

namespace mlinalg::structures {

    /**
     * @brief Dynamic Vector class for representing both row and column vectors in n-dimensional space
     */
    template <Number number>
    class Vector<number, Dynamic> : public VectorBase<Vector<number, Dynamic>, number> {
       public:
        using Base = VectorBase<Vector<number, Dynamic>, number>;
        static constexpr auto elements = Dynamic;

        // ===========================
        // Constructors and Destructor
        // ===========================

        Vector() = default;
        explicit Vector(size_t size) : n{size}, row(size) {}

        Vector(const std::initializer_list<number>& list) : n{list.size()}, row{list} {}

        template <typename Iterator>
        Vector(Iterator begin, Iterator end) : n(std::distance(begin, end)), row{begin, end} {}

        /**
         * @brief Copy construct a dynamic Vector from a static vector
         *
         * @param other Static vector to copy
         */
        template <int nN>
        explicit Vector(const Vector<number, nN>& other)
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
            if (nRows != 1)
                throw StackError<std::invalid_argument>("Matrix must have 1 row to be converted to a vector");
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

        ~Vector() = default;

        // ====================
        // Arithmetic Operators
        // ====================

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
                throw StackError<std::invalid_argument>(
                    "Matrix must have the same number of columns as the vector size");

            auto asCols{std::move(mat.colToVectorSet())};
            // If the matrix has only one row it is equivalent to a transposed vector
            // And thus the result is a matrix with the cols of this matrix being the result of the dot product of the
            // vector and the cols of the matrix
            if (numRows == 1) {
                Matrix<number, Dynamic, Dynamic> resMat(this->n, this->n);
                for (size_t i{}; i < n; i++) {
                    resMat[i] = *this;
                    resMat[i] *= mat.col(i)[0];
                }
                return resMat;
            } else {
                Vector<number, Dynamic> resVec(this->n);
                for (size_t i{}; i < n; i++) {
                    resVec[i] = *this * mat.col(i);
                }
                return resVec;
            }
        }

        // =================
        // Vector Operations
        // =================

        /**
         * @brief Transpose a row vector to a column vector
         *
         * @return Matrix<number, 1, n>
         */
        [[nodiscard]] Matrix<number, Dynamic, Dynamic> T() const {
            return vectorTranspose<number, Dynamic, Dynamic>(row);
        }

        // ========================
        // Miscellaneous Operations
        // ========================

        /**
         * @brief Size of the vector, i.e the number of elements in the vector
         *
         * @return the size of the vector
         */
        [[nodiscard]] size_t size() const override { return n; }

        /**
         * @brief Add an element to the end of the vector
         *
         * @param v The element to add
         */
        void pushBack(const number& v) {
            row.push_back(v);
            n = row.size();
        }

        /**
         * @brief Emplace an element to the end of the vector
         *
         * @param args The arguments to construct the element
         * @return A reference to the newly added element
         */
        template <Number... Numbers>
            requires(sizeof...(Numbers) > 0 && (std::is_convertible_v<Numbers, number> && ...))
        auto emplaceBack(Numbers&&... args) {
            auto res = row.emplace_back(std::forward<Numbers>(args)...);
            n = row.size();
            return res;
        }

        /**
         * @brief Resize the vector to a new size
         *
         * @param newSize The new size of the vector
         */
        void resize(size_t newSize) {
            if (newSize < n) {
                row.resize(newSize);
            } else if (newSize > n) {
                row.resize(newSize, number{});
            }
            n = newSize;
        }

        /**
         * @brief Reserve space for a new size in the vector
         *
         * @param newSize The new size to reserve space for
         */
        void reserve(size_t newSize) { return row.reserve(newSize); }

        /**
         * @brief Remove the element at the given index from the vector
         *
         * @param idx The index of the element to remove
         * @return The removed element
         */
        number remove(size_t idx) {
            if (idx >= n) throw StackError<std::out_of_range>("Index out of range");
            auto val = row[idx];
            row.erase(row.begin() + idx);
            --n;
            return val;
        }

       private:
        template <Number num, int mM, int nN>
        friend class Matrix;

        template <typename D, Number num>
        friend class MatrixBase;

        template <typename D, Number num>
        friend class VectorBase;

        size_t n;
        VectorRowDynamic<number> row{};
    };
}  // namespace mlinalg::structures
