/**
 * @file Matrix.hpp
 * @brief Header file for the Matrix class
 */

// FIX: Fix matrix-vector and vector-matrix multiplication order

#pragma once
#include <functional>

#include "Aliases.hpp"
#include "MatrixBase.hpp"
#include "MatrixOps.hpp"

namespace mlinalg::structures {

    /**
     * @brief Matrix class for representing MxN matrices
     *
     * @param m Number of rows
     * @param n Number of columns
     */
    template <Number number, int m, int n>
    class Matrix : public MatrixBase<Matrix<number, m, n>, number> {
       public:
        using Base = Matrix<number, m, n>;
        using Backing = TDArray<number, m, n>;
        using iterator = typename Backing::iterator;
        using value_type = typename Backing::value_type;
        using size_type = typename Backing::size_type;
        using MatrixColumns = Columns<number, m, n, Base>;

        static constexpr int rows = m;
        static constexpr int cols = n;

        // ===========================
        // Constructors and Destructor
        // ===========================

        Matrix() = default;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

        // Constructor to keep consistency with the Dynamic Matrix specialization to allow them to be used
        // interchangeably
        Matrix(size_t nRows, size_t nCols) {}  // NOLINT

#pragma GCC diagnostic pop

        /**
         * @brief Construct a new Matrix object from an initializer list of row vectors
         *
         * @param rows  Initializer list of row vectors
         */
        constexpr Matrix(const std::initializer_list<std::initializer_list<number>>& rows) {
            static_assert(m > 0, "Number of rows cannot be 0");
            static_assert(n > 0, "Number of colmns cannot be 0");
            for (int i{}; i < m; i++) {
                matrix.at(i) = Row<number, n>{*(rows.begin() + i)};
            }
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

        /**
         * @brief Default destructor
         */
        ~Matrix() = default;

        // ====================
        // Arithmetic Operators
        // ====================

        /**
         * @brief Matrix multiplication by a vector
         *
         * @param vec The vector to multiply by
         * @return The vector resulting from the multiplication of size m
         */
        template <int nOther>
        Vector<number, m> operator*(const Vector<number, nOther>& vec) const {
            if constexpr (nOther != n)
                throw runtime_error("The columns of the matrix must be equal to the size of the vector");
            return MatrixVectorMultiplication<number, m, n, nOther>(matrix, vec.row);
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

        // =================
        // Matrix Operations
        // =================

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

        // ========================
        // Miscellaneous Operations
        // ========================

        template <OffsetPair rowOffset = {0, m}, OffsetPair colOffset = {0, -1}, StridePair stride = {1, 1},
                  int nM = (rowOffset.end - rowOffset.start + stride.row - 1) / stride.row,
                  int nN = (colOffset.end - colOffset.start + stride.col - 1) / stride.col>
        auto view(OffsetPair /*rowOffsetArg*/ = {}, OffsetPair /*colOffsetArg*/ = {}, StridePair /*strideArg*/ = {}) {
            return View<number, m, n, nM, nN, colOffset, stride.col>(matrix, rowOffset, colOffset, stride);
        }

        /**
         * @brief Number of rows in the matrix
         *
         * @return The number of rows in the matrix
         */
        [[nodiscard]] constexpr size_t numRows() const { return static_cast<size_t>(m); }

        [[nodiscard]] constexpr size_t size() const { return static_cast<size_t>(m); }
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

       private:
        template <Number num, int mM, int nN>
        friend class Matrix;

        template <Number num, int nN>
        friend class Vector;

        template <typename D, Number num>
        friend class MatrixBase;

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
        TDArray<number, m, n> matrix{m};

        MatrixColumns columns{this};
    };

}  // namespace mlinalg::structures

namespace mlinalg::structures {
    /**
     * @brief Dynamic Matrix class for representing MxN matrices
     *
     * @param m  Number of rows
     * @param n  Number of columns
     */
    template <Number number>
    class Matrix<number, Dynamic, Dynamic> : public MatrixBase<Matrix<number, Dynamic, Dynamic>, number> {
       public:
        using Base = Matrix<number, Dynamic, Dynamic>;
        using Backing = TDArrayDynamic<number>;
        using iterator = typename Backing::iterator;
        using value_type = typename Backing::value_type;
        using size_type = typename Backing::size_type;
        using MatrixColumns = Columns<number, Dynamic, Dynamic, Base>;

        static constexpr int rows = Dynamic;
        static constexpr int cols = Dynamic;

        // ===========================
        // Constructors and Destructor
        // ===========================

        Matrix(size_t m, size_t n) : m(m), n(n) {
            if (m <= 0) throw invalid_argument("Matrix must have at least one row");
            if (n <= 0) throw invalid_argument("Matrix must have at least one column");
            matrix.reserve(m);
            matrix.resize(m, Vector<number, Dynamic>(n));
        }

        Matrix(const std::initializer_list<std::initializer_list<number>>& rows)
            : m{rows.size()}, n{rows.begin()->size()}, columns{this} {
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
            : m{m}, n{n}, columns{this} {
            matrix.reserve(m);
            for (size_t i{}; i < other.numRows(); ++i) {
                matrix.emplace_back(other.at(i));
            }
        }

        /**
         * @brief Copy construct a new Matrix object
         *
         * @param other Matrix to copy
         */
        Matrix(const Matrix& other) : m{other.m}, n{other.n}, matrix{other.matrix}, columns{this} {
            if (m <= 0) throw invalid_argument("Matrix must have at least one row");
            if (n <= 0) throw invalid_argument("Matrix must have at least one column");
        }

        /**
         * @brief Move construct a new Matrix object
         *
         * @param other  Matrix to move
         */
        Matrix(Matrix&& other) noexcept : m{other.m}, n{other.n}, matrix{std::move(other.matrix)}, columns{this} {
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
            columns = MatrixColumns{this};
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
            columns = MatrixColumns{this};
            return *this;
        }

        /**
         * @brief Default destructor
         */
        ~Matrix() = default;

        // ====================
        // Arithmetic Operators
        // ====================

        /**
         * @brief Matrix multiplication by a vector
         *
         * @param vec The vector to multiply by
         * @return The vector resulting from the multiplication of size m
         */

        template <int nOther>
        Vector<number, Dynamic> operator*(const Vector<number, nOther>& vec) const {
            return MatrixVectorMultiplication<number, Dynamic, Dynamic, Dynamic>(matrix, vec.row);
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

        // =================
        // Matrix Operations
        // =================

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

        // ========================
        // Miscellaneous Operations
        // ========================

        /**
         * @brief Number of rows in the matrix
         *
         * @return
         */
        [[nodiscard]] size_t numRows() const { return m; }

        [[nodiscard]] size_t size() const { return m; }

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

        template <int mM>
        void pushRow(const Row<number, mM>& v) {
            if (v.size() != n) {
                throw invalid_argument("Row size must match the number of columns in the matrix");
            }

            matrix.push_back(v);
            m++;
        }

        template <int nN>
        void pushCol(const Row<number, nN>& v) {
            if (v.size() != m) {
                throw invalid_argument("Column size must match the number of rows in the matrix");
            }

            for (size_t i{}; i < m; ++i) matrix[i].pushBack(v[i]);
            n++;
        }

        void resize(size_t newM, size_t newN) {
            matrix.resize(newM, Row<number, Dynamic>(newN));
            for (size_t i{}; i < newM; ++i) {
                matrix[i].resize(newN);
            }
            m = newM;
            n = newN;
        }

        void reserve(size_t newM, size_t newN) {
            matrix.reserve(newM);
            for (size_t i{}; i < newM; ++i) {
                matrix[i].reserve(newN);
            }
        }

        Row<number, Dynamic> removeRow(size_t idx) {
            if (idx >= m) throw StackError<std::out_of_range>("Index out of range");
            Row<number, Dynamic> row = matrix[idx];
            matrix.erase(matrix.begin() + idx);
            --m;
            return row;
        }

        Row<number, Dynamic> removeCol(size_t idx) {
            if (idx >= n) throw StackError<std::out_of_range>("Index out of range");
            Row<number, Dynamic> col(m);
            for (size_t i{}; i < m; ++i) {
                col[i] = matrix[i].remove(idx);
            }
            --n;
            return col;
        }

       private:
        template <Number num, int mM, int nN>
        friend class Matrix;

        template <Number num, int nN>
        friend class Vector;

        template <typename D, Number num>
        friend class MatrixBase;

        size_t m;
        size_t n;
        TDArrayDynamic<number> matrix;

        MatrixColumns columns{this};
    };

}  // namespace mlinalg::structures
