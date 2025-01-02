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

#include "../Concepts.hpp"
#include "../Helpers.hpp"
#include "Vector.hpp"

using std::vector, std::array, std::optional, std::unique_ptr, std::shared_ptr;
namespace rg = std::ranges;

using SizePair = std::pair<int, int>;

namespace mlinalg::structures {
    template <Number number, int m, int n>
    class Matrix;

    /**
     * @brief Type alias for a variant of a Vector and a Matrix
     *
     * This is used to represent the result of a matrix or vector transposition. As the transpose of a vector is a 1xM
     * matrix and the transpose of a matrix is an NxM matrix, this variant is used to represent either of these
     */
    template <Number number, int m, int n>
    using TransposeVariant = std::variant<Vector<number, m>, Matrix<number, n, m>>;

    /**
     * @brief Type alias for a Vector as a row in a Matrix
     */
    template <Number number, int n>
    using Row = Vector<number, n>;

    template <Number number>
    using RowDynamic = Vector<number, -1>;

    namespace {
        template <Container T, Container U>
        void checkMatrixOperandRowSize(const T& matrix, const U& otherMatrix) {
            if (matrix.size() != otherMatrix.size())
                throw std::invalid_argument("Matrices must be of the same dimensions");
        }

        template <Container T, Container U>
        void checkMatrixOperandSize(const T& matrix, const U& otherMatrix) {
            if (matrix.size() != otherMatrix.size())
                throw std::invalid_argument("Matrices must be of the same dimensions");

            if (matrix.at(0).size() != otherMatrix.at(0).size())
                throw std::invalid_argument("Matrices must be of the same dimensions");
        }

        template <Container T>
        auto& matrixRowAt(T& matrix, int i) {
            return matrix.at(i);
        }

        template <Container T>
        auto matrixRowAtConst(const T& matrix, int i) {
            return matrix.at(i);
        }

        template <Number number, Container T>
        number& matrixAt(T& matrix, int i, int j) {
            return matrix.at(i).at(j);
        }

        template <Number number, int m, int n, Container T>
        vector<Vector<number, m>> matrixColsToVectorSet(const T& matrix) {
            // TODO: Fix this not actually checking dynamic arrays when cofactoring
            if constexpr (m == -1 || n == -1) {
                const auto& nRows = matrix.size();
                const auto& nCols = matrix.at(0).size();
                vector<RowDynamic<number>> res;
                res.reserve(nCols);
                for (int i{}; i < nCols; i++) {
                    Vector<number, Dynamic> vec(nRows);
                    for (int j{}; j < nRows; j++) {
                        vec.at(j) = matrix.at(j).at(i);
                    }
                    res.emplace_back(std::move(vec));
                }
                return res;
            } else {
                vector<Vector<number, m>> res;
                res.reserve(n);
                for (int i{}; i < n; i++) {
                    Vector<number, m> vec;
                    for (int j{}; j < m; j++) {
                        vec.at(j) = matrix.at(j).at(i);
                    }
                    res.emplace_back(std::move(vec));
                }
                return res;
            }
        }

        template <Number number, int m, int n, Container T>
        vector<Vector<number, m>> matrixRowsToVectorSet(const T& matrix) {
            constexpr int vSize = (m == -1 || n == -1) ? Dynamic : n;
            const auto& nRows = matrix.size();
            vector<Vector<number, vSize>> res;
            res.reserve(nRows);
            for (const auto& row : matrix) res.emplace_back(row);
            return res;
        }

        template <Number number, int m, int n, Container T>
        Matrix<number, m, n> matrixScalarMult(const T& matrix, const number& scalar) {
            const auto& nRows = matrix.size();
            const auto& nCols = matrix.at(0).size();
            Matrix<number, m, n> res(nRows, nCols);
            auto asRowVectorSet{std::move(matrixRowsToVectorSet<number, m, n>(matrix))};
            for (int i{}; i < nRows; i++) {
                res.at(i) = asRowVectorSet.at(i) * scalar;
            }
            return res;
        }

        template <Number number, int m, int n, Container T>
        Matrix<number, m, n> matrixScalarDiv(const T& matrix, const number& scalar) {
            const auto& nRows = matrix.size();
            const auto& nCols = matrix.at(0).size();
            Matrix<number, m, n> res(nRows, nCols);
            auto asRowVectorSet{std::move(matrixRowsToVectorSet<number, m, n>(matrix))};
            for (int i{}; i < nRows; i++) {
                res.at(i) = asRowVectorSet.at(i) / scalar;
            }
            return res;
        }

        template <Number number, int m, int n, Container T, Container U>
        Matrix<number, m, n> matrixAdd(const T& matrix, const U& otherMatrix) {
            checkMatrixOperandSize(matrix, otherMatrix);
            const auto& nRows = matrix.size();
            const auto& nCols = matrix.at(0).size();
            Matrix<number, m, n> res(nRows, nCols);
            for (int i{}; i < nRows; i++) res.at(i) = matrix.at(i) + otherMatrix.at(i);
            return res;
        }

        template <Number number, int m, int n, Container T, Container U>
        Matrix<number, m, n> matrixSub(const T& matrix, const U& otherMatrix) {
            checkMatrixOperandSize(matrix, otherMatrix);
            const auto& nRows = matrix.size();
            const auto& nCols = matrix.at(0).size();
            Matrix<number, m, n> res(nRows, nCols);
            for (int i{}; i < nRows; i++) res.at(i) = matrix.at(i) - otherMatrix.at(i);
            return res;
        }

        template <Container T>
        std::string matrixStringRepr(const T& matrix) {
            std::stringstream ss{};
            const auto& nRows = matrix.size();

            size_t maxWidth = 0;
            for (const auto& row : matrix) {
                for (const auto& elem : row) {
                    std::stringstream temp_ss;
                    temp_ss << elem;
                    maxWidth = std::max(maxWidth, temp_ss.str().length());
                }
            }

            if (nRows == 1) {
                ss << "[ ";
                for (const auto& elem : matrix.at(0)) ss << " " << std::setw(maxWidth) << elem << " ";
                ss << "]\n";
            } else {
                int i{};
                for (const auto& row : matrix) {
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

        /**
         * @brief Matrix multiplication by a vector implementation function
         *
         * @param vec The vector to multiply by
         * @return The vector resulting from the multiplication
         */
        template <Number number, int m, int n, int mRes = m, Container T>
        Vector<number, mRes> multMatByVec(const T& matrix, const Vector<number, n>& vec) {
            if (matrix.at(0).size() != vec.size())
                throw std::invalid_argument("The columns of the matrix must be equal to the size of the vector");

            constexpr int vSize = (m == -1 || n == -1) ? Dynamic : m;
            const auto& nRows = matrix.size();
            const auto& nCols = matrix.at(0).size();
            Vector<number, vSize> res(nRows);
            auto asCols{std::move(matrixColsToVectorSet<number, m, n>(matrix))};
            int i{};
            for (auto& col : asCols) {
                const auto& mult = vec.at(i);
                col *= mult;
                i++;
            }

            for (int i{}; i < nRows; i++) {
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
        template <Number number, int m, int n, int nOther, Container T, Container U>
        Matrix<number, m, nOther> multMatByDef(const T& matrix, const U& otherMatrix) {
            if (matrix.at(0).size() != otherMatrix.size())
                throw std::invalid_argument(
                    "The columns of the first matrix must be equal to the rows of the second matrix");

            constexpr auto isDynamic = m == -1 || n == -1;
            constexpr auto DynamicPair = SizePair{Dynamic, Dynamic};

            constexpr int vSize = isDynamic ? Dynamic : m;
            constexpr auto resSizeP = isDynamic ? DynamicPair : SizePair{m, nOther};
            constexpr auto sizeP = isDynamic ? DynamicPair : SizePair{m, n};

            auto otherColVecSet{std::move(matrixColsToVectorSet<number, m, n>(otherMatrix))};
            vector<Vector<number, vSize>> res;
            const auto& nOtherCols = otherMatrix.at(0).size();
            res.reserve(nOtherCols);
            for (const auto& col : otherColVecSet) {
                auto multRes{multMatByVec<number, sizeP.first, sizeP.second>(matrix, col)};
                res.emplace_back(multRes);
            }
            return helpers::fromColVectorSet<number, resSizeP.first, resSizeP.second>(res);
        }

        /**
         * @brief Transpose a mxn matrix to a nxm matrix
         *
         * @return The transposed matrix of size nxm
         */
        template <Number number, int m, int n, Container T>
        TransposeVariant<number, m, n> TransposeMatrix(const T& matrix) {
            constexpr auto isDynamic = m == -1 || n == -1;
            constexpr auto DynamicPair = SizePair{Dynamic, Dynamic};

            constexpr int vSize = isDynamic ? Dynamic : m;
            constexpr auto sizeP = isDynamic ? DynamicPair : SizePair{m, n};

            const auto& nRows = matrix.size();
            const auto& nCols = matrix.at(0).size();

            auto mutateMatrix = [&matrix, &nRows, &nCols, &sizeP](auto& variant) {
                if constexpr (std::is_same_v<std::decay_t<decltype(variant)>,
                                             Matrix<number, sizeP.first, sizeP.second>>) {
                    for (int i{}; i < nRows; i++)
                        for (int j{}; j < nCols; j++) variant.at(j).at(i) = matrix.at(i).at(j);
                }
            };

            auto mutateVector = [&matrix, &nRows, &nCols, &vSize](auto& variant) {
                if constexpr (std::is_same_v<std::decay_t<decltype(variant)>, Vector<number, vSize>>) {
                    for (int i{}; i < nRows; i++) variant.at(i) = matrix.at(i).at(0);
                }
            };

            TransposeVariant<number, sizeP.first, sizeP.second> res(
                std::in_place_index<1>, Matrix<number, sizeP.first, sizeP.second>(nCols, nRows));
            if (nCols != 1) {
                res = Matrix<number, sizeP.second, sizeP.first>(nCols, nRows);
                std::visit(mutateMatrix, res);
                return res;
            }

            res = Vector<number, vSize>(nRows);
            std::visit(mutateVector, res);
            return res;
        }

        /**
         * @brief Augment the matrix with another matrix
         *
         * @param other The matrix to augment with
         * @return The augmented matrix of size mx(n + nN)
         */
        template <Number number, int m, int n, int nN, Container T, Container U>
        Matrix<number, m, nN + n> MatrixAugmentMatrix(const T& matrix, const U& otherMatrix) {
            checkMatrixOperandRowSize(matrix, otherMatrix);
            constexpr auto isDynamic = m == -1 || n == -1;
            constexpr auto DynamicPair = SizePair{Dynamic, Dynamic};

            constexpr int vSize = isDynamic ? Dynamic : m;
            constexpr auto sizeP = isDynamic ? DynamicPair : SizePair{m, n + nN};

            const auto& nRows = matrix.size();
            const auto& nCols = matrix.at(0).size();
            const auto& nOtherCols = otherMatrix.at(0).size();
            Matrix<number, sizeP.first, sizeP.second> res(nRows, nCols + nOtherCols);
            for (int i{}; i < nRows; i++) {
                auto& row{res.at(i)};
                const auto& thisRow{matrix.at(i)};
                const auto& otherRow{otherMatrix.at(i)};
                for (int j{}; j < (nCols + nOtherCols); j++) {
                    if (j < nCols)
                        row.at(j) = thisRow.at(j);
                    else
                        row.at(j) = otherRow.at((j - static_cast<int>(nOtherCols)));
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
        template <Number number, int m, int n, Container MatrixContainer, Container VectorContainer>
        Matrix<number, m, n + 1> MatrixAugmentVector(const MatrixContainer& matrix, const VectorContainer& vec) {
            constexpr auto isDynamic = m == -1 || n == -1;
            constexpr auto DynamicPair = SizePair{Dynamic, Dynamic};

            constexpr auto sizeP = isDynamic ? DynamicPair : SizePair{m, n + 1};

            const auto& nRows = matrix.size();
            const auto& nCols = matrix.at(0).size();
            const auto& nSize = vec.size();
            Matrix<number, sizeP.first, sizeP.second> res(nRows, nCols + 1);
            for (int i{}; i < nRows; i++) {
                auto& row{res.at(i)};
                const auto& thisRow{matrix.at(i)};
                int j{};
                for (; j < nCols; j++) {
                    row.at(j) = thisRow.at(j);
                }
                row.at(j) = vec.at(i);
            }
            return res;
        }

        /**
         * @brief Subset the matrix by removing a row and a column
         *
         * @param i Row index to remove
         * @param j Column index to remove
         * @return The subsetted matrix of size (m - 1)x(n - 1)
         */
        template <Number number, int m, int n, Container T>
        // TODO: Find a way to account for negatives and do nothing when they are found
        Matrix<number, m - 1, n - 1> MatrixSubset(const T& matrix, const std::optional<int>& i,
                                                  const std::optional<int> j) {
            const auto& nRows = matrix.size();
            const auto& nCols = matrix.at(0).size();
            if (nRows != nCols) throw std::runtime_error("Matrix must be square to find a subset");
            if (nRows <= 1 || nCols <= 1) throw std::runtime_error("Matrix must be at least 2x2 to find a subset");

            constexpr auto isDynamic = m == 0 || n == 0;
            constexpr auto DynamicPair = SizePair{Dynamic, Dynamic};

            constexpr auto sizeP = isDynamic ? DynamicPair : SizePair{m - 1, n - 1};

            Matrix<number, sizeP.first, sizeP.second> res(nRows - 1, nCols - 1);
            int resRow = 0;
            for (int k = 0; k < nRows; ++k) {
                if (i.has_value() && i.value() == k) continue;  // Skip the row to be removed

                int resCol = 0;
                for (int z = 0; z < nCols; ++z) {
                    if (j.has_value() && j.value() == z) continue;  // Skip the column to be removed

                    res.at(resRow, resCol) = matrix.at(k).at(z);
                    ++resCol;
                }
                ++resRow;
            }
            return res;
        }

        /**
         * @brief Calculate the determinant of a 2x2 matrix
         *
         * @return The determinant of the matrix
         */
        template <Number number, Container T>
        number MatrixDet2x2(const T& matrix) {
            return (matrix.at(0).at(0) * matrix.at(1).at(1)) - (matrix.at(0).at(1) * matrix.at(1).at(0));
        }

        enum By { ROW = 0, COL };

        /**
         * @brief Pick the row or column with the most zeros as a cofactor row or column
         */
        template <Number number, int m, int n, Container T>
        std::pair<By, int> pickCofactorRowOrCol(const T& matrix) {
            int maxRowZeros{};
            int rowPos{};

            int maxColZeros{};
            int colPos{};

            int pos{};
            for (const auto& row : matrix) {
                auto count = rg::count_if(row, [](auto x) { return x == 0; });
                if (count > maxRowZeros) {
                    maxRowZeros = count;
                    rowPos = pos;
                }
                pos++;
            }

            pos = 0;
            for (const auto& col : matrixColsToVectorSet<number, m, n>(matrix)) {
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
        template <Number number, int m, int n, Container T>
        number cofactorCommon(const T& matrix, int i, int j) {
            number res{};
            auto a = matrix.at(i).at(j);
            if (a == 0) return 0;
            auto A_ij = MatrixSubset<number, m, n>(matrix, i, j);
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
        template <Number number, int m, int n, Container T>
        number cofactorRow(const T& matrix, int row) {
            number res{};
            const auto& nCols = matrix.at(0).size();
            int i{row};
            for (int j{0}; j < nCols; j++) {
                res += cofactorCommon<number, m, n>(matrix, i, j);
            }
            return res;
        }

        /**
         * @brief Calculate the cofactor for a column
         *
         * @param col The column index
         * @return The cofactor for the column
         */
        template <Number number, int m, int n, Container T>
        number cofactorCol(const T& matrix, int col) {
            number res{};
            const auto& nCols = matrix.at(0).size();
            int j{col};
            for (int i{0}; i < nCols; i++) {
                res += cofactorCommon<number, m, n>(matrix, i, j);
            }
            return res;
        }

        /**
         * @brief Main cofactor calculation function
         *
         * @return The determinant of the matrix
         */
        template <Number number, int m, int n, Container T>
        number MatrixCofactor(const T& matrix) {
            auto [by, val] = pickCofactorRowOrCol<number, m, n>(matrix);
            if (by == ROW)
                return cofactorRow<number, m, n>(matrix, val);
            else
                return cofactorCol<number, m, n>(matrix, val);
        }

    }  // namespace

    /**
     * @brief Matrix class for representing NxM matrices
     *
     * @param m Number of rows
     * @param n Number of columns
     */
    template <Number number, int m, int n>
    class Matrix {
       public:
        /**
         * @brief Access the ith row of the matrix
         *
         * @param i The index of the row to access
         * @return A reference to the ith row
         */
        Row<number, n>& at(int i) { return matrixRowAt(matrix, i); }

        /**
         * @brief Const access the ith row of the matrix
         *
         * @param i The index of the row to access
         * @return A const reference to the ith row
         */
        Row<number, n> at(int i) const { return matrixRowAtConst(matrix, i); }

        /**
         * @brief Access the element at the ith row and jth column
         *
         * @param i The index of the row
         * @param j The index of the column
         * @return A reference to the element at the ith row and jth column
         */
        number& at(int i, int j) { return matrixAt<number>(matrix, i, j); }

        Matrix() = default;

        // Constructor to keep consistency with the Dynamic Matrix specialization to allow them to be used
        // interchangeably
        Matrix(int nRows, int nCols) {}

        /**
         * @brief Construct a new Matrix object from an initializer list of row vectors
         *
         * @param rows  Initializer list of row vectors
         */
        constexpr Matrix(const std::initializer_list<std::initializer_list<number>>& rows) {
            for (int i{}; i < m; i++) matrix.at(i) = Row<number, n>{*(rows.begin() + i)};
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
         * @brief Matrix division by a scalar
         *
         * @param scalar A scalar of the same type as the matrix
         * @return The matrix resulting from the division
         */
        Matrix operator/(const number& scalar) const { return matrixScalarDiv<number, m, n>(matrix, scalar); }

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
         * @brief Matrix subtraction
         *
         * @param other The matrix to subtract
         * @return The matrix resulting from the subtraction
         */
        Matrix operator-(const Matrix<number, m, n>& other) const {
            return matrixSub<number, m, n>(matrix, other.matrix);
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
        Vector<number, m> operator*(const Vector<number, n>& vec) const {
            return multMatByVec<number, m, n>(matrix, vec);
        }

        /**
         * @brief Matrix multiplication by a matrix
         *
         * @param other
         * @return
         */
        template <int nOther>
        Matrix<number, m, nOther> operator*(const Matrix<number, n, nOther>& other) const {
            return multMatByDef<number, m, n, nOther>(matrix, other.matrix);
        }

        /**
         * @brief Matrix multiplication by a transposed matrix
         *
         * @param other The transposed matrix to multiply by
         * @return The matrix resulting from the multiplication
         */
        template <int nOther>
        Matrix<number, m, nOther> operator*(const TransposeVariant<number, n, nOther>& other) const {
            return multMatByDef<number, m, n, nOther>(matrix, helpers::extractMatrixFromTranspose(other).matrix);
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
        [[nodiscard]] int numRows() const { return m; }

        /**
         * @brief Number of columns in the matrix
         *
         * @return
         */
        [[nodiscard]] int numCols() const { return n; }

        explicit operator std::string() const { return matrixStringRepr(matrix); }

        friend std::ostream& operator<<(std::ostream& os, const Matrix<number, m, n>& system) {
            os << std::string(system);
            return os;
        }

        /**
         * @brief Transpose a mxn matrix to a nxm matrix
         *
         * @return The transposed matrix of size nxm
         */
        TransposeVariant<number, m, n> T() const { return TransposeMatrix<number, m, n>(matrix); }

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
            if (m != n) throw std::runtime_error("Finding determinant of rectangular matrices is not defined");
            if constexpr (m == 2 && n == 2)
                return MatrixDet2x2<number>(matrix);
            else
                // TODO: Find a fix for this that allows dynamic matrices to the subset without m and n going to -2
                return MatrixCofactor<number, 0, 0>(matrix);
        }

        /**
         * @brief Subset the matrix by removing a row and a column
         *
         * @param i Row index to remove
         * @param j Column index to remove
         * @return The subsetted matrix of size (m - 1)x(n - 1)
         */
        Matrix<number, m - 1, n - 1> subset(std::optional<int> i, std::optional<int> j) const {
            return MatrixSubset<number, m, n>(matrix, i, j);
        }

       private:
        template <Number num, int nN>
        friend class Vector;

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
            if constexpr (n <= 0) throw std::invalid_argument("Matrix n must be greater than zero");
            if constexpr (m <= 0) throw std::invalid_argument("Matrix m must be greater than zero");
        }

        /**
         * @brief Backing array for the matrix
         */
        array<Row<number, n>, m> matrix{};
    };

    // Vector(std::array<number, n>) -> Vector<number, n>;

    // template <Number number, int m, int n, int nOther>
    // Matrix<number, m, nOther> operator*(Matrix<number, m, n> lhs, Matrix<number, n, nOther> rhs) {
    //     return lhs * rhs;
    // }

    template <Number number, int m, int n, int nOther>
    TransposeVariant<number, m, nOther> operator*(TransposeVariant<number, m, n> lhs, Matrix<number, n, nOther> rhs) {
        if (std::holds_alternative<Vector<number, m>>(lhs)) {
            auto vec = std::get<Vector<number, m>>(lhs);
            return vec * rhs;
        } else {
            auto mat = std::get<Matrix<number, m, n>>(lhs);
            return mat * rhs;
        }
    }

    template <Number number, int m, int n>
    Matrix<number, m, n> operator*(const number& scalar, Matrix<number, m, n> rhs) {
        return rhs * scalar;
    }

    template <Number number, int m, int n>
    Matrix<number, m, n> operator*(Vector<number, n> vec, Matrix<number, m, n> rhs) {
        Matrix<number, m, n> res;
        for (int i{}; i < n; i++) {
            const auto& mult = vec.at(i);
            for (int j{}; j < m; j++) res.at(j).at(i) = mult * rhs.matrix.at(j).at(i);
        }
        return res;
    }

}  // namespace mlinalg::structures

namespace mlinalg::structures {
    template <Number number>
    class Matrix<number, -1, -1> {
       public:
        Matrix(int m, int n) : m(m), n(n) {
            if (m <= 0) throw std::invalid_argument("Matrix must have at least one row");
            if (n <= 0) throw std::invalid_argument("Matrix must have at least one column");
            matrix.reserve(m);
            matrix.resize(m, Vector<number, Dynamic>(n));
        }

        Matrix(const std::initializer_list<std::initializer_list<number>>& rows)
            : m{rows.size()}, n{rows.begin()->size()} {
            if (m <= 0) throw std::invalid_argument("Matrix must have at least one row");
            if (n <= 0) throw std::invalid_argument("Matrix must have at least one column");

            for (const auto& row : rows) {
                if (row.size() != n) throw std::invalid_argument("All rows must have the same number of columns");
            }

            matrix.reserve(m);
            for (const auto& row : rows) {
                matrix.emplace_back(row);
            }
        }

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
         * @brief Access the element at the ith row and jth column
         *
         * @param i The index of the row
         * @param j The index of the column
         * @return A reference to the element at the ith row and jth column
         */
        number& at(int i, int j) { return matrixAt<number>(matrix, i, j); }

        /**
         * @brief Copy construct a new Matrix object
         *
         * @param other Matrix to copy
         */
        Matrix(const Matrix& other) : matrix{other.matrix}, m{other.m}, n{other.n} {}

        /**
         * @brief Move construct a new Matrix object
         *
         * @param other  Matrix to move
         */
        Matrix(Matrix&& other) noexcept
            : matrix{std::move(other.matrix)}, m{std::move(other.m)}, n{std::move(other.n)} {}

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
            m = std::move(other.m);
            n = std::move(other.n);
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
         * @brief Matrix division by a scalar
         *
         * @param scalar A scalar of the same type as the matrix
         * @return The matrix resulting from the division
         */
        Matrix<number, Dynamic, Dynamic> operator/(const number& scalar) const {
            return matrixScalarDiv<number, Dynamic, Dynamic>(matrix, scalar);
        }

        /**
         * @brief Matrix addition
         *
         * @param other The matrix to add
         * @return The matrix resulting from the addition
         */
        Matrix<number, Dynamic, Dynamic> operator+(const Matrix<number, Dynamic, Dynamic>& other) const {
            return matrixAdd<number, Dynamic, Dynamic>(matrix, other.matrix);
        }

        /**
         * @brief Matrix subtraction
         *
         * @param other The matrix to subtract
         * @return The matrix resulting from the subtraction
         */
        Matrix<number, Dynamic, Dynamic> operator-(const Matrix<number, Dynamic, Dynamic>& other) const {
            return matrixSub<number, Dynamic, Dynamic>(matrix, other.matrix);
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
        Vector<number, Dynamic> operator*(const Vector<number, Dynamic>& vec) const {
            return multMatByVec<number, Dynamic, Dynamic>(matrix, vec);
        }

        /**
         * @brief Matrix multiplication by a matrix
         *
         * @param other
         * @return
         */
        Matrix<number, Dynamic, Dynamic> operator*(const Matrix<number, Dynamic, Dynamic>& other) const {
            return multMatByDef<number, Dynamic, Dynamic, Dynamic>(matrix, other.matrix);
        }

        /**
         * @brief Matrix multiplication by a transposed matrix
         *
         * @param other The transposed matrix to multiply by
         * @return The matrix resulting from the multiplication
         */
        template <int nOther>
        Matrix<number, Dynamic, Dynamic> operator*(const TransposeVariant<number, Dynamic, Dynamic>& other) const {
            return multMatByDef<number, Dynamic, Dynamic, Dynamic>(helpers::extractMatrixFromTranspose(other));
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
        [[nodiscard]] int numRows() const { return m; }

        /**
         * @brief Number of columns in the matrix
         *
         * @return
         */
        [[nodiscard]] int numCols() const { return n; }

        explicit operator std::string() const { return matrixStringRepr(matrix); }

        friend std::ostream& operator<<(std::ostream& os, const Matrix<number, Dynamic, Dynamic>& system) {
            os << std::string(system);
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
        Matrix<number, Dynamic, Dynamic> subset(std::optional<int> i, std::optional<int> j) const {
            return MatrixSubset<number, 0, 0>(matrix, i, j);
        }

        /**
         * @brief Determinant of the matrix
         *
         * @return The determinant of the matrix
         */
        number det() const {
            if (m != n) throw std::runtime_error("Finding determinant of rectangular matrices is not defined");
            if (m == 2 && n == 2)
                return MatrixDet2x2<number>(matrix);
            else
                return MatrixCofactor<number, 0, 0>(matrix);
        }

       private:
        template <Number num, int nN>
        friend class Vector;

        size_t m;
        size_t n;
        vector<RowDynamic<number>> matrix;
    };

    // template <Number number>
    // Matrix<number, Dynamic, Dynamic> operator*(const Matrix<number, Dynamic, Dynamic>& lhs,
    //                                            const Matrix<number, Dynamic, Dynamic>& rhs) {
    //     return lhs * rhs;
    // }
}  // namespace mlinalg::structures
