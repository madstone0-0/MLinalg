/**
 * @file Matrix.hpp
 * @brief Header file for the Matrix class
 */

#pragma once

#include <cassert>
#include <cstddef>
#ifdef DEBUG
#include "../Logging.hpp"
#endif

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <memory>
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
#include "../Numeric.hpp"
#include "Vector.hpp"

using std::vector, std::array, std::optional, std::unique_ptr, std::shared_ptr;
namespace rg = std::ranges;

using SizePair = std::pair<int, int>;

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

    using SlicePair = std::pair<size_t, size_t>;

    template <Number number, int m, int n>
    class Matrix;

    template <Number number, int m, int n>
    struct MatrixView;

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

    template <Number number, int m, int n>
    using TDArray = array<Row<number, n>, m>;

    template <Number number>
    using TDArrayDynamic = vector<RowDynamic<number>>;

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
        auto& matrixRowAt(T& matrix, size_t i) {
            return matrix.at(i);
        }

        template <Container T>
        auto matrixRowAtConst(const T& matrix, size_t i) {
            return matrix.at(i);
        }

        template <Number number, Container T>
        number& matrixAt(T& matrix, size_t i, size_t j) {
            return matrix.at(i).at(j);
        }

        template <Number number, Container T>
        number matrixAtConst(const T& matrix, size_t i, size_t j) {
            return matrix.at(i).at(j);
        }

        template <Number number, int m, int n, Container T>
        vector<Vector<number, m>> matrixColsToVectorSet(const T& matrix) {
            // TODO: Fix this not actually checking dynamic arrays when cofactoring
            if constexpr (m == -1 || n == -1) {
                const size_t& nRows = matrix.size();
                const size_t& nCols = matrix.at(0).size();
                vector<RowDynamic<number>> res;
                res.reserve(nCols);
                for (size_t i{}; i < nCols; i++) {
                    Vector<number, Dynamic> vec(nRows);
                    for (size_t j{}; j < nRows; j++) {
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
        vector<Vector<number, n>> matrixRowsToVectorSet(const T& matrix) {
            constexpr int vSize = (m == -1 || n == -1) ? Dynamic : n;
            const auto& nRows = matrix.size();
            vector<Vector<number, vSize>> res;
            res.reserve(nRows);
            for (const auto& row : matrix) res.emplace_back(row);
            return res;
        }

        template <Number number, Container T, Container U>
        bool matrixEqual(const T& matrix, const U& otherMatrix) {
            const auto& nRows{matrix.size()};
            const auto& nCols{matrix.at(0).size()};
            const auto& nRowsOther{otherMatrix.size()};
            const auto& nColsOther{otherMatrix.at(0).size()};
            if (nRows != nRowsOther || nCols != nColsOther) return false;
            for (size_t i{}; i < nRows; i++)
                for (size_t j{}; j < nCols; j++) {
                    if (!fuzzyCompare(matrix.at(i).at(j), otherMatrix.at(i).at(j))) return false;
                }
            return true;
        }

        template <Number number, int m, int n, Container T>
        Matrix<number, m, n> matrixScalarMult(const T& matrix, const number& scalar) {
            const auto& nRows = matrix.size();
            const auto& nCols = matrix.at(0).size();
            Matrix<number, m, n> res(nRows, nCols);
            auto asRowVectorSet{std::move(matrixRowsToVectorSet<number, m, n>(matrix))};
            for (size_t i{}; i < nRows; i++) {
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
            for (size_t i{}; i < nRows; i++) {
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
            for (size_t i{}; i < nRows; i++) res.at(i) = matrix.at(i) + otherMatrix.at(i);
            return res;
        }

        template <Number number, int m, int n, Container T, Container U>
        Matrix<number, m, n> matrixSub(const T& matrix, const U& otherMatrix) {
            checkMatrixOperandSize(matrix, otherMatrix);
            const auto& nRows = matrix.size();
            const auto& nCols = matrix.at(0).size();
            Matrix<number, m, n> res(nRows, nCols);
            for (size_t i{}; i < nRows; i++) res.at(i) = matrix.at(i) - otherMatrix.at(i);
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
                for (const auto& elem : matrix.at(0)) ss << " " << std::setw(static_cast<int>(maxWidth)) << elem << " ";
                ss << "]\n";
            } else {
                int i{};
                for (const auto& row : matrix) {
                    if (i == 0) {
                        ss << "⎡";
                        for (const auto& elem : row) ss << " " << std::setw(static_cast<int>(maxWidth)) << elem << " ";
                        ss << "⎤\n";
                    } else if (i == static_cast<int>(nRows) - 1) {
                        ss << "⎣";
                        for (const auto& elem : row) ss << " " << std::setw(static_cast<int>(maxWidth)) << elem << " ";
                        ss << "⎦\n";
                    } else {
                        ss << "|";
                        for (const auto& elem : row) ss << " " << std::setw(static_cast<int>(maxWidth)) << elem << " ";
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

            Vector<number, vSize> res(nRows);
            auto asCols{std::move(matrixColsToVectorSet<number, m, n>(matrix))};
            int i{};
            for (auto& col : asCols) {
                const auto& mult = vec.at(i);
                col *= mult;
                i++;
            }

            for (size_t i{}; i < nRows; i++) {
                number sumRes{};
                for (const auto& col : asCols) sumRes += col.at(i);
                res.at(i) = sumRes;
            }

            return res;
        }

        /**
         * @brief Matrix multiplication by a matrix by the definition of matrix multiplication
         *
         * @param matrix The matrix to multiply
         * @param other The matrix to multiply by
         * @return The matrix resulting from the multiplication
         */
        template <Number number, int m, int n, int mOther, int nOther, Container T, Container U>
        Matrix<number, m, nOther> multMatByDef(const T& matrix, const U& otherMatrix) {
            if (matrix.at(0).size() != static_cast<size_t>(otherMatrix.size()))
                throw std::invalid_argument(
                    "The columns of the first matrix must be equal to the rows of the second matrix");

            constexpr bool isDynamic = m == Dynamic || n == Dynamic || mOther == Dynamic || nOther == Dynamic;
            constexpr auto DynamicPair = SizePair{Dynamic, Dynamic};

            constexpr int vSize = isDynamic ? Dynamic : m;
            constexpr auto resSizeP = isDynamic ? DynamicPair : SizePair{m, nOther};
            constexpr auto sizeP = isDynamic ? DynamicPair : SizePair{m, n};

            auto otherColVecSet{std::move(matrixColsToVectorSet<number, mOther, nOther>(otherMatrix))};
            vector<Vector<number, vSize>> res;
            const auto& nOtherCols = otherMatrix.at(0).size();
            res.reserve(nOtherCols);
            for (const auto& col : otherColVecSet) {
                auto multRes{multMatByVec<number, sizeP.first, sizeP.second>(matrix, col)};
                res.emplace_back(multRes);
            }

            return helpers::fromColVectorSet<number, resSizeP.first, resSizeP.second>(res);
        }


        template <Number number, int m, int n, size_t i, size_t j, Container T>
        MatrixView<number, m, n> View(T& matrix, size_t rowOffset = 0, size_t colOffset = 0, size_t rowStride = 1,
                                      size_t colStride = 1) {
            const auto rows = matrix.size();
            const auto cols = matrix.at(0).size();

            if (i > rows || j > cols) throw std::out_of_range("View dimensions exceed matrix size");
            if (rowOffset >= rows || colOffset >= cols) throw std::out_of_range("Offset out of range");
            return MatrixView<number, m, n>{&matrix, rowOffset, colOffset, rowStride, colStride};
        }

        template <Number number, int m, int n, Container T>
        Matrix<number, m - 1, n - 1> MatrixSubset(const T& matrix, const std::optional<int>& i,
                                                  const std::optional<int> j);

        template <size_t i0, size_t i1, size_t j0, size_t j1, Number number, int m, int n, Container T>
        Matrix<number, (i1 - i0), (j1 - j0)> MatrixSlice(const T& matrix) {
            if constexpr (rg::any_of(array<size_t, 4>{i0, i1, j0, j1}, [](auto x) { return x < 0; }))
                throw std::invalid_argument("Negative slicing not supported");

            if constexpr (rg::any_of(array<size_t, 2>{i1, j1}, [](auto x) { return x > n; }))
                throw std::invalid_argument("Cannot slice past matrix bounds");

            if constexpr (i0 > i1 || j0 > j1)
                throw std::invalid_argument("Start position cannot be greater than end position");
            constexpr size_t mSize{i1 - i0};
            constexpr size_t nSize{j1 - j0};
            Matrix<number, mSize, nSize> res{};

            auto isInRange = [](int x0, int x1, int y) { return y >= x0 && y < x1; };

            size_t insJ{};
            for (int i{}; i < m; i++) {
                if (!isInRange(i0, i1, i)) continue;
                insJ = 0;
                for (int j{}; j < n; j++) {
                    if (!isInRange(j0, j1, j)) continue;
                    res.at(i - i0).at(insJ) = matrix.at(i).at(j);
                    insJ++;
                }
            }
            return res;
        }

        template <Number number, Container T>
        Matrix<number, Dynamic, Dynamic> MatrixSlice(const T& matrix, const SlicePair& i, const SlicePair& j) {
            const auto& [i0, i1] = i;
            const auto& [j0, j1] = j;
            const auto& m = matrix.size();
            const auto& n = matrix.at(0).size();

            if (rg::any_of(array<size_t, 4>{i0, i1, j0, j1}, [](auto x) { return x < 0; }))
                throw std::invalid_argument("Negative slicing not supported");

            if (i1 > m || j1 > n) throw std::invalid_argument("Cannot slice past matrix bounds");

            if (i0 > i1 || j0 > j1) throw std::invalid_argument("Start position cannot be greater than end position");
            Matrix<number, Dynamic, Dynamic> res(i1 - i0, j1 - j0);

            auto isInRange = [](int x0, int x1, int y) { return y >= x0 && y < x1; };

            size_t insJ{};
            for (size_t i{}; i < m; i++) {
                if (!isInRange(i0, i1, i)) continue;
                insJ = 0;
                for (size_t j{}; j < n; j++) {
                    if (!isInRange(j0, j1, j)) continue;
                    res.at(i - i0).at(insJ) = matrix.at(i).at(j);
                    insJ++;
                }
            }
            return res;
        }

        template <Number number, int m, int n, int nOther, Container T, Container U>
        auto strassen(const T& A, const U& B) {
            if constexpr (n == 2 && m == 2 && nOther == 2) {
                auto m1 = [](const T& A, const U& B) {
                    return (A.at(0).at(0) + A.at(1).at(1)) * (B.at(0).at(0) + B.at(1).at(1));
                };
                auto m2 = [](const T& A, const U& B) {  //
                    return (A.at(1).at(0) + A.at(1).at(1)) * B.at(0).at(0);
                };
                auto m3 = [](const T& A, const U& B) {  //
                    return A.at(0).at(0) * (B.at(0).at(1) - B.at(1).at(1));
                };
                auto m4 = [](const T& A, const U& B) {  //
                    return A.at(1).at(1) * (B.at(1).at(0) - B.at(0).at(0));
                };
                auto m5 = [](const T& A, const U& B) {  //
                    return (A.at(0).at(0) + A.at(0).at(1)) * B.at(1).at(1);
                };
                auto m6 = [](const T& A, const U& B) {  //
                    return (A.at(1).at(0) - A.at(0).at(0)) * (B.at(0).at(0) + B.at(0).at(1));
                };
                auto m7 = [](const T& A, const U& B) {  //
                    return (A.at(0).at(1) - A.at(1).at(1)) * (B.at(1).at(0) + B.at(1).at(1));
                };
                // Base case
                if (A.size() <= 2 && A[0].size() <= 2 && B.size() <= 2 && B[0].size() <= 2) {
                    auto M1 = m1(A, B);
                    auto M2 = m2(A, B);
                    auto M3 = m3(A, B);
                    auto M4 = m4(A, B);
                    auto M5 = m5(A, B);
                    auto M6 = m6(A, B);
                    auto M7 = m7(A, B);
                    return Matrix<number, 2, 2>{
                        {M1 + M4 - M5 + M7, M3 + M5},  //
                        {M2 + M4, M1 + M3 - M2 + M6}   //
                    };
                }
            }  //
            else {
                auto merge = []<int nSize>(const auto& A00, const auto& A01, const auto& A10, const auto& A11) {
                    Matrix<number, nSize, nSize> res{};
                    for (int i = 0; i < nSize / 2; i++) {
                        for (int j = 0; j < nSize / 2; j++) {
                            res.at(i, j) = A00.at(i, j);
                            res.at(i, j + (nSize / 2)) = A01.at(i, j);
                            res.at(i + (nSize / 2), j) = A10.at(i, j);
                            res.at(i + (nSize / 2), j + (nSize / 2)) = A11.at(i, j);
                        }
                    }
                    return res;
                };

                constexpr int halfM = m / 2;
                constexpr int halfN = n / 2;
                constexpr int halfNOther = nOther / 2;

                // Split matrix A into 4 submatrices
                auto A00 = MatrixSlice<0, halfM, 0, halfN, number, m, n>(A);
                auto A01 = MatrixSlice<0, halfM, halfN, n, number, m, n>(A);
                auto A10 = MatrixSlice<halfM, m, 0, halfN, number, m, n>(A);
                auto A11 = MatrixSlice<halfM, m, halfN, n, number, m, n>(A);

                // Split matrix B into 4 submatrices
                auto B00 = MatrixSlice<0, halfM, 0, halfNOther, number, m, nOther>(B);
                auto B01 = MatrixSlice<0, halfM, halfNOther, nOther, number, m, nOther>(B);
                auto B10 = MatrixSlice<halfM, m, 0, halfNOther, number, m, nOther>(B);
                auto B11 = MatrixSlice<halfM, m, halfNOther, nOther, number, m, nOther>(B);

                auto M1 = strassen<number, A00.numRows(), A00.numCols(), B00.numCols()>((A00 + A11).getMatrix(),
                                                                                        (B00 + B11).getMatrix());
                auto M2 = strassen<number, A10.numRows(), A10.numCols(), B00.numCols()>((A10 + A11).getMatrix(),
                                                                                        B00.getMatrix());
                auto M3 = strassen<number, A00.numRows(), A00.numCols(), B01.numCols()>(A00.getMatrix(),
                                                                                        (B01 - B11).getMatrix());
                auto M4 = strassen<number, A11.numRows(), A11.numCols(), B10.numCols()>(A11.getMatrix(),
                                                                                        (B10 - B00).getMatrix());
                auto M5 = strassen<number, A00.numRows(), A00.numCols(), B00.numCols()>((A00 + A01).getMatrix(),
                                                                                        B11.getMatrix());
                auto M6 = strassen<number, A10.numRows(), A10.numCols(), B00.numCols()>((A10 - A00).getMatrix(),
                                                                                        (B00 + B01).getMatrix());
                auto M7 = strassen<number, A01.numRows(), A01.numCols(), B10.numCols()>((A01 - A11).getMatrix(),
                                                                                        (B10 + B11).getMatrix());

                auto C00 = M1 + M4 - M5 + M7;
                auto C01 = M3 + M5;
                auto C10 = M2 + M4;
                auto C11 = M1 + M3 - M2 + M6;

                constexpr int nSize{A00.numRows() * 2};
                return merge.template operator()<nSize>(C00, C01, C10, C11);
            }
        }

        template <Number number, int m, int n, int nOther, Container T, Container U>
        Matrix<number, m, nOther> multMatStrassen(const T& matrix, const U& otherMatrix) {
            if (matrix.at(0).size() != otherMatrix.size())
                throw std::invalid_argument(
                    "The columns of the first matrix must be equal to the rows of the second matrix");

            return strassen<number, m, n, nOther>(matrix, otherMatrix);
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

            const size_t& nRows = matrix.size();
            const size_t& nCols = matrix.at(0).size();

            auto mutateMatrix = [&matrix, &nRows, &nCols, sizeP](auto& variant) {
                if constexpr (std::is_same_v<std::decay_t<decltype(variant)>,
                                             Matrix<number, sizeP.second, sizeP.first>>) {
                    for (size_t i{}; i < nRows; i++)
                        for (size_t j{}; j < nCols; j++) variant.at(j).at(i) = matrix.at(i).at(j);
                }
            };

            auto mutateVector = [&matrix, &nRows](auto& variant) {
                if constexpr (std::is_same_v<std::decay_t<decltype(variant)>, Vector<number, vSize>>) {
                    for (size_t i{}; i < nRows; i++) variant.at(i) = matrix.at(i).at(0);
                }
            };

            TransposeVariant<number, sizeP.first, sizeP.second> res(
                std::in_place_index<1>, Matrix<number, sizeP.second, sizeP.first>(nCols, nRows));
            if (nCols != 1) {
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

            constexpr auto sizeP = isDynamic ? DynamicPair : SizePair{m, n + nN};

            const size_t& nRows = matrix.size();
            const size_t& nCols = matrix.at(0).size();
            const size_t& nOtherCols = otherMatrix.at(0).size();
            Matrix<number, sizeP.first, sizeP.second> res(nRows, nCols + nOtherCols);
            for (size_t i{}; i < nRows; i++) {
                auto& row{res.at(i)};
                const auto& thisRow{matrix.at(i)};
                const auto& otherRow{otherMatrix.at(i)};
                for (size_t j{}; j < (nCols + nOtherCols); j++) {
                    if (j < nCols)
                        row.at(j) = thisRow.at(j);
                    else
                        row.at(j) = otherRow.at(j - nCols);
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

            const size_t& nRows = matrix.size();
            const size_t& nCols = matrix.at(0).size();

            Matrix<number, sizeP.first, sizeP.second> res(nRows, nCols + 1);
            for (size_t i{}; i < nRows; i++) {
                auto& row{res.at(i)};
                const auto& thisRow{matrix.at(i)};
                size_t j{};
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
            const int& nRows = matrix.size();
            const int& nCols = matrix.at(0).size();
            if (nRows != nCols) throw std::runtime_error("Matrix must be square to find a subset");
            if (nRows <= 1 || nCols <= 1) throw std::runtime_error("Matrix must be at least 2x2 to find a subset");

            constexpr auto isDynamic = m == 0 || n == 0;
            constexpr auto DynamicPair = SizePair{Dynamic, Dynamic};

            constexpr auto sizeP = isDynamic ? DynamicPair : SizePair{m - 1, n - 1};

            Matrix<number, sizeP.first, sizeP.second> res(nRows - 1, nCols - 1);
            int resRow = 0;
            for (int k = 0; k < nRows; k++) {
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

        enum By : uint8_t { ROW = 0, COL };

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
            auto compFunc = [](number x) { return fuzzyCompare(x, number(0)); };

            for (const auto& row : matrix) {
                auto count = rg::count_if(row, compFunc);
                if (count > maxRowZeros) {
                    maxRowZeros = count;
                    rowPos = pos;
                }
                pos++;
            }

            pos = 0;
            for (const auto& col : matrixColsToVectorSet<number, m, n>(matrix)) {
                auto count = rg::count_if(col, compFunc);
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
            if (fuzzyCompare(a, number(0))) return 0;
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
            const size_t& nCols = matrix.at(0).size();
            int i{row};
            for (size_t j{0}; j < nCols; j++) {
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
            const size_t& nCols = matrix.at(0).size();
            int j{col};
            for (size_t i{0}; i < nCols; i++) {
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
     * @brief MatrixView class for representing a view of a matrix
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

        // Constructor to keep consistency with the Dynamic Matrix specialization to allow them to be used
        // interchangeably
        Matrix(int nRows, int nCols) {}  // NOLINT

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
        Row<number, n>& operator[](size_t i) { return matrixRowAt(matrix, i); }

        /**
         * @brief Const access the ith row of the matrix
         *
         * @param i The index of the row to access
         * @return The ith row
         */
        Row<number, n> operator[](size_t i) const { return matrixRowAtConst(matrix, i); }

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
        number& operator[](size_t i, size_t j) { return matrixAt<number>(matrix, i, j); }

        /**
         * @brief Const access the element at the ith row and jth column
         *
         * @param i The index of the row
         * @param j The index of the column
         * @return The element at the ith row and jth column
         */
        number operator[](size_t i, size_t j) const { return matrixAtConst<number>(matrix, i, j); }

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
            if (nOther != n)
                throw std::runtime_error("The columns of the matrix must be equal to the size of the vector");
            return multMatByVec<number, m, n>(matrix, vec);
        }

        template <int nOther>
        Vector<number, Dynamic> operator*(const Vector<number, nOther>& vec) const
            requires(nOther == Dynamic)
        {
            if (this->numCols() != vec.size())
                throw std::runtime_error("The columns of the matrix must be equal to the size of the vector");
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
            constexpr bool isDynamic = m == Dynamic || n == Dynamic || mOther == Dynamic || nOther == Dynamic;
            constexpr bool isNotSquare = m != n || (m != mOther && n != nOther) || mOther != nOther;
            constexpr bool isNotPow2 = (size_t(n) & (size_t(n) - 1)) != 0;  // Checks if n is not a power of 2

            if constexpr (isDynamic || isNotSquare || isNotPow2) {
#ifdef DEBUG
                logging::log("Using default matrix multiplication algorithm", "Matrix operator*", logging::Level::INF);
#endif
                return multMatByDef<number, m, n, mOther, nOther>(matrix, other.matrix);
            } else {
#ifdef DEBUG
                logging::log("Using strassen's matrix multiplication algorithm", "Matrix operator*",
                             logging::Level::INF);
#endif
                return multMatStrassen<number, m, n, nOther>(matrix, other.matrix);
            }
        }

        template <int mOther, int nOther>
        Matrix<number, Dynamic, Dynamic> operator*(const Matrix<number, mOther, nOther>& other) const
            requires((n == Dynamic && m == Dynamic) || (mOther == Dynamic && nOther == Dynamic))
        {
            return multMatByDef<number, Dynamic, Dynamic, Dynamic, Dynamic>(matrix, other.matrix);
        }

        /**
         * @brief Matrix multiplication by a transposed matrix
         *
         * @param other The transposed matrix to multiply by
         * @return The matrix resulting from the multiplication
         */
        template <int nOther>
        Matrix<number, m, nOther> operator*(const TransposeVariant<number, n, nOther>& other) const {
            return multMatByDef<number, m, n, n, nOther>(matrix, helpers::extractMatrixFromTranspose(other).matrix);
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

        /**
         * @brief Slice the matrix
         *
         * @return The sliced matrix of size (m - (i1 - i0))x(n - (j1 - j0))
         */
        template <size_t i0, size_t i1, size_t j0, size_t j1>
        Matrix<number, (i1 - i0), (j1 - j0)> slice() {
            return MatrixSlice<i0, i1, j0, j1, number, m, n>(matrix);
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
        MatrixView<number, m, n> view(size_t rowOffset = 0, size_t colOffset = 0, size_t rowStride = 1,
                                      size_t colStride = 1) {
            return View<number, m, n, i, j>(matrix, rowOffset, colOffset, rowStride, colStride);
        }

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
            if constexpr (n <= 0) throw std::invalid_argument("Matrix n must be greater than zero");
            if constexpr (m <= 0) throw std::invalid_argument("Matrix m must be greater than zero");
        }

        /**
         * @brief Backing array for the matrix
         */
        TDArray<number, m, n> matrix{};
    };

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

        template <int m, int n>
        explicit Matrix(const Matrix<number, m, n>& other)
            requires(m != -1 && n != -1)
            : m{m}, n{n} {
            matrix.reserve(m);
            for (const auto& row : other.matrix) matrix.emplace_back(row);
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
        number& operator[](size_t i, size_t j) { return matrixAt<number>(matrix, i, j); }

        /**
         * @brief Const access the element at the ith row and jth column
         *
         * @param i The index of the row
         * @param j The index of the column
         * @return The element at the ith row and jth column
         */
        number operator[](size_t i, size_t j) const { return matrixAtConst<number>(matrix, i, j); }

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
        template <int otherM, int otherN>
        Matrix<number, Dynamic, Dynamic> operator+(const Matrix<number, otherM, otherN>& other) const {
            return matrixAdd<number, Dynamic, Dynamic>(matrix, other.matrix);
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
            return multMatByDef<number, Dynamic, Dynamic, Dynamic, Dynamic>(matrix, other.matrix);
        }

        template <int m, int n, int otherM, int otherN>
        friend Matrix<number, Dynamic, Dynamic> operator*(const Matrix<number, otherM, otherN>& lhs,
                                                          const Matrix<number, m, n> rhs)
            requires((n == Dynamic && m == Dynamic) && (otherN != Dynamic && otherM != Dynamic))
        {
            return multMatByDef<number, Dynamic, Dynamic, Dynamic>(lhs.matrix, rhs.matrix);
        }

        /**
         * @brief Matrix multiplication by a transposed matrix
         *
         * @param other The transposed matrix to multiply by
         * @return The matrix resulting from the multiplication
         */
        template <int otherM, int otherN>
        Matrix<number, Dynamic, Dynamic> operator*(const TransposeVariant<number, otherM, Dynamic>& other) const {
            return multMatByDef<number, Dynamic, Dynamic, Dynamic>(helpers::extractMatrixFromTranspose(other));
        }

        template <int m, int n, int otherM, int otherN>
        friend Matrix<number, Dynamic, Dynamic> operator*(const TransposeVariant<number, otherM, otherN>& lhs,
                                                          const Matrix<number, m, n> rhs)
            requires((n == Dynamic && m == Dynamic) && (otherN != Dynamic && otherM != Dynamic))
        {
            return multMatByDef<number, Dynamic, Dynamic, Dynamic>(helpers::extractMatrixFromTranspose(lhs),
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
         * @brief Slice the matrix
         *
         * @param i SlicePair for the rows in the form {i0, i1}
         * @param j SlicePair for the columns in the form {j0, j1}
         * @return The sliced matrix of size (m - (i1 - i0))x(n - (j1 - j0))
         */
        Matrix<number, Dynamic, Dynamic> slice(const SlicePair& i, const SlicePair& j) {
            return MatrixSlice<number>(matrix, i, j);
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
