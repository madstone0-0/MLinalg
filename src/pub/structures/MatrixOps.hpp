/**
 * @file MatrixOps.hpp
 * @brief Header file for matrix operations
 */

#pragma once
#include <cmath>
#include <stdexcept>
#include <type_traits>

#include "VectorOps.hpp"

#ifdef __AVX__
#include <immintrin.h>
#endif

#ifdef DEBUG
#include "../Logging.hpp"
#endif

#include "../Concepts.hpp"
#include "../Helpers.hpp"
#include "../Numeric.hpp"
#include "Aliases.hpp"
#include "Vector.hpp"

using std::pair, std::invalid_argument, std::string, std::is_same_v, std::runtime_error, std::get, std::optional,
    std::array;
namespace rg = std::ranges;

namespace mlinalg::structures {
    using namespace mlinalg::stacktrace;
    using namespace helpers;

    template <Container T, Container U>
    inline void checkMatrixOperandRowSize(const T& matrix, const U& otherMatrix) {
        if (matrix.size() != otherMatrix.size())
            throw StackError<invalid_argument>("Matrices must be of the same dimensions");
    }

    template <Container T, Container U>
    inline void checkMatrixOperandSize(const T& matrix, const U& otherMatrix) {
        if (matrix.size() != otherMatrix.size())
            throw StackError<invalid_argument>("Matrices must be of the same dimensions");

        if (matrix[0].size() != otherMatrix[0].size())
            throw StackError<invalid_argument>("Matrices must be of the same dimensions");
    }

    template <Container T>
    inline auto& matrixRowAt(T& matrix, size_t i) {
        return matrix.at(i);
    }

    template <Container T>
    inline auto matrixRowAtConst(const T& matrix, size_t i) {
        return matrix.at(i);
    }

    template <Number number, Container T>
    inline number& matrixAt(T& matrix, size_t i, size_t j) {
        return matrix.at(i).at(j);
    }

    template <Number number, Container T>
    inline number matrixAtConst(const T& matrix, size_t i, size_t j) {
        return matrix.at(i).at(j);
    }

    template <Number number, int m, int n, Container T>
    inline auto matrixColsToVectorSet(const T& matrix)
        -> IsDynamicT<m, n, vector<Vector<number, Dynamic>>, vector<Vector<number, m>>> {
        if constexpr ((m == Dynamic || n == Dynamic) || (m == 0 || n == 0)) {
            const size_t& nRows = matrix.size();
            const size_t& nCols = matrix[0].size();
            vector<RowDynamic<number>> res;
            res.reserve(nCols);
            for (size_t i{}; i < nCols; i++) {
                Vector<number, Dynamic> vec(nRows);
                for (size_t j{}; j < nRows; j++) {
                    vec[j] = matrix[j][i];
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
                    vec[j] = matrix[j][i];
                }
                res.emplace_back(std::move(vec));
            }
            return res;
        }
    }

    template <Number number, int m, int n, Container T>
    inline auto matrixColsToVectorSet(Matrix<number, m, n> matrix) {
        vector<Vector<number, m>> res;
        res.reserve(n);
        for (size_t i{}; i < n; i++) {
            res.emplace_back(std::move(matrix.col(i).toVector()));
        }
        return res;
    }

    template <Number number, int m, int n, Container T>
    inline vector<Vector<number, n>> matrixRowsToVectorSet(const T& matrix) {
        constexpr int vSize = (m == Dynamic || n == Dynamic) ? Dynamic : n;
        const auto& nRows = matrix.size();
        vector<Vector<number, vSize>> res;
        res.reserve(nRows);
        for (const auto& row : matrix) res.emplace_back(row);
        return res;
    }

    template <Number number, Container T, Container U>
    inline bool matrixEqual(const T& matrix, const U& otherMatrix) {
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

    template <typename F, Container T>
    inline void matrixApply(T& matrix, F f) {
        for (auto& row : matrix) vectorApply(row, f);
    }

    template <typename F, Container T>
    inline void matrixApply(const T& matrix, F f) {
        for (auto& row : matrix) vectorApply(row, f);
    }

    template <typename F, Container T>
    inline void matrixRowApply(T& matrix, F f) {
        for (auto& row : matrix) f(row);
        ;
    }

    template <typename F, Container T>
    inline void matrixRowApply(const T& matrix, F f) {
        for (auto& row : matrix) f(row);
        ;
    }

    template <typename F, Container T, Container U, bool checkSizes = false>
    inline void matrixApply(T& matrix, const U& otherMatrix, F f) {
        const auto m = matrix.size();
        const auto n = matrix[0].size();
        const auto otherM = otherMatrix.size();
        const auto otherN = otherMatrix[0].size();
        if constexpr (checkSizes) {
            if (m != otherM || n != otherN) {
                throw StackError<invalid_argument>("Matrices must be of the same dimensions");
            }
        }
        auto i = matrix.begin();
        auto j = otherMatrix.begin();
        for (; i != matrix.end(); ++i, ++j) vectorApply(*i, *j, f);
    }

    template <typename F, Container T, Container U, bool checkSizes = false>
    inline void matrixApply(const T& matrix, const U& otherMatrix, F f) {
        const auto m = matrix.size();
        const auto n = matrix[0].size();
        const auto otherM = otherMatrix.size();
        const auto otherN = otherMatrix[0].size();
        if constexpr (checkSizes) {
            if (m != otherM || n != otherN) {
                throw StackError<invalid_argument>("Matrices must be of the same dimensions");
            }
        }
        auto i = matrix.begin();
        auto j = otherMatrix.begin();
        for (; i != matrix.end(); ++i, ++j) vectorApply(*i, *j, f);
    }

    template <typename F, Container T, Container U, bool checkSizes = false>
    inline void matrixRowApply(T& matrix, const U& otherMatrix, F f) {
        const auto m = matrix.size();
        const auto n = matrix[0].size();
        const auto otherM = otherMatrix.size();
        const auto otherN = otherMatrix[0].size();
        if constexpr (checkSizes) {
            if (m != otherM || n != otherN) {
                throw StackError<invalid_argument>("Matrices must be of the same dimensions");
            }
        }
        auto i = matrix.begin();
        auto j = otherMatrix.begin();
        for (; i != matrix.end(); ++i, ++j) f(*i, *j);
    }

    template <typename F, Container T, Container U, bool checkSizes = false>
    inline void matrixRowApply(const T& matrix, const U& otherMatrix, F f) {
        const auto m = matrix.size();
        const auto n = matrix[0].size();
        const auto otherM = otherMatrix.size();
        const auto otherN = otherMatrix[0].size();
        if constexpr (checkSizes) {
            if (m != otherM || n != otherN) {
                throw StackError<invalid_argument>("Matrices must be of the same dimensions");
            }
        }
        auto i = matrix.begin();
        auto j = otherMatrix.begin();
        for (; i != matrix.end(); ++i, ++j) f(*i, *j);
    }

    template <Number number, int m, int n, Container T>
    inline void matrixScalarMultI(T& matrix, const number& scalar) {
        matrixApply(matrix, [&](auto& x) { x *= scalar; });
    }

    template <Number number, int m, int n, Container T>
    inline void matrixScalarDivI(T& matrix, const number& scalar) {
        if (fuzzyCompare(scalar, number(0))) throw StackError<std::domain_error>("Division by zero");
        matrixApply(matrix, [&](auto& x) { x /= scalar; });
    }

    template <Number number, int m, int n, Container T, Container U>
    inline void matrixAddI(T& matrix, const U& otherMatrix) {
        checkMatrixOperandSize(matrix, otherMatrix);
        matrixApply(matrix, otherMatrix, [&](auto& x, const auto& y) { x += y; });
    }

    template <Number number, int m, int n, Container T, Container U>
    inline void matrixSubI(T& matrix, const U& otherMatrix) {
        checkMatrixOperandSize(matrix, otherMatrix);
        matrixApply(matrix, otherMatrix, [&](auto& x, const auto& y) { x -= y; });
    }

    template <Container T>
    inline string matrixStringRepr(const T& matrix) {
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
    inline Vector<number, mRes> multMatByVec(const T& matrix, const Vector<number, n>& vec) {
        if (matrix.at(0).size() != vec.size())
            throw StackError<invalid_argument>("The columns of the matrix must be equal to the size of the vector");

        constexpr int vSize = (m == Dynamic || n == Dynamic) ? Dynamic : m;
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

    template <Number number, int m, int n, int mRes = m, Container T, Container U>
    inline Vector<number, mRes> multMatByVec(const T& matrix, const U& vec) {
        if (matrix.at(0).size() != vec.size())
            throw StackError<invalid_argument>("The columns of the matrix must be equal to the size of the vector");

        constexpr int vSize = (m == Dynamic || n == Dynamic) ? Dynamic : m;
        const auto& nRows = matrix.size();

        Vector<number, vSize> res(nRows);
        const auto nR = matrix.size();
        const auto nC = matrix[0].size();
        for (size_t i{}; i < nRows; ++i) {
            number sum{};
            for (size_t j{}; j < nC; ++j) {
                sum += matrix[i][j] * vec[j];
            }
            res[i] = sum;
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
    inline Matrix<number, m, nOther> multMatByDef(const T& matrix, const U& otherMatrix) {
        if (matrix.at(0).size() != static_cast<size_t>(otherMatrix.size()))
            throw StackError<invalid_argument>(
                "The columns of the first matrix must be equal to the rows of the second matrix");

        constexpr bool isDynamic = m == Dynamic || n == Dynamic || mOther == Dynamic || nOther == Dynamic;

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

    /**
     * @brief Matrix multiplication using the row-wise method to improve cache locality
     *
     * @param matrix The matrix to multiply
     * @param other The matrix to multiply by
     * @return The matrix resulting from the multiplication
     */
    template <Number number, int m, int n, int mOther, int nOther, Container T, Container U>
    inline auto multMatRowWise(const T& matrix, const U& otherMatrix) {
        if (matrix[0].size() != static_cast<size_t>(otherMatrix.size()))
            throw StackError<invalid_argument>(
                "The columns of the first matrix must be equal to the rows of the second matrix");

        const size_t nRows = matrix.size();
        const size_t nCols = matrix[0].size();
        const size_t nColsOther = otherMatrix[0].size();

        constexpr bool isDynamic = m == Dynamic || n == Dynamic || mOther == Dynamic || nOther == Dynamic;

        constexpr auto resSizeP = isDynamic ? DynamicPair : SizePair{m, nOther};

        constexpr int blockSize = 64 / sizeof(number);
        Matrix<number, resSizeP.first, resSizeP.second> res{nRows, nColsOther};

        for (size_t i{}; i < nRows; i++)
            for (size_t j{}; j < nCols; j++) {
                const number temp = matrix[i][j];
                for (size_t k{}; k < nColsOther; k += blockSize) {
                    const size_t kEnd = std::min(k + blockSize, nColsOther);
                    for (size_t kk{k}; kk < kEnd; kk++) {
                        res(i, kk) += temp * otherMatrix[j][kk];
                    }
                }
            }

        return res;
    }

#if defined(__AVX__) && defined(__FMA__)
    template <Number number, int m, int n, int nOther, int mRes = m, Container T, Container U>
    inline Vector<number, mRes> multVecSIMD(const T& matrix, const U& vec)
        requires(is_same_v<number, double>)
    {
        if (matrix.at(0).size() != vec.size())
            throw StackError<invalid_argument>("The columns of the matrix must be equal to the size of the vector");

        constexpr bool isDynamic = m == Dynamic || n == Dynamic || nOther == Dynamic;

        constexpr int vSize = isDynamic ? Dynamic : m;
        const auto& nRows = matrix.size();
        const auto& nCols = matrix[0].size();
        constexpr int vecSize = 4;            // AVX2 can handle 4 doubles
        constexpr int unrollFactor = 4;       // Unroll by 4 for better instruction pipelining
        constexpr int prefetchDistance = 64;  // Cache line size

        Vector<number, vSize> res(nRows);

        // Pre-load vector data into cache (temporal locality)
        for (size_t k = 0; k < nCols; k += prefetchDistance / sizeof(double)) {
            _mm_prefetch(reinterpret_cast<const char*>(&vec[k]), _MM_HINT_T0);
        }

        // Process rows in blocks for better cache utilization
        constexpr size_t blockSize = 8;  // Process 8 rows at a time

        for (size_t blockStart = 0; blockStart < nRows; blockStart += blockSize) {
            size_t blockEnd = std::min(blockStart + blockSize, nRows);

            // Prefetch next block of matrix rows
            if (blockEnd < nRows) {
                for (size_t i = blockEnd; i < std::min(blockEnd + blockSize, nRows); ++i) {
                    _mm_prefetch(reinterpret_cast<const char*>(&matrix[i][0]), _MM_HINT_T1);
                }
            }

            // Process current block
            for (size_t i = blockStart; i < blockEnd; ++i) {
                // Initialize multiple accumulators to reduce dependency chains
                __m256d sum1 = _mm256_setzero_pd();
                __m256d sum2 = _mm256_setzero_pd();
                __m256d sum3 = _mm256_setzero_pd();
                __m256d sum4 = _mm256_setzero_pd();

                size_t j = 0;
                const double* matrixRow = &matrix[i][0];
                const double* vecData = &vec[0];

                // Prefetch current row data
                _mm_prefetch(reinterpret_cast<const char*>(matrixRow), _MM_HINT_T0);

                // Main vectorized loop with unrolling
                for (; j + unrollFactor * vecSize <= nCols; j += unrollFactor * vecSize) {
                    // Prefetch ahead for both matrix and vector
                    _mm_prefetch(reinterpret_cast<const char*>(&matrixRow[j + prefetchDistance]), _MM_HINT_T0);
                    _mm_prefetch(reinterpret_cast<const char*>(&vecData[j + prefetchDistance]), _MM_HINT_T0);

                    // Unrolled computation with separate accumulators
                    __m256d a1 = _mm256_loadu_pd(&matrixRow[j]);
                    __m256d v1 = _mm256_loadu_pd(&vecData[j]);
                    sum1 = _mm256_fmadd_pd(a1, v1, sum1);  // FMA: a*v + sum

                    __m256d a2 = _mm256_loadu_pd(&matrixRow[j + vecSize]);
                    __m256d v2 = _mm256_loadu_pd(&vecData[j + vecSize]);
                    sum2 = _mm256_fmadd_pd(a2, v2, sum2);

                    __m256d a3 = _mm256_loadu_pd(&matrixRow[j + 2 * vecSize]);
                    __m256d v3 = _mm256_loadu_pd(&vecData[j + 2 * vecSize]);
                    sum3 = _mm256_fmadd_pd(a3, v3, sum3);

                    __m256d a4 = _mm256_loadu_pd(&matrixRow[j + 3 * vecSize]);
                    __m256d v4 = _mm256_loadu_pd(&vecData[j + 3 * vecSize]);
                    sum4 = _mm256_fmadd_pd(a4, v4, sum4);
                }

                // Handle remaining full vectors
                for (; j + vecSize <= nCols; j += vecSize) {
                    __m256d a = _mm256_loadu_pd(&matrixRow[j]);
                    __m256d v = _mm256_loadu_pd(&vecData[j]);
                    sum1 = _mm256_fmadd_pd(a, v, sum1);
                }

                // Combine all accumulators
                __m256d totalSum = _mm256_add_pd(_mm256_add_pd(sum1, sum2), _mm256_add_pd(sum3, sum4));

                // Optimized horizontal sum using hadd
                __m256d hsum = _mm256_hadd_pd(totalSum, totalSum);
                __m128d sum128 = _mm_add_pd(_mm256_castpd256_pd128(hsum), _mm256_extractf128_pd(hsum, 1));
                double result = _mm_cvtsd_f64(sum128);

                // Handle remaining scalar elements
                for (; j < nCols; ++j) {
                    result += matrixRow[j] * vecData[j];
                }

                res[i] = result;
            }
        }

        return res;
    }

    template <Number number, int m, int n, int nOther, int mRes = m, Container T, Container U>
    inline Vector<number, mRes> multVecSIMD(const T& matrix, const U& vec)
        requires(is_same_v<number, float>)
    {
        if (matrix.at(0).size() != vec.size())
            throw StackError<invalid_argument>("The columns of the matrix must be equal to the size of the vector");

        constexpr bool isDynamic = m == Dynamic || n == Dynamic || nOther == Dynamic;

        constexpr int vSize = isDynamic ? Dynamic : m;
        const auto& nRows = matrix.size();
        const auto& nCols = matrix[0].size();
        const auto& vCols = vec.size();
        const int vecSize{8};  // AVX can handle 8 floats

        Vector<number, vSize> res(nRows);
        // For each row in the matrix
        for (size_t i{}; i < nRows; ++i) {
            __m256 sum = _mm256_setzero_ps();
            size_t j{};
            for (; j + vecSize <= nCols; j += vecSize) {
                // Load the values into __m256d registers
                // Load 8 matrix elements
                __m256 avxA = _mm256_loadu_ps(&matrix[i][j]);

                // Load 8 vector elements
                __m256 avxB = _mm256_loadu_ps(&vec[j]);

                // Multiply and Accumulate
                __m256 prod = _mm256_mul_ps(avxA, avxB);
                sum = _mm256_add_ps(sum, prod);
            }

            // Horizontal sum of the 8 values in sum register
            __m128 sumHigh = _mm256_extractf128_ps(sum, 1);
            __m128 sumLow = _mm256_castps256_ps128(sum);
            __m128 sum128 = _mm_add_ps(sumHigh, sumLow);
            __m128 sum64 = _mm_hadd_ps(sum128, sum128);
            sum64 = _mm_hadd_ps(sum64, sum64);
            float result = _mm_cvtss_f32(sum64);

            // Handle remaining elements (if nCols is not multiple of 8)
            for (; j < nCols; ++j) {
                result += matrix[i][j] * vec[j];
            }

            // Place the result into the result vector
            res[i] = result;
        }

        return res;
    }

    template <Number number, int m, int n, int mOther, int nOther, Container T, Container U>
    constexpr auto multMatSIMD(const T& matrix, const U& otherMatrix)
        requires(is_same_v<number, float>)
    {
        if (matrix[0].size() != static_cast<size_t>(otherMatrix.size()))
            throw StackError<invalid_argument>(
                "The columns of the first matrix must be equal to the rows of the second matrix");

        const int nRows = matrix.size();
        const int nCols = matrix[0].size();
        const int nColsOther = otherMatrix[0].size();

        constexpr bool isDynamic = m == Dynamic || n == Dynamic || mOther == Dynamic || nOther == Dynamic;

        constexpr auto resSizeP = isDynamic ? DynamicPair : SizePair{m, nOther};

        Matrix<float, resSizeP.first, resSizeP.second> res{nRows, nColsOther};
        const int vecSize{16};  // AVX can handle 8 floats, but we unroll 2 iterations

        for (size_t i{}; i < nRows; i++) {
            for (size_t k{}; k < nCols; k++) {
                const float a = matrix[i][k];
                __m256 avxA = _mm256_set1_ps(a);

                auto kRow = otherMatrix[k].data();
                auto iRow = res[i].data();

                // Only vectorize if there are at least 8 columns.
                if (nColsOther >= vecSize) {
                    size_t j = 0;

                    // Prefetch next cache lines
                    _mm_prefetch(reinterpret_cast<const char*>(&kRow[j + 64]), _MM_HINT_T0);
                    _mm_prefetch(reinterpret_cast<const char*>(&iRow[j + 64]), _MM_HINT_T0);

                    // Process in chunks of 16 floats
                    for (; j + vecSize <= nColsOther; j += vecSize) {
                        __m256 avxB1 = _mm256_loadu_ps(&kRow[j]);
                        __m256 avxRes1 = _mm256_loadu_ps(&iRow[j]);
                        avxRes1 = _mm256_fmadd_ps(avxA, avxB1, avxRes1);
                        _mm256_storeu_ps(&iRow[j], avxRes1);

                        __m256 avxB2 = _mm256_loadu_ps(&kRow[j + 8]);
                        __m256 avxRes2 = _mm256_loadu_ps(&iRow[j + 8]);
                        avxRes2 = _mm256_fmadd_ps(avxA, avxB2, avxRes2);
                        _mm256_storeu_ps(&iRow[j + 8], avxRes2);
                    }
                    // Process remaining elements, if any, with scalar code.
                    for (; j < nColsOther; j++) {
                        iRow[j] += a * kRow[j];
                    }

                } else {
                    // If there are fewer than 8 columns, use scalar code entirely.
                    for (size_t j = 0; j < nColsOther; j++) {
                        iRow[j] += a * kRow[j];
                    }
                }
            }
        }
        return res;
    }

    template <Number number, int m, int n, int mOther, int nOther, Container T, Container U>
    constexpr auto multMatSIMD(const T& matrix, const U& otherMatrix)
        requires(is_same_v<number, double>)
    {
        if (matrix[0].size() != static_cast<size_t>(otherMatrix.size()))
            throw StackError<invalid_argument>(
                "The columns of the first matrix must be equal to the rows of the second matrix");

        const size_t nRows = matrix.size();
        const size_t nCols = matrix[0].size();
        const size_t nColsOther = otherMatrix[0].size();

        constexpr bool isDynamic = m == Dynamic || n == Dynamic || mOther == Dynamic || nOther == Dynamic;

        constexpr auto resSizeP = isDynamic ? DynamicPair : SizePair{m, nOther};

        Matrix<double, resSizeP.first, resSizeP.second> res{nRows, nColsOther};
        const int vecSize{8};  // AVX can handle 4 doubles, but we unroll 2 iterations

        for (size_t i{}; i < nRows; i++) {
            for (size_t k{}; k < nCols; k++) {
                const double a = matrix[i][k];
                __m256d avxA = _mm256_set1_pd(a);

                auto kRow = otherMatrix[k].data();
                auto iRow = res[i].data();

                // Only vectorize if there are at least 4 columns.
                if (nColsOther >= vecSize) {
                    size_t j = 0;
                    // Prefetch next cache lines
                    _mm_prefetch(reinterpret_cast<const char*>(&kRow[j + 64]), _MM_HINT_T0);
                    _mm_prefetch(reinterpret_cast<const char*>(&iRow[j + 64]), _MM_HINT_T0);

                    // Process in chunks of 8 doubles
                    for (; j + vecSize <= nColsOther; j += vecSize) {
                        __m256d avxB1 = _mm256_loadu_pd(&kRow[j]);
                        __m256d avxRes1 = _mm256_loadu_pd(&iRow[j]);
                        avxRes1 = _mm256_fmadd_pd(avxA, avxB1, avxRes1);
                        _mm256_storeu_pd(&iRow[j], avxRes1);

                        __m256d avxB2 = _mm256_loadu_pd(&kRow[j + 4]);
                        __m256d avxRes2 = _mm256_loadu_pd(&iRow[j + 4]);
                        avxRes2 = _mm256_fmadd_pd(avxA, avxB2, avxRes2);
                        _mm256_storeu_pd(&iRow[j + 4], avxRes2);
                    }
                    // Process remaining elements, if any, with scalar code.
                    for (; j < nColsOther; j++) {
                        iRow[j] += a * kRow[j];
                    }
                } else {
                    // If there are fewer than 4 columns, use scalar code entirely.
                    for (size_t j = 0; j < nColsOther; j++) {
                        iRow[j] += a * kRow[j];
                    }
                }
            }
        }
        return res;
    }
#endif

    template <Number number, int m, int n, int mOther, int nOther, Container T, Container U>
    inline auto MatrixMultiplication(const T& matrix, const U& otherMatrix)
        -> IsDynamicTOther<m, n, mOther, nOther, Matrix<number, Dynamic, Dynamic>, Matrix<number, m, nOther>> {
#ifdef STRASSEN
        constexpr bool isDynamic = m == Dynamic || n == Dynamic || mOther == Dynamic || nOther == Dynamic;
        constexpr bool isNotSquare = m != n || (m != mOther && n != nOther) || mOther != nOther;
        constexpr bool isNotPow2 = (size_t(n) & (size_t(n) - 1)) != 0;  // Checks if n is not a power of 2

        if constexpr (isDynamic || isNotSquare || isNotPow2) {
#if defined(__AVX__) && defined(__FMA__)
            if constexpr (is_same_v<number, double> || is_same_v<number, float>) {
                return multMatSIMD<number, m, n, mOther, nOther>(matrix, otherMatrix);
            } else {
                return multMatRowWise<number, m, n, mOther, nOther>(matrix, otherMatrix);
            }

#else
            return multMatRowWise<number, m, n, mOther, nOther>(matrix, other.matrix);
#endif  // __AVX__

        } else {
#ifdef DEBUG
            logging::log("Using strassen's matrix multiplication algorithm", "Matrix operator*", logging::Level::INF);
#endif
            return multMatStrassen<number, m, n, nOther>(matrix, otherMatrix);
        }
#else

#if defined(__AVX__) && defined(__FMA__)
        if constexpr (is_same_v<number, double> || is_same_v<number, float>) {
            return multMatSIMD<number, m, n, mOther, nOther>(matrix, otherMatrix);
        } else {
            return multMatRowWise<number, m, n, mOther, nOther>(matrix, otherMatrix);
        }

#else
        return multMatRowWise<number, m, n, mOther, nOther>(matrix, otherMatrix);
#endif  // __AVX__
#endif
    }

    template <Number number, int m, int n, int nOther, int mRes = m, Container T, Container U>
    inline auto MatrixVectorMultiplication(const T& matrix, const U& vec) -> Vector<number, mRes> {
#if defined(__AVX__) && defined(__FMA__)
        if constexpr (is_same_v<number, double> || is_same_v<number, float>) {
            return multVecSIMD<number, m, n, nOther, mRes>(matrix, vec);
        } else {
            if constexpr (n != nOther) throw StackError<invalid_argument>("Matrix and vector sizes do not match");
            return multMatByVec<number, m, n, mRes>(matrix, vec);
        }
#else
        if constexpr (n != nOther) throw StackError<invalid_argument>("Matrix and vector sizes do not match");
        return multMatByVec<number, m, n, mRes>(matrix, vec);
#endif  // __AVX__
    }

    template <Number number, int m, int n, Container T>
    inline auto MatrixSubset(const T& matrix, const optional<int>& i, const optional<int> j)
        -> IsDynamicT<m, n, Matrix<number, Dynamic, Dynamic>, Matrix<number, m - 1, n - 1>>;

    template <int i0, int i1, int j0, int j1, Number number, int m, int n, Container T>
    inline Matrix<number, (i1 - i0), (j1 - j0)> MatrixSlice(const T& matrix) {
        if constexpr (rg::any_of(array<int, 4>{i0, i1, j0, j1}, [](auto x) { return x < 0; }))
            throw StackError<invalid_argument>("Negative slicing not supported");

        if constexpr (rg::any_of(array<size_t, 2>{i1, j1}, [](auto x) { return x > n; }))
            throw StackError<invalid_argument>("Cannot slice past matrix bounds");

        if constexpr (i0 > i1 || j0 > j1)
            throw StackError<invalid_argument>("Start position cannot be greater than end position");
        constexpr int mSize{i1 - i0};
        constexpr int nSize{j1 - j0};
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
    inline Matrix<number, Dynamic, Dynamic> MatrixSlice(const T& matrix, const SizeTPair& i, const SizeTPair& j) {
        const auto& [i0, i1] = i;
        const auto& [j0, j1] = j;
        const auto& m = matrix.size();
        const auto& n = matrix.at(0).size();

        if (rg::any_of(array<size_t, 4>{i0, i1, j0, j1}, [](auto x) { return x < 0; }))
            throw StackError<invalid_argument>("Negative slicing not supported");

        if (i1 > m || j1 > n) throw StackError<invalid_argument>("Cannot slice past matrix bounds");

        if (i0 > i1 || j0 > j1)
            throw StackError<invalid_argument>("Start position cannot be greater than end position");
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

    template <Container T, Container U>
    inline auto m1(const T& A, const U& B) {
        // Returns (A[0][0] + A[1][1]) * (B[0][0] + B[1][1])
        return (A.at(0).at(0) + A.at(1).at(1)) * (B.at(0).at(0) + B.at(1).at(1));
    }

    template <Container T, Container U>
    inline auto m2(const T& A, const U& B) {
        // Returns (A[1][0] + A[1][1]) * B[0][0]
        return (A.at(1).at(0) + A.at(1).at(1)) * B.at(0).at(0);
    }

    template <Container T, Container U>
    inline auto m3(const T& A, const U& B) {
        // Returns A[0][0] * (B[0][1] - B[1][1])
        return A.at(0).at(0) * (B.at(0).at(1) - B.at(1).at(1));
    }

    template <Container T, Container U>
    inline auto m4(const T& A, const U& B) {
        // Returns A[1][1] * (B[1][0] - B[0][0])
        return A.at(1).at(1) * (B.at(1).at(0) - B.at(0).at(0));
    }

    template <Container T, Container U>
    inline auto m5(const T& A, const U& B) {
        // Returns (A[0][0] + A[0][1]) * B[1][1]
        return (A.at(0).at(0) + A.at(0).at(1)) * B.at(1).at(1);
    }

    template <Container T, Container U>
    inline auto m6(const T& A, const U& B) {
        // Returns (A[1][0] - A[0][0]) * (B[0][0] + B[0][1])
        return (A.at(1).at(0) - A.at(0).at(0)) * (B.at(0).at(0) + B.at(0).at(1));
    }

    template <Container T, Container U>
    inline auto m7(const T& A, const U& B) {
        // Returns (A[0][1] - A[1][1]) * (B[1][0] + B[1][1])
        return (A.at(0).at(1) - A.at(1).at(1)) * (B.at(1).at(0) + B.at(1).at(1));
    }

    template <Number number, int m, int n, int nOther, Container T, Container U>
    inline Matrix<number, 2, 2> strassen(const T& A, const U& B)
        requires(n == 2 && m == 2 && nOther == 2)
    {
        // Base case
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

    template <Number number, int m, int n, int nOther, Container T, Container U>
    inline auto strassen(const T& A, const U& B) {
        if constexpr (n == 2 && m == 2 && nOther == 2) {
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

            // // Split matrix A into 4 submatrices
            // auto A00 = View<number, m, n, halfM, halfN>(A);
            // auto A01 = View<number, m, n, halfM, halfN>(A, 0, halfN);
            // auto A10 = View<number, m, n, halfM, halfN>(A, halfM, 0);
            // auto A11 = View<number, m, n, halfM, halfN>(A, halfM, halfN);
            //
            // // Split matrix B into 4 submatrices
            // auto B00 = View<number, m, nOther, halfM, halfNOther>(B);
            // auto B01 = View<number, m, nOther, halfM, halfNOther>(B, 0, halfNOther);
            // auto B10 = View<number, m, nOther, halfM, halfNOther>(B, halfM, 0);
            // auto B11 = View<number, m, nOther, halfM, halfNOther>(B, halfM, halfNOther);

            auto M1 = strassen<number, A00.numRows(), A00.numCols(), B00.numCols()>((A00 + A11).getMatrix(),
                                                                                    (B00 + B11).getMatrix());
            auto M2 =
                strassen<number, A10.numRows(), A10.numCols(), B00.numCols()>((A10 + A11).getMatrix(), B00.getMatrix());
            auto M3 =
                strassen<number, A00.numRows(), A00.numCols(), B01.numCols()>(A00.getMatrix(), (B01 - B11).getMatrix());
            auto M4 =
                strassen<number, A11.numRows(), A11.numCols(), B10.numCols()>(A11.getMatrix(), (B10 - B00).getMatrix());
            auto M5 =
                strassen<number, A00.numRows(), A00.numCols(), B00.numCols()>((A00 + A01).getMatrix(), B11.getMatrix());
            auto M6 = strassen<number, A10.numRows(), A10.numCols(), B00.numCols()>((A10 - A00).getMatrix(),
                                                                                    (B00 + B01).getMatrix());
            auto M7 = strassen<number, A01.numRows(), A01.numCols(), B10.numCols()>((A01 - A11).getMatrix(),
                                                                                    (B10 + B11).getMatrix());

            // auto M1 = strassen<number, halfM, halfN, halfNOther>(*((A00 + A11).matrix), *((B00 + B11).matrix));
            // auto M2 = strassen<number, halfM, halfN, halfNOther>(*((A10 + A11).matrix), *(B00.matrix));
            // auto M3 = strassen<number, halfM, halfN, halfNOther>(*(A00.matrix), *((B01 - B11).matrix));
            // auto M4 = strassen<number, halfM, halfN, halfNOther>(*(A11.matrix), *((B10 - B00).matrix));
            // auto M5 = strassen<number, halfM, halfN, halfNOther>(*((A00 + A01).matrix), *(B11.matrix));
            // auto M6 = strassen<number, halfM, halfN, halfNOther>(*((A10 - A00).matrix), *((B00 + B01).matrix));
            // auto M7 = strassen<number, halfM, halfN, halfNOther>(*((A01 - A11).matrix), *((B10 + B11).matrix));

            auto C00 = M1 + M4 - M5 + M7;
            auto C01 = M3 + M5;
            auto C10 = M2 + M4;
            auto C11 = M1 + M3 - M2 + M6;

            constexpr int nSize{m};
            return merge.template operator()<nSize>(C00, C01, C10, C11);
        }
    }

    template <Number number, int m, int n, int nOther, Container T, Container U>
    inline Matrix<number, m, nOther> multMatStrassen(const T& matrix, const U& otherMatrix) {
        if (matrix.at(0).size() != otherMatrix.size())
            throw StackError<invalid_argument>(
                "The columns of the first matrix must be equal to the rows of the second matrix");

        return strassen<number, m, n, nOther>(matrix, otherMatrix);
    }

    /**
     * @brief Transpose a mxn matrix to a nxm matrix
     *
     * @return The transposed matrix of size nxm
     */
    template <Number number, int m, int n, Container T>
    inline TransposeVariant<number, m, n> TransposeMatrix(const T& matrix) {
        constexpr auto isDynamic = m == Dynamic || n == Dynamic;

        constexpr auto vSize = [&]() -> auto {
            if constexpr (isDynamic) return Dynamic;
            if constexpr (n != 1)
                return n;
            else
                return m;
        }();
        constexpr auto sizeP = isDynamic ? DynamicPair : SizePair{m, n};

        const size_t& nRows = matrix.size();
        const size_t& nCols = matrix.at(0).size();

        auto mutateMatrix = [&](auto& variant) {
            if constexpr (is_same_v<std::decay_t<decltype(variant)>, Matrix<number, sizeP.second, sizeP.first>>) {
                for (size_t i{}; i < nRows; i++)
                    for (size_t j{}; j < nCols; j++) variant[j][i] = matrix[i][j];
            }
        };

        auto mutateVector = [&](auto& variant) {
            if constexpr (is_same_v<std::decay_t<decltype(variant)>, Vector<number, vSize>>) {
                if (nCols != 1)
                    for (size_t i{}; i < nCols; ++i) variant[i] = matrix[0][i];
                else
                    for (size_t i{}; i < nRows; ++i) variant[i] = matrix[i][0];
            }
        };

        TransposeVariant<number, sizeP.first, sizeP.second> res(
            std::in_place_index<1>, Matrix<number, sizeP.second, sizeP.first>(nCols, nRows));
        if (nRows != 1 && nCols != 1) {
            std::visit(mutateMatrix, res);
            return res;
        }

        const auto runtimeVSize = nCols != 1 ? nCols : nRows;
        res = Vector<number, vSize>(runtimeVSize);
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
    inline auto MatrixAugmentMatrix(const T& matrix, const U& otherMatrix)
        -> IsDynamicT<m, n, Matrix<number, Dynamic, Dynamic>, Matrix<number, m, n + nN>> {
        checkMatrixOperandRowSize(matrix, otherMatrix);
        constexpr auto isDynamic = m == Dynamic || n == Dynamic;

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
    inline auto MatrixAugmentVector(const MatrixContainer& matrix, const VectorContainer& vec)
        -> IsDynamicT<m, n, Matrix<number, Dynamic, Dynamic>, Matrix<number, m, n + 1>> {
        constexpr auto isDynamic = m == Dynamic || n == Dynamic;

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
    inline auto MatrixSubset(const T& matrix, const optional<int>& i, const optional<int> j)
        -> IsDynamicT<m, n, Matrix<number, Dynamic, Dynamic>, Matrix<number, m - 1, n - 1>> {
        const int& nRows = matrix.size();
        const int& nCols = matrix.at(0).size();
        if (nRows != nCols) throw StackError<runtime_error>("Matrix must be square to find a subset");
        if (nRows <= 1 || nCols <= 1) throw StackError<runtime_error>("Matrix must be at least 2x2 to find a subset");

        constexpr auto isDynamic = m == Dynamic || n == Dynamic;

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
    inline number MatrixDet2x2(const T& matrix) {
        return (matrix.at(0).at(0) * matrix.at(1).at(1)) - (matrix.at(0).at(1) * matrix.at(1).at(0));
    }

    enum By : uint8_t { ROW = 0, COL };

    /**
     * @brief Pick the row or column with the most zeros as a cofactor row or column
     */
    template <Number number, int m, int n, Container T>
    inline pair<By, int> pickCofactorRowOrCol(const T& matrix) {
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
    inline number cofactorCommon(const T& matrix, int i, int j) {
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
    inline number cofactorRow(const T& matrix, int row) {
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
    inline number cofactorCol(const T& matrix, int col) {
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
    inline number MatrixCofactor(const T& matrix) {
        auto [by, val] = pickCofactorRowOrCol<number, m, n>(matrix);
        if (by == ROW)
            return cofactorRow<number, m, n>(matrix, val);
        else
            return cofactorCol<number, m, n>(matrix, val);
    }

    template <Number number, Container T>
    inline number MatrixTrace(const T& matrix) {
        const auto nR = matrix.size();
        const auto nC = matrix.at(0).size();
        if (nR != nC) throw StackError<invalid_argument>("Matrix must be square to calculate trace");
        number sum{};
        for (size_t i{}; i < nR; i++) {
            sum += matrix[i][i];
        }
        return sum;
    }

    template <Number number, int m, int n>
    inline void matrixClear(Matrix<number, m, n>& A) {
        for (size_t i{}; i < A.numRows(); i++) {
            for (size_t j{}; j < A.numCols(); j++) {
                A(i, j) = number{};
            }
        }
    }

    // =====================
    // Induced Matrix Norms
    // =====================

    /**
     * @brief Calculate the L1 norm of a matrix
     *
     * @param matrix The matrix to calculate the norm of
     * @return The L1 norm of the matrix
     */
    template <Number number, int m, int n>
    inline number L1Norm(const Matrix<number, m, n>& matrix) {
        auto asCols{std::move(matrix.colToVectorSet())};
        number max{std::numeric_limits<number>::lowest()};
        for (const auto& col : asCols) {
            auto l1Norm = col.l1();
            if (l1Norm > max) max = l1Norm;
        }
        return max;
    }

    /**
     * @brief Calculate the L-inf norm of a matrix
     *
     * @param matrix  The matrix to calculate the norm of
     * @return The L-inf norm of the matrix
     */
    template <Number number, int m, int n>
    inline number LInfNorm(const Matrix<number, m, n>& matrix) {
        auto asRows{std::move(matrix.rowToVectorSet())};
        number max{std::numeric_limits<number>::lowest()};
        for (const auto& row : asRows) {
            auto l1Norm = row.l1();
            if (l1Norm > max) max = l1Norm;
        }
        return max;
    }

    // =====================
    // General Matrix Norms
    // =====================

    /**
     * @brief Calculate the Frobenius norm of a matrix
     *
     * @param matrix The matrix to calculate the norm of
     * @return The Frobenius norm of the matrix
     */
    template <Number number, Container T>
    inline number FrobenisNorm(const T& matrix) {
        const auto numRows = matrix.size();
        const auto numCols = matrix.at(0).size();
        number sum{};
        for (size_t i{}; i < numRows; i++)
            for (size_t j{}; j < numCols; j++) {
                sum += matrix[i][j] * matrix[i][j];
            }
        return std::sqrt(sum);
    }

}  // namespace mlinalg::structures
