/**
 * @file Operations.hpp
 * @brief Declarations and implementations of the operations that can be performed on the structures defined in
 * Structures.hpp.
 */

#pragma once
#include <sys/types.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iterator>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <vector>

#include "Concepts.hpp"
#include "Helpers.hpp"
#include "Logging.hpp"
#include "Numeric.hpp"
#include "structures/Aliases.hpp"
#include "structures/Matrix.hpp"
#include "structures/Vector.hpp"

using std::vector, std::array, std::optional, std::nullopt;

namespace mlinalg {
    using namespace structures;

    /**
     * @brief Linear System type alias.
     */
    template <Number number, int m, int n>
    using LinearSystem = Matrix<number, m, n>;

    template <Number num, int n>
    Vector<num, n> extractSolutionVector(const Vector<optional<num>, n>& solutions) {
        if (rg::any_of(solutions, [](const auto& val) { return !val.has_value(); })) {
            throw StackError<std::runtime_error>("Cannot extract solution vector from incomplete solutions");
        }
        const auto size = solutions.size();

        Vector<num, n> res(size);
        for (size_t i{}; i < size; i++) {
            res.at(i) = solutions.at(i).value();
        }
        return res;
    }

    /**
     * @brief Checks if a linear system is in echelon form.
     *
     * @param system  The linear system to check.
     * @param n      The number of columns in the system.
     * @param m     The number of rows in the system.
     * @param pivots The pivots of the system.
     * @return true if the system is in echelon form, false otherwise.
     */
    template <Number number, int m, int n>
    bool isInEchelonForm(const LinearSystem<number, m, n>& system, const RowOptional<number, m>& pivots) {
        const size_t& nRows = system.numRows();

        for (size_t i{}; i < pivots.size(); i++)
            for (size_t j{i + 1}; j < nRows; j++)
                if (!fuzzyCompare(system.at(j, i), number(0))) return false;
        return true;
    }

    /**
     * @brief Checks if a linear system is inconsistent. Assumes the linear system is in the form [A | b].
     *
     * @param system The linear system to check.
     * @return true if the system is inconsistent, false otherwise.
     */
    template <Number number, int m, int n>
    bool isInconsistent(const LinearSystem<number, m, n>& system) {
        const auto& nCols = system.numCols();

        for (const auto& row : system) {
            size_t zeroCount{};
            for (size_t i{}; i < (nCols - 1); i++)
                if (fuzzyCompare(row.at(i), number(0))) zeroCount++;

            if (zeroCount == (nCols - 1) && !fuzzyCompare(row.back(), number(0))) return true;
        }
        return false;
    }

    /**
     * @brief Checks if a linear system is in reduced echelon form.
     *
     * @param system The linear system to check.
     * @param n    The number of columns in the system.
     * @param m  The number of rows in the system.
     * @param pivots The pivots of the system.
     * @return true if the system is in reduced echelon form, false otherwise.
     */
    template <Number number, int m, int n>
    bool isInReducedEchelonForm(const LinearSystem<number, m, n>& system, const RowOptional<number, m>& pivots) {
        for (int i{1}; i < pivots.size(); i++) {
            if (!pivots.at(i).has_value()) continue;
            for (int j{i - 1}; j >= 0; j--)
                if (system.at(j).at(i) != 0) return false;
        }
        return true;
    }

    /**
     * @brief Solves a linear equation using the given row and solutions.
     *
     * @param row The row to solve.
     * @param varPos The position of the variable to solve for.
     * @param n The number of elements in the row.
     * @param solutions The already found values of other variables in the row.
     * @return The solution to the equation if it exists, nullopt otherwise.
     */
    template <Number number, int n>
    optional<number> solveEquation(const Row<number, n>& row, size_t varPos,
                                   const ConditionalRowOptional<number, n, n>& solutions) {
        const auto rowSize = row.size();
        number rightSum{row.back()};

        for (size_t i{}; i < rowSize - 1; i++) {
            optional<number> var = solutions.at(i);
            if (i != varPos) {
                number res{};
                if (var.has_value()) {
                    res = var.value() * row.at(i) * number(-1);
                } else {
                    res = row.at(i) * number(-1);
                }
                rightSum += res;
            }
        }

        const number& coeff = row.at(varPos);
        if (fuzzyCompare(coeff, number(0))) {
            if (fuzzyCompare(rightSum, number(0)) && fuzzyCompare(coeff, number(0)))
                return 0;
            else
                return nullopt;
        }

        const number& sol = rightSum / coeff;
        return sol;
    }

    /**
     * @brief Gets the pivots of a linear system.
     *
     * @param system The linear system to get the pivots from.
     * @return The pivots of the system, if they exist.
     */
    template <Number number, int m, int n>
    RowOptional<number, m> getPivots(const LinearSystem<number, m, n>& system, bool partial = true) {
        const size_t& nRows = system.numRows();
        const size_t& nCols = system.numCols();

        std::set<size_t> seenCols{};
        size_t pivRow{};
        size_t pivCol{};
        RowOptional<number, m> pivots(nRows);
        if (partial) {
            while (pivRow < nRows && pivCol < nCols) {
                // Check if current pivot is (effectively) zero
                if (fuzzyCompare(system.at(pivRow, pivCol), number(0))) {
                    // Find the first row below with a non-zero entry in this column
                    size_t kRow = pivRow;
                    for (kRow = pivRow + 1; kRow < nRows; kRow++) {
                        if (!fuzzyCompare(system.at(kRow, pivCol), number(0))) {
                            break;
                        }
                    }

                    if (kRow == nRows) {  // No non-zero found; move to next column
                        pivCol++;
                        continue;
                    }
                }
                pivots.at(pivRow) = system.at(pivRow, pivCol);
                pivRow++;
                pivCol++;
            }
        } else {
            for (size_t idx{}; idx < nRows; idx++) {
                const auto& row = system.at(idx);
                for (size_t i{pivCol}; i < nCols - 1; i++) {
                    pivCol++;
                    if (!fuzzyCompare(row.at(i), number(0))) {
                        pivots.at(pivRow) = row.at(i);
                        pivRow++;
                        break;
                    }
                }
            }
        }
        return pivots;
    }

    /**
     * @brief Gets the pivot locations of a linear system.
     *
     * @param system The linear system to get the pivots from.
     * @return The pivot locations if they exist, where the the value is the column position and the index of the value
     * is the row position
     */
    template <Number number, int m, int n>
    RowOptional<size_t, m> getPivotLocations(const LinearSystem<number, m, n>& system, bool partial = true) {
        const size_t& nRows = system.numRows();
        const size_t& nCols = system.numCols();

        std::set<size_t> seenCols{};
        size_t pivRow{};
        size_t pivCol{};
        RowOptional<size_t, m> pivots(nRows);
        if (partial) {
            while (pivRow < nRows && pivCol < nCols) {
                // Check if current pivot is (effectively) zero
                if (fuzzyCompare(system.at(pivRow, pivCol), number(0))) {
                    // Find the first row below with a non-zero entry in this column
                    size_t kRow = pivRow;
                    for (kRow = pivRow + 1; kRow < nRows; kRow++) {
                        if (!fuzzyCompare(system.at(kRow, pivCol), number(0))) {
                            break;
                        }
                    }

                    if (kRow == nRows) {  // No non-zero found; move to next column
                        pivCol++;
                        continue;
                    }
                }
                pivots.at(pivRow) = pivCol;
                pivRow++;
                pivCol++;
            }
        } else {
            for (size_t idx{}; idx < nRows; idx++) {
                const auto& row = system.at(idx);
                for (size_t i{pivCol}; i < (nCols - 1); i++) {
                    pivCol++;
                    if (!fuzzyCompare(row.at(i), number(0))) {
                        pivots.at(pivRow) = i;
                        pivRow++;
                        break;
                    }
                }
            }
        }
        return pivots;
    }

    /**
     * @brief Gets the pivot row of a linear system.
     *
     * @param system The linear system to get the pivot row from.
     * @param startRow The row to start searching from.
     * @param col The column to search for the pivot in.
     * @return The pivot row if it exists, nullopt otherwise.
     */
    template <Number number, int m, int n>
    optional<size_t> getPivotRow(const LinearSystem<number, m, n>& system, size_t startRow, size_t col) {
        const auto& nRows = system.numRows();
        const auto& nCols = system.numCols();

        for (size_t row = startRow; row < nRows; ++row) {
            if (system.at(row).at(col) != 0) {
                return row;
            }
        }
        return nullopt;  // No pivot found in this column
    }

    /**
     * @brief Rearrange a linear system to have the most zeros in the bottom rows.
     *
     * @param system The linear system to rearrange.
     * @return The rearranged linear system.
     */
    template <Number number, int m, int n>
    LinearSystem<number, m, n> rearrangeSystem(const LinearSystem<number, m, n>& system) {
        auto getZeroCount = [](const Row<number, n>& row) {
            const auto& size = row.size();
            size_t mid = floor((size - 1) / 2.);
            size_t count{};
            for (size_t i{}; i < size - 1; i++)
                if (fuzzyCompare(row.at(i), number(0)) && i <= mid) count++;
            return count;
        };

        auto cmp = [](const SizeTPair& a, const SizeTPair& b) { return (a.second < b.second); };

        auto getRowSortedCounts = [&getZeroCount, &cmp](const LinearSystem<number, m, n> sys) {
            const size_t& nRows = sys.numRows();
            std::multimap<size_t, size_t> rowCounts{};
            vector<SizeTPair> res{};

            for (size_t i{}; i < nRows; i++) rowCounts.emplace(i, getZeroCount(sys.at(i)));
            auto it = rowCounts.begin();
            for (; it != rowCounts.end(); it++) res.emplace_back(it->first, it->second);
            std::sort(res.begin(), res.end(), cmp);

            return res;
        };

        LinearSystem<number, m, n> rearranged{system};
        auto rowCounts = getRowSortedCounts(system);
        for (size_t i{}; i < rowCounts.size(); i++) {
            const auto& [idx, count] = rowCounts.at(i);
            if (i != idx) {
                std::swap(rearranged.at(i), rearranged.at(idx));
            }
            rowCounts = getRowSortedCounts(rearranged);
        }
        return rearranged;
    }

    /**
     * @brief Find the identity matrix of a square linear system
     * @return  The identity matrix of the system.
     */
    template <Number number, int m>
    Matrix<number, m, m> I() {
        Matrix<number, m, m> identity{};
        for (size_t i{}; i < m; i++) {
            identity.at(i).at(i) = 1;
        }
        return identity;
    }

    /**
     * @brief Find the identity matrix of a square linear system
     *
     * @return  The identity matrix of the system.
     */
    template <Number number, int m>
    Matrix<number, m, m> I(int nRows) {
        if constexpr (m == -1) {
            Matrix<number, m, m> identity(nRows, nRows);
            for (int i{}; i < nRows; i++) {
                identity.at(i, i) = 1;
            }
            return identity;
        } else {
            return I<number, m>();
        }
    }

    /**
     * @brief Checks if a matrix is isHermitian.
     *
     * A matrix is Hermitian if it is equal to its own transpose in the case of
     * a real matrix or if it is equal to the conjugate transpose in the case of a complex matrix.
     *
     * @param A the matrix to check
     * @return true if the matrix is Hermitian, false otherwise.
     */
    template <Number number, int m, int n>
    bool isHermitian(const Matrix<number, m, n>& A) {
        const auto& [nR, nC] = A.shape();
        if (nR != nC) throw StackError<std::invalid_argument>{"Matrix must be square"};
        return A == helpers::extractMatrixFromTranspose(A.T());
    }

    /**
     * @brief Checks if a matrix is orthogonal.
     *
     * The columns of a matrix are orthogonal if the dot product of any two distinct columns is zero.
     * This can be checked by computing the product of the transpose of the matrix and the matrix itself,
     * which should yield the identity matrix if the columns are orthogonal.
     *
     * @param A the matrix to check
     * @return true if the matrix is orthogonal, false otherwise.
     */
    template <Number number, int m, int n>
    bool isOrthogonal(const Matrix<number, m, n>& A) {
        const auto& [nR, nC] = A.shape();
        const auto& ATA = helpers::extractMatrixFromTranspose(A.T()) * A;
        const auto& Id = I<number, n>(nC);

        auto minus = ATA - Id;
        const auto& [nR2, nC2] = minus.shape();
        for (size_t i{}; i < nR2; i++) {
            for (size_t j{}; j < nC2; j++) {
                if (!fuzzyCompare(abs(minus(i, j)), number(0))) return false;
            }
        }
        return true;
    }

    /**
     * @brief Checks if a vector is a zero vector.
     *
     * @param v The vector to check.
     * @return true if the vector is a zero vector, false otherwise.
     */
    template <Number number, int n>
    constexpr bool isZeroVector(const Vector<number, n>& v) {
        for (const auto& i : v) {
            if (!fuzzyCompare(i, number(0))) return false;
        }
        return true;
    }

    /**
     * @brief Checks if a matrix is upper triangular.
     *
     * @param A The matrix to check.
     * @return true if the matrix is upper triangular, false otherwise.
     */
    template <Number number, int m, int n>
    bool isUpperTriangular(const Matrix<number, m, n>& A) {
        const auto [nRows, nCols] = A.shape();

        // Check all elements below the main diagonal
        for (size_t i{}; i < nRows; ++i) {
            for (size_t j{}; j < std::min(i, nCols); ++j) {
                if (!fuzzyCompare(A(i, j), number(0))) {
                    return false;
                }
            }
        }

        return true;
    }

    template <Number number, int m, int n>
    LinearSystem<number, m, n> rref(const LinearSystem<number, m, n>& system, bool identity = true);

    /**
     * @brief Computes the column space of a matrix, returning the basis vectors.
     *
     * @param A The matrix to compute the column space of.
     * @return The column space of the matrix.
     */
    template <Number number, int m, int n>
    auto colspace(const Matrix<number, m, n>& A) {
        const auto [nR, nC] = A.shape();
        auto reduced = A;
        // If the matrix is not in reduced echelon form, compute it
        if (!isInReducedEchelonForm(reduced, getPivots(reduced))) {
            reduced = rref(A);
        }

        // Find the indicies pivot columns in the reduced matrix
        vector<size_t> pivotPos{};
        pivotPos.reserve(nC);
        for (size_t i{}; i < nR; i++) {
            size_t j{i};
            while (j < nC) {
                if (!fuzzyCompare(reduced(i, j), number(0))) {
                    pivotPos.emplace_back(j);
                    break;
                }
                j++;
            }
        }
        auto cols = std::move(A.colToVectorSet());
        vector<Vector<number, m>> res{};
        res.reserve(nC);
        // Find the corresponding columns in the original matrix
        for (const auto idx : pivotPos) res.emplace_back(cols.at(idx));

        return res;
    }

    /**
     * @brief Computes the rank of a matrix, which is the number of linearly independent columns.
     * i.e. the cardinality of the column space.
     *
     * @param A
     * @return
     */
    template <Number number, int m, int n>
    size_t colrank(const Matrix<number, m, n>& A) {
        const auto& cspace{colspace(A)};
        return cspace.size();
    }

    /**
     * @brief Computes the row space of a matrix, returning the basis vectors.
     *
     * @param A The matrix to compute the row space of.
     * @return The row space of the matrix.
     */
    template <Number number, int m, int n>
    auto rowspace(const Matrix<number, m, n>& A) {
        const auto [nR, nC] = A.shape();
        auto reduced = A;
        // If the matrix is not in reduced echelon form, compute it
        if (!isInReducedEchelonForm(reduced, getPivots(reduced))) {
            reduced = rref(A);
        }

        // Find the indicies pivot rows in the reduced matrix
        vector<size_t> pivotPos{};
        pivotPos.reserve(nR);
        for (size_t i{}; i < nR; i++) {
            size_t j{i};
            while (j < nC) {
                if (!fuzzyCompare(reduced(i, j), number(0))) {
                    pivotPos.emplace_back(i);
                    break;
                }
                j++;
            }
        }

        const auto& cols = A.rowToVectorSet();
        vector<Vector<number, n>> res{};
        res.reserve(nR);
        // Find the corresponding rows in the original matrix
        for (const auto idx : pivotPos) res.emplace_back(cols.at(idx));

        return res;
    }

    /**
     * @brief Computes the rank of a matrix, which is the number of linearly independent rows.
     * i.e. the cardinality of the row space.
     *
     * @param A The matrix to compute the rank of.
     * @return The rank of the matrix.
     */
    template <Number number, int m, int n>
    size_t rowrank(const Matrix<number, m, n>& A) {
        const auto& rspace{rowspace(A)};
        return rspace.size();
    }

    /**
     * @brief Determined if a linear system in the form
     * [A | b]
     * is underdetermined.
     *
     * Assumes the system is already in row echelon form.
     * @param sys The linear system to check.
     * @return true if the system is underdetermined, false otherwise.
     */
    template <Number number, int m, int n>
    bool isSystemUnderdetermined(const LinearSystem<number, m, n>& sys) {
        // Compute row rank of the system.
        size_t rank{rowrank(sys)};
        // The system is underdetermined if there are more unknowns than pivot rows.
        return rank < sys.numCols() - 1;
    }

    /**
     * @brief Find the identity matrix of a square linear system
     * @return  The identity matrix of the system.
     */
    template <Number number, int m>
    Matrix<number, m, m> I() {
        Matrix<number, m, m> identity{};
        for (size_t i{}; i < m; i++) {
            identity.at(i).at(i) = 1;
        }
        return identity;
    }

    /**
     * @brief Find the identity matrix of a square linear system
     *
     * @return  The identity matrix of the system.
     */
    template <Number number, int m>
    Matrix<number, m, m> I(int nRows) {
        if constexpr (m == -1) {
            Matrix<number, m, m> identity(nRows, nRows);
            for (int i{}; i < nRows; i++) {
                identity.at(i, i) = 1;
            }
            return identity;
        } else {
            return I<number, m>();
        }
    }

    /**
     * @brief Creates a zero matrix of the given size.
     *
     * @tparam m The number of rows in the matrix.
     * @tparam n The number of columns in the matrix.
     * @return  The zero matrix of the given size.
     */
    template <Number number, int m, int n>
    Matrix<number, m, n> matrixZeros() {
        return Matrix<number, m, n>{};
    }

    /**
     * @brief Creates a zero matrix of the given size.
     *
     * @tparam m The number of rows in the matrix.
     * @tparam n The number of columns in the matrix.
     * @param nRows The number of rows in the matrix.
     * @param nCols The number of columns in the matrix.
     * @return  The zero matrix of the given size.
     */
    template <Number number, int m, int n>
    Matrix<number, m, n> matrixZeros(int nRows, int nCols) {
        if constexpr (m == Dynamic || n == Dynamic) {
            return Matrix<number, m, n>(nRows, nCols);
        } else {
            return matrixZeros<number, m, n>();
        }
    }

    /**
     * @brief Creates a zero vector of the given size.
     *
     * @tparam n The size of the vector.
     * @return The zero vector of the given size.
     */
    template <Number number, int n>
    Vector<number, n> vectorZeros() {
        return Vector<number, n>{};
    }

    /**
     * @brief Creates a zero vector of the given size.
     *
     * @tparam n The size of the vector.
     * @param size
     */
    template <Number number, int n>
    Vector<number, n> vectorZeros(int size) {
        if constexpr (n == Dynamic) {
            return Vector<number, n>(size);
        } else {
            return vectorZeros<number, n>();
        }
    }

    /**
     * @brief Creates a matrix of ones of the given size.
     *
     * @tparam m The number of rows in the matrix.
     * @tparam n The number of columns in the matrix.
     * @return The matrix of ones of the given size.
     */
    template <Number number, int m, int n>
    Matrix<number, m, n> matrixOnes() {
        Matrix<number, m, n> res{};
        for (size_t i{}; i < m; i++)
            for (size_t j{}; j < n; j++) res(i, j) = 1;
        return res;
    }

    /**
     * @brief Creates a matrix of ones of the given size.
     *
     * @tparam m The number of rows in the matrix.
     * @tparam n The number of columns in the matrix.
     * @param nRows The number of rows in the matrix.
     * @param nCols The number of columns in the matrix.
     * @return The matrix of ones of the given size.
     */
    template <Number number, int m, int n>
    Matrix<number, m, n> matrixOnes(int nRows, int nCols) {
        if constexpr (m == Dynamic || n == Dynamic) {
            Matrix<number, m, n> res(nRows, nCols);
            for (size_t i{}; i < (size_t)nRows; i++)
                for (size_t j{}; j < (size_t)nCols; j++) res(i, j) = 1;
            return res;
        } else {
            return matrixOnes<number, m, n>();
        }
    }

    /**
     * @brief Creates a vector of ones of the given size.
     *
     * @tparam n The size of the vector.
     * @return The vector of ones of the given size.
     */
    template <Number number, int n>
    Vector<number, n> vectorOnes() {
        Vector<number, n> res{};
        for (size_t i{}; i < n; i++) res[i] = 1;
        return res;
    }

    /**
     * @brief Creates a vector of ones of the given size.
     *
     * @tparam n The size of the vector.
     * @param size The size of the vector.
     */
    template <Number number, int n>
    Vector<number, n> vectorOnes(int size) {
        if constexpr (n == Dynamic) {
            Vector<number, n> res(size);
            for (int i{}; i < size; i++) res[i] = 1;
            return res;
        } else {
            return vectorOnes<number, n>();
        }
    }

    /**
     * @brief Creates a random vector of the given size.
     *
     * @tparam n The size of the vector.
     * @param min The minimum value of the random numbers.
     * @param max The maximum value of the random numbers.
     * @param seed The seed for the random number generator.
     * @return The random vector of the given size.
     */
    template <Number num, int n>
    Vector<num, n> vectorRandom(int min = 0, int max = 100, Seed seed = std::nullopt) {
        Vector<num, n> vec{};
        for (int i{}; i < n; i++) {
            vec.at(i) = helpers::rng<num>(min, max, seed);
        }
        return vec;
    }

    /**
     * @brief Creates a random vector of the given size.
     *
     * @tparam n The size of the vector.
     * @param min The minimum value of the random numbers.
     * @param max The maximum value of the random numbers.
     * @param seed The seed for the random number generator.
     * @param size The size of the vector.
     * @return The random vector of the given size.
     */
    template <Number number, int n>
    Vector<number, n> vectorRandom(int size, int min = 0, int max = 100, Seed seed = std::nullopt) {
        if constexpr (n == Dynamic) {
            Vector<number, n> vec(size);
            for (int i{}; i < size; i++) {
                vec.at(i) = helpers::rng<number>(min, max, seed);
            }
            return vec;
        } else {
            return vectorRandom<number, n>(min, max, seed);
        }
    }

    /**
     * @brief Creates a random matrix of the given size.
     *
     * @tparam m The number of rows in the matrix.
     * @tparam n The number of columns in the matrix.
     * @param min The minimum value of the random numbers.
     * @param max The maximum value of the random numbers.
     * @param seed The seed for the random number generator.
     * @return The random matrix of the given size.
     */
    template <Number num, int m, int n>
    Matrix<num, m, n> matrixRandom(int min = 0, int max = 100, Seed seed = std::nullopt) {
        Matrix<num, m, n> res{};
        for (int i{}; i < m; i++) {
            for (int j{}; j < n; j++) res(i, j) = helpers::rng<num>(min, max, seed);
        }
        return res;
    }

    /**
     * @brief Creates a random matrix of the given size.
     *
     * @tparam m The number of rows in the matrix.
     * @tparam n The number of columns in the matrix.
     * @param numRows The number of rows in the matrix.
     * @param numCols The number of columns in the matrix.
     * @param min The minimum value of the random numbers.
     * @param max The maximum value of the random numbers.
     * @param seed The seed for the random number generator.
     * @return The random matrix of the given size.
     */
    template <Number number, int m, int n>
    Matrix<number, m, n> matrixRandom(int numRows, int numCols, int min = 0, int max = 100, Seed seed = std::nullopt) {
        if constexpr (n == Dynamic || m == Dynamic) {
            Matrix<number, m, n> res(numRows, numCols);
            for (int i{}; i < m; i++) {
                for (int j{}; j < n; j++) res(i, j) = helpers::rng<number>(min, max, seed);
            }
            return res;
        } else {
            return matrixRandom<number, m, n>(min, max, seed);
        }
    }

    /**
     * @brief Compute the LU decomposition of a square matrix A using the Schur complement method.
     *
     * @param A The square matrix to decompose.
     * @return A pair containing the lower triangular matrix L and the upper triangular matrix U such that \f$ A = LU
     * \f$.
     */
    template <Number number>
    auto LU(const Matrix<number, 1, 1>& A) {
        auto L = matrixZeros<number, 1, 1>(1, 1);
        auto U = A;
        L(0, 0) = 1;
        U(0, 0) = A(0, 0);
        return pair{L, U};
    }

    template <Number number, int n>
    auto LU(const Matrix<number, n, n>& A) {
        const auto [nR, nC] = A.shape();
        if (nR != nC) throw StackError<std::invalid_argument>{"Matrix A must be square"};
        const int numRows = nR;
        auto L = matrixZeros<number, n, n>(numRows, numRows);
        auto U = A;

        // Base case for 1x1 matrix
        if (nC == 1) {
            L(0, 0) = 1;
            U(0, 0) = A(0, 0);
            return pair{L, U};
        } else {
            const auto& a11 = A(0, 0);
            const auto& A12 = A.template slice<0, 1, 1, n>({0, 1}, {1, numRows});
            const auto& A21 = A.template slice<1, n, 0, 1>({1, numRows}, {0, 1});
            const auto& A22 = A.template slice<1, n, 1, n>({1, numRows}, {1, numRows});

            L(0, 0) = 1;
            U(0, 0) = a11;

            // Calculate U12
            for (int j = 1; j < numRows; ++j) {
                U(0, j) = A(0, j);
            }

            // Calculate L21
            for (int i = 1; i < numRows; ++i) {
                L(i, 0) = A(i, 0) / (a11 + EPSILON);
                // Set the corresponding U values to 0 for lower triangular portion
                U(i, 0) = 0;
            }

            // Calculate the Schur complement: A22 - L21 * U12
            auto S = A22;
            for (int i = 1; i < numRows; ++i) {
                for (int j = 1; j < numRows; ++j) {
                    S(i - 1, j - 1) = A(i, j) - L(i, 0) * U(0, j);
                }
            }
            const auto& S22{S};

            // Recursive LU decomposition on the Schur complement
            if (numRows > 2) {
                const auto& [L22, U22] = LU(S22);

                // Copy the subL and subU into the appropriate positions of L and U
                for (int i = 0; i < numRows - 1; ++i) {
                    for (int j = 0; j < numRows - 1; ++j) {
                        L(i + 1, j + 1) = L22(i, j);
                        U(i + 1, j + 1) = U22(i, j);
                    }
                }
            } else if (numRows == 2) {
                // For a 2x2 matrix, the Schur complement is a 1x1 matrix
                L(1, 1) = 1;
                U(1, 1) = S22(0, 0);
            }

            return pair{L, U};
        }
    }

    enum class QRType : uint8_t { Full, Thin };

    template <QRType type, Number number, int m, int n>
    using QType = std::conditional_t<type == QRType::Full, Matrix<number, m, m>, Matrix<number, m, std::min(m, n)>>;

    template <QRType type, Number number, int m, int n>
    using RType = std::conditional_t<type == QRType::Full, Matrix<number, m, n>, Matrix<number, std::min(m, n), n>>;

    template <QRType type, Number number, int m, int n>
    using QRPair = pair<QType<type, number, m, n>, RType<type, number, m, n>>;

    /**
     * @brief Extend a set of orthonormal vectors to form a complete orthonormal basis.
     *
     * Given k orthonormal vectors in R^m (where k < m), this function finds
     * additional (m-k) orthonormal vectors to complete the basis for R^m.
     *
     * @param orthonormalVectors The existing orthonormal vectors (k vectors in R^m)
     * @param dimension The target dimension m
     * @return Complete set of m orthonormal vectors
     */
    template <Number number, int m>
    vector<Vector<number, m>> extendToCompleteBasis(const vector<Vector<number, m>>& qs, size_t dim) {
        vector<Vector<number, m>> basis = qs;
        auto n = qs.size();

        // We need (dimension - currentSize) more vectors
        for (size_t i = n; i < dim; ++i) {
            // Start with a standard basis vector
            Vector<number, m> candidate(dim);

            // Try each standard basis vector until we find one that's not
            // in the span of existing vectors
            bool found = false;
            for (size_t basisIdx = 0; basisIdx < dim && !found; ++basisIdx) {
                // Create e_basisIndex (standard basis vector)
                candidate = Vector<number, m>{};
                candidate[basisIdx] = number(1);

                // Orthogonalize against all existing vectors using Gram-Schmidt
                Vector<number, m> orth = candidate;
                for (const auto& v : basis) {
                    number proj = orth.dot(v);
                    orth -= proj * v;
                }

                // Check if the result is non-zero (not in span of existing vectors)
                number norm = orth.length();
                if (!fuzzyCompare(norm, number(0))) {
                    // Normalize and add to basis
                    basis.emplace_back(orth / norm);
                    found = true;
                }
            }

            if (!found) {
                throw StackError<std::logic_error>{
                    "Failed to extend to complete orthonormal basis - this shouldn't happen"};
            }
        }

        return basis;
    }

    /**
     * @brief Find an orthonormal basis for the column space of a matrix using the Gram-Schmidt process.
     *
     * Using the iteration:
     *
     * \f[
     *  \mathbf{v}_p  = \mathbf{x}_p - \frac{\mathbf{x}_p\cdot \mathbf{v}_1}{\mathbf{v}_1\cdot \mathbf{v}_1}
     * \mathbf{v}_1 -
     * \frac{\mathbf{x}_p \cdot  \mathbf{v}_2}{\mathbf{v}_2\cdot \mathbf{v}_2} \mathbf{v}_2 - \ldots -
     * \frac{\mathbf{x}_p
     * \cdot
     *   \mathbf{v}_{p-1} }{\mathbf{v}_{p-1} \cdot \mathbf{v}_{p-1}} \mathbf{v}_{p-1}
     * \f]
     * Or in summation notation:
     * \f[
     *  v_p
     *     = x_p
     *     - \sum_{j=1}^{p-1} r_{j,p}\,v_j
     * \f]
     * Where \f[r_{j,p} = \frac{x_p \cdot v_j}{v_j \cdot v_j}\f]
     *
     * @param A The matrix to find the orthonormal basis for.
     * @param R Optional output matrix to store the upper triangular matrix from the Gram-Schmidt process.
     * @return A vector of orthonormal vectors that form the basis for the column space of A.
     */
    template <QRType type, Number number, int m, int n>
    vector<Vector<number, m>> GSOrth(const Matrix<number, m, n>& A, RType<type, number, m, n>* R = nullptr) {
        const auto& [nRows, nCols] = A.shape();
        const auto& asCols = A.colToVectorSet();

        vector<Vector<number, m>> qs;
        qs.reserve(nCols);

        const auto& v0 = asCols[0];
        const auto& norm0 = v0.length();
        qs.push_back(v0 / norm0);
        if (R) {
            R->at(0, 0) = norm0;
        }

        for (size_t i{1}; i < nCols; ++i) {
            auto vi = asCols[i];

            for (size_t j{}; j < i; ++j) {
                number rji = vi.dot(qs[j]);
                if (R) {
                    R->at(j, i) = rji;
                }
                vi -= rji * qs[j];
            }

            const auto& norm = vi.length();
            if (fuzzyCompare(norm, number(0)))
                throw StackError<std::logic_error>{"Matrix A is singular or has linearly dependent columns"};

            if (R) {
                R->at(i, i) = norm;
            }
            qs.push_back(vi / norm);
        }

        if constexpr (type == QRType::Full) {
            auto fullRows = std::min(nRows, nCols);
            if (nRows > nCols) {
                // If the matrix is full, we need to add orthogonal vectors to fill the remaining rows
                const auto& completeBasis = extendToCompleteBasis<number, m>(qs, nRows);
                qs.clear();
                qs.reserve(nRows);
                for (const auto& vec : completeBasis) {
                    qs.emplace_back(vec);
                }

                if (R)
                    for (size_t i{nCols}; i < fullRows; ++i) R->at(i, i) = 0;
            }
        }

        return qs;
    }

    /**
     * @brief Compute the QR decomposition of a square matrix A using the Gram-Schmidt process.
     *
     * We factorize \f$A=Q\,R\f$ by orthonormalizing the columns \f$\{x_1,\dots,x_n\}\f$ of \f$A\f$:
     *
     * \f[
     *   v_p
     *     = x_p
     *     - \sum_{j=1}^{p-1} r_{j,p}\,v_j,
     *   \quad\text{where}\quad
     *   r_{j,p} = \frac{x_p \cdot v_j}{v_j \cdot v_j},
     * \f]
     *
     * then normalize
     *
     * \f[
     *   q_p = \frac{v_p}{\|v_p\|},
     *   \quad
     *   r_{p,p} = \|v_p\|.
     * \f]
     *
     * The orthonormal vectors \f$\{q_1,\dots,q_n\}\f$ form the columns of \f$Q\f$, and
     * all coefficients \f$r_{j,p}\f$ (for \f$j\le p\f$) assemble into the upper‑triangular matrix \f$R\f$.
     * Hence,
     *
     * \f[
     *   A = Q\,R,\quad
     *   Q^\top Q = I,\quad
     *   R_{j,i} = 0\quad\text{for }j>i.
     * \f]
     *
     * @param A The square matrix to decompose.
     * @return A pair containing the orthogonal matrix Q and the upper triangular matrix R.
     */
    template <QRType type, Number number, int m, int n>
    QRPair<type, number, m, n> gsQR(const Matrix<number, m, n>& A) {
        const auto [nR, nC] = A.shape();

        if constexpr (type == QRType::Full) {
            auto R = RType<type, number, m, n>(nR, nC);

            const auto& qs = GSOrth<type>(A, &R);
            // If the matrix is full, we need to create a square Q matrix
            auto Q = I<number, m>(nR);
            for (size_t i{}; i < nC; ++i) {
                for (size_t j{}; j < nR; ++j) {
                    Q(j, i) = qs[i][j];
                }
            }
            return {Q, R};
        } else {
            auto R = RType<type, number, m, n>(nR, std::min(nR, nC));

            const auto& qs = GSOrth<type>(A, &R);
            // For thin QR decomposition, we can return the orthogonal matrix directly
            auto Q = QType<type, number, m, n>(nR, std::min(nR, nC));
            for (size_t i{}; i < nC; ++i) {
                for (size_t j{}; j < nR; ++j) {
                    Q(j, i) = qs[i][j];
                }
            }
            return {Q, R};
        }
    }

    /**
     * @brief Find the Householder matrix for a given vector.
     *
     * The Householder matrix is used to reflect a vector across a hyperplane. And is
     * defined as:
     *
     * \f[
     * P = I - 2 \frac{\mathbf{v}\mathbf{v}^T}{\mathbf{v}^T \mathbf{v}}
     * \f]
     *
     * @param v The vector to find the Householder matrix for.
     * @return The Householder matrix.
     */
    template <Number number, int n>
    Matrix<number, n, n> houseHolder(const Vector<number, n>& v) {
        if (isZeroVector(v))
            throw StackError<std::logic_error>{"Vector v cannot be the zero vector"};  // v cannot be the zero vector
        const auto size = v.size();
        const auto& vT = v.T();
        auto P{I<number, n>(size)};
        // P = I - 2/v^T v * (v v^T)
        P = P - (2 / (vT * v).at(0)) * (v * vT);
        return P;
    }

    /**
     * @brief Compute the Householder reduction of a linear system.
     *
     * Applies a sequence of Householder reflections to zero out sub‑diagonal entries of the input matrix,
     * transforming it into an upper‑triangular (or upper‑Hessenberg) form.  For an m×n matrix \f$A\f$, we
     * construct at each step \f$i=1,\dots,\min(m,n)\f$ a vector
     *
     * \f[
     *   u_i = x_i - \|x_i\|\,e_1,\quad
     *   H_i = I - 2\,\frac{u_i\,u_i^T}{u_i^T u_i},
     * \f]
     *
     * where \f$x_i\f$ is the \f$i\f$th column of the current working matrix below the diagonal, and \f$e_1\f$
     * is the first standard basis vector of appropriate dimension.  We then update
     *
     * \f[
     *   A^{(i+1)} = H_i\,A^{(i)},
     * \quad
     *   Q^{(i+1)} = Q^{(i)}\,H_i^T,
     * \f]
     *
     * accumulating \f$Q = H_1^T H_2^T \cdots H_k^T\f$ so that ultimately
     *
     * \f[
     *   Q^T A = R
     * \f]
     *
     * with \f$Q\f$ orthogonal and \f$R\f$ upper‑triangular (or upper‑Hessenberg if \f$m>n\f$).
     *
     * @param system  The input linear system matrix \f$A\in\mathbb{R}^{m\times n}\f$ to be reduced.
     * @return A std::pair containing
     *         - \f$Q\in\mathbb{R}^{m\times m}\f$: the orthogonal matrix from accumulated reflections,
     *         - \f$R\in\mathbb{R}^{m\times n}\f$: the resulting upper‑triangular (or upper‑Hessenberg) matrix.
     */
    template <Number number, int m, int n>
    auto houseHolderRed(const LinearSystem<number, m, n>& system) {
        const auto [numRows, numCols] = system.shape();
        const size_t nR = numRows;
        const size_t nC = numCols;

        auto B{system};
        auto U = I<number, m>(numRows);
        auto V = I<number, n>(numCols);

        const size_t p{std::min(nR, nC)};
        for (size_t k{}; k < p; k++) {
            // --------------------------------------------------------------------
            // Left house holder reduction: zero out below the diagonal in column k
            // --------------------------------------------------------------------
            Vector<number, Dynamic> x(numRows - k);
            for (size_t i{k}; i < nR; i++) {
                x[i - k] = B(i, k);
            }

            // Compute reflector if x is not already zero
            if (!isZeroVector(x)) {
                // Determine norm and sign to avoid cancellation
                auto normX = x.length();
                number sign = (fuzzyCompare(x[0], number(0)) || x[0] > number(0)) ? number(1) : number(-1);
                // Form the Householder vector
                x[0] = x[0] + sign * normX;
                auto v = x;
                // Small Householder matrix of size (numRows-k)x(numRows-k)
                auto H = houseHolder(v);

                // Generate identity of full size and replace the lower right block with the
                // small Householder matrix
                auto Qk = I<number, m>(numRows);
                for (size_t i{k}; i < nR; i++)
                    for (size_t j{k}; j < nR; j++) Qk(i, j) = H(i - k, j - k);

                B = Qk * B;
                U = U * Qk;
            }

            // -----------------------------------------------------------------------
            // Right house holder reduction: zero out above the superdiagonal in row k
            // -----------------------------------------------------------------------
            if (static_cast<int>(k) < static_cast<int>(numCols - 1)) {
                Vector<number, Dynamic> x(numCols - k - 1);
                for (size_t j{k + 1}; j < numCols; j++) x[j - k - 1] = B(k, j);

                if (!isZeroVector(x)) {
                    auto normX = x.length();
                    number sign = (fuzzyCompare(x[0], number(0)) || x[0] > number(0)) ? number(1) : number(-1);
                    x[0] = x[0] + sign * normX;
                    auto v = x;
                    auto H = houseHolder(v);  // size (numCols-k-1)x(numCols-k-1)

                    // Embed H_small into an identity matrix for the full column range
                    auto Qk = I<number, n>(numCols);
                    for (size_t i{k + 1}; i < numCols; i++)
                        for (size_t j{k + 1}; j < numCols; j++) Qk(i, j) = H(i - k - 1, j - k - 1);

                    B = B * Qk;
                    V = V * Qk;
                }
            }
        }
        return std::tuple{U, B, V};
    }

    /**
     * @brief Computes the QR decomposition of a matrix A using the Householder transformation.
     *
     * The Householder transformation is used to zero out the elements below the diagonal of each column,
     * resulting in an upper triangular matrix R. The orthogonal matrix Q is built up from the Householder reflectors.
     *
     * @param A
     * @return
     */
    template <QRType type, Number number, int m, int n>
    QRPair<type, number, m, n> houseHolderQR(const Matrix<number, m, n>& A) {
        const auto [numRows, numCols] = A.shape();
        auto R = A;                      // will become upper‑triangular
        auto Q = I<number, m>(numRows);  // accumulate left reflectors

        const size_t p = std::min(numRows, numCols);
        for (size_t k = 0; k < p; ++k) {
            // ---------------------------------------------------
            // Left Householder step: zero out below-diagonal in col k
            // ---------------------------------------------------
            Vector<number, Dynamic> x(numRows - k);
            for (size_t i = k; i < numRows; ++i) {
                x[i - k] = R(i, k);
            }

            if (!isZeroVector(x)) {
                auto normX = x.length();
                number sign = (fuzzyCompare(x[0], number(0)) || x[0] > number(0)) ? number(1) : number(-1);

                x[0] += sign * normX;
                auto v = x;
                auto H = houseHolder(v);  // small (numRows-k)×(numRows-k)

                // embed H into full-size identity Qk
                auto Qk = I<number, m>(numRows);
                for (size_t i = k; i < numRows; ++i)
                    for (size_t j = k; j < numRows; ++j) Qk(i, j) = H(i - k, j - k);

                // apply reflector on the left
                R = Qk * R;
                Q = Q * Qk;
            }
        }

        if constexpr (type == QRType::Full) {
            return {Q, R};
        } else {
            // Extract thin QR decomposition
            const auto rank = std::min(numRows, numCols);

            // QThin: m×min(m,n) - first min(m,n) columns of Q
            auto QThin = QType<type, number, m, n>(numRows, rank);
            for (size_t i{}; i < numRows; ++i) {
                for (size_t j{}; j < rank; ++j) {
                    QThin(i, j) = Q(i, j);
                }
            }

            // RThin: min(m,n)×n - first min(m,n) rows of R
            auto RThin = RType<type, number, m, n>(rank, numCols);
            for (size_t i{}; i < rank; ++i) {
                for (size_t j{}; j < numCols; ++j) {
                    RThin(i, j) = R(i, j);
                }
            }

            return std::pair{QThin, RThin};
        }
    }

    enum class QRMethod : uint8_t {
        GramSchmidt,
        Householder,
    };

    /**
     * @brief Computes the QR decomposition of a matrix A using the specified method.
     *
     * @param A  The matrix to decompose.
     * @param method  The method to use for QR decomposition (default is Gram-Schmidt).
     * @return A pair containing the orthogonal matrix Q and the upper triangular matrix R.
     */
    template <QRType type, Number number, int m, int n>
    QRPair<type, number, m, n> QR(const Matrix<number, m, n>& A, QRMethod method = QRMethod::GramSchmidt) {
        switch (method) {
            case QRMethod::GramSchmidt:
                return gsQR<type>(A);
            case QRMethod::Householder:
                return houseHolderQR<type>(A);
            default:
                throw StackError<std::invalid_argument>{"Unknown QR method"};
        }
    }

    /**
     * @brief Checks if a matrix is singular, using LU decomposition to check if any
     * of the diagonal elements of U are zero, i.e. indicating the determinant is zero.
     *
     * @param A
     * @return
     */
    template <Number number, int n>
    bool isSingular(const Matrix<number, n, n>& A) {
        const auto [nR, nC] = A.shape();
        if (nR != nC) throw StackError<std::invalid_argument>{"Matrix A must be square"};

        const auto& [L, U] = LU(A);

        for (int i = 0; i < nR; ++i) {
            const auto& abs = std::abs(U(i, i));
            if (fuzzyCompare(abs, number(0)) || abs < EPSILON) return true;
            // if (std::abs(U(i, i)) < EPSILON_FIXED) return true;
        }
        return false;
    }

    /**
     * @brief Find the row echelon form of a square linear system
     *
     * @param system The linear system to find the reduced row echelon form of.
     * @return The reduced row echelon form of the system.
     */
    template <Number number, int m, int n>
    LinearSystem<number, m, n> refSq(const LinearSystem<number, m, n>& sys) {
        LinearSystem<number, m, n> system{sys};
        const size_t& nCols = system.numCols();

        for (size_t j{}; j < nCols; j++) {
            if (fuzzyCompare(system.at(j, j), number(0))) {
                number big{abs(system.at(j, j))};
                size_t kRow{j};

                for (size_t k{j + 1}; k < (nCols - 1); k++) {
                    auto val = abs(system.at(k, j));
                    if (val > big) {
                        big = val;
                        kRow = k;
                    }
                }

                if (kRow != j) {
                    auto temp = system.at(j);
                    system.at(j) = system.at(kRow);
                    system.at(kRow) = temp;
                }
            }

            auto pivot = system.at(j, j);

            if (fuzzyCompare(pivot, number(0))) {
                continue;
            }

            for (size_t i{j + 1}; i < nCols; i++) {
                auto permute = system.at(i, j) / pivot;
                system.at(i) -= permute * system.at(j);
            }
        }
        return system;
    }

    /**
     * @brief Find the row echelon form of a rectangular linear system
     *
     * @param system The linear system to find the row echelon form of.
     * @return The row echelon form of the system.
     */
    template <Number number, int m, int n>
    LinearSystem<number, m, n> refRec(const LinearSystem<number, m, n>& sys) {
        LinearSystem<number, m, n> system{sys};

        const size_t& nRows = system.numRows();
        const size_t& nCols = system.numCols();

        size_t pivRow{};
        size_t pivCol{};
        while (pivRow < nRows && pivCol < nCols) {
            // Check if current pivot is (effectively) zero
            if (fuzzyCompare(system.at(pivRow, pivCol), number(0))) {
                // Find the first row below with a non-zero entry in this column
                size_t kRow = pivRow;
                for (kRow = pivRow + 1; kRow < nRows; kRow++) {
                    if (!fuzzyCompare(system.at(kRow, pivCol), number(0))) {
                        break;
                    }
                }

                if (kRow == nRows) {  // No non-zero found; move to next column
                    pivCol++;
                    continue;
                } else {  // Swap rows to bring the non-zero entry up
                    auto temp = system.at(pivRow);
                    system.at(pivRow) = system.at(kRow);
                    system.at(kRow) = temp;
                }
            }

            for (size_t i{pivRow + 1}; i < nRows; i++) {
                auto permute = system.at(i, pivCol) / system.at(pivRow, pivCol);
                system.at(i) -= permute * system.at(pivRow);
            }

            pivRow++;
            pivCol++;
        }
        return system;
    }

    /**
     * @brief Find the row echelon form of a linear system
     *
     * Chooses the appropriate ref function based on the dimensions of the system.
     *
     * @param system The linear system to find the row echelon form of.
     * @return The row echelon form of the system.
     */
    template <Number number, int m, int n>
    LinearSystem<number, m, n> ref(const LinearSystem<number, m, n>& system) {
        const auto& nRows = system.numRows();
        const auto& nCols = system.numCols();

        if (nRows == nCols && nRows > 5) return refRec(system);
        if (nRows == nCols) return refSq(system);
        return refRec(system);
    }

    /**
     * @brief Find the reduced row echelon form of a rectangular linear system
     *
     * @param system The linear system to find the reduced row echelon form of.
     * @param identity Whether to convert the system to identity form.
     * @return The reduced row echelon form of the system.
     */
    template <Number number, int m, int n>
    LinearSystem<number, m, n> rrefRec(const LinearSystem<number, m, n>& sys, bool identity = true) {
        LinearSystem<number, m, n> system{sys};

        int nRows = static_cast<int>(system.numRows());

        auto pivots = getPivots(system);
        if (!isInEchelonForm(system, pivots)) {
            system = ref(system);
        }
        auto pivotLocs = getPivotLocations(system);

        // Iterate over each row that has a pivot.
        for (int i = nRows - 1; i >= 0; i--) {
            auto pivotLoc = pivotLocs.at(i);
            if (!pivotLoc.has_value()) continue;
            auto pivCol = pivotLoc.value();

            // Normalize the pivot row if identity is desired.
            number pivotVal = system.at(i, pivCol);
            if (identity && !fuzzyCompare(pivotVal, number(1))) {
                system.at(i) /= pivotVal;
            }

            // Eliminate the pivot column from all rows above.
            for (int row = 0; row < i; row++) {
                if (!pivotLocs.at(i).has_value()) continue;
                number permute = system.at(row, pivotLocs.at(i).value());
                if (!fuzzyCompare(permute, number(0))) {
                    system.at(row) -= permute * system.at(i);
                }
            }
        }

        return system;
    }

    /**
     * @brief Find the reduced row echelon form of a square linear system
     *
     * @param system The linear system to find the reduced row echelon form of.
     * @param identity Whether to convert the system to identity form.
     * @return The reduced row echelon form of the system.
     */
    template <Number number, int m, int n>
    LinearSystem<number, m, n> rrefSq(const LinearSystem<number, m, n>& sys, bool identity = true) {
        LinearSystem<number, m, n> system{sys};

        const auto& nCols = static_cast<int>(system.numCols());

        RowOptional<number, m> pivots = getPivots(system);
        if (!isInEchelonForm(system, pivots)) system = ref(system);

        for (int j{nCols - 1}; j >= 0; j--) {
            if (!fuzzyCompare(system.at(j, j), number(0))) {
                if (identity) system.at(j) *= number(1) / system.at(j, j);

                for (int i(j - 1); i >= 0; i--) {
                    auto permute = system.at(i, j) / system.at(j, j);
                    system.at(i) -= permute * system.at(j);
                }
            }
        }

        return system;
    }

    /**
     * @brief Find the reduced row echelon form of a linear system
     *
     * Chooses the appropriate rref function based on the dimensions of the system.
     *
     * @param system The linear system to find the reduced row echelon form of.
     * @param identity Whether to convert the system to identity form.
     * @return The reduced row echelon form of the system.
     */
    template <Number number, int m, int n>
    LinearSystem<number, m, n> rref(const LinearSystem<number, m, n>& system, bool identity) {
        const auto& nRows = system.numRows();
        const auto& nCols = system.numCols();

        if (nRows == nCols && nRows > 5) return rrefRec(system, identity);
        if (nRows == nCols) return rrefSq(system, identity);
        return rrefRec(system, identity);
    }

    template <Number number, int m, int n>
    auto findSolutions(const LinearSystem<number, m, n>& sys) -> ConditionalOptionalRowOptional<number, m, n>;

    /**
     * @brief Find the solutions to a matrix equation in the form:
     *
     * \f[
     * A \mathbf{x} = \mathbf{b}
     * \f]
     *
     * @param A The matrix of the equation
     * @param b The vector of the equation
     * @return The solutions x to the equation
     */
    template <Number number, int m, int n>
    auto findSolutions(const Matrix<number, m, n>& A, const Vector<number, m>& b) {
        auto system = A.augment(b);
        return findSolutions(system);
    }

    /**
     * @brief Find the solutions to a linear system in the form:
     *
     * \f[
     * \begin{bmatrix}
     * A & \mathbf{b}
     * \end{bmatrix}
     * \]
     * \f]
     *
     * @param system The linear system to find the solutions of.
     * @return The solutions to the system if they exist, nullopt otherwise.
     */
    template <Number number, int m, int n>
    auto findSolutions(const LinearSystem<number, m, n>& sys) -> ConditionalOptionalRowOptional<number, m, n> {
        const auto [numRows, numCols] = sys.shape();
        ConditionalRowOptional<number, m, n> solutions(numCols - 1);
        LinearSystem<number, m, n> system{sys};
        system = rearrangeSystem(system);

        auto reducedEchelon = rref(system);

        if (isInconsistent(reducedEchelon)) return nullopt;

        // TODO: Check for underdetermined systems, find the general solution and calculate the values
        // that satisfy the system with the minimum-norm solution.
        if (isSystemUnderdetermined(reducedEchelon)) {
            throw StackError<std::runtime_error>(
                "Finding the solutions to an underdetermined system is not implemented");
        } else {
            for (auto i{solutions.rbegin()}; i != solutions.rend(); i++) {
                try {
                    auto index = std::distance(solutions.rbegin(), i);
                    solutions.at(index) = solveEquation(reducedEchelon.at(index), index, solutions);
                } catch (const std::out_of_range& e) {
                    continue;
                }
            }
        }
        return solutions;
    }

    /**
     * @brief Solve an upper triangular system of equations
     *
     * @param sys The upper triangular system of equations to solve.
     * @param b  The right-hand side vector of the system.
     * @return The solution vector x to the system.
     */
    template <Number number, int m, int n>
    Vector<number, m> solveUpperTriangular(const Matrix<number, m, n>& sys, const Vector<number, n>& b) {
        const auto& A = sys;
        const auto [numRows, numCols] = A.shape();
        const int nR = numRows;
        if (!isUpperTriangular(A))
            throw StackError<std::invalid_argument>("Matrix A must be upper triangular for this solve");

        Vector<number, m> x(nR);
        number back{};
        for (int i{nR - 1}; i >= 0; --i) {
            back = 0;
            for (int j{i + 1}; j <= nR - 1; ++j) {
                back += x[j] * (A(i, j));
            }
            x[i] = (b[i] - back) / (A(i, i));
        }
        return x;
    }

    /**
     * @brief Solve a linear system of equations using QR decomposition.
     *
     * @param A The matrix of the linear system.
     * @param b The right-hand side vector of the linear system.
     * @param method The method to use for QR decomposition (default is Gram-Schmidt).
     * @return The solution vector x to the system.
     */
    template <Number number, int m, int n>
    Vector<number, m> solveQR(const Matrix<number, m, n>& A, const Vector<number, n>& b, QRMethod method) {
        const auto [nR, nC] = A.shape();
        const auto bSize = b.size();
        if (nR != bSize) throw StackError<invalid_argument>("The matrix and vector are incompatible");

        const auto& [Q, R] = QR<QRType::Thin>(A, method);
        const auto& rhs = Q * b;
        const auto x = solveUpperTriangular(R, rhs);
        return x;
    }

    template <Number number, int m, int n>
    auto nulspace(const Matrix<number, m, n>& A) {
        const auto [nR, nC] = A.shape();
        const auto& zero{vectorZeros<number, m>((int)nR)};
        const auto& x0 = findSolutions(A, zero);
        if (!x0.has_value()) throw StackError<runtime_error>{"Cannot solve Ax = 0"};
        const auto& x = extractSolutionVector(x0.value());
        return x;
    }

    /**
     * @brief Find the inverse of a square linear system
     *
     * @param system The linear system to find the inverse of.
     * @return The inverse of the system if it exists, nullopt otherwise.
     */
    template <Number number, int m>
    optional<Matrix<number, m, m>> inverse(const LinearSystem<number, m, m>& system) {
        auto det = system.det();
        if (fuzzyCompare(det, number(0))) return nullopt;
        const auto& [nRows, nCols] = system.shape();
        if (nRows == 2)
            return (1. / det) * Matrix<number, m, m>{{system.at(1).at(1), -system.at(0).at(1)},
                                                     {-system.at(1).at(0), system.at(0).at(0)}};
        else {
            auto identity = I<number, m>(nRows);
            auto augmented = system.augment(identity);
            auto rrefAug = rref(augmented);
            auto inv = rrefAug.colToVectorSet();
            inv.erase(inv.begin(), inv.begin() + nRows);
            return mlinalg::structures::helpers::fromColVectorSet<number, m, m>(inv);
        }
    }

    /**
     * @brief Find the diagonal of a square matrix.
     *
     * @param matrix The matrix to find the diagonal of.
     * @return The diagonal of the matrix.
     */
    template <Number number, int n>
    vector<number> diag(const Matrix<number, n, n>& matrix) {
        const auto [nR, nC] = matrix.shape();
        if (nR != nC) throw StackError<runtime_error>("Matrix must be square to find a diagonal");

        vector<number> res;
        res.reserve(nR);
        for (size_t i{}; i < nR; i++) {
            res.push_back(matrix(i, i));
        }
        return res;
    }

    /**
     * @brief Create a diagonal matrix with the given entries on the diagonal.
     *
     * @param a The value to fill the diagonal with.
     * @return A diagonal matrix with the given value on the diagonal.
     */
    template <int n, Number number>
    Matrix<number, n, n> diagonal(number a) {
        Matrix<number, n, n> res{n, n};
        size_t i{};
        while (i < n) {
            res(i, i) = a;
            i++;
        }
        return res;
    }

    /**
     * @brief Create a diagonal matrix with the given entries on the diagonal.
     *
     * @param entries The entries to fill the diagonal with.
     * @return A diagonal matrix with the given entries on the diagonal.
     */
    template <int n, Number number>
    Matrix<number, n, n> diagonal(const std::initializer_list<number>& entries) {
        Matrix<number, n, n> res{n, n};
        size_t i{};
        for (const auto& entry : entries) {
            if (i >= n) throw StackError<std::out_of_range>{"Too many entries for diagonal matrix"};
            res(i, i) = entry;
            i++;
        }
        return res;
    }

    /**
     * @brief Create a diagonal matrix with the given entries on the diagonal.
     *
     * @tparam Itr The iterator type for the entries.
     * @param begin The beginning iterator for the entries.
     * @param end  The ending iterator for the entries.
     * @return A diagonal matrix with the given entries on the diagonal.
     */
    template <int n, Number number, typename Itr>
    Matrix<number, n, n> diagonal(Itr begin, Itr end) {
        auto dist = std::distance(begin, end);
        if (n != dist) throw StackError<std::out_of_range>{"Too many entries for diagonal matrix"};
        Matrix<number, n, n> res(n, n);

        size_t i{};
        for (auto itr{begin}; itr != end; itr++) {
            res(i, i) = *itr;
            i++;
        }
        return res;
    }

    /**
     * @brief Create a diagonal matrix with the given entries on the diagonal.
     *
     * @param entries The entries to fill the diagonal with.
     * @return A diagonal matrix with the given entries on the diagonal.
     */
    template <int n, Number number>
    Matrix<number, n, n> diagonal(const array<number, n>& entries) {
        Matrix<number, n, n> res{n, n};
        for (size_t i{}; i < n; i++) {
            res(i, i) = entries[i];
        }
        return res;
    }

    template <Number number, int n>
    auto eigenQR(const Matrix<number, n, n>& A, size_t iters = 10'000) {
        const auto [nR, nC] = A.shape();
        const int numRows = nR;
        const int numCols = nC;
        if (numRows != numCols) throw StackError<std::invalid_argument>{"Matrix A must be square"};
        if (isSingular(A)) throw StackError<std::invalid_argument>{"Matrix A is singular"};

        auto Ai = A;
        Matrix<number, n, n> Q;
        Matrix<number, n, n> R;
        auto Qprod{diagonal<n>(number(1))};
        for (size_t i{1}; i < iters; i++) {
            const auto& res = QR<QRType::Full>(Ai);
            Q = std::move(res.first);
            R = std::move(res.second);
            Qprod = Qprod * Q;

            Ai = R * Q;

            if (isUpperTriangular(A)) break;
        }
        logging::log(format("Qprod -> {}", Qprod), "eigenQR");

        const auto& values = diag(Ai);
        auto vectors{Qprod.colToVectorSet()};

        return pair{values, vectors};
    }

    /**
     * @brief Compute the Singular Value Decomposition (SVD) of a matrix using the eigenvalue decomposition of
     * \f$A^TA\f$.
     *
     * We factorize \f$A = U\,\Sigma\,V^T\f$ in three main steps:
     *
     * 1. Eigen‑decompose \f$A A^T\f$
     *
     *    Compute
     *    \f[
     *      A A^T = U \,\Lambda\, U^T,
     *    \f]
     *    where \f$\Lambda = \text{diag}(\lambda_1,\dots,\lambda_r)\f$ are the eigenvalues
     *    and the columns of \f$U = [\,u_1\,\dots\,u_r\,]\f$ are the corresponding orthonormal eigenvectors.
     *
     * 2. Construct \f$\Sigma\f$ and \f$V\f$
     *
     *    Let \f$\sigma_i = \sqrt{\lambda_i}\f$ and build
     *    \f[
     *      \Sigma = \text{diag}(\sigma_1,\dots,\sigma_r),
     *    \f]
     *    then compute each column \f$v_i\f$ of \f$V\f$ by
     *    \f[
     *      v_i = \frac{A^T\,u_i}{\sigma_i},
     *    \f]
     *    so that \f$V = [\,v_1\,\dots\,v_r\,]\f$ is orthonormal.
     *
     * 3. Construct \f$U\f$
     *
     *    The matrix \f$U\f$ is simply the collection of eigenvectors \f$\{u_i\}\f$ from step 1,
     *    now aligned with singular values \f$\sigma_i\f$.  Together,
     *    \f[
     *      A = U\,\Sigma\,V^T.
     *    \f]
     *
     * @param A The matrix to decompose.
     * @return A std::tuple containing
     *         - \f$U\in\mathbb{R}^{m\times r}\f$: the left singular vectors,
     *         - \f$\Sigma\in\mathbb{R}^{r\times r}\f$: the diagonal matrix of singular values,
     *         - \f$V\in\mathbb{R}^{n\times r}\f$: the right singular vectors.
     */
    template <Number number, int m, int n>
    auto svdEigen(const LinearSystem<number, m, n>& sys) {
        // Find the eigen values and eigenvectors of A*A^T
        const auto [nR, nC] = sys.shape();
        const auto& ATA{sys * helpers::extractMatrixFromTranspose(sys.T())};
        auto [l, v] = eigenQR(ATA);
        const auto& A = helpers::padMatrixToSquare<number, m, n>(sys);

        const auto& p = helpers::sortPermutation(l.begin(), l.end(), std::greater<>());
        helpers::applySortPermutation(l, p);
        helpers::applySortPermutation(v, p);
        logging::log(format("SVD Eigenvalues: {}", l), "svdEigen");
        logging::log(format("SVD Eigenvectors: {}", v), "svdEigen");

        // Construct the Sigma matrix from the eigenvalues by taking the square root of the eigenvalues.
        auto Sigma = matrixZeros<number, n, n>(nC, nC);
        const auto minDim = std::min(nR, nC);
        for (size_t i{}; i < minDim; i++) {
            Sigma(i, i) = l[i] > 0 ? sqrt(l[i]) : 0;
        }
        logging::log(format("SVD Sigma: {}", Sigma), "svdEigen");

        // Construct the V matrix from the eigenvectors.
        const auto& V = helpers::fromColVectorSet<number, ATA.rows, ATA.cols>(v);
        logging::log(format("SVD V: {}", V), "svdEigen");

        // Construct the U matrix using the eigenvectors and the Sigma matrix.
        auto U = matrixZeros<number, m, m>(nR, nR);
        for (size_t i{}; i < minDim; i++)
            // Normalize the eigenvectors and multiply by the corresponding singular value.
            if (!fuzzyCompare(Sigma(i, i), number(0))) {
                auto ui = (A * v[i]).normalize();
                for (size_t j = 0; j < m; j++) {
                    U(j, i) = ui[j];
                }
            }
        logging::log(format("SVD U: {}", U), "svdEigen");

        return std::tuple{U, Sigma, helpers::extractMatrixFromTranspose(V.T())};
    }

    /**
     * @brief Compute the Singular Value Decomposition (SVD) of a linear system.
     *
     * Geometrically, any linear map \f$A:\mathbb{R}^n\to\mathbb{R}^m\f$ can be viewed in three stages:
     * 1. **Rotate (or reflect) the input space** by \f$V^T\f$, aligning the standard basis to the principal input
     * directions.
     * 2. **Scale along each principal axis** by the singular values \f$\sigma_i\f$, stretching or shrinking the unit
     * sphere into an ellipsoid.
     * 3. **Rotate (or reflect) into the output space** by \f$U\f$, orienting the ellipsoid in \f$\mathbb{R}^m\f$.
     *
     * Algebraically, we write
     * \f[
     *   A = U\,\Sigma\,V^T,
     * \f]
     * where:
     * - \f$V\in\mathbb{R}^{n\times r}\f$ contains the right singular vectors (orthonormal directions in the domain),
     * - \f$\Sigma=\mathrm{diag}(\sigma_1,\dots,\sigma_r)\in\mathbb{R}^{r\times r}\f$ has nonnegative singular values
     * \f$\sigma_i\ge0\f$,
     * - \f$U\in\mathbb{R}^{m\times r}\f$ contains the left singular vectors (orthonormal directions in the codomain).
     *
     * In this form:
     * - Columns of \f$V\f$ span the directions along which \f$A\f$ acts by pure scaling.
     * - Each \f$\sigma_i\f$ is the length of the semi‑axis of the image ellipsoid \f$A(B^n)\f$, where \f$B^n\f$ is the
     * unit ball.
     * - Columns of \f$U\f$ give the orientations of those axes in the output space.
     *
     * The SVD thus exposes the “principal axes” of the transformation,
     * provides the best low‑rank approximations to \f$A\f$,
     * and underlies stable algorithms for solving least‑squares and computing pseudoinverses.
     *
     * @param system The linear system to decompose.
     * @return A std::tuple containing
     *         - \f$U\in\mathbb{R}^{m\times r}\f$: the left singular vectors,
     *         - \f$\Sigma\in\mathbb{R}^{r\times r}\f$: the diagonal matrix of singular values,
     *         - \f$V\in\mathbb{R}^{n\times r}\f$: the right singular vectors.
     */
    template <Number number, int m, int n>
    auto svd(const LinearSystem<number, m, n>& system) {
        return svdEigen(system);
    }

}  // namespace mlinalg
