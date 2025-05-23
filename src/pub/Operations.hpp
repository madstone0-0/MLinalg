/**
 * @file Operations.hpp
 * @brief Declarations and implementations of the operations that can be performed on the structures defined in
 * Structures.hpp.
 */

#pragma once
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <iterator>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "Concepts.hpp"
#include "Helpers.hpp"
#include "Numeric.hpp"
#include "Structures.hpp"
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

    template <Number number>
    auto LU(const Matrix<number, 1, 1>& A);

    template <Number number, int n>
    auto LU(const Matrix<number, n, n>& A);

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
            if (std::abs(U(i, i)) < EPSILON) return true;
        }
        return false;
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

    template <Number number, int n>
    auto QR(const Matrix<number, n, n>& A) {
        const auto [nR, nC] = A.shape();
        const int numRows = nR;
        const int numCols = nC;
        if (numRows != numCols) throw StackError<std::invalid_argument>{"Matrix A must be square"};

        auto U = matrixZeros<number, n, n>(numRows, numRows);
        auto R = matrixZeros<number, n, n>(numRows, numRows);

        const auto& asCols{A.colToVectorSet()};
        for (int i{}; i < numRows; i++) {
            // Squared norm of w_i
            const auto& wI{asCols[i]};
            // const auto& wI{A[i]};
            const auto wIN2{wI.dot(wI)};
            // const auto wIN2{std::pow(wI.length(), 2)};

            // Compute coefficients R_ji
            number sum{0};
            for (int j{}; j < i; j++) {
                const auto dot = wI.dot(U[j]);
                R(j, i) = dot;
                sum += dot * dot;
            }
            const auto& diag = wIN2 - sum;
            R(i, i) = sqrt(std::max(diag, EPSILON));

            if (fuzzyCompare(R(i, i), number(0))) throw StackError<std::logic_error>{"Matrix A is singular"};

            // Compute vector U_i using forward substitution
            auto vec = vectorZeros<number, n>(numRows);
            for (int j{}; j < i; j++) vec += R(j, i) * U[j];
            U[i] = (wI - vec) / R(i, i);
        }

        // Q = U.T
        const auto& Q = helpers::extractMatrixFromTranspose(U.T());
        return std::pair{Q, R};
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
        if (m == 2)
            return (1. / det) * Matrix<number, m, m>{{system.at(1).at(1), -system.at(0).at(1)},
                                                     {-system.at(1).at(0), system.at(0).at(0)}};
        else {
            const auto& nRows = system.numRows();
            auto identity = I<number, m>(nRows);
            auto augmented = system.augment(identity);
            auto rrefAug = rref(augmented);
            auto inv = rrefAug.colToVectorSet();
            inv.erase(inv.begin(), inv.begin() + nRows);
            return mlinalg::structures::helpers::fromColVectorSet<number, m, m>(inv);
        }
    }

    /**
     * @brief Find the dominant eigenvector of a square matrix using the power method.
     *
     * @param A The matrix to find the dominant eigenvector of.
     * @param iters The number of iterations to perform.
     * @return The dominant eigenvector of the matrix.
     */
    template <Number number, int m>
    Vector<double, m> dominantEigenvector(const Matrix<number, m, m>& A, size_t iters = 1000) {
        const auto [nR, nC] = A.shape();
        const int numRows = nR;
        const int numCols = nC;
        if (numRows != numCols) throw StackError<std::invalid_argument>{"Matrix A must be square"};

        // Using the iterative power method
        auto x0{vectorOnes<number, m>(numRows)};
        for (size_t i{}; i < iters; i++) {
            x0 = A * x0;
            number largestAbs{x0[0]};
            // Scaling
            for (size_t j{1}; j < x0.size(); j++)
                if (abs(x0[j]) > largestAbs) largestAbs = abs(x0[j]);
            x0 = (1 / largestAbs) * x0;
        }
        return x0;
    }

    /**
     * @brief Find the dominant eigenvalue of a square matrix using the Rayleigh quotient.
     *
     * @param A
     * @param eigenvector
     * @return
     */
    template <Number number, int n>
    auto dominantEigenvalue(const Matrix<number, n, n>& A, const Vector<number, n>& eigenvector) {
        // Eigenvalue found by the Rayleigh quotient:
        // eigenvalue = (A*x * x) / (x * x)
        const auto& x = eigenvector;
        const auto& num = (A * x) * x;
        const auto& denom = x * x;
        return num / denom;
    }

    template <Number number, int n>
    auto powerEigen(const Matrix<number, n, n>& A, size_t iters = 10'000, double tol = 0.001,
                    Seed seed = std::nullopt) {
        const auto [nR, nC] = A.shape();
        const int numRows = nR;
        const int numCols = nC;

        if (numRows != numCols) throw StackError<std::invalid_argument>{"Matrix A must be square"};
        auto x0{vectorRandom<number, n>(numRows, 0, 1, seed)};
        number l0{0};

        for (size_t i{}; i < iters; i++) {
            x0 = std::move(A * x0);
            number largestAbs{x0[0]};
            // Scaling
            for (size_t j{1}; j < x0.size(); j++)
                if (abs(x0[j]) > largestAbs) largestAbs = abs(x0[j]);
            x0 = std::move((1 / largestAbs) * x0);

            const auto& x = x0;
            const auto& num = (A * x) * x;
            const auto& denom = x * x;
            l0 = num / denom;
            const auto& error = ((A * x0) - (l0 * x0)).length();
            if (fuzzyCompare(error, tol) || error < tol) break;
        }

        return pair{x0, l0};
    }

    /**
     * @brief Deflate the matrix A by the outer product of the eigenvector and the eigenvalue.
     *
     * @param A The matrix to deflate.
     * @param eignevalue The eigenvalue to deflate by.
     * @param eigenvector The eigenvector to deflate by.
     * @return The deflated matrix.
     */
    template <Number number, int n>
    auto deflate(const Matrix<number, n, n>& A, number eignevalue, const Vector<number, n>& eigenvector) {
        const auto& rhs = eignevalue * (eigenvector * eigenvector.T());
        return A - rhs;
    }

    template <Number number, int n>
    auto shiftedInvPowerEigen(const Matrix<number, n, n>& M, std::optional<number> s = std::nullopt,
                              size_t iters = 10'000, double tol = 0.001, Seed seed = std::nullopt) {
        const auto [nR, nC] = M.shape();
        const int numRows = nR;
        const int numCols = nC;

        if (numRows != numCols) throw StackError<std::invalid_argument>{"Matrix A must be square"};

        auto x0{vectorRandom<number, n>(numRows, 0, 1, seed)};
        x0 = (1 / x0.length()) * x0;
        number l0{0};

        for (size_t i{}; i < iters; i++) {
            // Use Rayleigh quotient to update the shift if shift is not provided
            if (!s.has_value()) s = (x0.dot(M * x0)) / (x0.dot(x0));

            const auto& A = M - s.value() * I<number, n>(numRows);
            const auto& [L, U] = LU(A);

            const auto& zi = findSolutions(L, x0);
            if (!zi.has_value()) throw StackError<std::runtime_error>{"Cannot find solutions to L"};
            const auto& ziExt = extractSolutionVector(zi.value());

            const auto& y = findSolutions(U, ziExt);
            if (!y.has_value()) throw StackError<std::runtime_error>{"Cannot find solutions to U"};
            const auto& yi = extractSolutionVector<number, n>(y.value());

            x0 = (1 / yi.length()) * yi;
            l0 = (x0.dot(A * x0)) / (x0.dot(x0));

            const auto& error = ((A * x0) - (l0 * x0)).length();
            if (fuzzyCompare(error, tol) || error < tol) break;
        }

        return pair{x0, s.value()};
    }

    template <Number number, int n>
    auto eigen(const Matrix<number, n, n>& A, Seed seed = std::nullopt, size_t iters = 10'000, double tol = 0.001) {
        const auto [nR, nC] = A.shape();
        const int numRows = nR;
        const int numCols = nC;
        if (numRows != numCols) throw StackError<std::invalid_argument>{"Matrix A must be square"};
        if (isSingular(A)) throw StackError<std::invalid_argument>{"Matrix A is singular"};
        vector<number> values;
        vector<Vector<number, n>> vectors;

        // Find the largest eignevalue and eigenvector
        const auto& [v1, l1] = powerEigen(A, iters, tol);
        // Find the smallest eigenvalue and eigenvector
        const auto& [vn, ln] = shiftedInvPowerEigen(A, optional<number>(0), iters, tol, seed);

        values.emplace_back(l1);
        vectors.emplace_back(v1);
        values.emplace_back(ln);
        vectors.emplace_back(vn);

        // Use the shifted inverse power method to find the remaining eigenvalues and eigenvectors
        // by trying to find the next eigenvalue that is not equal to the previous one.
        auto M = A;
        auto shift = ln;
        while (values.size() < nR) {
            const auto& [v, l] = shiftedInvPowerEigen(M, optional<number>(nullopt), iters, tol, seed);
            M = deflate(M, l, v);
            if (fuzzyCompare(l, l1)) break;

            values.emplace_back(l);
            vectors.emplace_back(v);
            shift = l;
        }

        return pair{values, vectors};
    }

    /**
     * @brief Find the diagonal of a square matrix.
     *
     * @param matrix The matrix to find the diagonal of.
     * @return The diagonal of the matrix.
     */
    template <Number number, int n>
    Matrix<number, n, n> diag(const Matrix<number, n, n>& matrix) {
        const auto [nR, nC] = matrix.shape();
        if (nR != nC) throw StackError<runtime_error>("Matrix must be square to find a diagonal");

        constexpr auto isDynamic = n == Dynamic;

        constexpr auto sizeP = isDynamic ? DynamicPair : SizePair{n, n};

        Matrix<number, sizeP.first, sizeP.second> res(nR, nR);
        for (size_t i{}; i < nR; i++) {
            res(i, i) = matrix(i, i);
        }
        return res;
    }

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

    template <int n, Number number>
    Matrix<number, n, n> diagonal(const array<number, n>& entries) {
        Matrix<number, n, n> res{n, n};
        for (size_t i{}; i < n; i++) {
            res(i, i) = entries[i];
        }
        return res;
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

    template <Number number, int m, int n>
    auto svdGolubReinsch(const LinearSystem<number, m, n>& system) {
        const auto& [B, U, V] = houseHolderRed(system);
        const auto [numRows, numCols] = system.shape();
        int i{};
        while (i < numCols) {
            i++;
        }
    }

    template <Number number, int m, int n>
    auto svdEigen(const LinearSystem<number, m, n>& A) {
        const auto [nR, nC] = A.shape();
        const auto& ATA{A * A.T()};
        auto [l, v] = eigen(ATA);

        const auto& p = helpers::sortPermutation(l, std::greater<>());
        helpers::applySortPermutation(l, p);
        helpers::applySortPermutation(v, p);

        auto Sigma = matrixZeros<number, n, n>(nC, nC);
        const auto minDim = std::min(nR, nC);
        for (size_t i{}; i < minDim; i++) {
            Sigma(i, i) = l[i] > 0 ? sqrt(l[i]) : 0;
        }

        const auto& V = helpers::fromColVectorSet<number, n, n>(v);

        auto U = matrixZeros<number, m, m>(nR, nR);
        for (size_t i{}; i < minDim; i++)
            if (!fuzzyCompare(Sigma(i, i), number(0))) {
                auto ui = (A * v[i]).normalize();
                for (size_t j = 0; j < m; j++) {
                    U(j, i) = ui[j];
                }
            }

        return std::tuple{U, Sigma, helpers::extractMatrixFromTranspose(V.T())};
    }

    template <Number number, int m, int n>
    auto svd(const LinearSystem<number, m, n>& system) {
        // const auto [numRows, numCols] = system.shape();
        // const auto& sys = numRows >= numCols ? system : helpers::extractMatrixFromTranspose(system.T());
        return svdEigen(system);
    }

}  // namespace mlinalg
