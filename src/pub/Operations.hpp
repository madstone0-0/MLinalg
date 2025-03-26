/**
 * @file Operations.hpp
 * @brief Declarations and implementations of the operations that can be performed on the structures defined in
 * Structures.hpp.
 */

#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <stdexcept>
#include <vector>

#include "Concepts.hpp"
#include "Numeric.hpp"
#include "Structures.hpp"
#include "structures/Aliases.hpp"

using std::vector, std::array, std::optional, std::nullopt;

namespace mlinalg {
    using namespace structures;

    /**
     * @brief Linear System type alias.
     */
    template <Number number, int m, int n>
    using LinearSystem = Matrix<number, m, n>;

    /**
     * @brief Row of optional numbers type alias.
     */
    template <Number number, int m>
    using RowOptional = std::conditional_t<m == -1, RowDynamic<optional<number>>, Row<optional<number>, m>>;

    template <Number num, int n>
    Vector<num, n> extractSolutionVector(const Vector<optional<num>, n>& solutions) {
        if (rg::any_of(solutions, [](const auto& val) { return !val.has_value(); })) {
            throw std::runtime_error("Cannot extract solution vector from incomplete solutions");
        }

        Vector<num, n> res{};
        for (size_t i{}; i < n; i++) {
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
                                   const Row<optional<number>, n - 1>& solutions) {
        vector<number> leftSide{row.begin(), row.end() - 1};
        vector<number> rightSide{row.back()};
        for (size_t i{}; i < leftSide.size(); i++) {
            optional<number> var = solutions.at(i);
            if (i != varPos) {
                number res = var.has_value() ? var.value() * leftSide.at(i) * -1 : leftSide.at(i) * -1;
                rightSide.push_back(res);
            }
        }
        leftSide = vector{leftSide.at(varPos)};
        auto rightSimpl = std::accumulate(rightSide.begin(), rightSide.end(), number(0));
        if (fuzzyCompare(leftSide.at(0), number(0))) {
            if (fuzzyCompare(rightSimpl, number(0)) && fuzzyCompare(leftSide.at(0), number(0)))
                return 0;
            else
                return nullopt;
        }

        number sol = rightSimpl / leftSide.at(0);
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
                if (identity) system.at(j) *= 1 / system.at(j, j);

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
    LinearSystem<number, m, n> rref(const LinearSystem<number, m, n>& system, bool identity = true) {
        const auto& nRows = system.numRows();
        const auto& nCols = system.numCols();

        if (nRows == nCols && nRows > 5) return rrefRec(system, identity);
        if (nRows == nCols) return rrefSq(system, identity);
        return rrefRec(system, identity);
    }

    template <Number number, int m, int n>
    optional<RowOptional<number, n - 1>> findSolutions(const LinearSystem<number, m, n>& system);

    /**
     * @brief Find the solutions to a matrix equation in the form:
     * A * x = b
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
     * @brief Determined if a linear system in the form
     * [A | b]
     *is underdetermined.
     *
     * Assumes the system is already in row echelon form.
     * @param sys The linear system to check.
     * @return true if the system is underdetermined, false otherwise.
     */
    template <Number number, int m, int n>
    bool isSystemUnderdetermined(const LinearSystem<number, m, n>& sys) {
        // Compute the effective rank by counting nonzero rows.
        // (This assumes that the system is already in row echelon form,
        // so that all-zero rows, if any, appear at the bottom.)
        size_t rank{};
        for (const auto& row : sys) {
            // If the row is not all zeros, count it as a pivot row.
            if (!rg::all_of(row, [](const number& value) { return fuzzyCompare(value, number(0)); })) {
                ++rank;
            }
        }
        // The system is underdetermined if there are more unknowns than pivot rows.
        return rank < sys.numCols() - 1;
    }

    /**
     * @brief Find the solutions to a linear system in the form:
     * [A | b]
     *
     * @param system The linear system to find the solutions of.
     * @return The solutions to the system if they exist, nullopt otherwise.
     */
    template <Number number, int m, int n>
    optional<RowOptional<number, n - 1>> findSolutions(const LinearSystem<number, m, n>& sys) {
        RowOptional<number, n - 1> solutions{};
        LinearSystem<number, m, n> system{sys};
        system = rearrangeSystem(system);

        auto reducedEchelon = rref(system);

        if (isInconsistent(reducedEchelon)) return nullopt;

        // TODO: Check for underdetermined systems, find the general solution and calculate the values
        // that satisfy the system with the minimum-norm solution.
        if (isSystemUnderdetermined(reducedEchelon)) {
            throw std::runtime_error("Finding the solutions to an underdetermined system is not implemented");
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
     * @brief Find the identity matrix of a square linear system
     *
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

}  // namespace mlinalg
