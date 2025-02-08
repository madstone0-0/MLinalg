/**
 * @file Operations.hpp
 * @brief Declarations and implementations of the operations that can be performed on the structures defined in
 * Structures.hpp.
 */

#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <map>
#include <numeric>
#include <optional>
#include <ostream>
#include <set>
#include <stdexcept>
#include <vector>

#include "Structures.hpp"

using std::vector, std::array, std::optional;

constexpr double EPSILON = std::numeric_limits<double>::epsilon();

namespace mlinalg {
    using namespace structures;
    using std::nullopt;

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
        const auto& nRows = system.numRows();

        for (size_t i{}; i < pivots.size(); i++)
            for (size_t j{i + 1}; j < nRows; j++)
                if (system.at(j).at(i) != 0) return false;
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
            for (size_t i{}; i < nCols - 1; i++)
                if (std::abs(row.at(i)) <= EPSILON) zeroCount++;

            if (zeroCount == nCols - 1 && row.back() != 0) return true;
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
        auto rightSimpl = std::accumulate(rightSide.begin(), rightSide.end(), number{0});
        if (std::abs(leftSide.at(0)) <= EPSILON) {
            if (std::abs(rightSimpl) <= EPSILON && std::abs(leftSide.at(0)) <= EPSILON)
                return 0;
            else
                return std::nullopt;
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
    RowOptional<number, m> getPivots(const LinearSystem<number, m, n>& system) {
        const auto& nRows = system.numRows();
        const auto& nCols = system.numCols();

        std::set<size_t> seenCols{};
        size_t pivIdx{};
        size_t colPos{};
        RowOptional<number, m> pivots(nRows);
        for (size_t idx{}; idx < nRows; idx++) {
            const auto& row = system.at(idx);
            for (size_t i{colPos}; i < nCols - 1; i++) {
                colPos++;
                if (std::abs(row.at(i)) > EPSILON) {
                    pivots.at(pivIdx) = row.at(i);
                    pivIdx++;
                    break;
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
        return std::nullopt;  // No pivot found in this column
    }

    /**
     * @brief Rearrange a linear system to have the most zeros in the bottom rows.
     *
     * @param system The linear system to rearrange.
     * @return The rearranged linear system.
     */
    template <Number number, int m, int n>
    LinearSystem<number, m, n> rearrangeSystem(const LinearSystem<number, m, n>& system) {
        const auto& nRows = system.numRows();
        const auto& nCols = system.numCols();

        auto getZeroCount = [](const Row<number, n>& row) {
            const auto& size = row.size();
            int mid = floor((size - 1) / 2.);
            size_t count{};
            for (size_t i{}; i < size - 1; i++)
                if (std::abs(row.at(i)) <= EPSILON && i <= mid) count++;
            return count;
        };

        auto cmp = [](const std::pair<size_t, size_t>& a, const std::pair<size_t, size_t>& b) {
            return (a.second < b.second);
        };

        auto getRowSortedCounts = [&getZeroCount, &cmp](const LinearSystem<number, m, n> sys) {
            const auto& nRows = sys.numRows();
            const auto& nCols = sys.numCols();
            std::multimap<size_t, size_t> rowCounts{};
            vector<std::pair<size_t, size_t>> res{};

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
        const auto& nRows = system.numRows();
        const auto& nCols = system.numCols();

        for (size_t j{}; j < nCols; j++) {
            if (system.at(j, j) == 0) {
                number big{std::abs(system.at(j, j))};
                size_t kRow{j};

                for (size_t k{j + 1}; k < nCols - 1; k++) {
                    auto val = std::abs(system.at(k, j));
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

            if (std::abs(pivot) <= EPSILON) {
                throw std::runtime_error("Matrix is singular or numerically unstable");
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

        const auto& nRows = system.numRows();
        const auto& nCols = system.numCols();

        size_t pivRow{};
        size_t pivCol{};
        while (pivRow < nRows && pivCol < nCols) {
            number big{std::abs(system.at(pivRow, pivCol))};
            size_t kRow{pivRow};

            // Check if current pivot is (effectively) zero
            if (std::abs(system.at(pivRow, pivCol)) < EPSILON) {
                // Find the first row below with a non-zero entry in this column
                size_t kRow = pivRow;
                for (kRow = pivRow + 1; kRow < nRows; kRow++) {
                    if (std::abs(system.at(kRow, pivCol)) >= EPSILON) {
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

        const auto& nRows = system.numRows();
        const auto& nCols = system.numCols();

        auto pivots = getPivots(system);
        if (!isInEchelonForm(system, pivots)) system = ref(system);

        for (size_t col = nCols; col > 0; --col) {
            system = rearrangeSystem(system);
            size_t pivotRow = col - 1;

            if (pivotRow >= nRows) continue;

            if (std::abs(system.at(pivotRow).at(col - 1)) <= EPSILON) continue;

            if (identity) {
                auto pivotValue = system.at(pivotRow).at(col - 1);
                if (pivotValue != 0) {
                    for (size_t j = 0; j < nCols; ++j) {
                        system.at(pivotRow).at(j) /= pivotValue;
                    }
                }
            }

            for (size_t row = 0; row < pivotRow; ++row) {
                auto upperValue = system.at(row).at(col - 1);
                if (upperValue != 0) {
                    for (size_t j = 0; j < nCols; ++j) {
                        system.at(row).at(j) -= upperValue * system.at(pivotRow).at(j);
                    }
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

        const auto& nRows = system.numRows();
        const auto& nCols = system.numCols();

        RowOptional<number, m> pivots = getPivots(system);
        if (!isInEchelonForm(system, pivots)) system = ref(system);
        pivots = getPivots(system);

        while (!isInReducedEchelonForm(system, pivots)) {
            auto size = pivots.size() < nCols - 1 ? pivots.size() : nCols - 1;
            for (int i{1}; i < size; i++) {
                pivots = getPivots(system);
                if (!pivots.at(i).has_value()) continue;
                for (int j{i - 1}; j >= 0; j--) {
                    pivots = getPivots(system);
                    const auto& pivot = pivots.at(i);
                    auto upper = system.at(j).at(i);
                    if (std::abs(upper) <= EPSILON) continue;
                    if (std::abs(pivot.value()) <= EPSILON) continue;

                    auto permute = upper / pivot.value();
                    auto rowItems = system.at(j);
                    Row<number, n> permuted{permute * system.at(i)};
                    auto newRow = permuted - rowItems;
                    system.at(j) = newRow;
                }
            }
        }

        if (identity)
            for (size_t i{}; i < pivots.size(); i++) {
                try {
                    const auto& pivot{system.at(i).at(i)};
                    if (std::abs(pivot) <= EPSILON) continue;
                    if (std::abs(pivot - 1) > EPSILON) system.at(i) = system.at(i) * (1 / pivot);
                } catch (const std::out_of_range& e) {
                    continue;
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
    optional<RowOptional<number, m>> findSolutions(const LinearSystem<number, m, n>& system);

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
     * @brief Find the solutions to a linear system in the form:
     * [A | b]
     *
     * @param system The linear system to find the solutions of.
     * @return The solutions to the system if they exist, nullopt otherwise.
     */
    template <Number number, int m, int n>
    optional<RowOptional<number, m>> findSolutions(const LinearSystem<number, m, n>& system) {
        RowOptional<number, m> solutions{};
        system = rearrangeSystem(system);

        auto reducedEchelon = rref(system);

        if (isInconsistent(reducedEchelon)) return nullopt;

        for (auto i{solutions.rbegin()}; i != solutions.rend(); i++) {
            try {
                auto index = std::distance(solutions.rbegin(), i);
                solutions.at(index) = solveEquation(reducedEchelon.at(index), index, solutions);
            } catch (const std::out_of_range& e) {
                continue;
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
            for (size_t i{}; i < nRows; i++) {
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
    template <Number number, int m, int n>
    optional<Matrix<number, m, n>> inverse(const LinearSystem<number, m, n>& system) {
        auto det = system.det();
        if (std::abs(det) <= EPSILON) return std::nullopt;
        if (m == 2 && n == 2)
            return (1. / det) * Matrix<number, m, n>{{system.at(1).at(1), -system.at(0).at(1)},
                                                     {-system.at(1).at(0), system.at(0).at(0)}};
        else {
            const auto& nRows = system.numRows();
            auto identity = I<number, m>(nRows);
            auto augmented = system.augment(identity);
            auto rrefAug = rref(augmented);
            auto inv = rrefAug.colToVectorSet();
            inv.erase(inv.begin(), inv.begin() + nRows);
            return mlinalg::structures::helpers::fromColVectorSet<number, m, n>(inv);
        }
    }

}  // namespace mlinalg
