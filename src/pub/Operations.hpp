/**
 * @file Operations.hpp
 * @brief Declarations and implementations of the operations that can be performed on the structures defined in
 * Structures.hpp.
 */

#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <numeric>
#include <optional>
#include <ostream>
#include <vector>

#include "Structures.hpp"

using std::vector, std::array, std::optional;

namespace mlinalg {
    using namespace structures;
    using std::nullopt;

    /**
     * @brief Linear System type alias.
     */
    template <Number number, size_t m, size_t n>
    using LinearSystem = Matrix<number, m, n>;

    /**
     * @brief Row of optional numbers type alias.
     */
    template <Number number, size_t n>
    using RowOptional = Row<optional<number>, n - 1>;

    /**
     * @brief Checks if a linear system is in echelon form.
     *
     * @param system  The linear system to check.
     * @param n      The number of columns in the system.
     * @param m     The number of rows in the system.
     * @param pivots The pivots of the system.
     * @return true if the system is in echelon form, false otherwise.
     */
    template <Number number, size_t m, size_t n>
    bool isInEchelonForm(LinearSystem<number, m, n> system, Row<optional<number>, n - 1> pivots) {
        for (size_t i{}; i < pivots.size(); i++)
            for (size_t j{i + 1}; j < m; j++)
                if (system.at(j).at(i) != 0) return false;
        return true;
    }

    /**
     * @brief Checks if a linear system is inconsistent.
     *
     * @param system The linear system to check.
     * @return true if the system is inconsistent, false otherwise.
     */
    template <Number number, size_t m, size_t n>
    bool isInconsistent(LinearSystem<number, m, n> system) {
        for (const auto& row : system) {
            size_t zeroCount{};
            for (size_t i{}; i < n - 1; i++)
                if (row.at(i) == 0) zeroCount++;

            if (zeroCount == n && row.back() != 0) return true;
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
    template <Number number, size_t m, size_t n>
    bool isInReducedEchelonForm(LinearSystem<number, m, n> system, Row<optional<number>, n - 1> pivots) {
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
    template <Number number, size_t n>
    optional<number> solveEquation(Row<number, n> row, size_t varPos, Row<optional<number>, n - 1> solutions) {
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
        if (leftSide.at(0) == 0) {
            if (rightSimpl == 0 && leftSide.at(0) == 0)
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
    template <Number number, size_t m, size_t n>
    RowOptional<number, n> getPivots(LinearSystem<number, m, n> system) {
        RowOptional<number, n> pivots{};

        for (size_t idx{}; idx < m; idx++)
            for (size_t i{}; i < n - 1; i++) {
                const auto& row = system.at(idx);
                if (row.at(i) != 0 && pivots.at(i) == nullopt) {
                    pivots.at(i) = row.at(i);
                    break;
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
    template <Number number, size_t m, size_t n>
    optional<size_t> getPivotRow(const LinearSystem<number, m, n>& system, size_t startRow, size_t col) {
        for (size_t row = startRow; row < m; ++row) {
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
    template <Number number, size_t m, size_t n>
    LinearSystem<number, m, n> rearrangeSystem(LinearSystem<number, m, n> system) {
        auto getZeroCount = [](const Row<number, n>& row) {
            size_t mid = floor((n - 1) / 2.);
            size_t count{};
            for (size_t i{}; i < n - 1; i++)
                if (row.at(i) == 0 && i <= mid) count++;
            return count;
        };

        auto cmp = [](const std::pair<size_t, size_t>& a, const std::pair<size_t, size_t>& b) {
            return (a.second < b.second);
        };

        auto getRowSortedCounts = [&getZeroCount, &cmp](const LinearSystem<number, m, n> sys) {
            std::multimap<size_t, size_t> rowCounts{};
            vector<std::pair<size_t, size_t>> res{};
            for (size_t i{}; i < m; i++) rowCounts.emplace(i, getZeroCount(sys.at(i)));
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
    template <Number number, size_t m, size_t n>
    LinearSystem<number, m, n> refSq(LinearSystem<number, m, n> system) {
        RowOptional<number, n> pivots = getPivots(system);
        while (!isInEchelonForm(system, pivots)) {
            for (size_t i{}; i < pivots.size(); i++) {
                pivots = getPivots(system);
                const auto& pivot = pivots.at(i);
                if (!pivot.has_value()) continue;
                for (size_t j{i + 1}; j < m; j++) {
                    system = rearrangeSystem(system);
                    pivots = getPivots(system);
                    auto lower = system.at(j).at(i);
                    if (lower == 0) continue;
                    if (pivot == 0) continue;

                    auto permute = lower / pivot.value();
                    auto rowItems = system.at(j);
                    Row<number, n> permuted{permute * system.at(i)};
                    auto newRow = permuted - rowItems;
                    system.at(j) = newRow;
                }
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
    template <Number number, size_t m, size_t n>
    LinearSystem<number, m, n> refRec(LinearSystem<number, m, n> system) {
        for (size_t col = 0; col < n && col < m; ++col) {
            system = rearrangeSystem(system);
            size_t pivotRow = col;

            for (size_t row = pivotRow + 1; row < m; ++row) {
                if (system.at(row).at(col) != 0) {
                    auto factor = system.at(row).at(col) / system.at(pivotRow).at(col);
                    for (size_t j = col; j < n; ++j) {
                        system.at(row).at(j) -= factor * system.at(pivotRow).at(j);
                    }
                }
            }
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
    template <Number number, size_t m, size_t n>
    LinearSystem<number, m, n> ref(LinearSystem<number, m, n> system) {
        if constexpr (m == n && m > 5) return refRec(system);
        if constexpr (m == n) return refSq(system);
        return refRec(system);
    }

    /**
     * @brief Find the reduced row echelon form of a rectangular linear system
     *
     * @param system The linear system to find the reduced row echelon form of.
     * @param identity Whether to convert the system to identity form.
     * @return The reduced row echelon form of the system.
     */
    template <Number number, size_t m, size_t n>
    LinearSystem<number, m, n> rrefRec(LinearSystem<number, m, n> system, bool identity = true) {
        RowOptional<number, n> pivots = getPivots(system);
        if (!isInEchelonForm(system, pivots)) system = ref(system);

        for (size_t col = n; col > 0; --col) {
            system = rearrangeSystem(system);
            size_t pivotRow = col - 1;

            if (pivotRow >= m) continue;

            if (system.at(pivotRow).at(col - 1) == 0) continue;

            if (identity) {
                auto pivotValue = system.at(pivotRow).at(col - 1);
                if (pivotValue != 0) {
                    for (size_t j = 0; j < n; ++j) {
                        system.at(pivotRow).at(j) /= pivotValue;
                    }
                }
            }

            for (size_t row = 0; row < pivotRow; ++row) {
                auto upperValue = system.at(row).at(col - 1);
                if (upperValue != 0) {
                    for (size_t j = 0; j < n; ++j) {
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
    template <Number number, size_t m, size_t n>
    LinearSystem<number, m, n> rrefSq(LinearSystem<number, m, n> system, bool identity = true) {
        RowOptional<number, n> pivots = getPivots(system);
        if (!isInEchelonForm(system, pivots)) system = ref(system);
        pivots = getPivots(system);

        while (!isInReducedEchelonForm(system, pivots)) {
            auto size = pivots.size() < n - 1 ? pivots.size() : n - 1;
            for (int i{1}; i < size; i++) {
                pivots = getPivots(system);
                const auto& pivot = pivots.at(i);
                if (!pivot.has_value()) continue;
                for (int j{i - 1}; j >= 0; j--) {
                    pivots = getPivots(system);
                    auto upper = system.at(j).at(i);
                    if (upper == 0) continue;
                    if (pivot == 0) continue;

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
                    if (pivot == 0) continue;
                    if (pivot != 1) system.at(i) = system.at(i) * (1 / pivot);
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
    template <Number number, size_t m, size_t n>
    LinearSystem<number, m, n> rref(LinearSystem<number, m, n> system, bool identity = true) {
        if (m == n && m > 5) return rrefRec(system, identity);
        if (m == n) return rrefSq(system, identity);
        return rrefRec(system, identity);
    }

    template <Number number, size_t m, size_t n>
    optional<RowOptional<number, n>> findSolutions(LinearSystem<number, m, n> system);

    /**
     * @brief Find the solutions to a matrix equation in the form:
     * A * x = b
     *
     * @param A The matrix of the equation
     * @param b The vector of the equation
     * @return The solutions x to the equation
     */
    template <Number number, size_t m, size_t n>
    auto findSolutions(Matrix<number, m, n> A, Vector<number, m> b) {
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
    template <Number number, size_t m, size_t n>
    optional<RowOptional<number, n>> findSolutions(LinearSystem<number, m, n> system) {
        RowOptional<number, n> solutions{};
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
    template <Number number, size_t m>
    Matrix<number, m, m> I() {
        Matrix<number, m, m> identity{};
        for (size_t i{}; i < m; i++) {
            identity.at(i).at(i) = 1;
        }
        return identity;
    }

    /**
     * @brief Find the inverse of a square linear system
     *
     * @param system The linear system to find the inverse of.
     * @return The inverse of the system if it exists, nullopt otherwise.
     */
    template <Number number, size_t m, size_t n>
    optional<Matrix<number, m, n>> inverse(const LinearSystem<number, m, n>& system) {
        auto det = system.det();
        if (det == 0) return std::nullopt;
        if (m == 2 && n == 2)
            return (1 / det) * Matrix<number, m, n>{{system.at(1).at(1), -system.at(0).at(1)},
                                                    {-system.at(1).at(0), system.at(0).at(0)}};
        else {
            auto identity = I<number, m>();
            auto augmented = system.augment(identity);
            auto rrefAug = rref(augmented);
            auto inv = rrefAug.colToVectorSet();
            inv.erase(inv.begin(), inv.begin() + m);
            return mlinalg::structures::helpers::fromColVectorSet<number, m, n>(inv);
        }
    }

}  // namespace mlinalg
