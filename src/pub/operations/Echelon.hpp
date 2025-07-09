#pragma once
#include <map>
#include <optional>
#include <set>

#include "../structures/Aliases.hpp"
#include "Aliases.hpp"
#include "../Numeric.hpp"

namespace mlinalg {
    using std::optional, std::nullopt;

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
    LinearSystem<number, m, n> rref(const LinearSystem<number, m, n>& system, bool identity = true) {
        const auto& nRows = system.numRows();
        const auto& nCols = system.numCols();

        if (nRows == nCols && nRows > 5) return rrefRec(system, identity);
        if (nRows == nCols) return rrefSq(system, identity);
        return rrefRec(system, identity);
    }

}  // namespace mlinalg
