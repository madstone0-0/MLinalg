/**
 * @file Matrix.hpp
 * @brief  Header file for matrix operations
 */

#pragma once
#include <optional>

#include "../Numeric.hpp"
#include "Aliases.hpp"
#include "Echelon.hpp"

namespace mlinalg {
    using std::optional, std::nullopt;

    /**
     * @brief Find the inverse of a square linear system
     *
     * @param system The linear system to find the inverse of.
     * @return The inverse of the system if it exists, nullopt otherwise.
     */
    template <Number number, Dim m>
    optional<Matrix<number, m, m>> inverse(const LinearSystem<number, m, m>& system) {
        auto det = system.det();
        if (fuzzyCompare(det, number(0))) return nullopt;
        const auto& [nRows, nCols] = system.shape();
        if (nRows == 2)
            return (1. / det) * Matrix<number, m, m>{
                                    {system(1, 1), -system(0, 1)},
                                    {-system(1, 0), system(0, 0)},
                                };
        else {
            auto identity = I<number, m>(nRows);
            auto augmented = system.augment(identity);
            auto rrefAug = rref(augmented);
            auto inv = rrefAug.colToVectorSet();
            inv.erase(inv.begin(), inv.begin() + nRows);
            return helpers::fromColVectorSet<number, m, m>(inv);
        }
    }

    /**
     * @brief Find the diagonal of a matrix.
     *
     * @param matrix The matrix to find the diagonal of.
     * @return The diagonal of the matrix.
     */
    template <Number number, Dim m, Dim n>
    Vector<number, m> diag(const Matrix<number, m, n>& matrix) {
        const auto [nR, nC] = matrix.shape();

        Vector<number, m> res(nR);
        for (size_t i{}; i < nR; i++) res[i] = matrix(i, i);

        return res;
    }
}  // namespace mlinalg
