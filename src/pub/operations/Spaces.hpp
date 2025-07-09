#pragma once

#include "../Concepts.hpp"
#include "../structures/Matrix.hpp"
#include "Aliases.hpp"
#include "Checks.hpp"
#include "Echelon.hpp"
#include "Solve.hpp"

namespace mlinalg {
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
     * @brief Checks if a linear system in the form
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
}  // namespace mlinalg
