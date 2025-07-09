#pragma once

#include <boost/type_index.hpp>
#include <format>
#include <type_traits>
#include <variant>

#include "../Concepts.hpp"
#include "../Helpers.hpp"
#include "../Logging.hpp"
#include "../Numeric.hpp"
#include "../structures/Aliases.hpp"
#include "../structures/Matrix.hpp"
#include "Aliases.hpp"

namespace mlinalg {
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
        for (int i{1}; i < static_cast<int>(pivots.size()); i++) {
            if (!pivots.at(i).has_value()) continue;
            for (int j{i - 1}; j >= 0; j--)
                if (!fuzzyCompare(system(j, i), number(0))) return false;
        }
        return true;
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

    template <Number number, int m, int n>
    struct OrthVisitor {
        const Matrix<number, m, n>& A;

        // HACK: Dirty
        bool operator()(const VectorTransposeVariant<number, m, n>& AT)
            requires((n == 1 || m == 1) || (n == Dynamic || m == Dynamic))
        {
            const auto& [nR, nC] = A.shape();
            const auto ATA = AT.T() * A;
            const auto& Id = I<number, n>(1);

            auto minus = ATA - Id;
            const auto& [nR2, nC2] = minus.shape();
            for (size_t i{}; i < nR2; i++) {
                for (size_t j{}; j < nC2; j++) {
                    if (!fuzzyCompare(abs(minus(i, j)), number(0))) return false;
                }
            }
            return true;
        }

        bool operator()(const VectorTransposeVariant<number, m, n>& AT) {
            logging::D(std::format("Why are we here -> {}x{}", m, n), "OrthVisitor");
            return false;
        }

        bool operator()(const MatrixTransposeVariant<number, m, n>& AT) {
            const auto& [nR, nC] = A.shape();
            const auto& ATA = AT * A;
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
    };

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
        OrthVisitor<number, m, n> v{A};
        return std::visit(v, A.T());
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

}  // namespace mlinalg
