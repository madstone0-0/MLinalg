#pragma once
#include "../Numeric.hpp"
#include "Aliases.hpp"

namespace mlinalg {
    /**
     * @brief Calculates the pseudoinverse of a matrix using the Moore-Penrose pseudoinverse.
     *
     * The Moore-Penrose pseudoinverse is a generalization of the matrix inverse that can be applied to non-square or
     * singular matrices. It is defined as the unique matrix \f$A^+\f$ that satisfies the following properties:
     * 1. \f$A A^+ A = A\f$
     * 2. \f$A^+ A A^+ = A^+\f$
     * 3. \f$(A A^+)^* = A A^+\
     * 4. \f$(A^+ A)^* = A^+ A\f$
     * Where \f$^*\f$ denotes the conjugate transpose.
     *
     * This implmentation calulates the pseudoinverse using the Singular Value Decomposition (SVD) of the matrix.
     * The SVD decomposes the matrix \f$A\f$ into three matrices
     * \f[
     *   A = U\,\Sigma\,V^T,
     * \f]
     * And the pseudoinverse is then calculated as:
     * \f[
     *   A^{+} = V\,\Sigma^{+}\,U^T,
     * \f]
     * Where \f$\Sigma^{+}\f$ is the transpose of the diagonal matrix \f$\Sigma\f$ with the reciprocal of the non-zero
     * singular values on the diagonal.
     *
     * @param A the matrix to calculate the pseudoinverse of.
     * @return A std::tuple containing
     *         - \f$V\in\mathbb{R}^{n\times r}\f$: the right singular vectors,
     *         - \f$\Sigma^{+}\in\mathbb{R}^{r \times r}\f$: the diagonal matrix of reciprocal singular values,
     *         - \f$U^T\in\mathbb{R}^{r\times m}\f$: the transpose of the left singular vectors.
     */
    template <Number number, int m, int n>
    auto moorePenrosePinv(const Matrix<number, m, n>& A) {
        const auto& [nR, nC] = A.shape();
        const auto& minDim{std::min(nR, nC)};
        auto [U, Sigma, VT] = svd(A);
        Sigma.apply([](auto& sigma) {
            if (!fuzzyCompare(sigma, number(0))) sigma = 1 / sigma;
        });
        const auto& SigmaPlus = helpers::extractMatrixFromTranspose(Sigma.T());
        const auto& UT = helpers::extractMatrixFromTranspose(U.T());
        const auto& V = helpers::extractMatrixFromTranspose(VT.T());
        return std::tuple{V, SigmaPlus, UT};
    }

    /**
     * @brief Calculates the pseudoinverse of a matrix
     *
     * @param A the matrix to calculate the pseudoinverse of.
     @ return The pseudoinverse of the matrix.
     */
    template <Number number, int m, int n>
    auto pinv(const Matrix<number, m, n>& A) {
        const auto& [V, SigmaPlus, UT] = moorePenrosePinv(A);
        return V * SigmaPlus * UT;
    }
}  // namespace mlinalg
