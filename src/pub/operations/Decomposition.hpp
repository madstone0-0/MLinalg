#pragma once
#include <format>
#include <utility>

#include "../Logging.hpp"
#include "../Numeric.hpp"
#include "Aliases.hpp"
#include "Builders.hpp"
#include "Matrix.hpp"

namespace mlinalg {
    using std::pair, std::tuple;

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

            L(0, 0) = 1;
            U(0, 0) = a11;

            // Calculate U12
            for (int j{1}; j < numRows; ++j) {
                U(0, j) = A(0, j);
            }

            // Calculate L21
            for (int i{1}; i < numRows; ++i) {
                L(i, 0) = (A(i, 0)) / (a11 + EPSILON);
                // Set the corresponding U values to 0 for lower triangular portion
                U(i, 0) = 0;
            }

            // Calculate the Schur complement: A22 - L21 * U12
            auto S = A.template slice<1, n, 1, n>({1, numRows}, {1, numRows});
            for (int i{1}; i < numRows; ++i) {
                for (int j{1}; j < numRows; ++j) {
                    S(i - 1, j - 1) = (A(i, j)) - (L(i, 0)) * (U(0, j));
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

        for (size_t i{}; i < nR; ++i) {
            const auto& abs = std::abs(U(i, i));
            if (fuzzyCompare(abs, number(0)) || abs < EPSILON) return true;
        }
        return false;
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
    auto extendToCompleteBasis(const vector<Vector<number, m>>& qs, size_t dim) {
        vector<Vector<number, m>> basis = qs;
        auto n = qs.size();

        if (n == 0)
            throw StackError<std::invalid_argument>{"At least one vector is required to extend to a complete basis"};

        for (size_t i = n; i < dim; ++i) {
            // Standard basis vector e_i
            Vector<number, m> candidate(dim);

            // Try each standard basis vector until we find one that's not
            // in the span of existing vectors
            bool found = false;
            for (size_t basisIdx{}; basisIdx < dim && !found; ++basisIdx) {
                // Create e_basisIndex (standard basis vector)
                candidate.clear();
                candidate[basisIdx] = number(1);

                // Orthogonalize against all existing vectors using Gram-Schmidt
                auto orth = candidate;
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
        qs.emplace_back(v0 / norm0);
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
            qs.emplace_back(vi / norm);
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
            auto R = RType<type, number, m, n>(std::min(nR, nC), nC);

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
        Vector<number, Dynamic> x(numRows);
        auto QkL = I<number, m>(numRows);
        auto QkR = I<number, n>(numCols);
        for (size_t k{}; k < p; k++) {
            // --------------------------------------------------------------------
            // Left house holder reduction: zero out below the diagonal in column k
            // --------------------------------------------------------------------
            x.resize(numRows - k);
            for (size_t i{k}; i < nR; i++) {
                x[i - k] = B(i, k);
            }

            // Compute reflector if x is not already zero
            if (!isZeroVector(x)) {
                // Determine norm and sign to avoid cancellation
                auto normX = x.length();
                number sign = (fuzzyCompare(x[0], number(0)) || x[0] > number(0)) ? number(1) : number(-1);
                // Form the Householder vector
                x[0] += sign * normX;
                // Small Householder matrix of size (numRows-k)x(numRows-k)
                const auto& H = houseHolder(x);

                // Generate identity of full size and replace the lower right block with the
                // small Householder matrix
                I(QkL);
                for (size_t i{k}; i < nR; ++i)
                    for (size_t j{k}; j < nR; ++j) QkL(i, j) = H(i - k, j - k);

                B = QkL * B;
                U = U * QkL;
            }

            // -----------------------------------------------------------------------
            // Right house holder reduction: zero out above the superdiagonal in row k
            // -----------------------------------------------------------------------
            if (static_cast<int>(k) < static_cast<int>(numCols - 1)) {
                x.resize(numCols - k - 1);
                for (size_t j{k + 1}; j < numCols; ++j) x[j - k - 1] = B(k, j);

                if (!isZeroVector(x)) {
                    auto normX = x.length();
                    number sign = (fuzzyCompare(x[0], number(0)) || x[0] > number(0)) ? number(1) : number(-1);
                    x[0] += sign * normX;
                    auto v = x;
                    const auto& H = houseHolder(v);  // size (numCols-k-1)x(numCols-k-1)

                    // Embed H_small into an identity matrix for the full column range
                    I(QkR);
                    for (size_t i{k + 1}; i < numCols; i++)
                        for (size_t j{k + 1}; j < numCols; j++) QkR(i, j) = H(i - k - 1, j - k - 1);

                    B = B * QkR;
                    V = V * QkR;
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
        auto Qk = I<number, m>(numRows);
        Vector<number, Dynamic> x(numRows);
        for (size_t k{}; k < p; ++k) {
            // ---------------------------------------------------
            // Left Householder step: zero out below-diagonal in col k
            // ---------------------------------------------------
            x.resize(numRows - k);
            for (size_t i = k; i < numRows; ++i) {
                x[i - k] = R(i, k);
            }

            if (!isZeroVector(x)) {
                auto normX = x.length();
                number sign = (fuzzyCompare(x[0], number(0)) || x[0] > number(0)) ? number(1) : number(-1);

                x[0] += sign * normX;
                const auto& H = houseHolder(x);  // small (numRows-k)×(numRows-k)

                // embed H into full-size identity Qk
                I(Qk);
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
     * @brief Compute the Schur matrix and the Orthogonal matrix Q such that
     * \f[
     * A = Q\,S\,Q^T
     * \f]
     *
     * @param A The square matrix to decompose.
     * @param checkSingular If true, checks if the matrix is singular before proceeding with the decomposition.
     * @param method  The method to use for QR decomposition (default is Householder).
     * @param iters The maximum number of iterations to perform for convergence (default is 10,000).
     * @return A pair containing the orthogonal matrix Q and the Schur matrix S.
     */
    template <QRType type, Number number, int n>
    auto schurCommon(const Matrix<number, n, n>& A, bool checkSingular = true, QRMethod method = QRMethod::Householder,
                     size_t iters = 10'000) {
        const auto [nR, nC] = A.shape();
        const int numRows = nR;
        const int numCols = nC;
        if (numRows != numCols) throw StackError<std::invalid_argument>{"Matrix A must be square"};
        if (checkSingular)
            if (isSingular(A)) throw StackError<std::invalid_argument>{"Matrix A is singular"};

        auto S = A;
        QType<type, number, n, n> Q(nC, nC);
        RType<type, number, n, n> R(nC, nC);
        auto QProd = I<number, n>(nC);

        bool converged{false};
        size_t i{1};
        for (; i < iters; ++i) {
            const auto& res = QR<type>(S, method);
            Q = res.first;
            R = res.second;
            QProd = QProd * Q;

            S = R * Q;

            if (isUpperTriangular(S)) {
                converged = true;
                logging::log(format("Converged after {} iterations", i), "schurCommon");
                break;
            }
        }
        if (!converged) logging::log(format("Did not converge after {} iterations", i), "schurCommon");

        return pair{QProd, S};
  }

    /**
     * @brief Compute the Schur decomposition of a square matrix A.
     *
     * @param A The square matrix to decompose.
     * @param checkSingular If true, checks if the matrix is singular before proceeding with the decomposition.
     * @param method The method to use for QR decomposition (default is Householder).
     * @param iters The maximum number of iterations to perform for convergence (default is 10,000).
     * @return A tuple containing:
     *         - \f$Q\in\mathbb{R}^{n\times n}\f$: the orthogonal matrix,
     *         - \f$S\in\mathbb{R}^{n\times n}\f$: the Schur matrix (upper triangular),
     *         - \f$Q^T\in\mathbb{R}^{n\times n}\f$: the transpose of the orthogonal matrix Q.
     */
    template <QRType type, Number number, int n>
    auto schur(const Matrix<number, n, n>& A, bool checkSingular = true, QRMethod method = QRMethod::Householder,
               size_t iters = 10'000) {
        const auto& [Q, S] = schurCommon<type>(A, checkSingular, method, iters);

        return std::tuple{Q, S, helpers::extractMatrixFromTranspose(Q.T())};
    }

    /**
     * @brief Compute the eigenvalues and eigenvectors of a square matrix A using the QR algorithm.
     *
     * Uses Schur decomposition to find the eigenvalues and eigenvectors, iteratively refining the
     * decomposition until convergence. The eigenvalues are the diagonal elements of the Schur matrix S,
     * and the eigenvectors are the columns of the orthogonal matrix Q.
     *
     * @param A The square matrix to decompose.
     * @param checkSingular If true, checks if the matrix is singular before proceeding with the decomposition.
     * @param method The method to use for QR decomposition (default is Householder).
     * @param iters The maximum number of iterations to perform for convergence (default is 10,000).
     * @return A pair containing:
     *         - A vector of eigenvalues,
     *         - A vector of eigenvectors (as column vectors).
     */
    template <QRType type, Number number, int n>
    auto eigenQR(const Matrix<number, n, n>& A, bool checkSingular = true, QRMethod method = QRMethod::Householder,
                 size_t iters = 10'000) {
        const auto [nR, nC] = A.shape();
        const auto& [Q, S] = schurCommon<type>(A, checkSingular, method, iters);

        const auto& values = diag(S);
        const auto& vectors = Q.colToVectorSet();

        return pair{values, vectors};
    }

    template <Number number, int n>
    auto eigenQR(const Matrix<number, n, n>& A, bool checkSingular = true, QRMethod method = QRMethod::Householder,
                 size_t iters = 10'000) {
        return eigenQR<QRType::Thin>(A, checkSingular, method, iters);
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
     * @param sys The matrix to decompose.
     * @return A std::tuple containing
     *         - \f$U\in\mathbb{R}^{m\times r}\f$: the left singular vectors,
     *         - \f$\Sigma\in\mathbb{R}^{r\times r}\f$: the diagonal matrix of singular values,
     *         - \f$V\in\mathbb{R}^{n\times r}\f$: the right singular vectors.
     */
    template <Number number, int m, int n>
    auto svdEigen(const LinearSystem<number, m, n>& A) {
        // Find the eigen values and eigenvectors of A*A^T
        const auto [nR, nC] = A.shape();
        const auto& ATA{helpers::extractMatrixFromTranspose(A.T() * A)};
        // logging::log(format("SVD ATA: {}", ATA), "svdEigen");

        vector<Vector<number, n>> v;
        vector<number> l;
        try {
            auto [lam, val] = eigenQR(ATA, false);
            l = std::move(lam);
            v = std::move(val);
            // const auto& A = helpers::padMatrixToSquare<number, m, n>(sys);
            // logging::log(format("A: {}", A), "svdEigen");

            const auto& p = helpers::sortPermutation(l.begin(), l.end(), std::greater<>());
            helpers::applySortPermutation(l, p);
            helpers::applySortPermutation(v, p);
            // logging::log(format("SVD Eigenvalues: {}", l), "svdEigen");
            // logging::log(format("SVD Eigenvectors: {}", v), "svdEigen");
        } catch (const std::exception& e) {
            logging::E(format("Error finding eigen values and vectors -> {}", e.what()), "svdEigen");
            throw e;
        }

        // Construct the Sigma matrix from the eigenvalues by taking the square root of the eigenvalues.
        const auto minDim = std::min(nR, nC);
        auto Sigma = matrixZeros<number, m, n>(nR, nC);
        try {
            for (size_t i{}; i < minDim; i++) {
                Sigma(i, i) = l[i] > number(0) ? sqrt(l[i]) : number(0);
            }
            // logging::log(format("SVD Sigma: {}", Sigma), "svdEigen");
        } catch (const std::exception& e) {
            logging::E(format("Error construction Sigma -> {}", e.what()), "svdEigen");
            throw e;
        }

        // Construct the V matrix from the eigenvectors.
        const auto& V = helpers::fromColVectorSet<number, ATA.rows, ATA.cols>(v);
        // logging::log(format("SVD V: {}", V), "svdEigen");

        // Construct the U matrix using the eigenvectors and the Sigma matrix.
        Matrix<number, m, m> U(nR, nR);
        try {
            vector<Vector<number, m>> us;
            us.reserve(minDim);
            size_t foundOrthCols{};
            for (size_t i{}; i < minDim; ++i) {
                // Normalize the eigenvectors and multiply by the corresponding singular value.
                const auto& sigma = Sigma(i, i);
                const auto& Avi = A * v[i];
                if (!fuzzyCompare(sigma, number(0))) {
                    const auto& ui = Avi / sigma;
                    us.emplace_back(ui);
                    foundOrthCols++;
                }
            }
            vector<Vector<number, m>> qs;
            // If the number of orthonormal columns found is less than the maximum dimension,
            // for example as a result of a zero singular value, generate the remaning columns
            // to complete the orthonormal basis.
            if (foundOrthCols < nR) {
                qs = extendToCompleteBasis(us, nR);
            } else
                qs = us;

            U = helpers::fromColVectorSet<number, m, m>(qs);
        } catch (const std::exception& e) {
            logging::E(format("Error constructiong U -> {}", e.what()), "svdEigen");
            throw e;
        }

        const auto& VT = helpers::extractMatrixFromTranspose(V.T());

        // logging::log(format("SVD U: {}", U), "svdEigen");

        // logging::log(format("A = U * Sigma * V^T: {}", U * Sigma * VT), "svdEigen");

        return std::tuple{U, Sigma, VT};
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
