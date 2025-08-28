/**
 * @file Solve.hpp
 * @brief Header file for solving linear systems of equations.
 */

#pragma once

#include <cstdint>
#include <optional>
#include <utility>

#include "../Concepts.hpp"
#include "../Helpers.hpp"
#include "../Numeric.hpp"
#include "../structures/Aliases.hpp"
#include "Aliases.hpp"
#include "Builders.hpp"
#include "Decomposition.hpp"
#include "Pseudoinverse.hpp"
#include "Spaces.hpp"

namespace mlinalg {
    using std::nullopt;

    /**
     * @brief Solves a linear equation using the given row and solutions.
     *
     * @param row The row to solve.
     * @param varPos The position of the variable to solve for.
     * @param n The number of elements in the row.
     * @param solutions The already found values of other variables in the row.
     * @return The solution to the equation if it exists, nullopt otherwise.
     */
    template <Number number, Dim n>
    optional<number> solveEquation(const Row<number, n>& row, size_t varPos,
                                   const ConditionalRowOptional<number, n, n>& solutions) {
        const auto rowSize = row.size();
        number rightSum{row.back()};

        for (size_t i{}; i < rowSize - 1; i++) {
            const auto& var = solutions.at(i);
            if (i != varPos) {
                number res{};
                if (var.has_value()) {
                    res = var.value() * row.at(i) * number(-1);
                } else {
                    continue;  // Skip if the variable is not known
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
     * @brief Solve an upper triangular system of equations
     *
     * @param sys The upper triangular system of equations to solve.
     * @param b  The right-hand side vector of the system.
     * @return The solution vector x to the system.
     */
    template <Number number, Dim m, Dim n>
    Vector<number, m> solveUpperTriangular(const Matrix<number, m, n>& sys, const Vector<number, n>& b) {
        const auto& A = sys;
        const auto [numRows, numCols] = A.shape();
        const Dim nR = numRows;
        if (!isUpperTriangular(A))
            throw StackError<std::invalid_argument>("Matrix A must be upper triangular for this solve");

        Vector<number, m> x(nR);
        number back{};
        for (int i{nR - 1}; i >= 0; --i) {
            back = 0;
            for (int j{i + 1}; j <= nR - 1; ++j) {
                back += x[j] * (A(i, j));
            }
            x[i] = (b[i] - back) / (A(i, i));
        }
        return x;
    }

    /**
     * @brief Solve a linear system of equations using QR decomposition.
     *
     * @param A The matrix of the linear system.
     * @param b The right-hand side vector of the linear system.
     * @param method The method to use for QR decomposition (default is Gram-Schmidt).
     * @return The solution vector x to the system.
     */
    template <Number number, Dim m, Dim n>
    Vector<number, m> solveQR(const Matrix<number, m, n>& A, const Vector<number, n>& b, QRMethod method) {
        const auto [nR, nC] = A.shape();
        const auto bSize = b.size();
        if (nR != bSize) throw StackError<invalid_argument>("The matrix and vector are incompatible");

        const auto& [Q, R] = QR<QRType::Thin>(A, method);
        const auto& rhs = Q * b;
        const auto& x = solveUpperTriangular(R, rhs);
        return x;
    }

    template <Number number, Dim m, Dim n>
    auto nulspace(const Matrix<number, m, n>& A) {
        const auto [nR, nC] = A.shape();
        const auto& zero{vectorZeros<number, m>((int)nR)};
        const auto& x0 = findSolutions(A, zero);
        if (!x0.has_value()) throw StackError<runtime_error>{"Cannot solve Ax = 0"};
        const auto& x = extractSolutionVector(x0.value());
        return x;
    }

    /**
     * @brief Solve a linear system of equations in the form:
     * \f[
     * A \mathbf{x} = \mathbf{b}
     * \f]
     * Using the pseudoinverse of the matrix A, i.e.:
     * \f[
     * \mathbf{x} = A^+ \mathbf{b}
     * \f]
     *
     * @param A The matrix of the linear system.
     * @param b The right-hand side vector of the linear system.
     * @return The solution vector x to the system.
     */
    template <Number number, Dim m, Dim n>
    auto solveLeastSquares(const Matrix<number, m, n>& A, const Vector<number, m>& b) {
        const auto& AInv = pinv(A);
        return AInv * b;
    }

    /**
     * @brief Solve a linear system of equations in the form:
     * \f[
     * \begin{bmatrix}
     * A & \mathbf{b}
     * \end{bmatrix}
     * \f]
     * Using the pseudoinverse of the matrix A, i.e.:
     * \f[
     * \mathbf{x} = A^+ \mathbf{b}
     * \f]
     *
     * @param sys The linear system to solve, represented as an augmented matrix.
     * @return The solution vector x to the system.
     */
    template <Number number, Dim m, Dim n>
    auto solveLeastSquares(const LinearSystem<number, m, n>& sys) {
        const auto& A = sys.template slice<0, m, 0, n - 1>();
        const auto& b = sys.template slice<0, m, n - 1, n>();
        const auto& bT = helpers::extractVectorFromTranspose(b.T());
        return solveLeastSquares(A, bT);
    }

    template <Number number, Dim m, Dim n>
    auto solveLeastSquares(const LinearSystem<number, m, n>& sys)
        requires(m == Dynamic || n == Dynamic)
    {
        const auto& [nR, nC] = sys.shape();
        const auto& A = sys.slice({0, nR}, {0, nC - 1});
        const auto& b = sys.slice({0, nR}, {nC - 1, nC});
        const auto& bT = helpers::extractVectorFromTranspose(b.T());
        return solveLeastSquares(A, bT);
    }

    enum class SolveMode : uint8_t {
        AUTO = 0,   // Defaults to exact, then least squares if exact fails
        EXACT = 1,  // Exact solution, rref and solve
        LSTS = 2,   // Least squares solution, pseudoinverse
    };

    /**
     * @brief Result of the solve operation, which can either be an exact solution or a least squares solution.
     * TODO: Add support for representing a system with infinite solutions in parametric form.
     *
     * @param sols The solutions to the system, which can be either exact or least squares.
     */
    template <SolveMode mode, Number number, Dim m, Dim n>
    struct SolveResult {
       public:
        using ExactSolutions = ConditionalOptionalRowOptional<number, m, n>;
        using LeastSquaresSolutions = ConditionalRow<number, m, n>;
        using Solutions = std::variant<ExactSolutions, LeastSquaresSolutions>;
        SolveMode UsedMode = mode;

        SolveResult(Solutions sols) : solutions{sols} {}

        ExactSolutions exactSolutions() const {
            if (std::holds_alternative<ExactSolutions>(solutions)) {
                return std::get<ExactSolutions>(solutions);
            }
            throw StackError<std::invalid_argument>("No exact solutions available");
        }

        LeastSquaresSolutions leastSquaresSolutions() const {
            if (std::holds_alternative<LeastSquaresSolutions>(solutions)) {
                return std::get<LeastSquaresSolutions>(solutions);
            }
            throw StackError<invalid_argument>("No least squares solutions available");
        }

       private:
        Solutions solutions;
    };

    /**
     * @brief Solve a linear system of equations in the form:
     * \f[
     * \begin{bmatrix}
     * A & \mathbf{b}
     * \end{bmatrix}
     * \f]
     *
     * Using Gauss-Jordan elimination to find the exact solutions.
     *
     * @param sys
     * @return
     */
    template <Number number, Dim m, Dim n>
    auto solveExact(const LinearSystem<number, m, n>& sys) -> ConditionalOptionalRowOptional<number, m, n> {
        const auto [numRows, numCols] = sys.shape();
        ConditionalRowOptional<number, m, n> solutions(numCols - 1);
        LinearSystem<number, m, n> system{sys};

        auto reducedEchelon = rref(system);

        if (isInconsistent(reducedEchelon)) {
            // return nullopt to fill the exactSolutions variant
            return nullopt;
        }

        if (isSystemUnderdetermined(reducedEchelon)) {
            throw StackError<std::invalid_argument>(
                "Underdetermined systems are not supported in this mode, use LSTS or AUTO instead");
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
     * @brief Find the solutions to a linear system in the form:
     *
     * \f[
     * \begin{bmatrix}
     * A & \mathbf{b}
     * \end{bmatrix}
     * \]
     * \f]
     *
     * @param system The linear system to find the solutions of.
     * @return The solutions to the system if they exist, nullopt otherwise.
     */
    template <SolveMode mode = SolveMode::EXACT, Number number, Dim m, Dim n>
    auto findSolutions(const LinearSystem<number, m, n>& sys) -> SolveResult<mode, number, m, n> {
        const auto [numRows, numCols] = sys.shape();
        using SolveResult = SolveResult<mode, number, m, n>;
        typename SolveResult::Solutions sols(std::in_place_type<typename SolveResult::LeastSquaresSolutions>,
                                             typename SolveResult::LeastSquaresSolutions(numRows - 1));
        switch (mode) {
                // AUTO mode tries to find an exact solution first, and if it fails, it falls back to least squares.
            case SolveMode::AUTO:
                try {
                    const auto& exactSols = solveExact<number, m, n>(sys);
                    if (!exactSols.has_value()) {
                        logging::D("The system is inconsistent, trying least squares solution", "findSolutions");
                        throw StackError<std::invalid_argument>(
                            "The system is inconsistent, trying least squares solution");
                    }
                    sols = std::move(exactSols);
                } catch (const std::invalid_argument& e) {
                    sols = solveLeastSquares(sys);
                }
                break;
            // EXACT mode tries to find an exact solution using Gauss-Jordan elimination.
            case SolveMode::EXACT: {
                sols = solveExact<number, m, n>(sys);
                break;
            }
                // LSTS mode uses the pseudoinverse to find a least squares solution.
            case SolveMode::LSTS: {
                sols = solveLeastSquares(sys);
                break;
            }
            default:
                throw StackError<std::invalid_argument>("Invalid solve mode should not happen");
        }
        return SolveResult(sols);
    }

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
    template <SolveMode mode = SolveMode::EXACT, Number number, Dim m, Dim n>
    auto findSolutions(const Matrix<number, m, n>& A, const Vector<number, m>& b)
        -> SolveResult<mode, number, m, n + 1> {
        const auto& [nR, nC] = A.shape();
        using SolveResult = SolveResult<mode, number, m, n + 1>;
        typename SolveResult::Solutions sols(std::in_place_type<typename SolveResult::LeastSquaresSolutions>,
                                             typename SolveResult::LeastSquaresSolutions(nR));
        switch (mode) {
            case SolveMode::AUTO: {
                try {
                    auto sys = A.augment(b);
                    const auto& exactSols = findSolutions<mode, number, sys.rows, sys.cols>(sys).exactSolutions();
                    if (!exactSols.has_value()) {
                        logging::D("The system is inconsistent, trying least squares solution", "findSolutions");
                        throw StackError<std::invalid_argument>(
                            "The system is inconsistent, trying least squares solution");
                    }
                    sols = std::move(exactSols);
                } catch (const std::invalid_argument& e) {
                    sols = solveLeastSquares(A, b);
                }
                break;
            }
            case SolveMode::EXACT: {
                auto system = A.augment(b);
                sols = findSolutions<mode, number, system.rows, system.cols>(system).exactSolutions();
                break;
            }
            case SolveMode::LSTS: {
                sols = solveLeastSquares(A, b);
                break;
            }
            default:
                throw StackError<std::invalid_argument>("Invalid solve mode should not happen");
        }
        return SolveResult(sols);
    }

    template <SolveMode mode = SolveMode::EXACT, Number number, Dim m, Dim n>
    auto findSolutions(const Matrix<number, m, n>& A, const Vector<number, m>& b) -> SolveResult<mode, number, m, n>
        requires(m == Dynamic || n == Dynamic)
    {
        const auto& [nR, nC] = A.shape();
        using SolveResult = SolveResult<mode, number, m, n>;
        typename SolveResult::Solutions sols(std::in_place_type<typename SolveResult::LeastSquaresSolutions>,
                                             typename SolveResult::LeastSquaresSolutions(nR));
        switch (mode) {
            case SolveMode::AUTO: {
                try {
                    auto sys = A.augment(b);
                    const auto& exactSols = findSolutions<mode, number, sys.rows, sys.cols>(sys).exactSolutions();
                    if (!exactSols.has_value()) {
                        logging::D("The system is inconsistent, trying least squares solution", "findSolutions");
                        throw StackError<std::invalid_argument>(
                            "The system is inconsistent, trying least squares solution");
                    }
                    sols = std::move(exactSols);
                } catch (const std::invalid_argument& e) {
                    sols = solveLeastSquares(A, b);
                }
                break;
            }
            case SolveMode::EXACT: {
                auto system = A.augment(b);
                sols = findSolutions<mode, number, system.rows, system.cols>(system).exactSolutions();
                break;
            }
            case SolveMode::LSTS: {
                sols = solveLeastSquares(A, b);
                break;
            }
            default:
                throw StackError<std::invalid_argument>("Invalid solve mode should not happen");
        }
        return SolveResult(sols);
    }

}  // namespace mlinalg
