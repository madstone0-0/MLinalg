/**
 * @file examples.cpp
 * @brief Demonstrates the functionalities and interface of the MLinalg library.
 *
 * This file provides various usage examples of the MLinalg linear algebra library.
 * It shows how to construct matrices and vectors, solve linear systems (including
 * both consistent and inconsistent cases), and perform operations such as addition,
 * multiplication, transposition, and computing row echelon forms.
 *
 */

#include <format>
#include <iostream>

#include "MLinalg.hpp"
#include "operations/Solve.hpp"

using std::cout;

/*
 * @brief Solves and prints the solution of a linear system given as a coefficient matrix and right-hand side vector.
 *
 * The function augments the coefficient matrix with the vector and attempts to solve the system using
 * MLinalg::findSolutions. It then prints the augmented matrix and the computed solution(s), or indicates if the system
 * is inconsistent.
 *
 * @tparam number Numeric type of the matrix elements.
 * @tparam m Number of rows.
 * @tparam n Number of columns.
 * @param A The coefficient matrix.
 * @param b The right-hand side vector.
 */
template <Number number, int m, int n>
void printSol(const mlinalg::Matrix<number, m, n>& A, const mlinalg::Vector<number, m>& b) {
    using namespace mlinalg;
    try {
        const auto& sols{findSolutions<SolveMode::AUTO>(A, b)};
        cout << std::format("For the system (augmented matrix):\n");
        cout << A.augment(b);
        cout << std::format("\nThe solutions are:\n");

        try {
            const auto& exactSols = sols.exactSolutions();
            const auto& x = extractSolutionVector(exactSols.value());
            cout << "Exact solutions: " << x << '\n';
        } catch (const std::exception& e) {
            const auto& leastSquaresSols = sols.leastSquaresSolutions();
            cout << "Least squares solutions: " << leastSquaresSols << '\n';
        }

        // if (!sol.has_value()) {
        //     cout << std::format("The system is inconsistent\n\n");
        //     return;
        // }
        // for (const auto& val : sol.value()) {
        //     if (val.has_value())
        //         cout << val.value() << " ";
        //     else
        //         cout << "None ";
        // }
        // cout << "\n";
    } catch (const std::exception& e) {
        cout << e.what() << "\n";
    }
}

/**
 * @brief Solves and prints the solution of a linear system provided as a LinearSystem object.
 *
 * This overload uses an MLinalg::LinearSystem object directly. It prints the system and then
 * its solution (if one exists), or indicates if the system is inconsistent.
 *
 * @tparam number Numeric type of the linear system.
 * @tparam m Number of rows.
 * @tparam n Number of columns.
 * @param system The linear system.
 */
template <Number number, int m, int n>
void printSol(const mlinalg::LinearSystem<number, m, n>& system) {
    using namespace mlinalg;
    try {
        const auto& sols{findSolutions<SolveMode::AUTO>(system)};
        cout << std::format("For the system:\n");
        cout << system;
        cout << std::format("\nThe solutions are:\n");

        try {
            const auto& exactSols = sols.exactSolutions();
            const auto& x = extractSolutionVector(exactSols.value());
            cout << "Exact solutions: " << x << '\n';
        } catch (const std::exception& e) {
            const auto& leastSquaresSols = sols.leastSquaresSolutions();
            cout << "Least squares solutions: " << leastSquaresSols << '\n';
        }

        // if (!sol.has_value()) {
        //     cout << std::format("The system is inconsistent\n\n");
        //     return;
        // }
        // for (const auto& val : sol.value()) {
        //     if (val.has_value())
        //         cout << val.value() << " ";
        //     else
        //         cout << "None ";
        // }
        // cout << "\n";
    } catch (const std::exception& e) {
        cout << e.what() << "\n";
    }
}

/**
 * @brief Main entry point demonstrating various MLinalg functionalities.
 *
 * This function creates several matrices, vectors, and linear systems to demonstrate:
 * - Construction of matrices and vectors.
 * - Solving linear systems (both consistent and inconsistent).
 * - Augmenting matrices with vectors.
 * - Matrix and scalar operations, such as multiplication, addition, and transposition.
 * - Calculation of row echelon forms (REF and RREF).
 *
 * @return int Exit code.
 */
int main() {
    using namespace mlinalg;

    // --- Linear System Examples ---

    // A consistent linear system of size 3x4 (3 equations, 4 unknowns)
    auto sys1 = LinearSystem<double, 3, 4>{{{1, -2, 1, 0}, {0, 2, -8, 8}, {5, 0, -5, 10}}};

    // Another consistent system with a different number of unknowns (3x5)
    LinearSystem<double, 3, 5> sys2{{
        {3, 0, -1, 0, 0},
        {8, 0, 0, -2, 0},
        {8, 2, -2, 1, 0},
    }};

    // A third system (3x4)
    LinearSystem<double, 3, 4> sys9 = {
        {2, 0, -6, -8},
        {0, 1, 2, 3},
        {3, 6, -2, -4},
    };

    // A square system (3x3)
    auto sys4 = LinearSystem<double, 3, 3>{
        {2, -1, 0},
        {-3, 1, -1},
        {2, -3, 4},
    };

    // A 3x5 system with more unknowns
    LinearSystem<double, 3, 5> sys3 = {
        {0.35, -0.1, -0.25, 0, 0},
        {-0.3, 0.9, -0.35, -0.25, 0},
        {-0.3, -0.15, 0.85, -0.4, 0},
        {-0.2, -0.1, -0.3, 0.6, 0},
    };

    // --- Matrix Examples ---

    // A general 4x3 matrix.
    Matrix<double, 4, 3> sys5{
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9},
        {10, 11, 12},
    };

    // Matrices for multiplication examples.
    Matrix<double, 2, 2> mult1{
        {2, 3},
        {1, -5},
    };
    Matrix<double, 2, 3> mult2{
        {4, 3, 6},
        {1, -2, 3},
    };

    // A 3x3 linear system.
    LinearSystem<double, 3, 3> sys6{
        {27.6, 30.2, 162},
        {3100, 6400, 23610},
        {250, 360, 1623},
    };

    // Another 3x4 linear system.
    LinearSystem<double, 3, 4> sys7{
        {1, 3, -5, 0},
        {1, 4, -8, 0},
        {-3, -7, 9, 0},
    };

    // A symmetric matrix for demonstration.
    auto symmetric = Matrix<double, 3, 3>{{1, 2, 3}, {2, 4, 5}, {3, 5, 6}};

    // A 3x6 system to demonstrate REF and RREF functionalities.
    auto sys20 = LinearSystem<double, 3, 6>{
        {-3, 6, -1, 1, -7, 0},
        {1, -2, 2, 3, -1, 0},
        {2, -4, 5, 8, -4, 0},
    };

    // An inconsistent system example (2 equations, 3 unknowns).
    // The second equation is not a multiple of the first, causing inconsistency.
    LinearSystem<double, 2, 3> incSys{{1, 2, 3}, {2, 4, 8.1}};

    // --- Demonstrations ---

    // Solve and display solutions for various systems.
    printSol(sys1);
    printSol(sys2);
    printSol(sys9);
    printSol(sys4);
    printSol(sys3);
    printSol(sys6);
    printSol(sys7);

    // Demonstrate solving an inconsistent system.
    cout << std::format("Demonstrating an inconsistent system:\n");
    printSol(incSys);

    // Display results of matrix multiplication.
    cout << std::format("\nMatrix multiplication:\nmult1 * mult2 =\n");
    cout << mult1 * mult2 << "\n";

    // Demonstrate exponentiation (raising a matrix to a power).
    cout << std::format("\nMatrix exponentiation:\nsys4 raised to the 4th power =\n");
    cout << sys4 * sys4 * sys4 * sys4 << "\n";

    // Demonstrate transposition.
    cout << std::format("\nTranspose operations:\nTranspose of sys1 =\n");
    cout << sys1.T() << "\n";

    // Solve a system provided as separate matrix and vector.
    cout << std::format("\nSolving system provided as separate matrix and vector:\n");
    printSol(Matrix<double, 3, 3>{{{1, -2, 1}, {0, 2, -8}, {5, 0, -5}}}, Vector<double, 3>{1, 2, 3});

    // Display basic operations on sys1.
    cout << std::format("\nsys1 =\n");
    cout << sys1 << "\n";
    cout << std::format("sys1 multiplied by 2 =\n");
    cout << sys1 * 2 << "\n";
    cout << std::format("2 multiplied by sys1 =\n");
    cout << 2. * sys1 << "\n";

    // Demonstrate operations involving row vectors and matrices.
    cout << std::format("\nRow vector operations:\nRow<double, 2>{{1, 2}}.T() * Matrix =\n");
    cout << Row<double, 2>{1, 2}.T() * Matrix<double, 2, 2>{{{2, 1}, {2, 1}}} << "\n";
    cout << std::format("Row<double, 2>{{1, 2}} * Matrix =\n");
    cout << Row<double, 2>{1, 2} * Matrix<double, 2, 2>{{{2, 1}, {2, 1}}} << "\n";

    // Demonstrate operations on symmetric matrices.
    cout << std::format("\nSymmetric matrix operations:\nSymmetric matrix:\n");
    cout << symmetric << "\n";
    cout << std::format("Transpose of symmetric matrix:\n");
    cout << symmetric.T() << "\n";
    cout << std::format("symmetric * symmetric.T() =\n");
    cout << symmetric * symmetric.T() << "\n";
    cout << std::format("symmetric.T() * symmetric =\n");
    cout << symmetric.T() * symmetric << "\n";

    // Demonstrate multiplication of two 2x2 matrices.
    auto A = Matrix<double, 2, 2>{{{5, 1}, {3, -2}}};
    auto B = Matrix<double, Dynamic, Dynamic>{{{2, 0}, {4, 3}}};
    cout << std::format("\nMultiplication of two 2x2 matrices:\nA * B =\n");
    cout << A * B << "\n";
    cout << std::format("B * A =\n");
    cout << B * A << "\n";

    // Demonstrate computing the row echelon forms.
    cout << std::format("\nRow echelon forms for sys20:\nREF =\n");
    cout << ref(sys20) << "\n";
    cout << std::format("RREF =\n");
    cout << rref(sys20) << "\n";

    // Demonstrate vector arithmetic.
    auto shear = Matrix<double, Dynamic, Dynamic>{{{1, 2}, {0, 1}}};
    auto refMat = Matrix<double, 2, 2>{{{-1, 0}, {0, 1}}};
    auto e1 = Vector<double, 2>{1, 0};
    auto e2 = Vector<double, 2>{0, 1};

    cout << std::format("\nVector arithmetic:\ne2 + 2 * e1 =\n");
    cout << (e2 + 2. * e1) << "\n";
    cout << std::format("shear + refMat =\n");
    cout << shear + refMat << "\n";

    // Demonstrate vector operations: length and dot product.
    auto vec = Vector<double, 3>{4, 3, 12};
    cout << std::format("\nVector operations:\nLength of vector = {}\n", vec.length());
    cout << std::format("Dot product of vector with itself = {}\n", vec.dot(vec));
    cout << std::format("vec.T() * vec =\n");
    cout << vec.T() * vec << "\n";

    // Demonstrate multiplication of a matrix with a vector.
    cout << std::format("\nMultiplication of a matrix with a vector:\nrefMat * e1 =\n");
    cout << refMat * e1 << "\n";

    return 0;
}
