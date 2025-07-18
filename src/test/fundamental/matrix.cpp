#include <MLinalg.hpp>
#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>

#include "Helpers.hpp"
#include "Numeric.hpp"
#include "structures/Aliases.hpp"
#include "structures/Vector.hpp"

using namespace Catch;
using namespace mlinalg;
using namespace mlinalg::structures;
using namespace mlinalg::structures::helpers;

TEST_CASE("Matrix", "[matrix]") {
    auto sys1 = mlinalg::LinearSystem<double, 3, 4>{{
        {1, -2, 1, 0},
        {0, 2, -8, 8},
        {5, 0, -5, 10},
    }};

    SECTION("Creation") {
        SECTION("Compile Time") {
            SECTION("Default constructor") {
                auto m = Matrix<int, 2, 2>{};
                REQUIRE(m.at(0, 0) == 0);
                REQUIRE(m.at(0, 1) == 0);
                REQUIRE(m.at(1, 0) == 0);
                REQUIRE(m.at(1, 1) == 0);
                REQUIRE(m.numRows() == 2);
                REQUIRE(m.numCols() == 2);
            }

            SECTION("Initializer list constructor") {
                auto m = Matrix<int, 2, 2>{{1, 2}, {3, 4}};
                REQUIRE(m.at(0, 0) == 1);
                REQUIRE(m.at(0, 1) == 2);
                REQUIRE(m.at(1, 0) == 3);
                REQUIRE(m.at(1, 1) == 4);
                REQUIRE(m.numRows() == 2);
                REQUIRE(m.numCols() == 2);
            }

            SECTION("Copy constructor") {
                auto m = Matrix<int, 2, 2>{{1, 2}, {3, 4}};
                auto m2 = Matrix<int, 2, 2>{m};
                REQUIRE(m2.at(0, 0) == 1);
                REQUIRE(m2.at(0, 1) == 2);
                REQUIRE(m2.at(1, 0) == 3);
                REQUIRE(m2.at(1, 1) == 4);
                REQUIRE(m2.numRows() == 2);
                REQUIRE(m2.numCols() == 2);
            }

            SECTION("Move constructor") {
                auto m = Matrix<int, 2, 2>{{1, 2}, {3, 4}};
                auto m2 = Matrix<int, 2, 2>{std::move(m)};
                REQUIRE(m2.at(0, 0) == 1);
                REQUIRE(m2.at(0, 1) == 2);
                REQUIRE(m2.at(1, 0) == 3);
                REQUIRE(m2.at(1, 1) == 4);
                REQUIRE(m2.numRows() == 2);
                REQUIRE(m2.numCols() == 2);
            }

            SECTION("Copy assignment") {
                auto m = Matrix<int, 2, 2>{{1, 2}, {3, 4}};
                auto m2 = Matrix<int, 2, 2>{};
                m2 = m;
                REQUIRE(m2.at(0, 0) == 1);
                REQUIRE(m2.at(0, 1) == 2);
                REQUIRE(m2.at(1, 0) == 3);
                REQUIRE(m2.at(1, 1) == 4);
                REQUIRE(m2.numRows() == 2);
                REQUIRE(m2.numCols() == 2);
            }

            SECTION("From Column Vector Set") {
                auto v1 = Vector<int, 2>{1, 2};
                auto v2 = Vector<int, 2>{3, 4};
                auto m = fromColVectorSet<int, 2, 2>({v1, v2});
                REQUIRE(m.at(0, 0) == 1);
                REQUIRE(m.at(0, 1) == 3);
                REQUIRE(m.at(1, 0) == 2);
                REQUIRE(m.at(1, 1) == 4);
                REQUIRE(m.numRows() == 2);
                REQUIRE(m.numCols() == 2);
            }

            SECTION("From Row Vector Set") {
                auto v1 = Vector<int, 2>{1, 2};
                auto v2 = Vector<int, 2>{3, 4};
                auto m = fromRowVectorSet<int, 2, 2>({v1, v2});
                REQUIRE(m.at(0, 0) == 1);
                REQUIRE(m.at(0, 1) == 2);
                REQUIRE(m.at(1, 0) == 3);
                REQUIRE(m.at(1, 1) == 4);
                REQUIRE(m.numRows() == 2);
                REQUIRE(m.numCols() == 2);
            }
        }

        SECTION("Dynamic") {
            SECTION("Initializer List Construction") {
                auto A = Matrix<int, Dynamic, Dynamic>{{1, 2, 3}, {1, 2, 3}};
                REQUIRE(A.at(0, 0) == 1);
                REQUIRE(A.at(0, 1) == 2);
                REQUIRE(A.at(0, 2) == 3);
                REQUIRE(A.at(1, 0) == 1);
                REQUIRE(A.at(1, 1) == 2);
                REQUIRE(A.at(1, 2) == 3);
                REQUIRE(A.numRows() == 2);
                REQUIRE(A.numCols() == 3);
            }

            SECTION("Copy constructor") {
                auto A = Matrix<int, Dynamic, Dynamic>{{1, 2, 3}, {1, 2, 3}};
                auto B = Matrix<int, Dynamic, Dynamic>{A};
                REQUIRE(B.at(0, 0) == 1);
                REQUIRE(B.at(0, 1) == 2);
                REQUIRE(B.at(0, 2) == 3);
                REQUIRE(B.at(1, 0) == 1);
                REQUIRE(B.at(1, 1) == 2);
                REQUIRE(B.at(1, 2) == 3);
                REQUIRE(B.numRows() == 2);
                REQUIRE(B.numCols() == 3);
            }

            SECTION("Move constructor") {
                auto A = Matrix<int, Dynamic, Dynamic>{{1, 2, 3}, {1, 2, 3}};
                auto B = Matrix<int, Dynamic, Dynamic>{std::move(A)};
                REQUIRE(B.at(0, 0) == 1);
                REQUIRE(B.at(0, 1) == 2);
                REQUIRE(B.at(0, 2) == 3);
                REQUIRE(B.at(1, 0) == 1);
                REQUIRE(B.at(1, 1) == 2);
                REQUIRE(B.at(1, 2) == 3);
                REQUIRE(B.numRows() == 2);
                REQUIRE(B.numCols() == 3);
            }

            SECTION("Copy assignment") {
                auto A = Matrix<int, Dynamic, Dynamic>{{1, 2, 3}, {1, 2, 3}};
                auto B = Matrix<int, Dynamic, Dynamic>{{1, 2, 3}};
                B = A;
                REQUIRE(B.at(0, 0) == 1);
                REQUIRE(B.at(0, 1) == 2);
                REQUIRE(B.at(0, 2) == 3);
                REQUIRE(B.at(1, 0) == 1);
                REQUIRE(B.at(1, 1) == 2);
                REQUIRE(B.at(1, 2) == 3);
                REQUIRE(B.numRows() == 2);
                REQUIRE(B.numCols() == 3);
            }

            SECTION("Move assignment") {
                auto A = Matrix<int, Dynamic, Dynamic>{{1, 2, 3}, {1, 2, 3}};
                auto B = Matrix<int, Dynamic, Dynamic>{{1, 2, 3}};
                B = std::move(A);
                REQUIRE(B.at(0, 0) == 1);
                REQUIRE(B.at(0, 1) == 2);
                REQUIRE(B.at(0, 2) == 3);
                REQUIRE(B.at(1, 0) == 1);
                REQUIRE(B.at(1, 1) == 2);
                REQUIRE(B.at(1, 2) == 3);
                REQUIRE(B.numRows() == 2);
                REQUIRE(B.numCols() == 3);
            }
        }
    }

    SECTION("Operations") {
        SECTION("Compile Time") {
            SECTION("Addition") {
                auto m1 = Matrix<int, 2, 2>{{1, 2}, {3, 4}};
                auto m2 = Matrix<int, 2, 2>{{5, 6}, {7, 8}};
                auto m3 = m1 + m2;
                REQUIRE(m3.at(0, 0) == 6);
                REQUIRE(m3.at(0, 1) == 8);
                REQUIRE(m3.at(1, 0) == 10);
                REQUIRE(m3.at(1, 1) == 12);

                // Inplace Addition
                auto m4 = Matrix<int, 2, 2>{{1, 2}, {3, 4}};
                auto m5 = Matrix<int, 2, 2>{{5, 6}, {7, 8}};
                m4 += m5;
                REQUIRE(m4.at(0, 0) == 6);
                REQUIRE(m4.at(0, 1) == 8);
                REQUIRE(m4.at(1, 0) == 10);
                REQUIRE(m4.at(1, 1) == 12);
            }

            SECTION("Subtraction") {
                auto m1 = Matrix<int, 2, 2>{{1, 2}, {3, 4}};
                auto m2 = Matrix<int, 2, 2>{{5, 6}, {7, 8}};
                auto m3 = m1 - m2;
                REQUIRE(m3.at(0, 0) == -4);
                REQUIRE(m3.at(0, 1) == -4);
                REQUIRE(m3.at(1, 0) == -4);
                REQUIRE(m3.at(1, 1) == -4);

                // Inplace Subtraction
                auto m4 = Matrix<int, 2, 2>{{1, 2}, {3, 4}};
                auto m5 = Matrix<int, 2, 2>{{5, 6}, {7, 8}};
                m4 -= m5;
                REQUIRE(m4.at(0, 0) == -4);
                REQUIRE(m4.at(0, 1) == -4);
                REQUIRE(m4.at(1, 0) == -4);
                REQUIRE(m4.at(1, 1) == -4);
            }

            SECTION("Matrix Multiplication") {
                // Test case 1: Multiply two 2x2 matrices
                {
                    auto m1 = Matrix<int, 2, 2>{{1, 2}, {3, 4}};
                    auto m2 = Matrix<int, 2, 2>{{5, 6}, {7, 8}};
                    auto m3 = m1 * m2;

                    // Expected result
                    Matrix<int, 2, 2> expected{{19, 22}, {43, 50}};

                    // Verify the result
                    for (int i = 0; i < static_cast<int>(m3.numRows()); i++) {
                        for (int j = 0; j < static_cast<int>(m3.numCols()); j++) {
                            REQUIRE(m3.at(i, j) == expected.at(i, j));
                        }
                    }
                }

                // Test case 2: Multiply by identity matrix
                {
                    auto m1 = Matrix<int, 2, 2>{{1, 2}, {3, 4}};
                    auto identity = Matrix<int, 2, 2>{{1, 0}, {0, 1}};
                    auto m2 = m1 * identity;

                    // Verify that the result is the same as the original matrix
                    for (int i = 0; i < static_cast<int>(m1.numRows()); i++) {
                        for (int j = 0; j < static_cast<int>(m1.numCols()); j++) {
                            REQUIRE(m2.at(i, j) == m1.at(i, j));
                        }
                    }
                }

                // Test case 3: Multiply by zero matrix
                {
                    auto m1 = Matrix<int, 2, 2>{{1, 2}, {3, 4}};
                    auto zero = Matrix<int, 2, 2>{{0, 0}, {0, 0}};
                    auto m2 = m1 * zero;

                    // Verify that the result is a zero matrix
                    for (int i = 0; i < static_cast<int>(m2.numRows()); i++) {
                        for (int j = 0; j < static_cast<int>(m2.numCols()); j++) {
                            REQUIRE(m2.at(i, j) == 0);
                        }
                    }
                }

                // Test case 4: Multiply non-square matrices (if supported)
                {
                    auto m1 = Matrix<int, 2, 3>{
                        {1, 2, 3}, {4, 5, 6}  //
                    };
                    auto m2 = Matrix<int, 3, 2>{
                        {7, 8},
                        {9, 10},
                        {11, 12},
                    };
                    auto m3 = m1 * m2;

                    // Expected result
                    Matrix<int, 2, 2> expected{{58, 64}, {139, 154}};

                    // Verify the result
                    for (int i = 0; i < static_cast<int>(m3.numRows()); i++) {
                        for (int j = 0; j < static_cast<int>(m3.numCols()); j++) {
                            REQUIRE(m3.at(i, j) == expected.at(i, j));
                        }
                    }
                }

                // Test case 5: Multiply larger matrices (e.g., 4x4)
                {
                    auto m1 = Matrix<int, 4, 4>{
                        {1, 2, 3, 4},
                        {5, 6, 7, 8},
                        {9, 10, 11, 12},
                        {13, 14, 15, 16},
                    };
                    auto m2 = Matrix<int, 4, 4>{
                        {17, 18, 19, 20}, {21, 22, 23, 24}, {25, 26, 27, 28}, {29, 30, 31, 32}  //
                    };
                    auto m3 = m1 * m2;

                    // Expected result
                    Matrix<int, 4, 4> expected{
                        {250, 260, 270, 280},
                        {618, 644, 670, 696},
                        {986, 1028, 1070, 1112},
                        {1354, 1412, 1470, 1528},
                    };

                    // Verify the result
                    for (int i = 0; i < static_cast<int>(m3.numRows()); i++) {
                        for (int j = 0; j < static_cast<int>(m3.numCols()); j++) {
                            REQUIRE(m3.at(i, j) == expected.at(i, j));
                        }
                    }
                }

                // Test case 3: Multiply by transposed zero matrix
                {
                    auto m1 = Matrix<int, 2, 2>{{1, 2}, {3, 4}};
                    auto zero = Matrix<int, 2, 2>{{0, 0}, {0, 0}};
                    auto m2 = m1 * zero.T();

                    // Verify that the result is a zero matrix
                    for (int i = 0; i < static_cast<int>(m2.numRows()); i++) {
                        for (int j = 0; j < static_cast<int>(m2.numCols()); j++) {
                            REQUIRE(m2.at(i, j) == 0);
                        }
                    }
                }
            }

            SECTION("Scalar multiplication") {
                {
                    auto m1 = Matrix<int, 2, 2>{{1, 2}, {3, 4}};
                    auto m2 = m1 * 2;
                    REQUIRE(m2.at(0, 0) == 2);
                    REQUIRE(m2.at(0, 1) == 4);
                    REQUIRE(m2.at(1, 0) == 6);
                    REQUIRE(m2.at(1, 1) == 8);

                    // Inplace Multiplication
                    auto m3 = Matrix<int, 2, 2>{{1, 2}, {3, 4}};
                    m3 *= 2;
                    REQUIRE(m3.at(0, 0) == 2);
                    REQUIRE(m3.at(0, 1) == 4);
                    REQUIRE(m3.at(1, 0) == 6);
                    REQUIRE(m3.at(1, 1) == 8);
                }
                {
                    auto m1 = sys1 * 2;
                    REQUIRE(m1.at(0, 0) == Approx(2));
                    REQUIRE(m1.at(0, 1) == Approx(-4));
                    REQUIRE(m1.at(0, 2) == Approx(2));
                    REQUIRE(m1.at(0, 3) == Approx(0));
                    REQUIRE(m1.at(1, 0) == Approx(0));
                    REQUIRE(m1.at(1, 1) == Approx(4));
                    REQUIRE(m1.at(1, 2) == Approx(-16));
                    REQUIRE(m1.at(1, 3) == Approx(16));
                    REQUIRE(m1.at(2, 0) == Approx(10));
                    REQUIRE(m1.at(2, 1) == Approx(0));
                    REQUIRE(m1.at(2, 2) == Approx(-10));
                    REQUIRE(m1.at(2, 3) == Approx(20));

                    // Inplace Multiplication
                    auto m2 = sys1;
                    m2 *= 2;
                    REQUIRE(m2.at(0, 0) == Approx(2));
                    REQUIRE(m2.at(0, 1) == Approx(-4));
                    REQUIRE(m2.at(0, 2) == Approx(2));
                    REQUIRE(m2.at(0, 3) == Approx(0));
                    REQUIRE(m2.at(1, 0) == Approx(0));
                    REQUIRE(m2.at(1, 1) == Approx(4));
                    REQUIRE(m2.at(1, 2) == Approx(-16));
                    REQUIRE(m2.at(1, 3) == Approx(16));
                    REQUIRE(m2.at(2, 0) == Approx(10));
                    REQUIRE(m2.at(2, 1) == Approx(0));
                    REQUIRE(m2.at(2, 2) == Approx(-10));
                    REQUIRE(m2.at(2, 3) == Approx(20));
                }
            }

            SECTION("Scalar division") {
                auto m1 = Matrix<int, 2, 2>{{1, 2}, {3, 4}};
                auto m2 = m1 / 2;
                REQUIRE(m2.at(0, 0) == 0);
                REQUIRE(m2.at(0, 1) == 1);
                REQUIRE(m2.at(1, 0) == 1);
                REQUIRE(m2.at(1, 1) == 2);

                // Inplace Division
                auto m3 = Matrix<int, 2, 2>{{1, 2}, {3, 4}};
                m3 /= 2;
                REQUIRE(m3.at(0, 0) == 0);
                REQUIRE(m3.at(0, 1) == 1);
                REQUIRE(m3.at(1, 0) == 1);
                REQUIRE(m3.at(1, 1) == 2);

                // Test division by zero
                auto m4 = Matrix<int, 2, 2>{{1, 2}, {3, 4}};
                REQUIRE_THROWS_AS(m4 / 0, std::domain_error);
            }

            SECTION("Norms") {
                SECTION("Frobenius Norm") {
                    auto m1 = Matrix<int, 2, 2>{{1, 2}, {3, 4}};
                    auto norm = m1.frob();
                    REQUIRE(norm == Approx(5.477225575051661));
                }

                SECTION("L1 Norm") {
                    auto m1 = Matrix<int, 2, 2>{{1, 2}, {3, 4}};
                    auto norm = m1.l1();
                    REQUIRE(norm == Approx(6));
                }

                SECTION("L-Infinity Norm") {
                    auto m1 = Matrix<int, 2, 2>{{1, 2}, {3, 4}};
                    auto norm = m1.lInf();
                    REQUIRE(norm == Approx(7));
                }
            }

            SECTION("Trace") {
                auto m1 = Matrix<int, 2, 2>{{1, 2}, {3, 4}};
                auto trace = m1.trace();
                REQUIRE(trace == 5);
            }

            SECTION("Transpose") {
                SECTION("Square Matrix") {
                    auto m1 = Matrix<int, 2, 2>{{1, 2}, {3, 4}};
                    auto m1T = extractMatrixFromTranspose(m1.T());
                    INFO("Square matrix transpose");
                    REQUIRE(m1T.at(0, 0) == 1);
                    REQUIRE(m1T.at(0, 1) == 3);
                    REQUIRE(m1T.at(1, 0) == 2);
                    REQUIRE(m1T.at(1, 1) == 4);

                    // Verify that transposing twice returns the original matrix
                    auto m1TT = extractMatrixFromTranspose(m1T.T());
                    for (size_t i = 0; i < m1.numRows(); ++i) {
                        for (size_t j = 0; j < m1.numCols(); ++j) {
                            INFO("Double transpose element (" << i << "," << j << ")");
                            REQUIRE(m1.at(i, j) == m1TT.at(i, j));
                        }
                    }
                }

                SECTION("Non-Square Matrix") {
                    auto m2 = Matrix<int, 2, 3>{{1, 2, 3}, {4, 5, 6}};
                    auto m2T = extractMatrixFromTranspose(m2.T());
                    INFO("Non-square matrix dimensions");
                    REQUIRE(m2T.numRows() == 3);
                    REQUIRE(m2T.numCols() == 2);

                    // Check element-wise correctness
                    REQUIRE(m2T.at(0, 0) == 1);
                    REQUIRE(m2T.at(0, 1) == 4);
                    REQUIRE(m2T.at(1, 0) == 2);
                    REQUIRE(m2T.at(1, 1) == 5);
                    REQUIRE(m2T.at(2, 0) == 3);
                    REQUIRE(m2T.at(2, 1) == 6);

                    // Verify double-transposition
                    auto m2TT = extractMatrixFromTranspose(m2T.T());
                    for (size_t i = 0; i < m2.numRows(); ++i) {
                        for (size_t j = 0; j < m2.numCols(); ++j) {
                            INFO("Double transpose element (" << i << "," << j << ")");
                            REQUIRE(m2.at(i, j) == m2TT.at(i, j));
                        }
                    }
                }

                SECTION("Matrix with Negative and Zero Elements") {
                    auto m3 = Matrix<int, 3, 3>{{0, -1, 2}, {3, 0, -4}, {5, -6, 0}};
                    auto m3T = extractMatrixFromTranspose(m3.T());
                    INFO("Matrix with negatives and zeros");
                    // Check that m3T(j,i) equals m3(i,j) for all elements
                    for (size_t i = 0; i < m3.numRows(); ++i) {
                        for (size_t j = 0; j < m3.numCols(); ++j) {
                            INFO("Comparing element (" << i << "," << j << ")");
                            REQUIRE(m3.at(i, j) == m3T.at(j, i));
                        }
                    }
                }

                SECTION("Linear System Transpose") {
                    auto sys1 = mlinalg::LinearSystem<double, 3, 4>{{
                        {1, -2, 1, 0},
                        {0, 2, -8, 8},
                        {5, 0, -5, 10},
                    }};
                    auto sys1T = extractMatrixFromTranspose(sys1.T());
                    INFO("Linear system transpose dimensions");
                    REQUIRE(sys1T.numRows() == 4);
                    REQUIRE(sys1T.numCols() == 3);

                    // Check element-wise that the transpose is correct.
                    for (size_t i = 0; i < sys1.numRows(); ++i) {
                        for (size_t j = 0; j < sys1.numCols(); ++j) {
                            INFO("Comparing sys1(" << i << "," << j << ") with sys1T(" << j << "," << i << ")");
                            REQUIRE(mlinalg::fuzzyCompare(sys1.at(i, j), sys1T.at(j, i)));
                        }
                    }
                }
            }

            SECTION("Matrix Vector Multiplication") {
                auto m = Matrix<int, 2, 2>{{1, 2}, {3, 4}};
                auto v = Vector<int, 2>{1, 2};

                // In this case v is a column vector
                auto v2 = m * v;
                REQUIRE(v2.at(0) == 5);
                REQUIRE(v2.at(1) == 11);

                // In this case v is a row vector
                auto v3 = v * m;
                REQUIRE(v3.at(0) == 7);
                REQUIRE(v3.at(1) == 10);
            }

            SECTION("Iteration") {
                auto m = Matrix<int, 2, 2>{{1, 2}, {3, 4}};
                const auto *it = m.begin();
                REQUIRE(*it == Vector<int, 2>{1, 2});
                REQUIRE(it != m.end());
                REQUIRE(*(++it) == Vector<int, 2>{3, 4});
                REQUIRE(++it == m.end());
            }

            SECTION("Determinant") {
                auto m = Matrix<int, 2, 2>{{1, 2}, {3, 4}};
                auto det = m.det();
                REQUIRE(det == -2);

                auto m2 = Matrix<int, 3, 3>{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
                auto det2 = m2.det();
                REQUIRE(det2 == 0);
            }

            SECTION("Augment") {
                SECTION("Matrix") {
                    auto m1 = Matrix<int, 2, 2>{{1, 2}, {3, 4}};
                    auto m2 = Matrix<int, 2, 2>{{5, 6}, {7, 8}};
                    auto m3 = m1.augment(m2);
                    REQUIRE(m3.at(0, 0) == 1);
                    REQUIRE(m3.at(0, 1) == 2);
                    REQUIRE(m3.at(0, 2) == 5);
                    REQUIRE(m3.at(0, 3) == 6);
                    REQUIRE(m3.at(1, 0) == 3);
                    REQUIRE(m3.at(1, 1) == 4);
                    REQUIRE(m3.at(1, 2) == 7);
                    REQUIRE(m3.at(1, 3) == 8);
                }

                SECTION("Vector") {
                    auto m1 = Matrix<int, 2, 2>{{1, 2}, {3, 4}};
                    auto v = Vector<int, 2>{5, 6};
                    auto m2 = m1.augment(v);
                    REQUIRE(m2.at(0, 0) == 1);
                    REQUIRE(m2.at(0, 1) == 2);
                    REQUIRE(m2.at(0, 2) == 5);
                    REQUIRE(m2.at(1, 0) == 3);
                    REQUIRE(m2.at(1, 1) == 4);
                    REQUIRE(m2.at(1, 2) == 6);
                }
            }

            SECTION("Comparisions") {
                // ------------------  Integer 32 (int) ------------------
                SECTION("Integer - Exact Equality") {
                    auto m1 = Matrix<int, 2, 2>{{1, 2}, {3, 4}};
                    auto m2 = Matrix<int, 2, 2>{{1, 2}, {3, 4}};
                    REQUIRE(m1 == m2);
                }

                SECTION("Integer - Exact Inequality") {
                    auto m1 = Matrix<int, 2, 2>{{1, 2}, {3, 4}};
                    auto m2 = Matrix<int, 2, 2>{{5, 6}, {7, 8}};
                    REQUIRE(m1 != m2);
                }

                // ------------------ Single Precision (float) ------------------
                SECTION("Float - Exact Equality") {
                    Matrix<float, 2, 2> m1{{1.0F, 2.0F}, {3.0F, 4.0F}};
                    Matrix<float, 2, 2> m2{{1.0F, 2.0F}, {3.0F, 4.0F}};
                    REQUIRE(m1 == m2);
                }

                SECTION("Float - Approximate Equality Within Tolerance") {
                    Matrix<float, 2, 2> m1{{1.000001F, 2.000001F}, {3.000001F, 4.000001F}};
                    Matrix<float, 2, 2> m2{{1.0F, 2.0F}, {3.0F, 4.0F}};
                    REQUIRE(m1 == m2);
                }

                SECTION("Float - Inequality Beyond Tolerance") {
                    Matrix<float, 2, 2> m1{{1.0F, 2.0F}, {3.0F, 4.0F}};
                    Matrix<float, 2, 2> m2{{1.0F, 2.0F}, {3.0F, 4.1F}};
                    REQUIRE(m1 != m2);
                }

                SECTION("Float - NaN Comparison") {
                    // NaN should not compare equal to NaN.
                    Matrix<float, 2, 2> m1{{std::numeric_limits<float>::quiet_NaN(), 2.0F}, {3.0F, 4.0F}};
                    Matrix<float, 2, 2> m2{{std::numeric_limits<float>::quiet_NaN(), 2.0F}, {3.0F, 4.0F}};
                    REQUIRE(m1 != m2);
                }

                SECTION("Float - Infinity Comparison") {
                    Matrix<float, 2, 2> m1{{std::numeric_limits<float>::infinity(), 2.0F}, {3.0F, 4.0F}};
                    Matrix<float, 2, 2> m2{{std::numeric_limits<float>::infinity(), 2.0F}, {3.0F, 4.0F}};
                    REQUIRE(m1 == m2);
                }

                // ------------------ Double Precision (double) ------------------
                SECTION("Double - Exact Equality") {
                    Matrix<double, 2, 2> m1{{1.0, 2.0}, {3.0, 4.0}};
                    Matrix<double, 2, 2> m2{{1.0, 2.0}, {3.0, 4.0}};
                    REQUIRE(m1 == m2);
                }

                SECTION("Double - Approximate Equality Within Tolerance") {
                    Matrix<double, 2, 2> m1{{1.000000001, 2.000000001}, {3.000000001, 4.000000001}};
                    Matrix<double, 2, 2> m2{{1.0, 2.0}, {3.0, 4.0}};
                    REQUIRE(m1 == m2);
                }

                SECTION("Double - Inequality Beyond Tolerance") {
                    Matrix<double, 2, 2> m1{{1.0, 2.0}, {3.0, 4.0}};
                    Matrix<double, 2, 2> m2{{1.0, 2.0}, {3.0, 4.0001}};
                    REQUIRE(m1 != m2);
                }

                SECTION("Double - NaN Comparison") {
                    Matrix<double, 2, 2> m1{{std::numeric_limits<double>::quiet_NaN(), 2.0}, {3.0, 4.0}};
                    Matrix<double, 2, 2> m2{{std::numeric_limits<double>::quiet_NaN(), 2.0}, {3.0, 4.0}};
                    REQUIRE(m1 != m2);
                }

                SECTION("Double - Infinity Comparison") {
                    Matrix<double, 2, 2> m1{{std::numeric_limits<double>::infinity(), 2.0}, {3.0, 4.0}};
                    Matrix<double, 2, 2> m2{{std::numeric_limits<double>::infinity(), 2.0}, {3.0, 4.0}};
                    REQUIRE(m1 == m2);
                }
            }

            SECTION("Subset") {
                auto m1 = Matrix<int, 3, 3>{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
                auto m2 = m1.subset(1, 1);
                REQUIRE(m2.at(0, 0) == 1);
                REQUIRE(m2.at(0, 1) == 3);
                REQUIRE(m2.at(1, 0) == 7);
                REQUIRE(m2.at(1, 1) == 9);
            }

            SECTION("Slice") {
                // Test case 1: Slice a 3x3 matrix to get a 2x2 submatrix

                auto m1 = Matrix<int, 3, 3>{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
                auto sliced1 = m1.slice<0, 2, 0, 2>();
                REQUIRE(sliced1.numRows() == 2);
                REQUIRE(sliced1.numCols() == 2);
                REQUIRE(sliced1.at(0, 0) == 1);
                REQUIRE(sliced1.at(0, 1) == 2);
                REQUIRE(sliced1.at(1, 0) == 4);
                REQUIRE(sliced1.at(1, 1) == 5);

                // Test case 2: Slice a 4x4 matrix to get a 2x2 submatrix from the top-left corner

                auto m2 = Matrix<int, 4, 4>{
                    {1, 2, 5, 6},
                    {3, 4, 7, 8},
                    {9, 10, 13, 14},
                    {11, 12, 15, 16},
                };
                constexpr int i1{m2.numRows() / 2};
                constexpr int j1{m2.numCols() / 2};
                auto sliced2 = m2.slice<0, i1, 0, j1>();
                REQUIRE(sliced2.numRows() == 2);
                REQUIRE(sliced2.numCols() == 2);
                REQUIRE(sliced2.at(0, 0) == 1);
                REQUIRE(sliced2.at(0, 1) == 2);
                REQUIRE(sliced2.at(1, 0) == 3);
                REQUIRE(sliced2.at(1, 1) == 4);

                // Test case 3: Slice a 4x4 matrix to get a single row

                auto m3 = Matrix<int, 4, 4>{
                    {1, 2, 3, 4},
                    {5, 6, 7, 8},
                    {9, 10, 11, 12},
                    {13, 14, 15, 16},
                };
                auto sliced3 = m3.slice<1, 2, 0, 4>();
                REQUIRE(sliced3.numRows() == 1);
                REQUIRE(sliced3.numCols() == 4);
                REQUIRE(sliced3.at(0, 0) == 5);
                REQUIRE(sliced3.at(0, 1) == 6);
                REQUIRE(sliced3.at(0, 2) == 7);
                REQUIRE(sliced3.at(0, 3) == 8);

                // Test case 4: Slice a 4x4 matrix to get a single column

                auto m4 = Matrix<int, 4, 4>{
                    {1, 2, 3, 4},
                    {5, 6, 7, 8},
                    {9, 10, 11, 12},
                    {13, 14, 15, 16},
                };
                auto sliced4 = m4.slice<0, 4, 2, 3>();
                REQUIRE(sliced4.numRows() == 4);
                REQUIRE(sliced4.numCols() == 1);
                REQUIRE(sliced4.at(0, 0) == 3);
                REQUIRE(sliced4.at(1, 0) == 7);
                REQUIRE(sliced4.at(2, 0) == 11);
                REQUIRE(sliced4.at(3, 0) == 15);

                // Test case 5: Slice the entire matrix

                auto m5 = Matrix<int, 2, 2>{{1, 2}, {3, 4}};
                auto sliced5 = m5.slice<0, 2, 0, 2>();
                REQUIRE(sliced5.numRows() == 2);
                REQUIRE(sliced5.numCols() == 2);
                REQUIRE(sliced5.at(0, 0) == 1);
                REQUIRE(sliced5.at(0, 1) == 2);
                REQUIRE(sliced5.at(1, 0) == 3);
                REQUIRE(sliced5.at(1, 1) == 4);

                // Test case 9: Slice with negative indices (should be caught at compile time)

                auto m9 = Matrix<int, 3, 3>{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
                // This should fail at compile time, so it's not included in the runtime tests.
                // static_assert(!std::is_invocable_v<decltype(m9.slice<-1, 2, 0, 2>()), decltype(m9)>);
            }

            SECTION("Indexing") {
                auto m = Matrix<int, 3, 3>{
                    {1, 2, 3},
                    {4, 5, 6},
                    {7, 8, 9},
                };
                REQUIRE(m(0, 0) == 1);
                REQUIRE(m(0, 1) == 2);
                REQUIRE(m(0, 2) == 3);
                REQUIRE(m(1, 0) == 4);
                REQUIRE(m(1, 1) == 5);
                REQUIRE(m(1, 2) == 6);
                REQUIRE(m(2, 0) == 7);
                REQUIRE(m(2, 1) == 8);
                REQUIRE(m(2, 2) == 9);

                REQUIRE(m.at(0, 0) == 1);
                REQUIRE(m.at(0, 1) == 2);
                REQUIRE(m.at(0, 2) == 3);
                REQUIRE(m.at(1, 0) == 4);
                REQUIRE(m.at(1, 1) == 5);
                REQUIRE(m.at(1, 2) == 6);
                REQUIRE(m.at(2, 0) == 7);
                REQUIRE(m.at(2, 1) == 8);
                REQUIRE(m.at(2, 2) == 9);

                REQUIRE(m[0] == Vector<int, 3>{1, 2, 3});
                REQUIRE(m[1] == Vector<int, 3>{4, 5, 6});
                REQUIRE(m[2] == Vector<int, 3>{7, 8, 9});

                REQUIRE(m.at(0) == Vector<int, 3>{1, 2, 3});
                REQUIRE(m.at(1) == Vector<int, 3>{4, 5, 6});
                REQUIRE(m.at(2) == Vector<int, 3>{7, 8, 9});
            }

            SECTION("Views") {
                auto A = Matrix<int, 6, 6>{
                    {1, 2, 3, 4, 5, 6},        //
                    {7, 8, 9, 10, 11, 12},     //
                    {13, 14, 15, 16, 17, 18},  //
                    {19, 20, 21, 22, 23, 24},  //
                    {25, 26, 27, 28, 29, 30},  //
                    {31, 32, 33, 34, 35, 36},  //
                };
                auto V1 = A.view<2, 2>();
                REQUIRE(V1(0, 0) == 1);
                REQUIRE(V1(0, 1) == 2);
                REQUIRE(V1(1, 0) == 7);
                REQUIRE(V1(1, 1) == 8);

                auto V2 = A.view<3, 2>();
                REQUIRE(V2(0, 0) == 1);
                REQUIRE(V2(0, 1) == 2);
                REQUIRE(V2(1, 0) == 7);
                REQUIRE(V2(1, 1) == 8);
                REQUIRE(V2(2, 0) == 13);
                REQUIRE(V2(2, 1) == 14);

                auto V3 = A.view<3, 3>(0, 3);
                REQUIRE(V3(0, 0) == 4);
                REQUIRE(V3(0, 1) == 5);
                REQUIRE(V3(0, 2) == 6);
                REQUIRE(V3(1, 0) == 10);
                REQUIRE(V3(1, 1) == 11);
                REQUIRE(V3(1, 2) == 12);
                REQUIRE(V3(2, 0) == 16);
                REQUIRE(V3(2, 1) == 17);
                REQUIRE(V3(2, 2) == 18);

                REQUIRE_THROWS(A.view<7, 7>());
                REQUIRE_THROWS(A.view<6, 6>(7, 7));
            }
        }

        SECTION("Dynamic") {
            SECTION("Addition") {
                auto m1 = Matrix<int, Dynamic, Dynamic>{{1, 2}, {3, 4}};
                auto m2 = Matrix<int, Dynamic, Dynamic>{{5, 6}, {7, 8}};
                auto m3 = m1 + m2;
                REQUIRE(m3.at(0, 0) == 6);
                REQUIRE(m3.at(0, 1) == 8);
                REQUIRE(m3.at(1, 0) == 10);
                REQUIRE(m3.at(1, 1) == 12);

                // Inplace Addition
                auto m4 = Matrix<int, Dynamic, Dynamic>{{1, 2}, {3, 4}};
                m4 += m2;
                REQUIRE(m4.at(0, 0) == 6);
                REQUIRE(m4.at(0, 1) == 8);
                REQUIRE(m4.at(1, 0) == 10);
                REQUIRE(m4.at(1, 1) == 12);
            }

            SECTION("Subtraction") {
                auto m1 = Matrix<int, Dynamic, Dynamic>{{1, 2}, {3, 4}};
                auto m2 = Matrix<int, Dynamic, Dynamic>{{5, 6}, {7, 8}};
                auto m3 = m1 - m2;
                REQUIRE(m3.at(0, 0) == -4);
                REQUIRE(m3.at(0, 1) == -4);
                REQUIRE(m3.at(1, 0) == -4);
                REQUIRE(m3.at(1, 1) == -4);

                // Inplace Subtraction
                auto m4 = Matrix<int, Dynamic, Dynamic>{{1, 2}, {3, 4}};
                m4 -= m2;
                REQUIRE(m4.at(0, 0) == -4);
                REQUIRE(m4.at(0, 1) == -4);
                REQUIRE(m4.at(1, 0) == -4);
                REQUIRE(m4.at(1, 1) == -4);
            }

            SECTION("Matrix Multiplication") {
                // Test case 1: Multiply two 2x2 matrices
                {
                    auto m1 = Matrix<int, Dynamic, Dynamic>{{1, 2}, {3, 4}};
                    auto m2 = Matrix<int, Dynamic, Dynamic>{{5, 6}, {7, 8}};
                    auto m3 = m1 * m2;

                    // Expected result
                    Matrix<int, Dynamic, Dynamic> expected{{19, 22}, {43, 50}};

                    // Verify the result
                    for (size_t i{}; i < m3.numRows(); i++) {
                        for (size_t j{}; j < m3.numCols(); j++) {
                            REQUIRE(m3.at(i, j) == expected.at(i, j));
                        }
                    }
                }

                // Test case 2: Multiply by identity matrix
                {
                    auto m1 = Matrix<int, Dynamic, Dynamic>{{1, 2}, {3, 4}};
                    auto identity = Matrix<int, Dynamic, Dynamic>{{1, 0}, {0, 1}};
                    auto m2 = m1 * identity;

                    // Verify that the result is the same as the original matrix
                    for (size_t i{}; i < m1.numRows(); i++) {
                        for (size_t j{}; j < m1.numCols(); j++) {
                            REQUIRE(m2.at(i, j) == m1.at(i, j));
                        }
                    }
                }

                // Test case 3: Multiply by zero matrix
                {
                    auto m1 = Matrix<int, Dynamic, Dynamic>{{1, 2}, {3, 4}};
                    auto zero = Matrix<int, Dynamic, Dynamic>{{0, 0}, {0, 0}};
                    auto m2 = m1 * zero;

                    // Verify that the result is a zero matrix
                    for (size_t i{}; i < m2.numRows(); i++) {
                        for (size_t j{}; j < m2.numCols(); j++) {
                            REQUIRE(m2.at(i, j) == 0);
                        }
                    }
                }

                // Test case 4: Multiply non-square matrices (if supported)
                {
                    auto m1 = Matrix<int, Dynamic, Dynamic>{
                        {1, 2, 3}, {4, 5, 6}  //
                    };
                    auto m2 = Matrix<int, Dynamic, Dynamic>{
                        {7, 8},
                        {9, 10},
                        {11, 12},
                    };
                    auto m3 = m1 * m2;

                    // Expected result
                    Matrix<int, Dynamic, Dynamic> expected{{58, 64}, {139, 154}};

                    // Verify the result
                    for (size_t i{}; i < m3.numRows(); i++) {
                        for (size_t j{}; j < m3.numCols(); j++) {
                            REQUIRE(m3.at(i, j) == expected.at(i, j));
                        }
                    }
                }

                // Test case 5: Multiply larger matrices (e.g., 4x4)
                {
                    auto m1 = Matrix<int, Dynamic, Dynamic>{
                        {1, 2, 3, 4},
                        {5, 6, 7, 8},
                        {9, 10, 11, 12},
                        {13, 14, 15, 16},
                    };
                    auto m2 = Matrix<int, Dynamic, Dynamic>{
                        {17, 18, 19, 20},
                        {21, 22, 23, 24},
                        {25, 26, 27, 28},
                        {29, 30, 31, 32},
                    };
                    auto m3 = m1 * m2;

                    // Expected result
                    Matrix<int, Dynamic, Dynamic> expected{
                        {250, 260, 270, 280},
                        {618, 644, 670, 696},
                        {986, 1028, 1070, 1112},
                        {1354, 1412, 1470, 1528},
                    };

                    // Verify the result
                    for (size_t i{}; i < m3.numRows(); i++) {
                        for (size_t j{}; j < m3.numCols(); j++) {
                            REQUIRE(m3.at(i, j) == expected.at(i, j));
                        }
                    }
                }
            }

            SECTION("Scalar multiplication") {
                auto m1 = Matrix<int, Dynamic, Dynamic>{{1, 2}, {3, 4}};
                auto m2 = m1 * 2;
                REQUIRE(m2.at(0, 0) == 2);
                REQUIRE(m2.at(0, 1) == 4);
                REQUIRE(m2.at(1, 0) == 6);
                REQUIRE(m2.at(1, 1) == 8);

                // Inplace Multiplication
                auto m3 = Matrix<int, Dynamic, Dynamic>{{1, 2}, {3, 4}};
                m3 *= 2;
                REQUIRE(m3.at(0, 0) == 2);
                REQUIRE(m3.at(0, 1) == 4);
                REQUIRE(m3.at(1, 0) == 6);
                REQUIRE(m3.at(1, 1) == 8);
            }

            SECTION("Scalar division") {
                auto m1 = Matrix<int, Dynamic, Dynamic>{{1, 2}, {3, 4}};
                auto m2 = m1 / 2;
                REQUIRE(m2.at(0, 0) == 0);
                REQUIRE(m2.at(0, 1) == 1);
                REQUIRE(m2.at(1, 0) == 1);
                REQUIRE(m2.at(1, 1) == 2);

                // Inplace Division
                auto m3 = Matrix<int, Dynamic, Dynamic>{{1, 2}, {3, 4}};
                m3 /= 2;
                REQUIRE(m3.at(0, 0) == 0);
                REQUIRE(m3.at(0, 1) == 1);
                REQUIRE(m3.at(1, 0) == 1);
                REQUIRE(m3.at(1, 1) == 2);
            }

            SECTION("Norms") {
                SECTION("Frobenius Norm") {
                    auto m1 = Matrix<int, Dynamic, Dynamic>{{1, 2}, {3, 4}};
                    auto norm = m1.frob();
                    REQUIRE(norm == Approx(5.477225575051661));
                }

                SECTION("L1 Norm") {
                    auto m1 = Matrix<int, Dynamic, Dynamic>{{1, 2}, {3, 4}};
                    auto norm = m1.l1();
                    REQUIRE(norm == Approx(6));
                }

                SECTION("L-Infinity Norm") {
                    auto m1 = Matrix<int, Dynamic, Dynamic>{{1, 2}, {3, 4}};
                    auto norm = m1.lInf();
                    REQUIRE(norm == Approx(7));
                }
            }

            SECTION("Trace") {
                auto m1 = Matrix<int, Dynamic, Dynamic>{{1, 2}, {3, 4}};
                auto trace = m1.trace();
                REQUIRE(trace == 5);
            }

            SECTION("Transpose") {
                auto m1 = Matrix<int, Dynamic, Dynamic>{{1, 2}, {3, 4}};
                auto m2 = extractMatrixFromTranspose(m1.T());
                REQUIRE(m2.at(0, 0) == 1);
                REQUIRE(m2.at(0, 1) == 3);
                REQUIRE(m2.at(1, 0) == 2);
                REQUIRE(m2.at(1, 1) == 4);
            }

            SECTION("Matrix Vector Multiplication") {
                auto m = Matrix<int, Dynamic, Dynamic>{{1, 2}, {3, 4}};
                auto v = Vector<int, Dynamic>{1, 2};
                auto v2 = m * v;
                REQUIRE(v2.at(0) == 5);
                REQUIRE(v2.at(1) == 11);
            }

            SECTION("Iteration") {
                auto m = Matrix<int, Dynamic, Dynamic>{{1, 2}, {3, 4}};
                auto it = m.begin();
                REQUIRE(*it == Vector<int, Dynamic>{1, 2});
                REQUIRE(it != m.end());
                REQUIRE(*(++it) == Vector<int, Dynamic>{3, 4});
                REQUIRE(++it == m.end());
            }

            SECTION("Determinant") {
                auto m = Matrix<int, Dynamic, Dynamic>{{1, 2}, {3, 4}};
                auto det = m.det();
                REQUIRE(det == -2);

                auto m2 = Matrix<int, Dynamic, Dynamic>{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
                auto det2 = m2.det();
                REQUIRE(det2 == 0);
            }

            SECTION("Augment") {
                SECTION("Matrix") {
                    auto m1 = Matrix<int, Dynamic, Dynamic>{{1, 2}, {3, 4}};
                    auto m2 = Matrix<int, Dynamic, Dynamic>{{5, 6}, {7, 8}};
                    auto m3 = m1.augment(m2);
                    REQUIRE(m3.at(0, 0) == 1);
                    REQUIRE(m3.at(0, 1) == 2);
                    REQUIRE(m3.at(0, 2) == 5);
                    REQUIRE(m3.at(0, 3) == 6);
                    REQUIRE(m3.at(1, 0) == 3);
                    REQUIRE(m3.at(1, 1) == 4);
                    REQUIRE(m3.at(1, 2) == 7);
                    REQUIRE(m3.at(1, 3) == 8);
                }

                SECTION("Vector") {
                    auto m1 = Matrix<int, Dynamic, Dynamic>{{1, 2}, {3, 4}};
                    auto v = Vector<int, Dynamic>{5, 6};
                    auto m2 = m1.augment(v);
                    REQUIRE(m2.at(0, 0) == 1);
                    REQUIRE(m2.at(0, 1) == 2);
                    REQUIRE(m2.at(0, 2) == 5);
                    REQUIRE(m2.at(1, 0) == 3);
                    REQUIRE(m2.at(1, 1) == 4);
                    REQUIRE(m2.at(1, 2) == 6);
                }
            }

            SECTION("Comparisions") {
                // ------------------  Integer 32 (int) ------------------
                SECTION("Integer - Exact Equality") {
                    auto m1 = Matrix<int, Dynamic, Dynamic>{{1, 2}, {3, 4}};
                    auto m2 = Matrix<int, Dynamic, Dynamic>{{1, 2}, {3, 4}};
                    REQUIRE(m1 == m2);
                }

                SECTION("Integer - Exact Inequality") {
                    auto m1 = Matrix<int, Dynamic, Dynamic>{{1, 2}, {3, 4}};
                    auto m2 = Matrix<int, Dynamic, Dynamic>{{5, 6}, {7, 8}};
                    REQUIRE(m1 != m2);
                }

                // ------------------ Single Precision (float) ------------------
                SECTION("Float - Exact Equality") {
                    Matrix<float, Dynamic, Dynamic> m1{{1.0F, 2.0F}, {3.0F, 4.0F}};
                    Matrix<float, Dynamic, Dynamic> m2{{1.0F, 2.0F}, {3.0F, 4.0F}};
                    REQUIRE(m1 == m2);
                }

                SECTION("Float - Approximate Equality Within Tolerance") {
                    Matrix<float, Dynamic, Dynamic> m1{{1.000001F, 2.000001F}, {3.000001F, 4.000001F}};
                    Matrix<float, Dynamic, Dynamic> m2{{1.0F, 2.0F}, {3.0F, 4.0F}};
                    REQUIRE(m1 == m2);
                }

                SECTION("Float - Inequality Beyond Tolerance") {
                    Matrix<float, Dynamic, Dynamic> m1{{1.0F, 2.0F}, {3.0F, 4.0F}};
                    Matrix<float, Dynamic, Dynamic> m2{{1.0F, 2.0F}, {3.0F, 4.1F}};
                    REQUIRE(m1 != m2);
                }

                SECTION("Float - NaN Comparison") {
                    // NaN should not compare equal to NaN.
                    Matrix<float, Dynamic, Dynamic> m1{{std::numeric_limits<float>::quiet_NaN(), 2.0F}, {3.0F, 4.0F}};
                    Matrix<float, Dynamic, Dynamic> m2{{std::numeric_limits<float>::quiet_NaN(), 2.0F}, {3.0F, 4.0F}};
                    REQUIRE(m1 != m2);
                }

                SECTION("Float - Infinity Comparison") {
                    Matrix<float, Dynamic, Dynamic> m1{{std::numeric_limits<float>::infinity(), 2.0F}, {3.0F, 4.0F}};
                    Matrix<float, Dynamic, Dynamic> m2{{std::numeric_limits<float>::infinity(), 2.0F}, {3.0F, 4.0F}};
                    REQUIRE(m1 == m2);
                }

                // ------------------ Double Precision (double) ------------------
                SECTION("Double - Exact Equality") {
                    Matrix<double, Dynamic, Dynamic> m1{{1.0, 2.0}, {3.0, 4.0}};
                    Matrix<double, Dynamic, Dynamic> m2{{1.0, 2.0}, {3.0, 4.0}};
                    REQUIRE(m1 == m2);
                }

                SECTION("Double - Approximate Equality Within Tolerance") {
                    Matrix<double, Dynamic, Dynamic> m1{{1.000000001, 2.000000001}, {3.000000001, 4.000000001}};
                    Matrix<double, Dynamic, Dynamic> m2{{1.0, 2.0}, {3.0, 4.0}};
                    REQUIRE(m1 == m2);
                }

                SECTION("Double - Inequality Beyond Tolerance") {
                    Matrix<double, Dynamic, Dynamic> m1{{1.0, 2.0}, {3.0, 4.0}};
                    Matrix<double, Dynamic, Dynamic> m2{{1.0, 2.0}, {3.0, 4.0001}};
                    REQUIRE(m1 != m2);
                }

                SECTION("Double - NaN Comparison") {
                    Matrix<double, Dynamic, Dynamic> m1{{std::numeric_limits<double>::quiet_NaN(), 2.0}, {3.0, 4.0}};
                    Matrix<double, Dynamic, Dynamic> m2{{std::numeric_limits<double>::quiet_NaN(), 2.0}, {3.0, 4.0}};
                    REQUIRE(m1 != m2);
                }

                SECTION("Double - Infinity Comparison") {
                    Matrix<double, Dynamic, Dynamic> m1{{std::numeric_limits<double>::infinity(), 2.0}, {3.0, 4.0}};
                    Matrix<double, Dynamic, Dynamic> m2{{std::numeric_limits<double>::infinity(), 2.0}, {3.0, 4.0}};
                    REQUIRE(m1 == m2);
                }
            }

            SECTION("Subset") {
                auto m1 = Matrix<int, Dynamic, Dynamic>{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
                auto m2 = m1.subset(1, 1);
                REQUIRE(m2.at(0, 0) == 1);
                REQUIRE(m2.at(0, 1) == 3);
                REQUIRE(m2.at(1, 0) == 7);
                REQUIRE(m2.at(1, 1) == 9);
            }

            SECTION("Slice") {
                // Test case 1: Slice a 3x3 matrix to get a 2x2 submatrix

                auto m1 = Matrix<int, Dynamic, Dynamic>{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
                auto sliced1 = m1.slice({0, 2}, {0, 2});
                REQUIRE(sliced1.numRows() == 2);
                REQUIRE(sliced1.numCols() == 2);
                REQUIRE(sliced1.at(0, 0) == 1);
                REQUIRE(sliced1.at(0, 1) == 2);
                REQUIRE(sliced1.at(1, 0) == 4);
                REQUIRE(sliced1.at(1, 1) == 5);

                // Test case 2: Slice a 4x4 matrix to get a 2x2 submatrix from the top-left corner

                auto m2 = Matrix<int, Dynamic, Dynamic>{
                    {1, 2, 5, 6},
                    {3, 4, 7, 8},
                    {9, 10, 13, 14},
                    {11, 12, 15, 16},
                };
                auto sliced2 = m2.slice({0, m2.numRows() / 2}, {0, m2.numCols() / 2});
                REQUIRE(sliced2.numRows() == 2);
                REQUIRE(sliced2.numCols() == 2);
                REQUIRE(sliced2.at(0, 0) == 1);
                REQUIRE(sliced2.at(0, 1) == 2);
                REQUIRE(sliced2.at(1, 0) == 3);
                REQUIRE(sliced2.at(1, 1) == 4);

                // Test case 3: Slice a 4x4 matrix to get a single row

                auto m3 = Matrix<int, Dynamic, Dynamic>{
                    {1, 2, 3, 4},
                    {5, 6, 7, 8},
                    {9, 10, 11, 12},
                    {13, 14, 15, 16},
                };
                auto sliced3 = m3.slice({1, 2}, {0, 4});
                REQUIRE(sliced3.numRows() == 1);
                REQUIRE(sliced3.numCols() == 4);
                REQUIRE(sliced3.at(0, 0) == 5);
                REQUIRE(sliced3.at(0, 1) == 6);
                REQUIRE(sliced3.at(0, 2) == 7);
                REQUIRE(sliced3.at(0, 3) == 8);

                // Test case 4: Slice a 4x4 matrix to get a single column

                auto m4 = Matrix<int, Dynamic, Dynamic>{
                    {1, 2, 3, 4},
                    {5, 6, 7, 8},
                    {9, 10, 11, 12},
                    {13, 14, 15, 16},
                };
                auto sliced4 = m4.slice({0, 4}, {2, 3});
                REQUIRE(sliced4.numRows() == 4);
                REQUIRE(sliced4.numCols() == 1);
                REQUIRE(sliced4.at(0, 0) == 3);
                REQUIRE(sliced4.at(1, 0) == 7);
                REQUIRE(sliced4.at(2, 0) == 11);
                REQUIRE(sliced4.at(3, 0) == 15);

                // Test case 5: Slice the entire matrix

                auto m5 = Matrix<int, Dynamic, Dynamic>{{1, 2}, {3, 4}};
                auto sliced5 = m5.slice({0, 2}, {0, 2});
                REQUIRE(sliced5.numRows() == 2);
                REQUIRE(sliced5.numCols() == 2);
                REQUIRE(sliced5.at(0, 0) == 1);
                REQUIRE(sliced5.at(0, 1) == 2);
                REQUIRE(sliced5.at(1, 0) == 3);
                REQUIRE(sliced5.at(1, 1) == 4);

                // Test case 9: Slice with negative indices (should be caught at compile time)

                auto m9 = Matrix<int, Dynamic, Dynamic>{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
                // This should fail at compile time, so it's not included in the runtime tests.
                // static_assert(!std::is_invocable_v<decltype(m9.slice<-1, 2, 0, 2>()), decltype(m9)>);
            }
        }

        SECTION("Cross") {
            SECTION("Addition") {
                auto m1 = Matrix<int, Dynamic, Dynamic>{{1, 2}, {3, 4}};
                auto m2 = Matrix<int, 2, 2>{{5, 6}, {7, 8}};
                auto m3 = m1 + m2;
                auto m4 = m2 + m1;
                REQUIRE(m3.at(0, 0) == 6);
                REQUIRE(m3.at(0, 1) == 8);
                REQUIRE(m3.at(1, 0) == 10);
                REQUIRE(m3.at(1, 1) == 12);

                REQUIRE(m4.at(0, 0) == 6);
                REQUIRE(m4.at(0, 1) == 8);
                REQUIRE(m4.at(1, 0) == 10);
                REQUIRE(m4.at(1, 1) == 12);
            }

            SECTION("Subtraction") {
                auto m1 = Matrix<int, Dynamic, Dynamic>{{1, 2}, {3, 4}};
                auto m2 = Matrix<int, 2, 2>{{5, 6}, {7, 8}};
                auto m3 = m1 - m2;
                auto m4 = m2 - m1;
                REQUIRE(m3.at(0, 0) == -4);
                REQUIRE(m3.at(0, 1) == -4);
                REQUIRE(m3.at(1, 0) == -4);
                REQUIRE(m3.at(1, 1) == -4);

                REQUIRE(m4.at(0, 0) == 4);
                REQUIRE(m4.at(0, 1) == 4);
                REQUIRE(m4.at(1, 0) == 4);
                REQUIRE(m4.at(1, 0) == 4);
            }

            SECTION("Matrix Multiplication") {
                auto m1 = Matrix<int, Dynamic, Dynamic>{{1, 2}, {3, 4}};
                auto m2 = Matrix<int, 2, 2>{{5, 6}, {7, 8}};
                auto m3 = m1 * m2;
                auto m4 = m2 * m1;
                REQUIRE(m3.at(0, 0) == 19);
                REQUIRE(m3.at(0, 1) == 22);
                REQUIRE(m3.at(1, 0) == 43);
                REQUIRE(m3.at(1, 1) == 50);

                REQUIRE(m4.at(0, 0) == 23);
                REQUIRE(m4.at(0, 1) == 34);
                REQUIRE(m4.at(1, 0) == 31);
                REQUIRE(m4.at(1, 1) == 46);
            }

            SECTION("Matrix Vector Multiplication") {
                auto m1 = Matrix<int, Dynamic, Dynamic>{{1, 2}, {3, 4}};
                auto v1 = Vector<int, 2>{1, 2};
                auto r1 = m1 * v1;
                auto m2 = Matrix<int, 2, 2>{{1, 2}, {3, 4}};
                auto v2 = Vector<int, Dynamic>{1, 2};
                auto r2 = m2 * v2;

                REQUIRE(r1.at(0) == 5);
                REQUIRE(r1.at(1) == 11);

                REQUIRE(r2.at(0) == 5);
                REQUIRE(r2.at(1) == 11);
            }
            //
            // SECTION("Augment") {
            //     SECTION("Matrix") {
            //         auto m1 = Matrix<int, Dynamic, Dynamic>{{1, 2}, {3, 4}};
            //         auto m2 = Matrix<int, 2, 2>{{5, 6}, {7, 8}};
            //         auto m3 = m1.augment(m2);
            //         REQUIRE(m3.at(0, 0) == 1);
            //         REQUIRE(m3.at(0, 1) == 2);
            //         REQUIRE(m3.at(0, 2) == 5);
            //         REQUIRE(m3.at(0, 3) == 6);
            //         REQUIRE(m3.at(1, 0) == 3);
            //         REQUIRE(m3.at(1, 1) == 4);
            //         REQUIRE(m3.at(1, 2) == 7);
            //         REQUIRE(m3.at(1, 3) == 8);
            //     }
            //
            //     SECTION("Vector") {
            //         auto m1 = Matrix<int, Dynamic, Dynamic>{{1, 2}, {3, 4}};
            //         auto v = Vector<int, 2>{5, 6};
            //         auto m2 = m1.augment(v);
            //         REQUIRE(m2.at(0, 0) == 1);
            //         REQUIRE(m2.at(0, 1) == 2);
            //         REQUIRE(m2.at(0, 2) == 5);
            //         REQUIRE(m2.at(1, 0) == 3);
            //         REQUIRE(m2.at(1, 1) == 4);
            //         REQUIRE(m2.at(1, 2) == 6);
            //     }
            // }

            SECTION("Comparisions") {
                SECTION("Equality") {
                    auto m1 = Matrix<int, Dynamic, Dynamic>{{1, 2}, {3, 4}};
                    auto m2 = Matrix<int, 2, 2>{{1, 2}, {3, 4}};
                    REQUIRE(m1 == m2);
                    REQUIRE(m2 == m1);
                }
                //
                //     SECTION("Inequality") {
                //         auto m1 = Matrix<int, Dynamic, Dynamic>{{1, 2}, {3, 4}};
                //         auto m2 = Matrix<int, 2, 2>{{5, 6}, {7, 8}};
                //         REQUIRE(m1 != m2);
                //         REQUIRE(m2 != m1);
                //     }
                //
                //     SECTION("Less than") {
                //         auto m1 = Matrix<int, Dynamic, Dynamic>{{1, 2}, {3, 4}};
                //         auto m2 = Matrix<int, 2, 2>{{5, 6}, {7, 8}};
                //         REQUIRE(m1 < m2);
                //         REQUIRE_FALSE(m2 < m1);
                //     }
                //
                //     SECTION("Less than or equal") {
                //         auto m1 = Matrix<int, Dynamic, Dynamic>{{1, 2}, {3, 4}};
                //         auto m2 = Matrix<int, 2, 2>{{1, 2}, {3, 4}};
                //         REQUIRE(m1 <= m2);
                //         REQUIRE(m2 <= m1);
                //     }
                //
                //     SECTION("Greater than") {
                //         auto m1 = Matrix<int, Dynamic, Dynamic>{{5, 6}, {7, 8}};
                //         auto m2 = Matrix<int, 2, 2>{{1, 2}, {3, 4}};
                //         REQUIRE(m1 > m2);
                //         REQUIRE_FALSE(m2 > m1);
                //     }
                //
                //     SECTION("Greater than or equal") {
                //         auto m1 = Matrix<int, Dynamic, Dynamic>{{1, 2}, {3, 4}};
                //         auto m2 = Matrix<int, 2, 2>{{1, 2}, {3, 4}};
                //         REQUIRE(m1 >= m2);
                //         REQUIRE(m2 >= m1);
                //     }
            }
        }
    }

    SECTION("Robustness") {
        SECTION("Compile Time") {
            SECTION("Indexing") {
                SECTION("Out-of-Bounds Access") {
                    SECTION("Accessing invalid index throws exception") {
                        auto m = Matrix<int, 2, 2>{{1, 2}, {3, 4}};
                        REQUIRE_THROWS_AS(m.at(-1, 0), std::out_of_range);
                        REQUIRE_THROWS_AS(m.at(2, 2), std::out_of_range);
                    }

                    SECTION("Accessing valid indices does not throw") {
                        auto m = Matrix<int, 2, 2>{{1, 2}, {3, 4}};
                        REQUIRE_NOTHROW(m.at(0, 0));
                        REQUIRE_NOTHROW(m.at(1, 1));
                    }
                }
            }

            SECTION("Chained Operations") {
                SECTION("Matrix addition followed by multiplication") {
                    auto m1 = Matrix<int, 2, 2>{{1, 2}, {3, 4}};
                    auto m2 = Matrix<int, 2, 2>{{5, 6}, {7, 8}};
                    auto m3 = Matrix<int, 2, 2>{{2, 2}, {2, 2}};
                    auto result = (m1 + m2) * m3;
                    REQUIRE(result.at(0, 0) == 28);
                    REQUIRE(result.at(0, 1) == 28);
                    REQUIRE(result.at(1, 0) == 44);
                    REQUIRE(result.at(1, 1) == 44);
                }

                SECTION("Matrix subtraction followed by scalar multiplication") {
                    auto m1 = Matrix<int, 2, 2>{{5, 6}, {7, 8}};
                    auto m2 = Matrix<int, 2, 2>{{1, 2}, {3, 4}};
                    auto result = (m1 - m2) * 3;
                    REQUIRE(result.at(0, 0) == 12);
                    REQUIRE(result.at(0, 1) == 12);
                    REQUIRE(result.at(1, 0) == 12);
                    REQUIRE(result.at(1, 1) == 12);
                }
            }

            SECTION("Matrix multiplication with identity matrix") {
                auto m1 = Matrix<int, 2, 2>{{1, 2}, {3, 4}};
                auto identity = Matrix<int, 2, 2>{{1, 0}, {0, 1}};
                auto result = m1 * identity;
                REQUIRE(result.at(0, 0) == 1);
                REQUIRE(result.at(0, 1) == 2);
                REQUIRE(result.at(1, 0) == 3);
                REQUIRE(result.at(1, 1) == 4);
            }

            SECTION("Floating-Point Precision") {
                SECTION("Matrix multiplication with floating-point precision") {
                    auto m1 = Matrix<double, 2, 2>{{1.5, 2.5}, {3.5, 4.5}};
                    auto m2 = Matrix<double, 2, 2>{{5.5, 6.5}, {7.5, 8.5}};
                    auto result = m1 * m2;
                    REQUIRE(result.at(0, 0) == Approx(27.0).epsilon(0.001));
                    REQUIRE(result.at(0, 1) == Approx(31.0).epsilon(0.001));
                    REQUIRE(result.at(1, 0) == Approx(53.0).epsilon(0.001));
                    REQUIRE(result.at(1, 1) == Approx(61.0).epsilon(0.001));
                }

                SECTION("Scalar division with floating-point precision") {
                    auto m = Matrix<double, 2, 2>{{10.0, 20.0}, {30.0, 40.0}};
                    auto result = m / 3.0;
                    REQUIRE(result.at(0, 0) == Approx(3.3333).epsilon(0.001));
                    REQUIRE(result.at(0, 1) == Approx(6.6667).epsilon(0.001));
                    REQUIRE(result.at(1, 0) == Approx(10.0).epsilon(0.001));
                    REQUIRE(result.at(1, 1) == Approx(13.3333).epsilon(0.001));
                }
            }

            SECTION("Performance") {
                using namespace std::chrono;

                SECTION("Matrix Addition Performance") {
                    constexpr int N = 1000;  // Large matrix size
                    Matrix<int, N, N> m1;
                    Matrix<int, N, N> m2;

                    // Initialize matrices with large data
                    for (int i = 0; i < N; ++i) {
                        for (int j = 0; j < N; ++j) {
                            m1.at(i, j) = i + j;
                            m2.at(i, j) = i - j;
                        }
                    }

                    auto start = high_resolution_clock::now();
                    auto result = m1 + m2;
                    auto end = high_resolution_clock::now();
                    auto duration = duration_cast<milliseconds>(end - start).count();

                    REQUIRE(result.at(0, 0) == 0);  // Simple check
                    REQUIRE(duration < 100);        // Ensure it runs within 100 ms

                    // BENCHMARK("Matrix Addition Performance") { m1 + m2; };
                }

                SECTION("Matrix Multiplication Performance") {
                    const int N = 100;  // Matrix multiplication is more expensive, smaller N
                    Matrix<long long, N, N> m1{};
                    Matrix<long long, N, N> m2{};

                    // Initialize matrices with large data
                    for (int i{}; i < N; i++) {
                        for (int j{}; j < N; j++) {
                            m1.at(i, j) = i + j;
                            m2.at(i, j) = i - j;
                        }
                    }

                    auto start = high_resolution_clock::now();
                    auto result = m1 * m2;
                    auto end = high_resolution_clock::now();

                    auto duration = duration_cast<milliseconds>(end - start).count();

                    REQUIRE(duration < 500);  // Ensure it runs within 500 ms

                    // BENCHMARK("Matrix Multiplication Performance") { m1 *m2; };
                }

                SECTION("Matrix Scalar Multiplication Performance") {
                    const int N = 1000;  // Scalar multiplication is simple, test larger matrices
                    Matrix<int, N, N> m1;

                    // Initialize matrix with large data
                    for (int i = 0; i < N; ++i) {
                        for (int j = 0; j < N; ++j) {
                            m1.at(i, j) = i + j;
                        }
                    }

                    auto start = high_resolution_clock::now();
                    auto result = m1 * 5;
                    auto end = high_resolution_clock::now();
                    auto duration = duration_cast<milliseconds>(end - start).count();

                    REQUIRE(result.at(0, 0) == 0);  // Simple check
                    REQUIRE(duration < 5000);       // Ensure it runs within 5 seconds

                    // BENCHMARK("Matrix Scalar Multiplication Performance") { m1 * 5; };
                }
            }
        }

        SECTION("Dynamic") {
            SECTION("Indexing") {
                SECTION("Out-of-Bounds Access") {
                    SECTION("Accessing invalid index throws exception") {
                        auto m = Matrix<int, Dynamic, Dynamic>{{1, 2}, {3, 4}};
                        REQUIRE_THROWS_AS(m.at(-1, 0), std::out_of_range);
                        REQUIRE_THROWS_AS(m.at(2, 2), std::out_of_range);
                    }

                    SECTION("Accessing valid indices does not throw") {
                        auto m = Matrix<int, Dynamic, Dynamic>{{1, 2}, {3, 4}};
                        REQUIRE_NOTHROW(m.at(0, 0));
                        REQUIRE_NOTHROW(m.at(1, 1));
                    }
                }
            }

            SECTION("Chained Operations") {
                SECTION("Matrix addition followed by multiplication") {
                    auto m1 = Matrix<int, Dynamic, Dynamic>{{1, 2}, {3, 4}};
                    auto m2 = Matrix<int, Dynamic, Dynamic>{{5, 6}, {7, 8}};
                    auto m3 = Matrix<int, Dynamic, Dynamic>{{2, 2}, {2, 2}};
                    auto result = (m1 + m2) * m3;
                    REQUIRE(result.at(0, 0) == 28);
                    REQUIRE(result.at(0, 1) == 28);
                    REQUIRE(result.at(1, 0) == 44);
                    REQUIRE(result.at(1, 1) == 44);
                }

                SECTION("Matrix subtraction followed by scalar multiplication") {
                    auto m1 = Matrix<int, Dynamic, Dynamic>{{5, 6}, {7, 8}};
                    auto m2 = Matrix<int, Dynamic, Dynamic>{{1, 2}, {3, 4}};
                    auto result = (m1 - m2) * 3;
                    REQUIRE(result.at(0, 0) == 12);
                    REQUIRE(result.at(0, 1) == 12);
                    REQUIRE(result.at(1, 0) == 12);
                    REQUIRE(result.at(1, 1) == 12);
                }
            }

            SECTION("Matrix multiplication with identity matrix") {
                auto m1 = Matrix<int, Dynamic, Dynamic>{{1, 2}, {3, 4}};
                auto identity = Matrix<int, Dynamic, Dynamic>{{1, 0}, {0, 1}};
                auto result = m1 * identity;
                REQUIRE(result.at(0, 0) == 1);
                REQUIRE(result.at(0, 1) == 2);
                REQUIRE(result.at(1, 0) == 3);
                REQUIRE(result.at(1, 1) == 4);
            }

            SECTION("Floating-Point Precision") {
                SECTION("Matrix multiplication with floating-point precision") {
                    auto m1 = Matrix<double, Dynamic, Dynamic>{{1.5, 2.5}, {3.5, 4.5}};
                    auto m2 = Matrix<double, Dynamic, Dynamic>{{5.5, 6.5}, {7.5, 8.5}};
                    auto result = m1 * m2;
                    REQUIRE(result.at(0, 0) == Approx(27.0).epsilon(0.001));
                    REQUIRE(result.at(0, 1) == Approx(31.0).epsilon(0.001));
                    REQUIRE(result.at(1, 0) == Approx(53.0).epsilon(0.001));
                    REQUIRE(result.at(1, 1) == Approx(61.0).epsilon(0.001));
                }

                SECTION("Scalar division with floating-point precision") {
                    auto m = Matrix<double, Dynamic, Dynamic>{{10.0, 20.0}, {30.0, 40.0}};
                    auto result = m / 3.0;
                    REQUIRE(result.at(0, 0) == Approx(3.3333).epsilon(0.001));
                    REQUIRE(result.at(0, 1) == Approx(6.6667).epsilon(0.001));
                    REQUIRE(result.at(1, 0) == Approx(10.0).epsilon(0.001));
                    REQUIRE(result.at(1, 1) == Approx(13.3333).epsilon(0.001));
                }
            }

            SECTION("Performance") {
                using namespace std::chrono;

                SECTION("Matrix Addition Performance") {
                    const int N = 1000;  // Large matrix size
                    Matrix<int, N, N> m1;
                    Matrix<int, N, N> m2;

                    // Initialize matrices with large data
                    for (int i = 0; i < N; ++i) {
                        for (int j = 0; j < N; ++j) {
                            m1.at(i, j) = i + j;
                            m2.at(i, j) = i - j;
                        }
                    }

                    auto start = high_resolution_clock::now();
                    auto result = m1 + m2;
                    auto end = high_resolution_clock::now();
                    auto duration = duration_cast<milliseconds>(end - start).count();

                    REQUIRE(result.at(0, 0) == 0);  // Simple check
                    REQUIRE(duration < 100);        // Ensure it runs within 100 ms

                    // BENCHMARK("Matrix Addition Performance") { m1 + m2; };
                }

                // SECTION("Matrix Multiplication Performance") {
                //     const int N = 100;  // Matrix multiplication is more expensive, smaller N
                //     Matrix<long long, N, N> m1{};
                //     Matrix<long long, N, N> m2{};
                //
                //     // Initialize matrices with large data
                //     for (long i = 0; i < N; ++i) {
                //         for (long j = 0; j < N; ++j) {
                //             m1.at(i, j) = i + j;
                //             m2.at(i, j) = i - j;
                //         }
                //     }
                //
                //     auto start = high_resolution_clock::now();
                //     auto result = m1 * m2;
                //     auto end = high_resolution_clock::now();
                //     auto duration = duration_cast<milliseconds>(end - start).count();
                //
                //     REQUIRE(result.at(0, 0) == 0);  // Simple check
                //     REQUIRE(duration < 500);        // Ensure it runs within 100 ms
                //
                //     // BENCHMARK("Matrix Multiplication Performance") { m1 *m2; };
                // }

                SECTION("Matrix Scalar Multiplication Performance") {
                    const int N = 1000;  // Scalar multiplication is simple, test larger matrices
                    Matrix<int, Dynamic, Dynamic> m1(N, N);

                    // Initialize matrix with large data
                    for (int i = 0; i < N; ++i) {
                        for (int j = 0; j < N; ++j) {
                            m1.at(i, j) = i + j;
                        }
                    }

                    auto start = high_resolution_clock::now();
                    auto result = m1 * 5;
                    auto end = high_resolution_clock::now();
                    auto duration = duration_cast<milliseconds>(end - start).count();

                    REQUIRE(result.at(0, 0) == 0);  // Simple check
                    REQUIRE(duration < 5000);       // Ensure it runs within 5 seconds

                    // BENCHMARK("Matrix Scalar Multiplication Performance") { m1 * 5; };
                }
            }
        }
    }
}
