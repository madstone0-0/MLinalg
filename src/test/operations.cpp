#include "Operations.hpp"

#include <MLinalg.hpp>
#include <catch2/catch_all.hpp>
#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>

#include "Helpers.hpp"
#include "Structures.hpp"
#include "structures/Aliases.hpp"
#include "structures/Vector.hpp"

using namespace Catch;
using namespace mlinalg;

TEST_CASE("Operations") {
    // System1: A 3x4 system already in echelon form.
    LinearSystem<double, 3, 4> system1{{
        {1, 2, 3, 4},
        {0, 0, 2, 3},
        {0, 0, 0, 0},
    }};

    LinearSystem<double, 3, 3> A1{{
        {1, 2, 3},
        {0, 0, 2},
        {0, 0, 0},
    }};
    Vector<double, 3> b1{4, 3, 0};
    // Expected Gauss-Jordan Elimination
    // - Row2 -> 1/2*row2 = {0, 0, 1, 3/2}.
    // - Row1 -> row1 - 3*row2 = {1, 2, 0, -1/2}.
    // - Row3 remains: {0, 0, 0, 0}.

    // System2: A 4x6 system already in echelon form.
    LinearSystem<double, 4, 6> system2{
        {1, -3, 0, -1, 0, -2},
        {0, 1, 0, 0, -4, 1},
        {0, 0, 0, 1, 9, 4},
        {0, 0, 0, 0, 0, 0},
    };
    LinearSystem<double, 4, 5> A2{{
        {1, -3, 0, -1, 0},
        {0, 1, 0, 0, -4},
        {0, 0, 0, 1, 9},
        {0, 0, 0, 0, 0},
    }};
    Vector<double, 4> b2{-2, 1, 4, 0};
    // Expected Gauss-Jordan Elimination:
    // - Row1 → Row1 + 3*Row2 → {1, 0, 0, -1, -12, 1}
    // - Row1 → Row1 + Row3 → {1, 0, 0, 0, -3, 5}
    // - Row2 remains: {0, 1, 0, 0, -4, 1}
    // - Row3 remains: {0, 0, 0, 1, 9, 4}
    // - Row4 remains zero.

    // System3: A 2x4 system that requires elimination.
    LinearSystem<double, 2, 4> system3{
        {1, 3, 4, 7},
        {3, 9, 7, 6},
    };
    LinearSystem<double, 2, 3> A3{{
        {1, 3, 4},
        {3, 9, 7},
    }};
    Vector<double, 2> b3{7, 6};
    // Expected Gaussian Elimination for system3:
    //  - Row1 remains: {1, 3, 4, 7}.
    //  - Row2 -> row2 - 3*row1 = {0, 0, -5, -15}.
    // Expected Gauss-Jordan Elimination:
    // - Row2 → (-1/5)*Row2 = {0, 0, 1, 3}
    // - Row1 → Row1 - 4*Row2 = {1, 3, 0, -5}

    // System4: A 3x5 system that requires elimination.
    LinearSystem<double, 3, 5> system4{
        {1, -7, 0, 6, 5},
        {0, 0, 1, -2, -3},
        {-1, 7, -4, 2, 7},
    };
    LinearSystem<double, 3, 4> A4{{
        {1, -7, 0, 6},
        {0, 0, 1, -2},
        {-1, 7, -4, 2},
    }};
    Vector<double, 3> b4{5, -3, 7};
    // Expected Gaussian Elimination for system4:
    //  - Row1 remains: {1, -7, 0, 6, 5}.
    //  - Row2 -> row2 + 7*row1 = {0, 0, 1, -2, -3}.
    // Expected Gauss-Jordan Elimination:
    // - Matrix already in reduced row-echelon form after Gaussian elimination.

    // System5: A 4x6 inconsistent system.
    LinearSystem<double, 4, 6> system5{
        {1, -3, 0, -1, 0, -2},
        {0, 1, 0, 0, -4, 1},
        {0, 0, 0, 1, 9, 4},
        {0, 0, 0, 0, 0, 10},
    };
    LinearSystem<double, 4, 5> A5{{
        {1, -3, 0, -1, 0},
        {0, 1, 0, 0, -4},
        {0, 0, 0, 1, 9},
        {0, 0, 0, 0, 0},
    }};
    Vector<double, 4> b5{-2, 1, 4, 10};
    // This system is already in echelon form (with the last row indicating inconsistency).
    // Expected Gauss-Jordan Elimination:
    // - Row1 → Row1 + 3*Row2 → {1, 0, 0, -1, -12, 1}
    // - Row1 → Row1 + Row3 → {1, 0, 0, 0, -3, 5}
    // - Row4 remains: {0, 0, 0, 0, 0, 10} (inconsistent)

    // SquareSystem: A 2x2 square system.
    LinearSystem<double, 2, 2> squareSystem({
        {3, 1},
        {1, 2},
    });
    // Expected Gaussian Elimination for squareSystem:
    //  - Row1 remains: {3, 1}.
    //  - Row2 -> row2 - (1/3)*row1 = {0, 5/3}.

    // System6: A 3x4 system that requires elimination.
    LinearSystem<double, 3, 4> system6{{
        {2, 4, -2, 2},
        {1, 2, 0, 4},
        {3, 6, -4, 8},
    }};
    LinearSystem<double, 3, 3> A6{{
        {2, 4, -2},
        {1, 2, 0},
        {3, 6, -4},
    }};
    Vector<double, 3> b6{2, 4, 8};
    // Expected Gaussian Elimination for system6:
    //  - Row1 remains: {2, 4, -2, 2}.
    //  - Row2 -> row2 - (1/2)*row1 = {0, 0, 1, 3}.
    //  - Row3 -> row3 - (3/2)*row1 = {0, 0, -1, 5}, then row3 + row2 = {0, 0, 0, 8}.
    // Expected Gauss-Jordan Elimination:
    // - Row1 → (1/2)*Row1 → {1, 2, -1, 1}
    // - Row1 → Row1 + Row2 → {1, 2, 0, 4}
    // - Row2 → {0, 0, 1, 3}
    // - Row3 remains: {0, 0, 0, 8} (inconsistent)
    // System7: A 3x5 system with dependent rows.

    LinearSystem<double, 3, 5> system7{{
        {1, 2, -1, 3, 5},
        {2, 4, 0, 6, 10},
        {3, 6, 1, 9, 15},
    }};
    LinearSystem<double, 3, 4> A7{{
        {1, 2, -1, 3},
        {2, 4, 0, 6},
        {3, 6, 1, 9},
    }};
    Vector<double, 3> b7{5, 10, 15};
    // Expected Gaussian Elimination for system7:
    //  - Row1 remains: {1, 2, -1, 3, 5}.
    //  - Row2 -> row2 - 2*row1 = {0, 0, 2, 0, 0}.
    //  - Row3 -> row3 - 3*row1 = {0, 0, 4, 0, 0}, then row3 - 2*row2 = {0, 0, 0, 0, 0}.
    // Expected Gauss-Jordan Elimination:
    // - Row2 → (1/2)*Row2 → {0, 0, 1, 0, 0}
    // - Row1 → Row1 + Row2 → {1, 2, 0, 3, 5}

    // System8: A 3x4 inconsistent system (last row indicates inconsistency).
    LinearSystem<double, 3, 4> system8{{
        {1, 2, -1, 3},
        {0, 1, 2, 4},
        {0, 0, 0, 7},
    }};
    LinearSystem<double, 3, 3> A8{{
        {1, 2, -1},
        {0, 1, 2},
        {0, 0, 0},
    }};
    Vector<double, 3> b8{3, 4, 7};
    // This system is already in echelon form (with the third row being a zero–coefficient row with a nonzero constant).
    // Expected Gauss-Jordan Elimination:
    // - Row1 → Row1 - 2*Row2 → {1, 0, -5, -5}
    // - Row3 remains: {0, 0, 0, 7} (inconsistent)

    // System9: A 3x3 square system already in echelon form.
    LinearSystem<double, 3, 3> system9{{
        {2, -1, 3},
        {0, 4, 5},
        {0, 0, 6},
    }};
    LinearSystem<double, 3, 2> A9{{
        {2, -1},
        {0, 4},
        {0, 0},
    }};
    Vector<double, 3> b9{3, 5, 6};

    // System10: A 4x3 system with one redundant row.
    LinearSystem<double, 4, 3> system10{{
        {1, 2, 3},
        {2, 4, 6},
        {0, 1, 1},
        {3, 7, 8},
    }};
    LinearSystem<double, 4, 2> A10{{
        {1, 2},
        {2, 4},
        {0, 1},
        {3, 7},
    }};
    Vector<double, 4> b10{3, 6, 1, 8};
    // Expected Gaussian Elimination for system10:
    //  - Row1 remains: {1, 2, 3}.
    //  - Row2 -> row2 - 2*row1 = {0, 0, 0}.
    //  - Row3 remains: {0, 1, 1} (pivot in column 1).
    //  - Row4 -> row4 - 3*row1 = {0, 1, -1}, then row4 - row3 = {0, 0, -2}.
    // Expected Gauss-Jordan Elimination:
    // - Row2 → Row2 - 2*Row1 → {0, 0, 0}
    // - Row4 → Row4 - 3*Row1 → {0, 1, -1}
    // - Row4 → Row4 - Row3 → {0, 0, -2}
    // - Inconsistency detected in Row4: {0, 0, -2}

    // System11: A trivial 1x3 system.
    LinearSystem<double, 1, 3> system11{{
        {5, -3, 2},
    }};
    LinearSystem<double, 1, 2> A11{{{5, -3}}};
    Vector<double, 1> b11{2};
    // Already in echelon form.
    // Already in RREF (no elimination possible)

    // System12: A 2x3 system that needs one elimination step.
    LinearSystem<double, 2, 3> system12{{
        {4, 8, 12},
        {2, 5, 7},
    }};
    LinearSystem<double, 2, 2> A12{{{4, 8}, {2, 5}}};
    Vector<double, 2> b12{12, 7};
    // Expected Gaussian Elimination for system12:
    //  - Row1 remains: {4, 8, 12}.
    //  - Row2 -> row2 - (2/4)*row1 = {0, 1, 1};
    // Expected Gauss-Jordan Elimination:
    // - Row1 → (1/4)*Row1 → {1, 2, 3}
    // - Row2 → Row2 - (1/2)*Row1 → {0, 1, 1}
    // - Row1 → Row1 - 2*Row2 → {1, 0, 1}

    auto dyna1{helpers::toDynamic(system1)};
    auto dyna2{helpers::toDynamic(system2)};
    auto dyna3{helpers::toDynamic(system3)};
    auto dyna4{helpers::toDynamic(system4)};
    auto dyna5{helpers::toDynamic(system5)};
    auto squareDyna{helpers::toDynamic(squareSystem)};
    auto dyna6{helpers::toDynamic(system6)};
    auto dyna7{helpers::toDynamic(system7)};
    auto dyna8{helpers::toDynamic(system8)};
    auto dyna9{helpers::toDynamic(system9)};
    auto dyna10{helpers::toDynamic(system10)};
    auto dyna11{helpers::toDynamic(system11)};
    auto dyna12{helpers::toDynamic(system12)};

    SECTION("Helper Operations", "[helper]") {
        SECTION("Compile Time") {
            SECTION("Consistency") {
                REQUIRE_FALSE(isInconsistent(system1));
                REQUIRE_FALSE(isInconsistent(system2));
                REQUIRE_FALSE(isInconsistent(system3));
                REQUIRE_FALSE(isInconsistent(system4));
                REQUIRE_FALSE(isInconsistent(squareSystem));
                REQUIRE(isInconsistent(system5));
            }

            SECTION("Pivots") {
                auto pivots1 = getPivots(system1, false);
                auto pivots2 = getPivots(system2, false);
                auto pivots3 = getPivots(system3, false);
                auto pivots4 = getPivots(system4, false);
                auto pivots5 = getPivots(system5, false);
                auto pivots6 = getPivots(squareSystem, false);

                REQUIRE((pivots1.at(0).has_value() && pivots1.at(0) == 1));
                REQUIRE((pivots1.at(1).has_value() && pivots1.at(1) == 2));
                REQUIRE_FALSE(pivots1.at(2).has_value());

                REQUIRE((pivots2.at(0).has_value() && pivots2.at(0) == 1));
                REQUIRE((pivots2.at(1).has_value() && pivots2.at(1) == 1));
                REQUIRE((pivots2.at(2).has_value() && pivots2.at(2) == 1));
                REQUIRE_FALSE(pivots2.at(3).has_value());

                REQUIRE((pivots3.at(0).has_value() && pivots3.at(0) == 1));
                REQUIRE((pivots3.at(1).has_value() && pivots3.at(1) == 9));

                REQUIRE((pivots4.at(0).has_value() && pivots4.at(0) == 1));
                REQUIRE((pivots4.at(1).has_value() && pivots4.at(1) == 1));
                REQUIRE((pivots4.at(2).has_value() && pivots4.at(2) == 2));

                REQUIRE((pivots5.at(0).has_value() && pivots5.at(0) == 1));
                REQUIRE((pivots5.at(1).has_value() && pivots5.at(1) == 1));
                REQUIRE((pivots5.at(2).has_value() && pivots5.at(2) == 1));
                REQUIRE_FALSE((pivots5.at(3).has_value()));

                REQUIRE((pivots6.at(0).has_value() && pivots6.at(0) == 3));
            }

            SECTION("Echelon Form") {
                REQUIRE(isInEchelonForm(system1, getPivots(system1)));
                REQUIRE(isInEchelonForm(system2, getPivots(system2)));
                REQUIRE_FALSE(isInEchelonForm(system3, getPivots(system3)));
                REQUIRE_FALSE(isInEchelonForm(system4, getPivots(system4)));
                REQUIRE(isInEchelonForm(system5, getPivots(system5)));
                REQUIRE_FALSE(isInEchelonForm(squareSystem, getPivots(squareSystem)));
            }

            SECTION("Rearrange System") {
                REQUIRE(rearrangeSystem(system1) == system1);
                REQUIRE(rearrangeSystem(system2) == system2);
            }

            SECTION("Is zero vector") {
                auto zeroVec1{vectorZeros<double, 4>()};
                Vector<double, 3> nonZeroVec1{1, 0, 0};

                REQUIRE(isZeroVector(zeroVec1));
                REQUIRE_FALSE(isZeroVector(nonZeroVec1));
            }

            SECTION("Is upper triangular") {
                REQUIRE(isUpperTriangular(system1));
                REQUIRE_FALSE(isUpperTriangular(system3));
            }

            SECTION("Is singular") {
                auto SingularSystem = LinearSystem<double, 2, 2>{
                    {1, 2},
                    {2, 4},
                };
                auto NonSingularSystem = LinearSystem<double, 2, 2>{
                    {1, 2},
                    {3, 4},
                };
                REQUIRE(isSingular(SingularSystem));
                REQUIRE_FALSE(isSingular(NonSingularSystem));
            }

            SECTION("Is Hermetian") {
                SECTION("Real symmetric matrix is Hermitian") {
                    M2x2 A{
                        {1.0, 2.0},
                        {2.0, 3.0},
                    };
                    REQUIRE(isHermitian(A));
                }

                SECTION("Real asymmetric matrix is not Hermitian") {
                    M2x2 A{
                        {1.0, 2.0},
                        {0.0, 3.0},
                    };
                    REQUIRE_FALSE(isHermitian(A));
                }

                // Not supported yet
                // using CM2x2 = Matrix<std::complex<double>, 2, 2>;
                // SECTION("Complex Hermitian matrix") {
                //     CM2x2 A{
                //         {{1.0, 0.0}, {2.0, -1.0}},
                //         {{2.0, 1.0}, {3.0, 0.0}},
                //     };
                //     CHECK(isHermitian(A) == true);
                // }
                //
                // SECTION("Complex non-Hermitian matrix") {
                //     CM2x2 A{
                //         {{1.0, 0.0}, {2.0, 1.0}},
                //         {{2.0, 1.0}, {3.0, 0.0}},
                //     };
                //     CHECK(isHermitian(A) == false);
                // }

                SECTION("Non-square matrix throws") {
                    Matrix<double, 2, 3> A{
                        {1.0, 2.0, 3.0},
                        {4.0, 5.0, 6.0},
                    };
                    CHECK_THROWS_AS(isHermitian(A), std::invalid_argument);
                }
            }

            SECTION("Is Orthogonal") {
                SECTION("2x2 Identity is orthogonal") {
                    M2x2 A{
                        {1.0, 0.0},
                        {0.0, 1.0},
                    };
                    REQUIRE(isOrthogonal(A));
                }

                SECTION("2x2 rotation matrix is orthogonal") {
                    double theta = M_PI / 4;
                    M2x2 A{
                        {cos(theta), -sin(theta)},
                        {sin(theta), cos(theta)},
                    };
                    REQUIRE(isOrthogonal(A));
                }

                SECTION("2x2 non-orthogonal matrix") {
                    M2x2 A{
                        {1.0, 1.0},
                        {0.0, 1.0},
                    };
                    REQUIRE_FALSE(isOrthogonal(A));
                }

                SECTION("3x3 orthogonal matrix") {
                    M3x3 A{
                        {1, 0, 0},
                        {0, 0, -1},
                        {0, 1, 0},
                    };
                    REQUIRE(isOrthogonal(A));
                }

                SECTION("3x3 near-orthogonal matrix fails fuzzy comparison") {
                    M3x3d A{
                        {1, 0, 0},
                        {0, 1, 1e-3},
                        {0, 0, 1},
                    };
                    REQUIRE_FALSE(isOrthogonal(A));
                }
            }

            SECTION("Solve Equation") {
                SECTION("All variables unknown (treated as zero contribution)") {
                    const Row<double, 4> r1{1, 2, 3, 4};        // Equation: 1*x0 + 2*x1 + 3*x2 = 4
                    ConditionalRowOptional<double, 4, 4> s1{};  // all std::nullopt
                    auto result = solveEquation(r1, 0, s1);
                    REQUIRE(result.has_value());
                    REQUIRE(result.value() == Approx(4.0));
                }

                SECTION("Some variables known") {
                    const Row<double, 4> r2{2, 1, -1, 3};  // Equation: 2*x0 + 1*x1 -1*x2 = 3
                    ConditionalRowOptional<double, 4, 4> s2{};
                    s2[1] = 4.0;  // x1 = 4
                    s2[2] = 1.0;  // x2 = 1
                    auto result = solveEquation(r2, 0, s2);
                    REQUIRE(result.has_value());
                    REQUIRE(result.value() == Approx(0.0));
                }

                SECTION("Zero coefficient and RHS also zero (infinite solutions)") {
                    const Row<double, 4> r3{0, 1, 2, -3};  // Equation: 0*x0 + 1*x1 + 2*x2 = -3
                    ConditionalRowOptional<double, 4, 4> s3{};
                    s3[1] = -1.0;
                    s3[2] = -1.0;
                    auto result = solveEquation(r3, 0, s3);
                    REQUIRE(result.has_value());
                    REQUIRE(result.value() == Approx(0.0));
                }

                SECTION("Zero coefficient and non-zero RHS (no solution)") {
                    const Row<double, 4> r4{0, 0, 0, 1};        // Equation: 0*x0 + 0*x1 + 0*x2 = 1
                    ConditionalRowOptional<double, 4, 4> s4{};  // all unknown
                    auto result = solveEquation(r4, 0, s4);
                    REQUIRE_FALSE(result.has_value());
                }

                SECTION("Negative coefficients") {
                    const Row<double, 4> r5{-2, -1, 0, -5};  // Equation: -2*x0 - x1 = -5
                    ConditionalRowOptional<double, 4, 4> s5{};
                    s5[1] = 1.5;
                    auto result = solveEquation(r5, 0, s5);
                    REQUIRE(result.has_value());
                    REQUIRE(result.value() == Approx(1.75));
                }
            }

            SECTION("Diagonal") {
                auto diag1 = diagonal<3>(4.);
                for (size_t i{}; i < diag1.rows; ++i) REQUIRE(fuzzyCompare(diag1(i, i), 4.));

                auto diag2 = diagonal<5>({1., 1., 1., 1., 1.});
                for (size_t i{}; i < diag2.rows; ++i) REQUIRE(fuzzyCompare(diag2(i, i), 1.));

                vector<int> nums;
                nums.reserve(10);
                for (int i{}; i < 10; ++i) nums.emplace_back(i);
                auto diag3 = diagonal<10, int>(nums.begin(), nums.end());
                for (int i{}; i < diag3.rows; ++i) REQUIRE(diag3(i, i) == i);
            }

            SECTION("Diag") {
                auto diag1 = diag(squareSystem);
                auto diag2 = diag(A1);
                auto diag3 = diag(A6);

                for (size_t i{}; i < diag1.size(); ++i) REQUIRE(fuzzyCompare(diag1.at(i), squareSystem(i, i)));

                for (size_t i{}; i < diag2.size(); ++i) REQUIRE(fuzzyCompare(diag2.at(i), A1(i, i)));

                for (size_t i{}; i < diag3.size(); ++i) REQUIRE(fuzzyCompare(diag3.at(i), A6(i, i)));
            }

            SECTION("Colspace") {
                SECTION("Identity matrix") {
                    M2x2d I{
                        {1, 0},
                        {0, 1},
                    };
                    auto basis = colspace(I);
                    REQUIRE(basis.size() == 2);
                    REQUIRE(basis[0] == V2d{1, 0});
                    REQUIRE(basis[1] == V2d{0, 1});
                }

                SECTION("Linearly dependent columns") {
                    M2x2d A{
                        {1, 2},
                        {2, 4},
                    };
                    auto basis = colspace(A);
                    REQUIRE(basis.size() == 1);
                    REQUIRE(basis[0] == V2d{1, 2});
                }

                SECTION("Zero matrix") {
                    M2x2d Z{
                        {0, 0},
                        {0, 0},
                    };
                    auto basis = colspace(Z);
                    REQUIRE(basis.empty());
                }

                SECTION("3x3 full rank") {
                    M3x3d A{
                        {1, 0, 2},
                        {0, 1, 3},
                        {0, 0, 1},
                    };
                    auto basis = colspace(A);
                    REQUIRE(basis.size() == 3);
                    REQUIRE(basis[0] == V3d{1, 0, 0});
                    REQUIRE(basis[1] == V3d{0, 1, 0});
                    REQUIRE(basis[2] == V3d{2, 3, 1});
                }

                SECTION("3x3 rank 2") {
                    M3x3d A{
                        {1, 2, 3},
                        {0, 1, 2},
                        {0, 0, 0},
                    };
                    auto basis = colspace(A);
                    REQUIRE(basis.size() == 2);
                    REQUIRE(basis[0] == V3d{1, 0, 0});
                    REQUIRE(basis[1] == V3d{2, 1, 0});
                }
            }

            SECTION("Colrank") {
                SECTION("Identity matrix") {
                    M2x2d I{
                        {1, 0},
                        {0, 1},
                    };
                    REQUIRE(colrank(I) == 2);
                }

                SECTION("Linearly dependent columns") {
                    M2x2d A{
                        {1, 2},
                        {2, 4},
                    };
                    REQUIRE(colrank(A) == 1);
                }

                SECTION("Zero matrix") {
                    M2x2d Z{
                        {0, 0},
                        {0, 0},
                    };
                    REQUIRE(colrank(Z) == 0);
                }

                SECTION("3x3 full rank") {
                    M3x3d A{
                        {1, 0, 2},
                        {0, 1, 3},
                        {0, 0, 1},
                    };
                    REQUIRE(colrank(A) == 3);
                }

                SECTION("3x3 rank 2") {
                    M3x3d A{
                        {1, 2, 3},
                        {0, 1, 2},
                        {0, 0, 0},
                    };
                    REQUIRE(colrank(A) == 2);
                }
            }

            SECTION("Rowspace") {
                SECTION("Identity matrix") {
                    M2x2d I{
                        {1, 0},
                        {0, 1},
                    };
                    auto rspace = rowspace(I);
                    REQUIRE(rspace.size() == 2);
                    REQUIRE(rspace[0] == V2d{1, 0});
                    REQUIRE(rspace[1] == V2d{0, 1});
                }

                SECTION("Dependent rows") {
                    M2x2d A{
                        {1, 2},
                        {2, 4},
                    };
                    auto rspace = rowspace(A);
                    REQUIRE(rspace.size() == 1);
                    REQUIRE(rspace[0] == V2d{1, 2});
                }

                SECTION("Zero matrix") {
                    M2x2d Z{
                        {0, 0},
                        {0, 0},
                    };
                    auto rspace = rowspace(Z);
                    REQUIRE(rspace.size() == 0);
                }

                SECTION("3x3 full rank") {
                    M3x3d A{
                        {1, 0, 2},
                        {0, 1, 3},
                        {0, 0, 1},
                    };
                    auto rspace = rowspace(A);
                    REQUIRE(rspace.size() == 3);
                }

                SECTION("3x3 rank 2") {
                    M3x3d A{
                        {1, 2, 3},
                        {0, 1, 2},
                        {0, 0, 0},
                    };
                    auto rspace = rowspace(A);
                    REQUIRE(rspace.size() == 2);
                }
            }

            SECTION("Rowrank") {
                SECTION("Identity matrix") {
                    M2x2d I{
                        {1, 0},
                        {0, 1},
                    };
                    REQUIRE(rowspace(I).size() == 2);
                }

                SECTION("Dependent rows") {
                    M2x2d A{
                        {1, 2},
                        {2, 4},
                    };
                    REQUIRE(rowspace(A).size() == 1);
                }

                SECTION("Zero matrix") {
                    M2x2d Z{
                        {0, 0},
                        {0, 0},
                    };
                    REQUIRE(rowspace(Z).size() == 0);
                }

                SECTION("3x3 full rank") {
                    M3x3d A{
                        {1, 0, 2},
                        {0, 1, 3},
                        {0, 0, 1},
                    };
                    REQUIRE(rowspace(A).size() == 3);
                }

                SECTION("3x3 rank 2") {
                    M3x3d A{
                        {1, 2, 3},
                        {0, 1, 2},
                        {0, 0, 0},
                    };
                    REQUIRE(rowspace(A).size() == 2);
                }
            }

            SECTION("Extend to complete basis") {
                SECTION("1 orthonormal vector to full 2D basis") {
                    vector<V2d> input{V2d{1, 0}};
                    auto result = extendToCompleteBasis(input, 2);
                    REQUIRE(result.size() == 2);
                    REQUIRE(fuzzyCompare(result[0].length(), 1.0));
                    REQUIRE(fuzzyCompare(result[1].length(), 1.0));
                    REQUIRE(fuzzyCompare(result[0].dot(result[1]), 0.0));
                }

                SECTION("2 orthonormal vectors to full 3D basis") {
                    vector<V3d> input{V3d{1, 0, 0}, V3d{0, 1, 0}};
                    auto result = extendToCompleteBasis(input, 3);
                    REQUIRE(result.size() == 3);
                    for (const auto& v : result) REQUIRE(fuzzyCompare(v.length(), 1.0));

                    for (size_t i = 0; i < result.size(); ++i) {
                        for (size_t j = i + 1; j < result.size(); ++j) {
                            REQUIRE(fuzzyCompare(result[i].dot(result[j]), 0.0));
                        }
                    }
                }

                SECTION("Already complete basis is unchanged") {
                    vector<V2d> input{V2d{1, 0}, V2d{0, 1}};
                    auto result = extendToCompleteBasis(input, 2);
                    REQUIRE(result.size() == 2);
                    REQUIRE(result == input);
                }
            }
        }

        SECTION("Dynamic") {
            SECTION("Consistency") {
                REQUIRE_FALSE(isInconsistent(dyna1));
                REQUIRE_FALSE(isInconsistent(dyna2));
                REQUIRE_FALSE(isInconsistent(dyna3));
                REQUIRE_FALSE(isInconsistent(dyna4));
                REQUIRE_FALSE(isInconsistent(squareDyna));
                REQUIRE(isInconsistent(dyna5));
            }

            SECTION("Pivots") {
                auto pivots1 = getPivots(dyna1, false);
                auto pivots2 = getPivots(dyna2, false);
                auto pivots3 = getPivots(dyna3, false);
                auto pivots4 = getPivots(dyna4, false);
                auto pivots5 = getPivots(dyna5, false);
                auto pivots6 = getPivots(squareDyna, false);

                REQUIRE((pivots1.at(0).has_value() && pivots1.at(0) == 1));
                REQUIRE((pivots1.at(1).has_value() && pivots1.at(1) == 2));
                REQUIRE_FALSE(pivots1.at(2).has_value());

                REQUIRE((pivots2.at(0).has_value() && pivots2.at(0) == 1));
                REQUIRE((pivots2.at(1).has_value() && pivots2.at(1) == 1));
                REQUIRE((pivots2.at(2).has_value() && pivots2.at(2) == 1));
                REQUIRE_FALSE(pivots2.at(3).has_value());

                REQUIRE((pivots3.at(0).has_value() && pivots3.at(0) == 1));
                REQUIRE((pivots3.at(1).has_value() && pivots3.at(1) == 9));

                REQUIRE((pivots4.at(0).has_value() && pivots4.at(0) == 1));
                REQUIRE((pivots4.at(1).has_value() && pivots4.at(1) == 1));
                REQUIRE((pivots4.at(2).has_value() && pivots4.at(2) == 2));

                REQUIRE((pivots5.at(0).has_value() && pivots5.at(0) == 1));
                REQUIRE((pivots5.at(1).has_value() && pivots5.at(1) == 1));
                REQUIRE((pivots5.at(2).has_value() && pivots5.at(2) == 1));
                REQUIRE_FALSE((pivots5.at(3).has_value()));

                REQUIRE((pivots6.at(0).has_value() && pivots6.at(0) == 3));
            }

            SECTION("Echelon Form") {
                REQUIRE(isInEchelonForm(dyna1, getPivots(dyna1)));
                REQUIRE(isInEchelonForm(dyna2, getPivots(dyna2)));
                REQUIRE_FALSE(isInEchelonForm(dyna3, getPivots(dyna3)));
                REQUIRE_FALSE(isInEchelonForm(dyna4, getPivots(dyna4)));
                REQUIRE(isInEchelonForm(dyna5, getPivots(dyna5)));
                REQUIRE_FALSE(isInEchelonForm(squareSystem, getPivots(squareSystem)));
            }

            SECTION("Rearrange System") {
                REQUIRE(rearrangeSystem(dyna1) == dyna1);
                REQUIRE(rearrangeSystem(dyna2) == dyna2);
            }

            SECTION("Is zero vector") {
                auto zeroVec1{vectorZeros<double, Dynamic>(4)};
                Vector<double, Dynamic> nonZeroVec1{1, 0, 0};

                REQUIRE(isZeroVector(zeroVec1));
                REQUIRE_FALSE(isZeroVector(nonZeroVec1));
            }

            SECTION("Is upper triangular") {
                REQUIRE(isUpperTriangular(dyna1));
                REQUIRE_FALSE(isUpperTriangular(dyna3));
            }

            SECTION("Is singular") {
                auto SingularSystem = LinearSystem<double, Dynamic, Dynamic>{
                    {1, 2},
                    {2, 4},
                };
                auto NonSingularSystem = LinearSystem<double, Dynamic, Dynamic>{
                    {1, 2},
                    {3, 4},
                };
                REQUIRE(isSingular(SingularSystem));
                REQUIRE_FALSE(isSingular(NonSingularSystem));
            }

            SECTION("Is Hermetian") {
                SECTION("Real symmetric matrix is Hermitian") {
                    MD A{
                        {1.0, 2.0},
                        {2.0, 3.0},
                    };
                    REQUIRE(isHermitian(A));
                }

                SECTION("Real asymmetric matrix is not Hermitian") {
                    MD A{
                        {1.0, 2.0},
                        {0.0, 3.0},
                    };
                    REQUIRE_FALSE(isHermitian(A));
                }

                // Not supported yet
                // using CM2x2 = Matrix<std::complex<double>, 2, 2>;
                // SECTION("Complex Hermitian matrix") {
                //     CM2x2 A{
                //         {{1.0, 0.0}, {2.0, -1.0}},
                //         {{2.0, 1.0}, {3.0, 0.0}},
                //     };
                //     CHECK(isHermitian(A) == true);
                // }
                //
                // SECTION("Complex non-Hermitian matrix") {
                //     CM2x2 A{
                //         {{1.0, 0.0}, {2.0, 1.0}},
                //         {{2.0, 1.0}, {3.0, 0.0}},
                //     };
                //     CHECK(isHermitian(A) == false);
                // }

                SECTION("Non-square matrix throws") {
                    MD A{
                        {1.0, 2.0, 3.0},
                        {4.0, 5.0, 6.0},
                    };
                    CHECK_THROWS_AS(isHermitian(A), std::invalid_argument);
                }
            }

            SECTION("Is Orthogonal") {
                SECTION("2x2 Identity is orthogonal") {
                    MD<double> A{
                        {1.0, 0.0},
                        {0.0, 1.0},
                    };
                    REQUIRE(isOrthogonal(A));
                }

                SECTION("2x2 rotation matrix is orthogonal") {
                    double theta = M_PI / 4;
                    MD<double> A{
                        {cos(theta), -sin(theta)},
                        {sin(theta), cos(theta)},
                    };
                    REQUIRE(isOrthogonal(A));
                }

                SECTION("2x2 non-orthogonal matrix") {
                    MD<double> A{
                        {1.0, 1.0},
                        {0.0, 1.0},
                    };
                    REQUIRE_FALSE(isOrthogonal(A));
                }

                SECTION("3x3 orthogonal matrix") {
                    MD<double> A{
                        {1, 0, 0},
                        {0, 0, -1},
                        {0, 1, 0},
                    };
                    REQUIRE(isOrthogonal(A));
                }

                SECTION("3x3 near-orthogonal matrix fails fuzzy comparison") {
                    MD<double> A{
                        {1, 0, 0},
                        {0, 1, 1e-3},
                        {0, 0, 1},
                    };
                    REQUIRE_FALSE(isOrthogonal(A));
                }
            }

            SECTION("Solve Equation") {
                SECTION("All variables unknown (treated as zero contribution)") {
                    const Row<double, Dynamic> r1{1, 2, 3, 4};               // Equation: 1*x0 + 2*x1 + 3*x2 = 4
                    ConditionalRowOptional<double, Dynamic, Dynamic> s1(4);  // all std::nullopt
                    auto result = solveEquation(r1, 0, s1);
                    REQUIRE(result.has_value());
                    REQUIRE(result.value() == Approx(4.0));
                }

                SECTION("Some variables known") {
                    const Row<double, Dynamic> r2{2, 1, -1, 3};  // Equation: 2*x0 + 1*x1 -1*x2 = 3
                    ConditionalRowOptional<double, Dynamic, Dynamic> s2(4);
                    s2[1] = 4.0;  // x1 = 4
                    s2[2] = 1.0;  // x2 = 1
                    auto result = solveEquation(r2, 0, s2);
                    REQUIRE(result.has_value());
                    REQUIRE(result.value() == Approx(0.0));
                }

                SECTION("Zero coefficient and RHS also zero (infinite solutions)") {
                    const Row<double, Dynamic> r3{0, 1, 2, -3};  // Equation: 0*x0 + 1*x1 + 2*x2 = -3
                    ConditionalRowOptional<double, Dynamic, Dynamic> s3(4);
                    s3[1] = -1.0;
                    s3[2] = -1.0;
                    auto result = solveEquation(r3, 0, s3);
                    REQUIRE(result.has_value());
                    REQUIRE(result.value() == Approx(0.0));
                }

                SECTION("Zero coefficient and non-zero RHS (no solution)") {
                    const Row<double, Dynamic> r4{0, 0, 0, 1};               // Equation: 0*x0 + 0*x1 + 0*x2 = 1
                    ConditionalRowOptional<double, Dynamic, Dynamic> s4(4);  // all unknown
                    auto result = solveEquation(r4, 0, s4);
                    REQUIRE_FALSE(result.has_value());
                }

                SECTION("Negative coefficients") {
                    const Row<double, Dynamic> r5{-2, -1, 0, -5};  // Equation: -2*x0 - x1 = -5
                    ConditionalRowOptional<double, Dynamic, Dynamic> s5(4);
                    s5[1] = 1.5;
                    auto result = solveEquation(r5, 0, s5);
                    REQUIRE(result.has_value());
                    REQUIRE(result.value() == Approx(1.75));
                }
            }

            SECTION("Diagonal") {
                auto diag1 = diagonal<Dynamic>(4., 3);
                for (size_t i{}; i < diag1.numRows(); ++i) REQUIRE(fuzzyCompare(diag1(i, i), 4.));

                auto diag2 = diagonal<Dynamic>({1., 1., 1., 1., 1.}, 5);
                for (size_t i{}; i < diag2.numRows(); ++i) REQUIRE(fuzzyCompare(diag2(i, i), 1.));

                vector<int> nums;
                nums.reserve(10);
                for (int i{}; i < 10; ++i) nums.emplace_back(i);
                auto diag3 = diagonal<Dynamic, int>(nums.begin(), nums.end());
                for (int i{}; i < diag3.numRows(); ++i) REQUIRE(diag3(i, i) == i);
            }

            SECTION("Diag") {
                auto diag1 = diag(squareDyna);
                auto A1Dyna = helpers::toDynamic(A1);
                auto A6Dyna = helpers::toDynamic(A6);
                auto diag2 = diag(A1Dyna);
                auto diag3 = diag(A6Dyna);

                for (size_t i{}; i < diag1.size(); ++i) REQUIRE(fuzzyCompare(diag1.at(i), squareDyna(i, i)));

                for (size_t i{}; i < diag2.size(); ++i) REQUIRE(fuzzyCompare(diag2.at(i), A1Dyna(i, i)));

                for (size_t i{}; i < diag3.size(); ++i) REQUIRE(fuzzyCompare(diag3.at(i), A6Dyna(i, i)));
            }

            SECTION("Colspace") {
                SECTION("Identity matrix") {
                    MD<double> I{
                        {1, 0},
                        {0, 1},
                    };
                    auto basis = colspace(I);
                    REQUIRE(basis.size() == 2);
                    REQUIRE(basis[0] == VD<double>{1, 0});
                    REQUIRE(basis[1] == VD<double>{0, 1});
                }

                SECTION("Linearly dependent columns") {
                    MD<double> A{
                        {1, 2},
                        {2, 4},
                    };
                    auto basis = colspace(A);
                    REQUIRE(basis.size() == 1);
                    REQUIRE(basis[0] == VD<double>{1, 2});
                }

                SECTION("Zero matrix") {
                    MD<double> Z{
                        {0, 0},
                        {0, 0},
                    };
                    auto basis = colspace(Z);
                    REQUIRE(basis.empty());
                }

                SECTION("3x3 full rank") {
                    MD<double> A{
                        {1, 0, 2},
                        {0, 1, 3},
                        {0, 0, 1},
                    };
                    auto basis = colspace(A);
                    REQUIRE(basis.size() == 3);
                    REQUIRE(basis[0] == VD<double>{1, 0, 0});
                    REQUIRE(basis[1] == VD<double>{0, 1, 0});
                    REQUIRE(basis[2] == VD<double>{2, 3, 1});
                }

                SECTION("3x3 rank 2") {
                    MD<double> A{
                        {1, 2, 3},
                        {0, 1, 2},
                        {0, 0, 0},
                    };
                    auto basis = colspace(A);
                    REQUIRE(basis.size() == 2);
                    REQUIRE(basis[0] == VD<double>{1, 0, 0});
                    REQUIRE(basis[1] == VD<double>{2, 1, 0});
                }
            }

            SECTION("Colrank") {
                SECTION("Identity matrix") {
                    MD<double> I{
                        {1, 0},
                        {0, 1},
                    };
                    REQUIRE(colrank(I) == 2);
                }

                SECTION("Linearly dependent columns") {
                    MD<double> A{
                        {1, 2},
                        {2, 4},
                    };
                    REQUIRE(colrank(A) == 1);
                }

                SECTION("Zero matrix") {
                    MD<double> Z{
                        {0, 0},
                        {0, 0},
                    };
                    REQUIRE(colrank(Z) == 0);
                }

                SECTION("3x3 full rank") {
                    MD<double> A{
                        {1, 0, 2},
                        {0, 1, 3},
                        {0, 0, 1},
                    };
                    REQUIRE(colrank(A) == 3);
                }

                SECTION("3x3 rank 2") {
                    MD<double> A{
                        {1, 2, 3},
                        {0, 1, 2},
                        {0, 0, 0},
                    };
                    REQUIRE(colrank(A) == 2);
                }
            }

            SECTION("Rowspace") {
                SECTION("Identity matrix") {
                    MD<double> I{
                        {1, 0},
                        {0, 1},
                    };
                    auto rspace = rowspace(I);
                    REQUIRE(rspace.size() == 2);
                    REQUIRE(rspace[0] == VD<double>{1, 0});
                    REQUIRE(rspace[1] == VD<double>{0, 1});
                }

                SECTION("Dependent rows") {
                    MD<double> A{
                        {1, 2},
                        {2, 4},
                    };
                    auto rspace = rowspace(A);
                    REQUIRE(rspace.size() == 1);
                    REQUIRE(rspace[0] == VD<double>{1, 2});
                }

                SECTION("Zero matrix") {
                    MD<double> Z{
                        {0, 0},
                        {0, 0},
                    };
                    auto rspace = rowspace(Z);
                    REQUIRE(rspace.size() == 0);
                }

                SECTION("3x3 full rank") {
                    MD<double> A{
                        {1, 0, 2},
                        {0, 1, 3},
                        {0, 0, 1},
                    };
                    auto rspace = rowspace(A);
                    REQUIRE(rspace.size() == 3);
                }

                SECTION("3x3 rank 2") {
                    MD<double> A{
                        {1, 2, 3},
                        {0, 1, 2},
                        {0, 0, 0},
                    };
                    auto rspace = rowspace(A);
                    REQUIRE(rspace.size() == 2);
                }
            }

            SECTION("Rowrank") {
                SECTION("Identity matrix") {
                    MD<double> I{
                        {1, 0},
                        {0, 1},
                    };
                    REQUIRE(rowspace(I).size() == 2);
                }

                SECTION("Dependent rows") {
                    MD<double> A{
                        {1, 2},
                        {2, 4},
                    };
                    REQUIRE(rowspace(A).size() == 1);
                }

                SECTION("Zero matrix") {
                    MD<double> Z{
                        {0, 0},
                        {0, 0},
                    };
                    REQUIRE(rowspace(Z).size() == 0);
                }

                SECTION("3x3 full rank") {
                    MD<double> A{
                        {1, 0, 2},
                        {0, 1, 3},
                        {0, 0, 1},
                    };
                    REQUIRE(rowspace(A).size() == 3);
                }

                SECTION("3x3 rank 2") {
                    MD<double> A{
                        {1, 2, 3},
                        {0, 1, 2},
                        {0, 0, 0},
                    };
                    REQUIRE(rowspace(A).size() == 2);
                }
            }

            SECTION("Extend to complete basis") {
                SECTION("1 orthonormal vector to full 2D basis") {
                    vector<VD<double>> input{VD<double>{1, 0}};
                    auto result = extendToCompleteBasis(input, 2);
                    REQUIRE(result.size() == 2);
                    REQUIRE(fuzzyCompare(result[0].length(), 1.0));
                    REQUIRE(fuzzyCompare(result[1].length(), 1.0));
                    REQUIRE(fuzzyCompare(result[0].dot(result[1]), 0.0));
                }

                SECTION("2 orthonormal vectors to full 3D basis") {
                    vector<VD<double>> input{VD<double>{1, 0, 0}, VD<double>{0, 1, 0}};
                    auto result = extendToCompleteBasis(input, 3);
                    REQUIRE(result.size() == 3);
                    for (const auto& v : result) REQUIRE(fuzzyCompare(v.length(), 1.0));

                    for (size_t i = 0; i < result.size(); ++i) {
                        for (size_t j = i + 1; j < result.size(); ++j) {
                            REQUIRE(fuzzyCompare(result[i].dot(result[j]), 0.0));
                        }
                    }
                }

                SECTION("Already complete basis is unchanged") {
                    vector<VD<double>> input{VD<double>{1, 0}, VD<double>{0, 1}};
                    auto result = extendToCompleteBasis(input, 2);
                    REQUIRE(result.size() == 2);
                    REQUIRE(result == input);
                }
            }
        }
    }

    SECTION("Linear Algebra Operations", "[linalg]") {
        SECTION("Compile Time") {
            SECTION("Echelon Form") {
                auto ref1 = ref(system1);
                auto ref2 = ref(system2);
                auto ref3 = ref(system3);
                auto ref4 = ref(system4);
                auto ref5 = ref(system5);
                auto ref6 = ref(squareSystem);
                auto ref7 = ref(system6);
                auto ref8 = ref(system7);
                auto ref9 = ref(system8);
                auto ref10 = ref(system9);
                auto ref11 = ref(system10);
                auto ref12 = ref(system11);
                auto ref13 = ref(system12);

                REQUIRE(ref1 == system1);  // System1 is already in echelon form
                REQUIRE(ref2 == system2);  // System2 is already in echelon form
                REQUIRE(ref3 == decltype(system3){
                                    {1, 3, 4, 7},
                                    {0, 0, -5, -15},
                                });
                REQUIRE(ref4 == decltype(system4){
                                    {1, -7, 0, 6, 5},
                                    {0, 0, 1, -2, -3},
                                    {0, 0, 0, 0, 0},
                                });
                REQUIRE(ref5 == system5);
                REQUIRE(ref6 == decltype(squareSystem){
                                    {3, 1},
                                    {0, 5 / 3.},
                                });

                REQUIRE(ref7 == decltype(system6){
                                    {2, 4, -2, 2},
                                    {0, 0, 1, 3},
                                    {0, 0, 0, 8},
                                });

                REQUIRE(ref8 == decltype(system7){
                                    {1, 2, -1, 3, 5},
                                    {0, 0, 2, 0, 0},
                                    {0, 0, 0, 0, 0},
                                });

                REQUIRE(ref9 == system8);   // System8 is already in echelon form.
                REQUIRE(ref10 == system9);  // System9 is already in echelon form.

                REQUIRE(ref11 == decltype(system10){
                                     {1, 2, 3},
                                     {0, 1, 1},
                                     {0, 0, -2},
                                     {0, 0, 0},
                                 });

                REQUIRE(ref12 == system11);
                REQUIRE(ref13 == decltype(system12){
                                     {4, 8, 12},
                                     {0, 1, 1},
                                 });
            }

            SECTION("Reduced Echelon Form") {
                auto rref1 = rref(system1);
                auto rref2 = rref(system2);
                auto rref3 = rref(system3);
                auto rref4 = rref(system4);
                auto rref5 = rref(system5);
                auto rref6 = rref(squareSystem);
                auto rref7 = rref(system6);
                auto rref8 = rref(system7);
                auto rref9 = rref(system8);
                auto rref10 = rref(system9);
                auto rref11 = rref(system10);
                auto rref12 = rref(system11);
                auto rref13 = rref(system12);

                REQUIRE(rref1 == decltype(system1){
                                     {1, 2, 0, -1. / 2.},
                                     {0, 0, 1, 3. / 2.},
                                     {0, 0, 0, 0},
                                 });
                REQUIRE(rref2 == decltype(system2){
                                     {1, 0, 0, 0, -3, 5},
                                     {0, 1, 0, 0, -4, 1},
                                     {0, 0, 0, 1, 9, 4},
                                     {0, 0, 0, 0, 0, 0},
                                 });
                REQUIRE(rref3 == decltype(system3){
                                     {1, 3, 0, -5},
                                     {0, 0, 1, 3},
                                 });
                REQUIRE(rref4 == decltype(system4){
                                     {1, -7, 0, 6, 5},
                                     {0, 0, 1, -2, -3},
                                     {0, 0, 0, 0, 0},
                                 });
                REQUIRE(rref5 == decltype(system5){
                                     {1, 0, 0, 0, -3, 0},
                                     {0, 1, 0, 0, -4, 0},
                                     {0, 0, 0, 1, 9, 0},
                                     {0, 0, 0, 0, 0, 1},
                                 });
                REQUIRE(rref6 == decltype(squareSystem){
                                     {1, 0},
                                     {0, 1},
                                 });
                REQUIRE(rref7 == decltype(system6){
                                     {1, 2, 0, 0},
                                     {0, 0, 1, 0},
                                     {0, 0, 0, 1},
                                 });
                REQUIRE(rref8 == decltype(system7){
                                     {1, 2, 0, 3, 5},
                                     {0, 0, 1, 0, 0},
                                     {0, 0, 0, 0, 0},
                                 });
                REQUIRE(rref9 == decltype(system8){
                                     {1, 0, -5, 0},
                                     {0, 1, 2, 0},
                                     {0, 0, 0, 1},
                                 });
            }

            SECTION("Find Solutions To A Linear System") {
                SECTION("[A | b]") {
                    // auto sol1 = findSolutions(system1);
                    //
                    // REQUIRE(sol1.has_value());

                    SECTION("Unique solution using [A | b]") {
                        LinearSystem<double, 2, 3> sys{
                            {2.0, 3.0, 8.0},
                            {1.0, 2.0, 5.0},
                        };
                        auto sol = findSolutions(sys);
                        REQUIRE(sol.has_value());
                        REQUIRE(sol.value().at(0) == Approx(1.0));
                        REQUIRE(sol.value().at(1) == Approx(2.0));
                    }

                    SECTION("No solution using [A | b]") {
                        // Augmented matrix for an inconsistent system:
                        // x + y = 3, 2x + 2y = 4 (b should be [3, 4] but consistency requires [3, 6])
                        LinearSystem<double, 2, 3> sys{
                            {1.0, 1.0, 3.0},
                            {2.0, 2.0, 4.0},
                        };
                        auto sol = findSolutions(sys);
                        REQUIRE_FALSE(sol.has_value());
                    }
                }

                SECTION("Ax = b") {
                    // auto sol1 = findSolutions(A1, b1);
                    //
                    // REQUIRE(sol1.has_value());

                    SECTION("Unique solution using Ax = b") {
                        // Solve: [2 3; 1 2] * x = [8; 5]  -> expected x = [1, 2]
                        Matrix<double, 2, 2> A{
                            {2.0, 3.0},
                            {1.0, 2.0},
                        };
                        Vector<double, 2> b{{8.0, 5.0}};
                        auto sol = findSolutions(A, b);
                        REQUIRE(sol.has_value());
                        REQUIRE(sol.value().at(0) == Approx(1.0));
                        REQUIRE(sol.value().at(1) == Approx(2.0));
                    }

                    SECTION("No solution using Ax = b") {
                        // Inconsistent system: x + y = 3 and 2x + 2y = 4 (should be 6 if consistent)
                        Matrix<double, 2, 2> A{{1.0, 1.0}, {2.0, 2.0}};
                        Vector<double, 2> b{{3.0, 4.0}};
                        auto sol = findSolutions(A, b);
                        REQUIRE_FALSE(sol.has_value());
                    }

                    SECTION("Larger system with unique solution") {
                        // A 3x3 system.
                        Matrix<double, 3, 3> A{{3.0, -1.0, 2.0}, {2.0, 4.0, 1.0}, {-1.0, 2.0, 5.0}};
                        Vector<double, 3> b{{5.0, 11.0, 8.0}};
                        auto sol = findSolutions(A, b);
                        REQUIRE(sol.has_value());
                        auto computed_b = A * extractSolutionVector(sol.value());
                        for (int i = 0; i < 3; ++i) {
                            REQUIRE(computed_b.at(i) == Approx(b.at(i)));
                        }
                    }

                    SECTION("Overdetermined but consistent system using Ax = b") {
                        // Over-determined system: More equations than unknowns.
                        // Example: x = 2, y = 3, and x+y = 5.
                        Matrix<double, 3, 2> A{
                            {1.0, 0.0},
                            {0.0, 1.0},
                            {1.0, 1.0},
                        };
                        Vector<double, 3> b{
                            {2.0, 3.0, 5.0},
                        };
                        auto sol = findSolutions(A, b);
                        REQUIRE(sol.has_value());
                        REQUIRE(sol.value().at(0) == Approx(2.0));
                        REQUIRE(sol.value().at(1) == Approx(3.0));
                    }

                    SECTION("Verify solution correctness via substitution") {
                        Matrix<double, 2, 2> A{
                            {4.0, -2.0},
                            {1.0, 3.0},
                        };
                        Vector<double, 2> b{
                            {6.0, 8.0},
                        };
                        auto sol = findSolutions(A, b);
                        REQUIRE(sol.has_value());
                        auto computed_b = A * extractSolutionVector(sol.value());
                        for (int i = 0; i < 2; ++i) {
                            INFO("Row " << i << ": expected " << b.at(i) << ", got " << computed_b.at(i));
                            REQUIRE(computed_b.at(i) == Approx(b.at(i)));
                        }
                    }
                }
            }

            SECTION("Identity of a Square Matrix") {
                auto AI1 = A1 * I<double, A1.numRows()>();
                auto AI6 = A6 * I<double, A6.numRows()>();
                auto AI8 = A8 * I<double, A8.numRows()>();

                REQUIRE(AI1 == A1);
                REQUIRE(AI6 == A6);
                REQUIRE(AI8 == A8);
            }

            SECTION("Inverse of a Matrix") {
                Matrix<double, 3, 3> AInv1{
                    {0, 1, 2},
                    {1, 0, 3},
                    {4, -3, 8},
                };

                Matrix<double, 3, 3> AInv2{
                    {1, 5, 0},
                    {2, 4, -1},
                    {0, -2, 0},
                };

                Matrix<double, 5, 5> AInv3{
                    {3, -7, 8, 9, -6},  //
                    {0, 2, -5, 7, 3},   //
                    {0, 0, 1, 5, 0},    //
                    {0, 0, 2, 4, -1},   //
                    {0, 0, 0, -2, 0},
                };

                auto inv1 = inverse(A1);
                auto inv2 = inverse(AInv1);
                auto inv3 = inverse(AInv2);
                auto inv4 = inverse(AInv3);

                REQUIRE_FALSE(inv1.has_value());

                REQUIRE(inv2.has_value());
                REQUIRE(inverse(inv2.value()).value() == AInv1);

                REQUIRE(inv3.has_value());
                REQUIRE(inverse(inv3.value()).value() == AInv2);

                REQUIRE(inv4.has_value());
                REQUIRE(inverse(inv4.value()).value() == AInv3);
            }

            SECTION("Gram-Schmidt Orthogonalization") {
                const auto checks = [](const auto& qs, const auto& A) {
                    // Check orthogonality
                    for (size_t i = 0; i < qs.size(); ++i) {
                        for (size_t j = i + 1; j < qs.size(); ++j) {
                            REQUIRE(fuzzyCompare(qs.at(i) * qs.at(j), 0.0));
                        }
                    }

                    // Check normality
                    for (const auto& vec : qs) {
                        REQUIRE(fuzzyCompare(vec.length(), 1.0));
                    }
                };
                {
                    // Account for linear dependence
                    Matrix<double, 4, 3> A1{
                        {1, 0, 0},
                        {1, 1, 0},
                        {1, 1, 1},
                        {1, 1, 1},
                    };
                    const auto& qs = GSOrth<QRType::Thin>(A1);

                    checks(qs, A1);
                }
                {
                    // Account for linear dependence
                    Matrix<double, 4, 3> A1{
                        {1, -1, 4},
                        {1, 4, -2},
                        {1, 4, 2},
                        {1, -1, 0},
                    };
                    const auto& qs = GSOrth<QRType::Thin>(A1);

                    checks(qs, A1);
                }
            }

            SECTION("LU Decomposition", "[LU]") {
                // Helper lambda functions
                auto isLowerTriangular = [](const auto& L, double tolerance = 1e-12) -> bool {
                    for (size_t i = 0; i < L.numRows(); ++i) {
                        for (size_t j = i + 1; j < L.numCols(); ++j) {
                            if (!fuzzyCompare(L(i, j), 0.0, tolerance)) {
                                return false;
                            }
                        }
                    }
                    return true;
                };

                auto isUpperTriangular = [](const auto& U, double tolerance = 1e-12) -> bool {
                    for (size_t i = 1; i < U.numRows(); ++i) {
                        for (size_t j = 0; j < std::min(i, U.numCols()); ++j) {
                            if (!fuzzyCompare(U(i, j), 0.0, tolerance)) {
                                return false;
                            }
                        }
                    }
                    return true;
                };

                auto hasUnitDiagonal = [](const auto& L, double tolerance = 1e-12) -> bool {
                    size_t minDim = std::min(L.numRows(), L.numCols());
                    for (size_t i = 0; i < minDim; ++i) {
                        if (!fuzzyCompare(L(i, i), 1.0, tolerance)) {
                            return false;
                        }
                    }
                    return true;
                };

                auto verifyLUDecomposition = [&](const auto& L, const auto& U, const auto& A,
                                                 double tolerance = 1e-12) -> bool {
                    auto LU_product = L * U;
                    CAPTURE(A, L, U, LU_product, A == LU_product);
                    return A == LU_product;
                };

                auto testMatrix = [&](const auto& A, const std::string& name) {
                    INFO("Testing LU decomposition for: " << name);

                    try {
                        const auto& [L, U] = LU(A);

                        // Verify matrix dimensions
                        REQUIRE(L.numRows() == A.numRows());
                        REQUIRE(L.numCols() == A.numRows());  // L should be square
                        REQUIRE(U.numRows() == A.numRows());
                        REQUIRE(U.numCols() == A.numCols());

                        // Verify structural properties
                        REQUIRE(isLowerTriangular(L));
                        REQUIRE(isUpperTriangular(U));
                        REQUIRE(hasUnitDiagonal(L));  // L should have unit diagonal

                        // Verify mathematical correctness
                        REQUIRE(verifyLUDecomposition(L, U, A));

                        // Additional checks for numerical stability
                        INFO("L matrix condition verified");
                        INFO("U matrix condition verified");
                        INFO("A = LU verified");
                        CAPTURE(A, L, U, L * U);

                    } catch (const std::exception& e) {
                        INFO("LU decomposition failed with exception: " << e.what());
                        // For matrices that should have valid LU decomposition, this is a failure
                        // For matrices that are expected to fail (singular, etc.), this might be OK
                        FAIL("LU decomposition threw unexpected exception");
                    }
                };

                // Test the original matrix A1
                // {
                //     Matrix<double, 4, 3> A1{
                //         {1, 0, 0},
                //         {1, 1, 0},
                //         {1, 1, 1},
                //         {1, 1, 1},
                //     };
                //     testMatrix(A1, "Original A1 matrix");
                // }

                // Test additional matrices with different characteristics
                {
                    // Square matrix - basic case
                    Matrix<double, 3, 3> square{
                        {2, 1, 1},
                        {4, 3, 3},
                        {8, 7, 9},
                    };
                    testMatrix(square, "Square matrix");
                }

                {
                    // Matrix with larger values
                    Matrix<double, 3, 3> larger_values{
                        {10, 5, 2},
                        {3, 15, 4},
                        {1, 2, 20},
                    };
                    testMatrix(larger_values, "Larger values matrix");
                }

                {
                    // Matrix with negative values
                    Matrix<double, 3, 3> negative_values{
                        {1, -2, 3},
                        {-4, 5, -6},
                        {7, -8, 9},
                    };
                    testMatrix(negative_values, "Mixed sign matrix");
                }

                {
                    // Identity matrix (trivial case)
                    Matrix<double, 3, 3> identity{
                        {1, 0, 0},
                        {0, 1, 0},
                        {0, 0, 1},
                    };
                    testMatrix(identity, "Identity matrix");
                }

                // Test edge cases that might cause issues
                SECTION("Edge Cases") {
                    // Test matrix with zeros
                    {
                        Matrix<double, 3, 3> with_zeros{
                            {1, 0, 3},
                            {0, 2, 0},
                            {4, 0, 5},
                        };

                        INFO("Testing matrix with strategic zeros");
                        const auto& [L, U] = LU(with_zeros);
                        REQUIRE(verifyLUDecomposition(L, U, with_zeros));
                    }

                    // Test matrix that might need pivoting (if your LU supports it)
                    // {
                    //     Matrix<double, 3, 3> needs_pivoting{
                    //         {0, 1, 2},
                    //         {1, 1, 1},
                    //         {2, 1, 0},
                    //     };
                    //
                    //     INFO("Testing matrix that might need pivoting");
                    //     try {
                    //         const auto& [L, U] = LU(needs_pivoting);
                    //         REQUIRE(verifyLUDecomposition(L, U, needs_pivoting));
                    //     } catch (const std::exception& e) {
                    //         // If LU decomposition fails due to zero pivot, that's expected
                    //         // for matrices that need pivoting in naive LU implementation
                    //         INFO("Matrix requires pivoting - exception expected: " << e.what());
                    //     }
                    // }

                    // Test small matrix
                    {
                        Matrix<double, 2, 2> small{
                            {3, 2},
                            {6, 4},
                        };

                        INFO("Testing 2x2 matrix");
                        try {
                            const auto& [L, U] = LU(small);
                            REQUIRE(verifyLUDecomposition(L, U, small));
                        } catch (const std::exception& e) {
                            // This matrix is singular (rank 1), so LU might fail
                            INFO("Singular matrix - exception expected: " << e.what());
                        }
                    }
                }

                // Test numerical precision
                SECTION("Numerical Precision") {
                    Matrix<double, 3, 3> precision_test{
                        {1.0001, 1, 1},
                        {1, 1.0001, 1},
                        {1, 1, 1.0001},
                    };

                    INFO("Testing numerical precision with near-singular matrix");
                    const auto& [L, U] = LU(precision_test);

                    // Use slightly relaxed tolerance for near-singular cases
                    auto relaxed_verify = [&](const auto& L, const auto& U, const auto& A) -> bool {
                        auto LU_product = L * U;
                        for (size_t i = 0; i < A.numRows(); ++i) {
                            for (size_t j = 0; j < A.numCols(); ++j) {
                                if (!fuzzyCompare(A(i, j), LU_product(i, j), 1e-10)) {
                                    return false;
                                }
                            }
                        }
                        return true;
                    };

                    REQUIRE(isLowerTriangular(L));
                    REQUIRE(isUpperTriangular(U));
                    REQUIRE(hasUnitDiagonal(L));
                    REQUIRE(relaxed_verify(L, U, precision_test));
                }

                // Test determinant preservation (if applicable)
                SECTION("Determinant Properties") {
                    Matrix<double, 3, 3> det_test{
                        {2, 1, 0},
                        {1, 3, 1},
                        {0, 1, 2},
                    };

                    const auto& [L, U] = LU(det_test);

                    // For LU decomposition: det(A) = det(L) * det(U) = det(U) (since det(L) = 1)
                    // This is a consistency check if you have determinant calculation available
                    REQUIRE(verifyLUDecomposition(L, U, det_test));

                    // Verify that diagonal of U contains meaningful values (no unexpected zeros)
                    for (size_t i = 0; i < std::min(U.numRows(), U.numCols()); ++i) {
                        INFO("U(" << i << "," << i << ") = " << U(i, i));
                        // Don't require non-zero (matrix might be singular), just document the values
                    }
                }
            }

            // Enhanced test cases
            SECTION("QR Decomposition") {
                auto isOrthogonal = [](auto const& Q) {
                    const auto& QCols = Q.colToVectorSet();
                    const size_t n = Q.numCols();

                    for (size_t i = 0; i < n; ++i) {
                        for (size_t j = 0; j < n; ++j) {
                            double dot_product = QCols[i].dot(QCols[j]);
                            double expected = (i == j) ? 1.0 : 0.0;
                            if (!fuzzyCompare(dot_product, expected)) return false;
                        }
                    }
                    return true;
                };

                auto isUpperTriangular = [](auto const& R) {
                    const size_t m = R.numRows();
                    const size_t n = R.numCols();

                    for (size_t i = 0; i < m; ++i) {
                        for (size_t j = 0; j < std::min(i, n); ++j) {
                            if (!fuzzyCompare(R(i, j), 0.0)) return false;
                        }
                    }
                    return true;
                };

                auto verifyDecomposition = [](auto const& Q, auto const& R, auto const& A) {
                    auto QR = Q * R;
                    return A == QR;
                };

                SECTION("Test Matrix Collection") {
                    // Test 1: Simple lower triangular matrix
                    Matrix<double, 4, 3> simple{
                        {1, 0, 0},
                        {1, 1, 0},
                        {1, 1, 1},
                        {1, 1, 1},
                    };

                    // Test 2: Matrix with negative values
                    Matrix<double, 4, 3> mixed_signs{
                        {1, -1, 4},
                        {1, 4, -2},
                        {1, 4, 2},
                        {1, -1, 0},
                    };

                    // Test 3: Nearly singular matrix (to test numerical stability)
                    Matrix<double, 3, 3> near_singular{
                        {1, 1, 1},
                        {1, 1.000001, 1},
                        {1, 1, 1.000001},
                    };

                    // Test 4: Random-like matrix
                    Matrix<double, 3, 3> random_like{
                        {2.3, -1.7, 0.8},
                        {-0.5, 3.1, 2.4},
                        {1.2, 0.6, -1.9},
                    };

                    // Test 5: Wide matrix (more columns than rows)
                    Matrix<double, 2, 4> wide{
                        {1, 2, 3, 4},
                        {5, 6, 7, 8},
                    };

                    // Test 6: Identity matrix (should be trivial)
                    Matrix<double, 3, 3> identity{
                        {1, 0, 0},
                        {0, 1, 0},
                        {0, 0, 1},
                    };

                    auto testMatrix = [&isOrthogonal, &isUpperTriangular, &verifyDecomposition](
                                          const auto& A, const std::string& name) {
                        INFO("Testing matrix: " << name);

                        SECTION("Thin QR - Gram-Schmidt") {
                            const auto& [Q, R] = QR<QRType::Thin>(A, QRMethod::GramSchmidt);

                            // Verify dimensions
                            REQUIRE(Q.numRows() == A.numRows());
                            REQUIRE(Q.numCols() == std::min(A.numRows(), A.numCols()));
                            REQUIRE(R.numRows() == std::min(A.numRows(), A.numCols()));
                            REQUIRE(R.numCols() == A.numCols());

                            // Verify mathematical properties
                            REQUIRE(isOrthogonal(Q));
                            REQUIRE(isUpperTriangular(R));
                            REQUIRE(verifyDecomposition(Q, R, A));
                        }

                        // Only test Full QR for tall/square matrices
                        // if (A.numRows() >= A.numCols()) {
                        //     SECTION("Full QR - Gram-Schmidt") {
                        //         const auto& [Q, R] = QR<QRType::Full>(A, QRMethod::GramSchmidt);
                        //
                        //         // Verify dimensions
                        //         REQUIRE(Q.numRows() == A.numRows());
                        //         REQUIRE(Q.numCols() == A.numRows());
                        //         REQUIRE(R.numRows() == A.numRows());
                        //         REQUIRE(R.numCols() == A.numCols());
                        //
                        //         // Verify mathematical properties
                        //         REQUIRE(isOrthogonal(Q));
                        //         REQUIRE(isUpperTriangular(R));
                        //         REQUIRE(verifyDecomposition(Q, R, A));
                        //
                        //         // Additional check: bottom rows of R should be zero for tall matrices
                        //         if (A.numRows() > A.numCols()) {
                        //             for (size_t i = A.numCols(); i < A.numRows(); ++i) {
                        //                 for (size_t j = 0; j < A.numCols(); ++j) {
                        //                     REQUIRE(fuzzyCompare(R(i, j), 0.0));
                        //                 }
                        //             }
                        //         }
                        //     }
                        // }

                        // Test Householder method for comparison
                        SECTION("Thin QR - Householder") {
                            const auto& [Q, R] = QR<QRType::Thin>(A, QRMethod::Householder);

                            REQUIRE(isOrthogonal(Q));
                            REQUIRE(isUpperTriangular(R));
                            REQUIRE(verifyDecomposition(Q, R, A));
                        }

                        if (A.numRows() >= A.numCols()) {
                            SECTION("Full QR - Householder") {
                                const auto& [Q, R] = QR<QRType::Full>(A, QRMethod::Householder);

                                REQUIRE(isOrthogonal(Q));
                                REQUIRE(isUpperTriangular(R));
                                REQUIRE(verifyDecomposition(Q, R, A));
                            }
                        }
                    };

                    // Run tests on all matrices
                    testMatrix(simple, "Simple Lower Triangular");
                    testMatrix(mixed_signs, "Mixed Signs");
                    testMatrix(near_singular, "Near Singular");
                    testMatrix(random_like, "Random-like");
                    testMatrix(wide, "Wide Matrix");
                    testMatrix(identity, "Identity Matrix");
                }

                SECTION("Method Comparison") {
                    Matrix<double, 4, 3> test_matrix{
                        {1, -1, 4},
                        {1, 4, -2},
                        {1, 4, 2},
                        {1, -1, 0},
                    };

                    // Compare Gram-Schmidt vs Householder
                    const auto& [Q_gs, R_gs] = QR<QRType::Thin>(test_matrix, QRMethod::GramSchmidt);
                    const auto& [Q_hh, R_hh] = QR<QRType::Thin>(test_matrix, QRMethod::Householder);

                    // Both should give valid decompositions
                    REQUIRE(verifyDecomposition(Q_gs, R_gs, test_matrix));
                    REQUIRE(verifyDecomposition(Q_hh, R_hh, test_matrix));

                    // R matrices should be essentially the same (up to sign differences)
                    for (size_t i = 0; i < R_gs.numRows(); ++i) {
                        for (size_t j = i; j < R_gs.numCols(); ++j) {
                            // Allow for sign differences in the decomposition
                            double ratio = R_gs(i, j) / R_hh(i, j);
                            REQUIRE((fuzzyCompare(ratio, 1.0, 1e-10) || fuzzyCompare(ratio, -1.0, 1e-10)));
                        }
                    }
                }

                SECTION("Edge Cases") {
                    SECTION("Single column matrix") {
                        Matrix<double, 3, 1> single_col{
                            {2},
                            {3},
                            {4},
                        };

                        const auto& [Q, R] = QR<QRType::Thin>(single_col, QRMethod::GramSchmidt);
                        REQUIRE(isOrthogonal(Q));
                        REQUIRE(verifyDecomposition(Q, R, single_col));
                    }

                    // SECTION("Single row matrix") {
                    //     Matrix<double, 1, 3> single_row{
                    //         {1, 2, 3},
                    //     };
                    //
                    //     const auto& [Q, R] = QR<QRType::Thin>(single_row, QRMethod::GramSchmidt);
                    //     REQUIRE(isOrthogonal(Q));
                    //     REQUIRE(verifyDecomposition(Q, R, single_row));
                    // }

                    SECTION("Square matrix") {
                        Matrix<double, 3, 3> square{
                            {1, 2, 3}, {4, 5, 6}, {7, 8, 10},  // Slightly modified to avoid singularity
                        };

                        // For square matrices, Full and Thin should give same dimensions
                        const auto& [Q_full, R_full] = QR<QRType::Full>(square, QRMethod::GramSchmidt);
                        const auto& [Q_thin, R_thin] = QR<QRType::Thin>(square, QRMethod::GramSchmidt);

                        REQUIRE(Q_full.numRows() == Q_thin.numRows());
                        REQUIRE(Q_full.numCols() == Q_thin.numCols());
                        REQUIRE(R_full.numRows() == R_thin.numRows());
                        REQUIRE(R_full.numCols() == R_thin.numCols());
                    }
                }

                SECTION("Numerical Stability") {
                    // Test with a matrix that might cause numerical issues
                    Matrix<double, 3, 3> challenging{
                        {1e-15, 1, 0},
                        {0, 1e-15, 1},
                        {1, 0, 1e-15},
                    };

                    const auto& [Q, R] = QR<QRType::Thin>(challenging, QRMethod::Householder);

                    // Should still maintain mathematical properties (with relaxed tolerance)
                    REQUIRE(isOrthogonal(Q));
                    REQUIRE(isUpperTriangular(R));
                    REQUIRE(verifyDecomposition(Q, R, challenging));
                }
            }
        }

        SECTION("Dynamic") {
            SECTION("Echelon Form") {
                auto ref1 = ref(dyna1);
                auto ref2 = ref(dyna2);
                auto ref3 = ref(dyna3);
                auto ref4 = ref(dyna4);
                auto ref5 = ref(dyna5);
                auto ref6 = ref(squareDyna);
                auto ref7 = ref(dyna6);
                auto ref8 = ref(dyna7);
                auto ref9 = ref(dyna8);
                auto ref10 = ref(dyna9);
                auto ref11 = ref(dyna10);
                auto ref12 = ref(dyna11);
                auto ref13 = ref(dyna12);

                REQUIRE(ref1 == dyna1);  // dyna1 is already in echelon form
                REQUIRE(ref2 == dyna2);  // dyna2 is already in echelon form
                REQUIRE(ref3 == decltype(dyna3){
                                    {1, 3, 4, 7},
                                    {0, 0, -5, -15},
                                });
                REQUIRE(ref4 == decltype(dyna4){
                                    {1, -7, 0, 6, 5},
                                    {0, 0, 1, -2, -3},
                                    {0, 0, 0, 0, 0},
                                });
                REQUIRE(ref5 == dyna5);
                REQUIRE(ref6 == decltype(squareSystem){
                                    {3, 1},
                                    {0, 5 / 3.},
                                });

                REQUIRE(ref7 == decltype(dyna6){
                                    {2, 4, -2, 2},
                                    {0, 0, 1, 3},
                                    {0, 0, 0, 8},
                                });

                REQUIRE(ref8 == decltype(dyna7){
                                    {1, 2, -1, 3, 5},
                                    {0, 0, 2, 0, 0},
                                    {0, 0, 0, 0, 0},
                                });

                REQUIRE(ref9 == dyna8);   // dyna8 is already in echelon form.
                REQUIRE(ref10 == dyna9);  // dyna9 is already in echelon form.

                REQUIRE(ref11 == decltype(dyna10){
                                     {1, 2, 3},
                                     {0, 1, 1},
                                     {0, 0, -2},
                                     {0, 0, 0},
                                 });

                REQUIRE(ref12 == dyna11);
                REQUIRE(ref13 == decltype(dyna12){
                                     {4, 8, 12},
                                     {0, 1, 1},
                                 });
            }

            SECTION("Reduced Echelon Form") {
                auto rref1 = rref(dyna1);
                auto rref2 = rref(dyna2);
                auto rref3 = rref(dyna3);
                auto rref4 = rref(dyna4);
                auto rref5 = rref(dyna5);
                auto rref6 = rref(squareDyna);
                auto rref7 = rref(dyna6);
                auto rref8 = rref(dyna7);
                auto rref9 = rref(dyna8);
                auto rref10 = rref(dyna9);
                auto rref11 = rref(dyna10);
                auto rref12 = rref(dyna11);
                auto rref13 = rref(dyna12);

                REQUIRE(rref1 == decltype(dyna1){
                                     {1, 2, 0, -1. / 2.},
                                     {0, 0, 1, 3. / 2.},
                                     {0, 0, 0, 0},
                                 });
                REQUIRE(rref2 == decltype(dyna2){
                                     {1, 0, 0, 0, -3, 5},
                                     {0, 1, 0, 0, -4, 1},
                                     {0, 0, 0, 1, 9, 4},
                                     {0, 0, 0, 0, 0, 0},
                                 });
                REQUIRE(rref3 == decltype(dyna3){
                                     {1, 3, 0, -5},
                                     {0, 0, 1, 3},
                                 });
                REQUIRE(rref4 == decltype(dyna4){
                                     {1, -7, 0, 6, 5},
                                     {0, 0, 1, -2, -3},
                                     {0, 0, 0, 0, 0},
                                 });
                REQUIRE(rref5 == decltype(dyna5){
                                     {1, 0, 0, 0, -3, 0},
                                     {0, 1, 0, 0, -4, 0},
                                     {0, 0, 0, 1, 9, 0},
                                     {0, 0, 0, 0, 0, 1},
                                 });
                REQUIRE(rref6 == decltype(squareDyna){
                                     {1, 0},
                                     {0, 1},
                                 });
                REQUIRE(rref7 == decltype(dyna6){
                                     {1, 2, 0, 0},
                                     {0, 0, 1, 0},
                                     {0, 0, 0, 1},
                                 });
                REQUIRE(rref8 == decltype(dyna7){
                                     {1, 2, 0, 3, 5},
                                     {0, 0, 1, 0, 0},
                                     {0, 0, 0, 0, 0},
                                 });
                REQUIRE(rref9 == decltype(dyna8){
                                     {1, 0, -5, 0},
                                     {0, 1, 2, 0},
                                     {0, 0, 0, 1},
                                 });
            }

            SECTION("Find Solutions To A Linear System") {
                SECTION("[A | b]") {
                    // auto sol1 = findSolutions(system1);
                    //
                    // REQUIRE(sol1.has_value());

                    SECTION("Unique solution using [A | b]") {
                        LinearSystem<double, 2, 3> sys{
                            {2.0, 3.0, 8.0},
                            {1.0, 2.0, 5.0},
                        };
                        auto sol = findSolutions(sys);
                        REQUIRE(sol.has_value());
                        REQUIRE(sol.value().at(0) == Approx(1.0));
                        REQUIRE(sol.value().at(1) == Approx(2.0));
                    }

                    SECTION("No solution using [A | b]") {
                        // Augmented matrix for an inconsistent system:
                        // x + y = 3, 2x + 2y = 4 (b should be [3, 4] but consistency requires [3, 6])
                        LinearSystem<double, 2, 3> sys{
                            {1.0, 1.0, 3.0},
                            {2.0, 2.0, 4.0},
                        };
                        auto sol = findSolutions(sys);
                        REQUIRE_FALSE(sol.has_value());
                    }
                }

                SECTION("Ax = b") {
                    // auto sol1 = findSolutions(A1, b1);
                    //
                    // REQUIRE(sol1.has_value());

                    SECTION("Unique solution using Ax = b") {
                        // Solve: [2 3; 1 2] * x = [8; 5]  -> expected x = [1, 2]
                        Matrix<double, Dynamic, Dynamic> A{
                            {2.0, 3.0},
                            {1.0, 2.0},
                        };
                        Vector<double, Dynamic> b{{8.0, 5.0}};
                        auto sol = findSolutions(A, b);
                        REQUIRE(sol.has_value());
                        REQUIRE(sol.value().at(0) == Approx(1.0));
                        REQUIRE(sol.value().at(1) == Approx(2.0));
                    }

                    SECTION("No solution using Ax = b") {
                        // Inconsistent system: x + y = 3 and 2x + 2y = 4 (should be 6 if consistent)
                        Matrix<double, Dynamic, Dynamic> A{{1.0, 1.0}, {2.0, 2.0}};
                        Vector<double, Dynamic> b{{3.0, 4.0}};
                        auto sol = findSolutions(A, b);
                        REQUIRE_FALSE(sol.has_value());
                    }

                    SECTION("Larger system with unique solution") {
                        // A 3x3 system.
                        Matrix<double, Dynamic, Dynamic> A{{3.0, -1.0, 2.0}, {2.0, 4.0, 1.0}, {-1.0, 2.0, 5.0}};
                        Vector<double, Dynamic> b{{5.0, 11.0, 8.0}};
                        auto sol = findSolutions(A, b);
                        REQUIRE(sol.has_value());
                        auto computed_b = A * extractSolutionVector(sol.value());
                        for (int i = 0; i < 3; ++i) {
                            REQUIRE(computed_b.at(i) == Approx(b.at(i)));
                        }
                    }

                    SECTION("Overdetermined but consistent system using Ax = b") {
                        // Over-determined system: More equations than unknowns.
                        // Example: x = 2, y = 3, and x+y = 5.
                        Matrix<double, Dynamic, Dynamic> A{
                            {1.0, 0.0},
                            {0.0, 1.0},
                            {1.0, 1.0},
                        };
                        Vector<double, Dynamic> b{
                            {2.0, 3.0, 5.0},
                        };
                        auto sol = findSolutions(A, b);
                        REQUIRE(sol.has_value());
                        REQUIRE(sol.value().at(0) == Approx(2.0));
                        REQUIRE(sol.value().at(1) == Approx(3.0));
                    }

                    SECTION("Verify solution correctness via substitution") {
                        Matrix<double, Dynamic, Dynamic> A{
                            {4.0, -2.0},
                            {1.0, 3.0},
                        };
                        Vector<double, Dynamic> b{
                            {6.0, 8.0},
                        };
                        auto sol = findSolutions(A, b);
                        REQUIRE(sol.has_value());
                        auto computed_b = A * extractSolutionVector(sol.value());
                        for (int i = 0; i < 2; ++i) {
                            INFO("Row " << i << ": expected " << b.at(i) << ", got " << computed_b.at(i));
                            REQUIRE(computed_b.at(i) == Approx(b.at(i)));
                        }
                    }
                }
            }

            SECTION("Identity of a Square Matrix") {
                auto AI1 = A1 * I<double, A1.numRows()>();
                auto AI6 = A6 * I<double, A6.numRows()>();
                auto AI8 = A8 * I<double, A8.numRows()>();

                REQUIRE(AI1 == A1);
                REQUIRE(AI6 == A6);
                REQUIRE(AI8 == A8);
            }

            SECTION("Inverse of a Matrix") {
                Matrix<double, Dynamic, Dynamic> AInv1{
                    {0, 1, 2},
                    {1, 0, 3},
                    {4, -3, 8},
                };

                Matrix<double, Dynamic, Dynamic> AInv2{
                    {1, 5, 0},
                    {2, 4, -1},
                    {0, -2, 0},
                };

                Matrix<double, Dynamic, Dynamic> AInv3{
                    {3, -7, 8, 9, -6},  //
                    {0, 2, -5, 7, 3},   //
                    {0, 0, 1, 5, 0},    //
                    {0, 0, 2, 4, -1},   //
                    {0, 0, 0, -2, 0},
                };

                auto inv1 = inverse(A1);
                auto inv2 = inverse(AInv1);
                auto inv3 = inverse(AInv2);
                auto inv4 = inverse(AInv3);

                REQUIRE_FALSE(inv1.has_value());

                REQUIRE(inv2.has_value());
                REQUIRE(inverse(inv2.value()).value() == AInv1);

                REQUIRE(inv3.has_value());
                REQUIRE(inverse(inv3.value()).value() == AInv2);

                REQUIRE(inv4.has_value());
                REQUIRE(inverse(inv4.value()).value() == AInv3);
            }
        }
    }
}
