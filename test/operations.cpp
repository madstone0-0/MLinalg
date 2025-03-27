#include "Operations.hpp"

#include <MLinalg.hpp>
#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>

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
