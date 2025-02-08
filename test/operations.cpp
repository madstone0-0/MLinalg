#include "Operations.hpp"

#include <MLinalg.hpp>
#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>

#include "Structures.hpp"
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

    // System2: A 4x6 system already in echelon form.
    LinearSystem<double, 4, 6> system2{
        {1, -3, 0, -1, 0, -2},
        {0, 1, 0, 0, -4, 1},
        {0, 0, 0, 1, 9, 4},
        {0, 0, 0, 0, 0, 0},
    };

    // System3: A 2x4 system that requires elimination.
    LinearSystem<double, 2, 4> system3{
        {1, 3, 4, 7},
        {3, 9, 7, 6},
    };
    // Expected elimination for system3:
    //  - Row1 remains: {1, 3, 4, 7}.
    //  - Row2 -> row2 - 3*row1 = {0, 0, -5, -15}.

    // System4: A 3x5 system that requires elimination.
    LinearSystem<double, 3, 5> system4{
        {1, -7, 0, 6, 5},
        {0, 0, 1, -2, -3},
        {-1, 7, -4, 2, 7},
    };
    // Expected elimination for system4:
    //  - Row1 remains: {1, -7, 0, 6, 5}.
    //  - Row2 -> row2 + 7*row1 = {0, 0, 1, -2, -3}.

    // System5: A 4x6 inconsistent system.
    LinearSystem<double, 4, 6> system5{
        {1, -3, 0, -1, 0, -2},
        {0, 1, 0, 0, -4, 1},
        {0, 0, 0, 1, 9, 4},
        {0, 0, 0, 0, 0, 10},
    };
    // This system is already in echelon form (with the last row indicating inconsistency).

    // SquareSystem: A 2x2 square system.
    LinearSystem<double, 2, 2> squareSystem({
        {3, 1},
        {1, 2},
    });
    // Expected elimination for squareSystem:
    //  - Row1 remains: {3, 1}.
    //  - Row2 -> row2 - (1/3)*row1 = {0, 5/3}.

    // System6: A 3x4 system that requires elimination.
    LinearSystem<double, 3, 4> system6{{
        {2, 4, -2, 2},
        {1, 2, 0, 4},
        {3, 6, -4, 8},
    }};
    // Expected elimination for system6:
    //  - Row1 remains: {2, 4, -2, 2}.
    //  - Row2 -> row2 - (1/2)*row1 = {0, 0, 1, 3}.
    //  - Row3 -> row3 - (3/2)*row1 = {0, 0, -1, 5}, then row3 + row2 = {0, 0, 0, 8}.

    // System7: A 3x5 system with dependent rows.
    LinearSystem<double, 3, 5> system7{{
        {1, 2, -1, 3, 5},
        {2, 4, 0, 6, 10},
        {3, 6, 1, 9, 15},
    }};
    // Expected elimination for system7:
    //  - Row1 remains: {1, 2, -1, 3, 5}.
    //  - Row2 -> row2 - 2*row1 = {0, 0, 2, 0, 0}.
    //  - Row3 -> row3 - 3*row1 = {0, 0, 4, 0, 0}, then row3 - 2*row2 = {0, 0, 0, 0, 0}.

    // System8: A 3x4 inconsistent system (last row indicates inconsistency).
    LinearSystem<double, 3, 4> system8{{
        {1, 2, -1, 3},
        {0, 1, 2, 4},
        {0, 0, 0, 7},
    }};
    // This system is already in echelon form (with the third row being a zero–coefficient row with a nonzero constant).

    // System9: A 3x3 square system already in echelon form.
    LinearSystem<double, 3, 3> system9{{
        {2, -1, 3},
        {0, 4, 5},
        {0, 0, 6},
    }};

    // System10: A 4x3 system with one redundant row.
    LinearSystem<double, 4, 3> system10{{
        {1, 2, 3},
        {2, 4, 6},
        {0, 1, 1},
        {3, 7, 8},
    }};
    // Expected elimination for system10:
    //  - Row1 remains: {1, 2, 3}.
    //  - Row2 -> row2 - 2*row1 = {0, 0, 0}.
    //  - Row3 remains: {0, 1, 1} (pivot in column 1).
    //  - Row4 -> row4 - 3*row1 = {0, 1, -1}, then row4 - row3 = {0, 0, -2}.

    // System11: A trivial 1x3 system.
    LinearSystem<double, 1, 3> system11{{
        {5, -3, 2},
    }};
    // Already in echelon form.

    // System12: A 2x3 system that needs one elimination step.
    LinearSystem<double, 2, 3> system12{{
        {4, 8, 12},
        {2, 5, 7},
    }};
    // Expected elimination for system12:
    //  - Row1 remains: {4, 8, 12}.
    //  - Row2 -> row2 - (2/4)*row1 = {0, 1, 1};

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
                auto pivots1 = getPivots(system1);
                auto pivots2 = getPivots(system2);
                auto pivots3 = getPivots(system3);
                auto pivots4 = getPivots(system4);
                auto pivots5 = getPivots(system5);
                auto pivots6 = getPivots(squareSystem);

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
                auto pivots1 = getPivots(dyna1);
                auto pivots2 = getPivots(dyna2);
                auto pivots3 = getPivots(dyna3);
                auto pivots4 = getPivots(dyna4);
                auto pivots5 = getPivots(dyna5);
                auto pivots6 = getPivots(squareDyna);

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
        }
    }
}
