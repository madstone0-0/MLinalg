#include <MLinalg.hpp>
#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>

#include "Structures.hpp"
#include "structures/Vector.hpp"

using namespace Catch;
using namespace mlinalg;

TEST_CASE("Helper Operations", "[helper]") {
    SECTION("Consistency") {
        // Example System for Testing
        Matrix<double, 3, 4> system({{1, 2, 3, 4}, {0, 0, 0, 0}, {0, 0, 2, 3}});

        Matrix<double, 2, 2> squareSystem({{3, 1}, {1, 2}});
    }
}
