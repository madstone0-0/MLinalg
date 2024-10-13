#include <MLinalg.hpp>
#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>

#include "Structures.hpp"
#include "structures/Vector.hpp"

using namespace Catch;
using namespace mlinalg::structures;
using namespace mlinalg::structures::helpers;

TEST_CASE("Vector", "[vector]") {
    SECTION("Creation") {
        SECTION("Compile Time") {
            SECTION("Default constructor") {
                auto v = Vector<int, 2>{};
                REQUIRE(v.at(0) == 0);
                REQUIRE(v.at(1) == 0);
                REQUIRE(v.size() == 2);
            }

            SECTION("Initializer list constructor") {
                auto v = Vector<int, 2>{1, 2};
                REQUIRE(v.at(0) == 1);
                REQUIRE(v.at(1) == 2);
                REQUIRE(v.size() == 2);
            }

            SECTION("Copy constructor") {
                auto v = Vector<int, 2>{1, 2};
                auto v2 = Vector<int, 2>{v};
                REQUIRE(v2.at(0) == 1);
                REQUIRE(v2.at(1) == 2);
                REQUIRE(v2.size() == 2);
            }

            SECTION("Move constructor") {
                auto v = Vector<int, 2>{1, 2};
                auto v2 = Vector<int, 2>{std::move(v)};
                REQUIRE(v2.at(0) == 1);
                REQUIRE(v2.at(1) == 2);
                REQUIRE(v2.size() == 2);
            }

            SECTION("Copy assignment") {
                auto v = Vector<int, 2>{1, 2};
                auto v2 = Vector<int, 2>{};
                v2 = v;
                REQUIRE(v2.at(0) == 1);
                REQUIRE(v2.at(1) == 2);
                REQUIRE(v2.size() == 2);
            }

            SECTION("Move assignment") {
                auto v = Vector<int, 2>{1, 2};
                auto v2 = Vector<int, 2>{};
                v2 = std::move(v);
                REQUIRE(v2.at(0) == 1);
                REQUIRE(v2.at(1) == 2);
                REQUIRE(v2.size() == 2);
            }

            SECTION("Sanity check") { REQUIRE_THROWS(Vector<int, 0>{}); }
        }

        SECTION("Dynamic") {
            SECTION("Initializer List Constructor") {
                auto v = Vector<int, Dynamic>{1, 2, 3};
                REQUIRE(v.at(0) == 1);
                REQUIRE(v.at(1) == 2);
                REQUIRE(v.at(2) == 3);
                REQUIRE(v.size() == 3);
            }

            SECTION("Copy constructor") {
                auto v = Vector<int, Dynamic>{1, 2};
                auto v2 = Vector<int, Dynamic>{v};
                REQUIRE(v2.at(0) == 1);
                REQUIRE(v2.at(1) == 2);
                REQUIRE(v2.size() == 2);
            }

            SECTION("Move constructor") {
                auto v = Vector<int, Dynamic>{1, 2};
                auto v2 = Vector<int, Dynamic>{std::move(v)};
                REQUIRE(v2.at(0) == 1);
                REQUIRE(v2.at(1) == 2);
                REQUIRE(v2.size() == 2);
            }

            SECTION("Copy assignment") {
                auto v = Vector<int, Dynamic>{1, 2};
                auto v2 = Vector<int, Dynamic>{2};
                v2 = v;
                REQUIRE(v2.at(0) == 1);
                REQUIRE(v2.at(1) == 2);
                REQUIRE(v2.size() == 2);
            }

            SECTION("Move assignment") {
                auto v = Vector<int, Dynamic>{1, 2};
                auto v2 = Vector<int, Dynamic>{2};
                v2 = std::move(v);
                REQUIRE(v2.at(0) == 1);
                REQUIRE(v2.at(1) == 2);
                REQUIRE(v2.size() == 2);
            }

            SECTION("Sanity check") { REQUIRE_THROWS(Vector<int, 0>{}); }
        }
    }

    SECTION("Operations") {
        SECTION("Compile Time") {
            SECTION("Addition") {
                auto v1 = Vector<int, 2>{1, 2};
                auto v2 = Vector<int, 2>{3, 4};
                auto v3 = v1 + v2;
                REQUIRE(v3.at(0) == 4);
                REQUIRE(v3.at(1) == 6);
            }

            SECTION("Subtraction") {
                auto v1 = Vector<int, 2>{1, 2};
                auto v2 = Vector<int, 2>{3, 4};
                auto v3 = v1 - v2;
                REQUIRE(v3.at(0) == -2);
                REQUIRE(v3.at(1) == -2);
            }

            SECTION("Vector Multiplication") {
                auto v1 = Vector<int, 2>{1, 2};
                auto v2 = Vector<int, 2>{3, 4};
                auto v3 = v1 * v2;
                REQUIRE(v3.at(0) == 11);
            }

            // SECTION("Division") {
            //   auto v1 = Vector<int, 2>{1, 2};
            //   auto v2 = Vector<int, 2>{3, 4};
            //   auto v3 = v1 / v2;
            //   REQUIRE(v3.at(0) == 0);
            //   REQUIRE(v3.at(1) == 0);
            // }

            SECTION("Scalar multiplication") {
                auto v1 = Vector<int, 2>{1, 2};
                auto v2 = v1 * 2;
                REQUIRE(v2.at(0) == 2);
                REQUIRE(v2.at(1) == 4);
            }

            SECTION("Scalar division") {
                auto v1 = Vector<int, 2>{1, 2};
                auto v2 = v1 / 2;
                REQUIRE(v2.at(0) == 0);
                REQUIRE(v2.at(1) == 1);
            }

            SECTION("Dot product") {
                auto v1 = Vector<int, 2>{1, 2};
                auto v2 = Vector<int, 2>{3, 4};
                auto dot = v1.dot(v2);
                REQUIRE(dot == 11);
            }

            SECTION("Distance") {
                auto v1 = Vector<int, 2>{1, 2};
                auto v2 = Vector<int, 2>{3, 4};
                auto dist = v1.dist(v2);
                REQUIRE(dist == Approx(2.8284271247461903));
            }

            SECTION("Length") {
                auto v = Vector<int, 2>{1, 2};
                auto len = v.length();
                REQUIRE(len == Approx(2.23606797749979));
            }

            SECTION("Transpose") {
                auto v = Vector<int, 2>{1, 2};
                auto v2 = v.T();
                REQUIRE(v2.at(0).at(0) == 1);
                REQUIRE(v2.at(0).at(1) == 2);
            }

            SECTION("Iteration") {
                auto v = Vector<int, 2>{1, 2};
                const auto *it = v.begin();
                REQUIRE(*it == 1);
                ++it;
                REQUIRE(*it == 2);
                ++it;
                REQUIRE(it == v.end());
            }

            SECTION("Comparisions") {
                SECTION("Equality") {
                    auto v1 = Vector<int, 2>{1, 2};
                    auto v2 = Vector<int, 2>{1, 2};
                    REQUIRE(v1 == v2);
                }

                SECTION("Inequality") {
                    auto v1 = Vector<int, 2>{1, 2};
                    auto v2 = Vector<int, 2>{3, 4};
                    REQUIRE(v1 != v2);
                }

                SECTION("Less than") {
                    auto v1 = Vector<int, 2>{1, 2};
                    auto v2 = Vector<int, 2>{3, 4};
                    REQUIRE(v1 < v2);
                }

                SECTION("Less than or equal") {
                    auto v1 = Vector<int, 2>{1, 2};
                    auto v2 = Vector<int, 2>{1, 2};
                    REQUIRE(v1 <= v2);
                }

                SECTION("Greater than") {
                    auto v1 = Vector<int, 2>{3, 4};
                    auto v2 = Vector<int, 2>{1, 2};
                    REQUIRE(v1 > v2);
                }

                SECTION("Greater than or equal") {
                    auto v1 = Vector<int, 2>{1, 2};
                    auto v2 = Vector<int, 2>{1, 2};
                    REQUIRE(v1 >= v2);
                }
            }
        }

        SECTION("Dynamic") {
            SECTION("Addition") {
                auto v1 = Vector<int, Dynamic>{1, 2};
                auto v2 = Vector<int, Dynamic>{3, 4};
                auto v3 = v1 + v2;
                REQUIRE(v3.at(0) == 4);
                REQUIRE(v3.at(1) == 6);
            }

            SECTION("Subtraction") {
                auto v1 = Vector<int, Dynamic>{1, 2};
                auto v2 = Vector<int, Dynamic>{3, 4};
                auto v3 = v1 - v2;
                REQUIRE(v3.at(0) == -2);
                REQUIRE(v3.at(1) == -2);
            }

            SECTION("Vector Multiplication") {
                auto v1 = Vector<int, Dynamic>{1, 2};
                auto v2 = Vector<int, Dynamic>{3, 4};
                auto v3 = v1 * v2;
                REQUIRE(v3.at(0) == 11);
            }

            // SECTION("Division") {
            //   auto v1 = Vector<int, Dynamic>{1, 2};
            //   auto v2 = Vector<int, Dynamic>{3, 4};
            //   auto v3 = v1 / v2;
            //   REQUIRE(v3.at(0) == 0);
            //   REQUIRE(v3.at(1) == 0);
            // }

            SECTION("Scalar multiplication") {
                auto v1 = Vector<int, Dynamic>{1, 2};
                auto v2 = v1 * 2;
                REQUIRE(v2.at(0) == 2);
                REQUIRE(v2.at(1) == 4);
            }

            SECTION("Scalar division") {
                auto v1 = Vector<int, Dynamic>{1, 2};
                auto v2 = v1 / 2;
                REQUIRE(v2.at(0) == 0);
                REQUIRE(v2.at(1) == 1);
            }

            SECTION("Dot product") {
                auto v1 = Vector<int, Dynamic>{1, 2};
                auto v2 = Vector<int, Dynamic>{3, 4};
                auto dot = v1.dot(v2);
                REQUIRE(dot == 11);
            }

            SECTION("Distance") {
                auto v1 = Vector<int, Dynamic>{1, 2};
                auto v2 = Vector<int, Dynamic>{3, 4};
                auto dist = v1.dist(v2);
                REQUIRE(dist == Approx(2.8284271247461903));
            }

            SECTION("Length") {
                auto v = Vector<int, Dynamic>{1, 2};
                auto len = v.length();
                REQUIRE(len == Approx(2.23606797749979));
            }

            SECTION("Transpose") {
                auto v = Vector<int, Dynamic>{1, 2};
                auto v2 = v.T();
                REQUIRE(v2.at(0).at(0) == 1);
                REQUIRE(v2.at(0).at(1) == 2);
            }

            SECTION("Iteration") {
                auto v = Vector<int, Dynamic>{1, 2};
                auto it = v.cbegin();
                REQUIRE(*it == 1);
                ++it;
                REQUIRE(*it == 2);
                ++it;
                REQUIRE(it == v.end());
            }

            SECTION("Comparisions") {
                SECTION("Equality") {
                    auto v1 = Vector<int, Dynamic>{1, 2};
                    auto v2 = Vector<int, Dynamic>{1, 2};
                    REQUIRE(v1 == v2);
                }

                SECTION("Inequality") {
                    auto v1 = Vector<int, Dynamic>{1, 2};
                    auto v2 = Vector<int, Dynamic>{3, 4};
                    REQUIRE(v1 != v2);
                }

                SECTION("Less than") {
                    auto v1 = Vector<int, Dynamic>{1, 2};
                    auto v2 = Vector<int, Dynamic>{3, 4};
                    REQUIRE(v1 < v2);
                }

                SECTION("Less than or equal") {
                    auto v1 = Vector<int, Dynamic>{1, 2};
                    auto v2 = Vector<int, Dynamic>{1, 2};
                    REQUIRE(v1 <= v2);
                }

                SECTION("Greater than") {
                    auto v1 = Vector<int, Dynamic>{3, 4};
                    auto v2 = Vector<int, Dynamic>{1, 2};
                    REQUIRE(v1 > v2);
                }

                SECTION("Greater than or equal") {
                    auto v1 = Vector<int, Dynamic>{1, 2};
                    auto v2 = Vector<int, Dynamic>{1, 2};
                    REQUIRE(v1 >= v2);
                }
            }
        }
    }

    SECTION("Robustness") {
        SECTION("Compile Time") {
            SECTION("Indexing") {
                SECTION("Out-of-bounds Access") {
                    auto v = Vector<int, 2>{1, 2};
                    REQUIRE_THROWS_AS(v.at(2), std::out_of_range);
                    REQUIRE_THROWS_AS(v.at(-1), std::out_of_range);
                }

                SECTION("Out-of-bounds Modification") {
                    auto v = Vector<int, 2>{1, 2};
                    REQUIRE_THROWS_AS(v.at(2) = 3, std::out_of_range);
                    REQUIRE_THROWS_AS(v.at(-1) = 3, std::out_of_range);
                }
            }

            SECTION("Division by zero") {
                auto v = Vector<int, 2>{1, 2};
                REQUIRE_THROWS_AS(
                    v / 0,
                    std::domain_error);  // Ensure you throw std::domain_error in case of division by zero.
            }

            SECTION("Zero-length Vector") {
                auto v = Vector<int, 2>{0, 0};
                auto len = v.length();
                REQUIRE(len == 0);
            }

            SECTION("Large number operations") {
                auto large_value = std::numeric_limits<int>::max();
                auto v1 = Vector<int, 2>{large_value, large_value};
                auto v2 = v1 * 2;
                REQUIRE(v2.at(0) == large_value * 2);
                REQUIRE(v2.at(1) == large_value * 2);
            }

            SECTION("Small number operations") {
                auto small_value = std::numeric_limits<int>::min();
                auto v1 = Vector<int, 2>{small_value, small_value};
                auto v2 = v1 * 2;
                REQUIRE(v2.at(0) == small_value * 2);
                REQUIRE(v2.at(1) == small_value * 2);
            }

            SECTION("Floating point precision for length and distance") {
                auto v1 = Vector<double, 2>{1e-10, 1e-10};
                auto v2 = Vector<double, 2>{2e-10, 2e-10};
                auto len = v1.length();
                auto dist = v1.dist(v2);

                REQUIRE(len == Approx(1.41421356237e-10));
                REQUIRE(dist == Approx(1.41421356237e-10));
            }

            SECTION("Const and non-const iterators") {
                auto v = Vector<int, 2>{1, 2};

                // Non-const iterator
                const auto *it = v.begin();
                REQUIRE(*it == 1);
                ++it;
                REQUIRE(*it == 2);

                // Const iterator
                const auto &v_const = v;
                const auto *const_it = v_const.begin();
                REQUIRE(*const_it == 1);
                ++const_it;
                REQUIRE(*const_it == 2);
            }

            SECTION("Large vector operations performance") {
                const size_t large_size = 1e6;
                auto v1 = Vector<size_t, large_size>{};
                auto v2 = Vector<size_t, large_size>{};

                for (size_t i = 0; i < large_size; ++i) {
                    v1.at(i) = i;
                    v2.at(i) = i;
                }

                auto start = std::chrono::high_resolution_clock::now();
                auto v3 = v1 + v2;
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

                REQUIRE(v3.at(0) == 0);
                REQUIRE(v3.at(large_size - 1) == (large_size - 1) * 2);
                REQUIRE(duration < 100);  // Ensure it runs within 100 ms

                // BENCHMARK("Large vector addition performance") { v1 + v2; };
            }
        }

        SECTION("Dynamic") {
            SECTION("Indexing") {
                SECTION("Out-of-bounds Access") {
                    auto v = Vector<int, Dynamic>{1, 2};
                    REQUIRE_THROWS_AS(v.at(2), std::out_of_range);
                    REQUIRE_THROWS_AS(v.at(-1), std::out_of_range);
                }

                SECTION("Out-of-bounds Modification") {
                    auto v = Vector<int, Dynamic>{1, 2};
                    REQUIRE_THROWS_AS(v.at(2) = 3, std::out_of_range);
                    REQUIRE_THROWS_AS(v.at(-1) = 3, std::out_of_range);
                }
            }

            SECTION("Division by zero") {
                auto v = Vector<int, Dynamic>{1, 2};
                REQUIRE_THROWS_AS(
                    v / 0,
                    std::domain_error);  // Ensure you throw std::domain_error in case of division by zero.
            }

            SECTION("Zero-length Vector") {
                auto v = Vector<int, Dynamic>{0, 0};
                auto len = v.length();
                REQUIRE(len == 0);
            }

            SECTION("Large number operations") {
                auto large_value = std::numeric_limits<int>::max();
                auto v1 = Vector<int, Dynamic>{large_value, large_value};
                auto v2 = v1 * 2;
                REQUIRE(v2.at(0) == large_value * 2);
                REQUIRE(v2.at(1) == large_value * 2);
            }

            SECTION("Small number operations") {
                auto small_value = std::numeric_limits<int>::min();
                auto v1 = Vector<int, Dynamic>{small_value, small_value};
                auto v2 = v1 * 2;
                REQUIRE(v2.at(0) == small_value * 2);
                REQUIRE(v2.at(1) == small_value * 2);
            }

            SECTION("Floating point precision for length and distance") {
                auto v1 = Vector<double, 2>{1e-10, 1e-10};
                auto v2 = Vector<double, 2>{2e-10, 2e-10};
                auto len = v1.length();
                auto dist = v1.dist(v2);

                REQUIRE(len == Approx(1.41421356237e-10));
                REQUIRE(dist == Approx(1.41421356237e-10));
            }

            SECTION("Const and non-const iterators") {
                auto v = Vector<int, Dynamic>{1, 2};

                // Non-const iterator
                auto it = v.begin();
                REQUIRE(*it == 1);
                ++it;
                REQUIRE(*it == 2);

                // Const iterator
                const auto &v_const = v;
                auto const_it = v_const.cbegin();
                REQUIRE(*const_it == 1);
                ++const_it;
                REQUIRE(*const_it == 2);
            }

            SECTION("Large vector operations performance") {
                const size_t large_size = 1e6;
                auto v1 = Vector<size_t, large_size>{};
                auto v2 = Vector<size_t, large_size>{};

                for (size_t i = 0; i < large_size; ++i) {
                    v1.at(i) = i;
                    v2.at(i) = i;
                }

                auto start = std::chrono::high_resolution_clock::now();
                auto v3 = v1 + v2;
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

                REQUIRE(v3.at(0) == 0);
                REQUIRE(v3.at(large_size - 1) == (large_size - 1) * 2);
                REQUIRE(duration < 100);  // Ensure it runs within 100 ms

                // BENCHMARK("Large vector addition performance") { v1 + v2; };
            }
        }
    }
}

TEST_CASE("Matrix", "[matrix]") {
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
            }

            SECTION("Subtraction") {
                auto m1 = Matrix<int, 2, 2>{{1, 2}, {3, 4}};
                auto m2 = Matrix<int, 2, 2>{{5, 6}, {7, 8}};
                auto m3 = m1 - m2;
                REQUIRE(m3.at(0, 0) == -4);
                REQUIRE(m3.at(0, 1) == -4);
                REQUIRE(m3.at(1, 0) == -4);
                REQUIRE(m3.at(1, 1) == -4);
            }

            SECTION("Matrix Multiplication") {
                auto m1 = Matrix<int, 2, 2>{{1, 2}, {3, 4}};
                auto m2 = Matrix<int, 2, 2>{{5, 6}, {7, 8}};
                auto m3 = m1 * m2;
                REQUIRE(m3.at(0, 0) == 19);
                REQUIRE(m3.at(0, 1) == 22);
                REQUIRE(m3.at(1, 0) == 43);
                REQUIRE(m3.at(1, 1) == 50);
            }

            SECTION("Scalar multiplication") {
                auto m1 = Matrix<int, 2, 2>{{1, 2}, {3, 4}};
                auto m2 = m1 * 2;
                REQUIRE(m2.at(0, 0) == 2);
                REQUIRE(m2.at(0, 1) == 4);
                REQUIRE(m2.at(1, 0) == 6);
                REQUIRE(m2.at(1, 1) == 8);
            }

            SECTION("Scalar division") {
                auto m1 = Matrix<int, 2, 2>{{1, 2}, {3, 4}};
                auto m2 = m1 / 2;
                REQUIRE(m2.at(0, 0) == 0);
                REQUIRE(m2.at(0, 1) == 1);
                REQUIRE(m2.at(1, 0) == 1);
                REQUIRE(m2.at(1, 1) == 2);
            }

            SECTION("Transpose") {
                auto m1 = Matrix<int, 2, 2>{{1, 2}, {3, 4}};
                auto m2 = extractMatrixFromTranspose(m1.T());
                REQUIRE(m2.at(0, 0) == 1);
                REQUIRE(m2.at(0, 1) == 3);
                REQUIRE(m2.at(1, 0) == 2);
                REQUIRE(m2.at(1, 1) == 4);
            }

            SECTION("Matrix Vector Multiplication") {
                auto m = Matrix<int, 2, 2>{{1, 2}, {3, 4}};
                auto v = Vector<int, 2>{1, 2};
                auto v2 = m * v;
                REQUIRE(v2.at(0) == 5);
                REQUIRE(v2.at(1) == 11);
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
                SECTION("Equality") {
                    auto m1 = Matrix<int, 2, 2>{{1, 2}, {3, 4}};
                    auto m2 = Matrix<int, 2, 2>{{1, 2}, {3, 4}};
                    REQUIRE(m1 == m2);
                }

                SECTION("Inequality") {
                    auto m1 = Matrix<int, 2, 2>{{1, 2}, {3, 4}};
                    auto m2 = Matrix<int, 2, 2>{{5, 6}, {7, 8}};
                    REQUIRE(m1 != m2);
                }

                SECTION("Less than") {
                    auto m1 = Matrix<int, 2, 2>{{1, 2}, {3, 4}};
                    auto m2 = Matrix<int, 2, 2>{{5, 6}, {7, 8}};
                    REQUIRE(m1 < m2);
                }

                SECTION("Less than or equal") {
                    auto m1 = Matrix<int, 2, 2>{{1, 2}, {3, 4}};
                    auto m2 = Matrix<int, 2, 2>{{1, 2}, {3, 4}};
                    REQUIRE(m1 <= m2);
                }

                SECTION("Greater than") {
                    auto m1 = Matrix<int, 2, 2>{{5, 6}, {7, 8}};
                    auto m2 = Matrix<int, 2, 2>{{1, 2}, {3, 4}};
                    REQUIRE(m1 > m2);
                }

                SECTION("Greater than or equal") {
                    auto m1 = Matrix<int, 2, 2>{{1, 2}, {3, 4}};
                    auto m2 = Matrix<int, 2, 2>{{1, 2}, {3, 4}};
                    REQUIRE(m1 >= m2);
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
            }

            SECTION("Subtraction") {
                auto m1 = Matrix<int, Dynamic, Dynamic>{{1, 2}, {3, 4}};
                auto m2 = Matrix<int, Dynamic, Dynamic>{{5, 6}, {7, 8}};
                auto m3 = m1 - m2;
                REQUIRE(m3.at(0, 0) == -4);
                REQUIRE(m3.at(0, 1) == -4);
                REQUIRE(m3.at(1, 0) == -4);
                REQUIRE(m3.at(1, 1) == -4);
            }

            SECTION("Matrix Multiplication") {
                auto m1 = Matrix<int, Dynamic, Dynamic>{{1, 2}, {3, 4}};
                auto m2 = Matrix<int, Dynamic, Dynamic>{{5, 6}, {7, 8}};
                auto m3 = m1 * m2;
                REQUIRE(m3.at(0, 0) == 19);
                REQUIRE(m3.at(0, 1) == 22);
                REQUIRE(m3.at(1, 0) == 43);
                REQUIRE(m3.at(1, 1) == 50);
            }

            SECTION("Scalar multiplication") {
                auto m1 = Matrix<int, Dynamic, Dynamic>{{1, 2}, {3, 4}};
                auto m2 = m1 * 2;
                REQUIRE(m2.at(0, 0) == 2);
                REQUIRE(m2.at(0, 1) == 4);
                REQUIRE(m2.at(1, 0) == 6);
                REQUIRE(m2.at(1, 1) == 8);
            }

            SECTION("Scalar division") {
                auto m1 = Matrix<int, Dynamic, Dynamic>{{1, 2}, {3, 4}};
                auto m2 = m1 / 2;
                REQUIRE(m2.at(0, 0) == 0);
                REQUIRE(m2.at(0, 1) == 1);
                REQUIRE(m2.at(1, 0) == 1);
                REQUIRE(m2.at(1, 1) == 2);
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

                auto m2 = Matrix<int, 3, 3>{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
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
                SECTION("Equality") {
                    auto m1 = Matrix<int, Dynamic, Dynamic>{{1, 2}, {3, 4}};
                    auto m2 = Matrix<int, Dynamic, Dynamic>{{1, 2}, {3, 4}};
                    REQUIRE(m1 == m2);
                }

                SECTION("Inequality") {
                    auto m1 = Matrix<int, Dynamic, Dynamic>{{1, 2}, {3, 4}};
                    auto m2 = Matrix<int, Dynamic, Dynamic>{{5, 6}, {7, 8}};
                    REQUIRE(m1 != m2);
                }

                SECTION("Less than") {
                    auto m1 = Matrix<int, Dynamic, Dynamic>{{1, 2}, {3, 4}};
                    auto m2 = Matrix<int, Dynamic, Dynamic>{{5, 6}, {7, 8}};
                    REQUIRE(m1 < m2);
                }

                SECTION("Less than or equal") {
                    auto m1 = Matrix<int, Dynamic, Dynamic>{{1, 2}, {3, 4}};
                    auto m2 = Matrix<int, Dynamic, Dynamic>{{1, 2}, {3, 4}};
                    REQUIRE(m1 <= m2);
                }

                SECTION("Greater than") {
                    auto m1 = Matrix<int, Dynamic, Dynamic>{{5, 6}, {7, 8}};
                    auto m2 = Matrix<int, Dynamic, Dynamic>{{1, 2}, {3, 4}};
                    REQUIRE(m1 > m2);
                }

                SECTION("Greater than or equal") {
                    auto m1 = Matrix<int, Dynamic, Dynamic>{{1, 2}, {3, 4}};
                    auto m2 = Matrix<int, Dynamic, Dynamic>{{1, 2}, {3, 4}};
                    REQUIRE(m1 >= m2);
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
