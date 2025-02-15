#include <MLinalg.hpp>
#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>

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

                auto u1 = Vector<int, 1>{1};
                auto u2 = Vector<int, 1>{2};
                auto u3 = u1 + u2;
                REQUIRE(u3.at(0) == 3);
            }

            SECTION("Subtraction") {
                auto v1 = Vector<int, 2>{1, 2};
                auto v2 = Vector<int, 2>{3, 4};
                auto v3 = v1 - v2;
                REQUIRE(v3.at(0) == -2);
                REQUIRE(v3.at(1) == -2);

                auto u1 = Vector<int, 1>{1};
                auto u2 = Vector<int, 1>{2};
                auto u3 = u1 - u2;
                REQUIRE(u3.at(0) == -1);
            }

            SECTION("Vector Multiplication") {
                auto v1 = Vector<int, 2>{1, 2};
                auto v2 = Vector<int, 2>{3, 4};
                auto v3 = v1 * v2;
                REQUIRE(v3.at(0) == 11);

                auto u1 = Vector<int, 1>{1};
                auto u2 = Vector<int, 1>{2};
                auto u3 = u1 * u2;
                REQUIRE(u3.at(0) == 2);
            }

            SECTION("Scalar multiplication") {
                auto v1 = Vector<int, 2>{1, 2};
                auto v2 = v1 * 2;
                REQUIRE(v2.at(0) == 2);
                REQUIRE(v2.at(1) == 4);

                auto u1 = Vector<int, 1>{1};
                auto u2 = u1 * 2;
                REQUIRE(u2.at(0) == 2);
            }

            SECTION("Scalar division") {
                auto v1 = Vector<int, 2>{1, 2};
                auto v2 = v1 / 2;
                REQUIRE(v2.at(0) == 0);
                REQUIRE(v2.at(1) == 1);

                auto u1 = Vector<int, 1>{1};
                auto u2 = u1 / 2;
                REQUIRE(u2.at(0) == 0);
            }

            SECTION("Dot product") {
                auto v1 = Vector<int, 2>{1, 2};
                auto v2 = Vector<int, 2>{3, 4};
                auto dot = v1.dot(v2);
                REQUIRE(dot == 11);

                auto u1 = Vector<int, 1>{1};
                auto u2 = Vector<int, 1>{2};
                auto dot2 = u1.dot(u2);
                REQUIRE(dot2 == 2);
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

                auto u = Vector<int, 1>{1};
                auto u2 = u.T();
                REQUIRE(u2.at(0).at(0) == 1);
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
                // ------------------ Integer 32 (int) Tests ------------------
                SECTION("Integer - Equality") {
                    Vector<int, 2> v1{1, 2};
                    Vector<int, 2> v2{1, 2};
                    REQUIRE(v1 == v2);
                }

                SECTION("Integer - Inequality") {
                    Vector<int, 2> v1{1, 2};
                    Vector<int, 2> v2{3, 4};
                    REQUIRE(v1 != v2);
                }

                // ------------------ Single Precision (float) Tests ------------------
                SECTION("Float - Exact Equality") {
                    Vector<float, 2> v1{1.0F, 2.0F};
                    Vector<float, 2> v2{1.0F, 2.0F};
                    REQUIRE(v1 == v2);
                }

                SECTION("Float - Approximate Equality Within Tolerance") {
                    Vector<float, 2> v1{1.000001F, 2.000001F};
                    Vector<float, 2> v2{1.0F, 2.0F};
                    REQUIRE(v1 == v2);
                }

                SECTION("Float - Inequality Beyond Tolerance") {
                    Vector<float, 2> v1{1.0F, 2.0F};
                    Vector<float, 2> v2{1.0F, 2.1F};
                    REQUIRE(v1 != v2);
                }

                SECTION("Float - Infinity Comparison") {
                    Vector<float, 2> v1{std::numeric_limits<float>::infinity(), 2.0F};
                    Vector<float, 2> v2{std::numeric_limits<float>::infinity(), 2.0F};
                    REQUIRE(v1 == v2);
                }

                SECTION("Float - NaN Comparison") {
                    Vector<float, 2> v1{std::numeric_limits<float>::quiet_NaN(), 2.0F};
                    Vector<float, 2> v2{std::numeric_limits<float>::quiet_NaN(), 2.0F};
                    REQUIRE(v1 != v2);
                }

                // ------------------ Double Precision (double) Tests ------------------
                SECTION("Double - Exact Equality") {
                    Vector<double, 2> v1{1.0, 2.0};
                    Vector<double, 2> v2{1.0, 2.0};
                    REQUIRE(v1 == v2);
                }

                SECTION("Double - Approximate Equality Within Tolerance") {
                    Vector<double, 2> v1{1.000000001, 2.000000001};
                    Vector<double, 2> v2{1.0, 2.0};
                    REQUIRE(v1 == v2);
                }

                SECTION("Double - Inequality Beyond Tolerance") {
                    Vector<double, 2> v1{1.0, 2.0};
                    Vector<double, 2> v2{1.0, 2.0001};
                    REQUIRE(v1 != v2);
                }

                SECTION("Double - Infinity Comparison") {
                    Vector<double, 2> v1{std::numeric_limits<double>::infinity(), 2.0};
                    Vector<double, 2> v2{std::numeric_limits<double>::infinity(), 2.0};
                    REQUIRE(v1 == v2);
                }

                SECTION("Double - NaN Comparison") {
                    Vector<double, 2> v1{std::numeric_limits<double>::quiet_NaN(), 2.0};
                    Vector<double, 2> v2{std::numeric_limits<double>::quiet_NaN(), 2.0};
                    REQUIRE(v1 != v2);
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

                auto u1 = Vector<int, Dynamic>{1};
                auto u2 = Vector<int, Dynamic>{2};
                auto u3 = u1 + u2;
                REQUIRE(u3.at(0) == 3);
            }

            SECTION("Subtraction") {
                auto v1 = Vector<int, Dynamic>{1, 2};
                auto v2 = Vector<int, Dynamic>{3, 4};
                auto v3 = v1 - v2;
                REQUIRE(v3.at(0) == -2);
                REQUIRE(v3.at(1) == -2);

                auto u1 = Vector<int, Dynamic>{1};
                auto u2 = Vector<int, Dynamic>{2};
                auto u3 = u1 - u2;
                REQUIRE(u3.at(0) == -1);
            }

            SECTION("Vector Multiplication") {
                auto v1 = Vector<int, Dynamic>{1, 2};
                auto v2 = Vector<int, Dynamic>{3, 4};
                auto v3 = v1 * v2;
                REQUIRE(v3.at(0) == 11);

                auto u1 = Vector<int, Dynamic>{1};
                auto u2 = Vector<int, Dynamic>{2};
                auto u3 = u1 * u2;
            }

            SECTION("Scalar multiplication") {
                auto v1 = Vector<int, Dynamic>{1, 2};
                auto v2 = v1 * 2;
                REQUIRE(v2.at(0) == 2);
                REQUIRE(v2.at(1) == 4);

                auto u1 = Vector<int, Dynamic>{1};
                auto u2 = u1 * 2;
                REQUIRE(u2.at(0) == 2);
            }

            SECTION("Scalar division") {
                auto v1 = Vector<int, Dynamic>{1, 2};
                auto v2 = v1 / 2;
                REQUIRE(v2.at(0) == 0);
                REQUIRE(v2.at(1) == 1);

                auto u1 = Vector<int, Dynamic>{1};
                auto u2 = u1 / 2;
                REQUIRE(u2.at(0) == 0);
            }

            SECTION("Dot product") {
                auto v1 = Vector<int, Dynamic>{1, 2};
                auto v2 = Vector<int, Dynamic>{3, 4};
                auto dot = v1.dot(v2);
                REQUIRE(dot == 11);

                auto u1 = Vector<int, Dynamic>{1};
                auto u2 = Vector<int, Dynamic>{2};
                auto dot2 = u1.dot(u2);
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

                auto u = Vector<int, Dynamic>{1};
                auto u2 = u.T();
                REQUIRE(u2.at(0).at(0) == 1);
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
                // ------------------ Integer 32 (int) Tests ------------------
                SECTION("Integer - Equality") {
                    Vector<int, Dynamic> v1{1, 2};
                    Vector<int, Dynamic> v2{1, 2};
                    REQUIRE(v1 == v2);
                }

                SECTION("Integer - Inequality") {
                    Vector<int, Dynamic> v1{1, 2};
                    Vector<int, Dynamic> v2{3, 4};
                    REQUIRE(v1 != v2);
                }

                // ------------------ Single Precision (float) Tests ------------------
                SECTION("Float - Exact Equality") {
                    Vector<float, Dynamic> v1{1.0F, 2.0F};
                    Vector<float, Dynamic> v2{1.0F, 2.0F};
                    REQUIRE(v1 == v2);
                }

                SECTION("Float - Approximate Equality Within Tolerance") {
                    Vector<float, Dynamic> v1{1.000001F, 2.000001F};
                    Vector<float, Dynamic> v2{1.0F, 2.0F};
                    REQUIRE(v1 == v2);
                }

                SECTION("Float - Inequality Beyond Tolerance") {
                    Vector<float, Dynamic> v1{1.0F, 2.0F};
                    Vector<float, Dynamic> v2{1.0F, 2.1F};
                    REQUIRE(v1 != v2);
                }

                SECTION("Float - Infinity Comparison") {
                    Vector<float, Dynamic> v1{std::numeric_limits<float>::infinity(), 2.0F};
                    Vector<float, Dynamic> v2{std::numeric_limits<float>::infinity(), 2.0F};
                    REQUIRE(v1 == v2);
                }

                SECTION("Float - NaN Comparison") {
                    Vector<float, Dynamic> v1{std::numeric_limits<float>::quiet_NaN(), 2.0F};
                    Vector<float, Dynamic> v2{std::numeric_limits<float>::quiet_NaN(), 2.0F};
                    REQUIRE(v1 != v2);
                }

                // ------------------ Double Precision (double) Tests ------------------
                SECTION("Double - Exact Equality") {
                    Vector<double, Dynamic> v1{1.0, 2.0};
                    Vector<double, Dynamic> v2{1.0, 2.0};
                    REQUIRE(v1 == v2);
                }

                SECTION("Double - Approximate Equality Within Tolerance") {
                    Vector<double, Dynamic> v1{1.000000001, 2.000000001};
                    Vector<double, Dynamic> v2{1.0, 2.0};
                    REQUIRE(v1 == v2);
                }

                SECTION("Double - Inequality Beyond Tolerance") {
                    Vector<double, Dynamic> v1{1.0, 2.0};
                    Vector<double, Dynamic> v2{1.0, 2.0001};
                    REQUIRE(v1 != v2);
                }

                SECTION("Double - Infinity Comparison") {
                    Vector<double, Dynamic> v1{std::numeric_limits<double>::infinity(), 2.0};
                    Vector<double, Dynamic> v2{std::numeric_limits<double>::infinity(), 2.0};
                    REQUIRE(v1 == v2);
                }

                SECTION("Double - NaN Comparison") {
                    Vector<double, Dynamic> v1{std::numeric_limits<double>::quiet_NaN(), 2.0};
                    Vector<double, Dynamic> v2{std::numeric_limits<double>::quiet_NaN(), 2.0};
                    REQUIRE(v1 != v2);
                }
            }
        }

        SECTION("Cross") {
            SECTION("Addition") {
                auto v1 = Vector<int, Dynamic>{1, 2};
                auto v2 = Vector<int, 2>{3, 4};
                auto v3 = v1 + v2;
                auto v4 = v2 + v1;
                REQUIRE(v3.at(0) == 4);
                REQUIRE(v3.at(1) == 6);
                REQUIRE(v4.at(0) == 4);
                REQUIRE(v4.at(1) == 6);
            }

            SECTION("Subtraction") {
                auto v1 = Vector<int, Dynamic>{1, 2};
                auto v2 = Vector<int, 2>{3, 4};
                auto v3 = v1 - v2;
                auto v4 = v2 - v1;
                REQUIRE(v3.at(0) == -2);
                REQUIRE(v3.at(1) == -2);
                REQUIRE(v4.at(0) == 2);
                REQUIRE(v4.at(1) == 2);
            }

            SECTION("Vector Multiplication") {
                auto v1 = Vector<int, Dynamic>{1, 2};
                auto v2 = Vector<int, 2>{3, 4};
                auto v3 = v1 * v2;
                REQUIRE(v3.at(0) == 11);
            }

            SECTION("Dot product") {
                auto v1 = Vector<int, Dynamic>{1, 2};
                auto v2 = Vector<int, 2>{3, 4};
                auto dot = v1.dot(v2);
                REQUIRE(dot == 11);
            }

            SECTION("Distance") {
                auto v1 = Vector<int, Dynamic>{1, 2};
                auto v2 = Vector<int, 2>{3, 4};
                auto dist = v1.dist(v2);
                REQUIRE(dist == Approx(2.8284271247461903));
            }

            SECTION("Comparisions") {
                SECTION("Equality") {
                    auto v1 = Vector<int, Dynamic>{1, 2};
                    auto v2 = Vector<int, 2>{1, 2};
                    REQUIRE(v1 == v2);
                    REQUIRE(v2 == v1);
                }

                SECTION("Inequality") {
                    auto v1 = Vector<int, Dynamic>{1, 2};
                    auto v2 = Vector<int, 2>{3, 4};
                    REQUIRE(v1 != v2);
                    REQUIRE(v2 != v1);
                }
                //
                //     SECTION("Less than") {
                //         auto v1 = Vector<int, Dynamic>{1, 2};
                //         auto v2 = Vector<int, 2>{3, 4};
                //         REQUIRE(v1 < v2);
                //         REQUIRE(v2 > v1);
                //     }
                //
                //     SECTION("Less than or equal") {
                //         auto v1 = Vector<int, Dynamic>{1, 2};
                //         auto v2 = Vector<int, 2>{1, 2};
                //         REQUIRE(v1 <= v2);
                //         REQUIRE(v2 >= v1);
                //     }
                //
                //     SECTION("Greater than") {
                //         auto v1 = Vector<int, Dynamic>{3, 4};
                //         auto v2 = Vector<int, 2>{1, 2};
                //         REQUIRE(v1 > v2);
                //         REQUIRE(v2 < v1);
                //     }
                //
                //     SECTION("Greater than or equal") {
                //         auto v1 = Vector<int, Dynamic>{1, 2};
                //         auto v2 = Vector<int, 2>{1, 2};
                //         REQUIRE(v1 >= v2);
                //         REQUIRE(v2 <= v1);
                //     }
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
                // Test case 1: Multiply two 2x2 matrices
                {
                    auto m1 = Matrix<int, 2, 2>{{1, 2}, {3, 4}};
                    auto m2 = Matrix<int, 2, 2>{{5, 6}, {7, 8}};
                    auto m3 = m1 * m2;

                    // Expected result
                    Matrix<int, 2, 2> expected{{19, 22}, {43, 50}};

                    // Verify the result
                    for (int i = 0; i < m3.numRows(); i++) {
                        for (int j = 0; j < m3.numCols(); j++) {
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
                    for (int i = 0; i < m1.numRows(); i++) {
                        for (int j = 0; j < m1.numCols(); j++) {
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
                    for (int i = 0; i < m2.numRows(); i++) {
                        for (int j = 0; j < m2.numCols(); j++) {
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
                    for (int i = 0; i < m3.numRows(); i++) {
                        for (int j = 0; j < m3.numCols(); j++) {
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
                    for (int i = 0; i < m3.numRows(); i++) {
                        for (int j = 0; j < m3.numCols(); j++) {
                            REQUIRE(m3.at(i, j) == expected.at(i, j));
                        }
                    }
                }
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

            // SECTION("Iteration") {
            //     auto m = Matrix<int, 2, 2>{{1, 2}, {3, 4}};
            //     const auto *it = m.begin();
            //     REQUIRE(*it == Vector<int, 2>{1, 2});
            //     REQUIRE(it != m.end());
            //     REQUIRE(*(++it) == Vector<int, 2>{3, 4});
            //     REQUIRE(++it == m.end());
            // }

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
                // Test case 1: Multiply two 2x2 matrices
                {
                    auto m1 = Matrix<int, Dynamic, Dynamic>{{1, 2}, {3, 4}};
                    auto m2 = Matrix<int, Dynamic, Dynamic>{{5, 6}, {7, 8}};
                    auto m3 = m1 * m2;

                    // Expected result
                    Matrix<int, Dynamic, Dynamic> expected{{19, 22}, {43, 50}};

                    // Verify the result
                    for (int i = 0; i < m3.numRows(); i++) {
                        for (int j = 0; j < m3.numCols(); j++) {
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
                    for (int i = 0; i < m1.numRows(); i++) {
                        for (int j = 0; j < m1.numCols(); j++) {
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
                    for (int i = 0; i < m2.numRows(); i++) {
                        for (int j = 0; j < m2.numCols(); j++) {
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
                    for (int i = 0; i < m3.numRows(); i++) {
                        for (int j = 0; j < m3.numCols(); j++) {
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
                    for (int i = 0; i < m3.numRows(); i++) {
                        for (int j = 0; j < m3.numCols(); j++) {
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

                    bool correct{true};
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
