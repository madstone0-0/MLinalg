
#include "structures/Vector.hpp"

#include <MLinalg.hpp>
#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>

#include "Helpers.hpp"
#include "Numeric.hpp"
#include "structures/Aliases.hpp"

using namespace Catch;
using namespace mlinalg;
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

            SECTION("Sanity check") {
                REQUIRE_THROWS(Vector<int, 0>{});
                REQUIRE_THROWS(Vector<int, 3>{1});
            }
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

                // Inplace Addition
                auto v4 = Vector<int, 2>{1, 2};
                auto v5 = Vector<int, 2>{3, 4};
                v4 += v5;
                REQUIRE(v4.at(0) == 4);
                REQUIRE(v4.at(1) == 6);
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

                // Inplace subtraction
                auto v4 = Vector<int, 2>{1, 2};
                auto v5 = Vector<int, 2>{3, 4};
                v4 -= v5;
                REQUIRE(v4.at(0) == -2);
                REQUIRE(v4.at(1) == -2);
            }

            SECTION("Negation") {
                auto v = Vector<int, 2>{1, 2};
                auto v2 = -v;
                REQUIRE(v2.at(0) == -1);
                REQUIRE(v2.at(1) == -2);

                auto u = Vector<int, 1>{1};
                auto u2 = -u;
                REQUIRE(u2.at(0) == -1);
            }

            SECTION("Vector Multiplication") {
                auto v1 = Vector<int, 2>{1, 2};
                auto v2 = Vector<int, 2>{3, 4};
                auto dot1 = v1 * v2;
                REQUIRE(dot1 == 11);

                auto u1 = Vector<int, 1>{1};
                auto u2 = Vector<int, 1>{2};
                auto dot2 = u1 * u2;
                REQUIRE(dot2 == 2);

                auto w1 = Vector<int, 2>{1, 1};
                auto m = w1 * w1.T();
                REQUIRE(m.at(0, 0) == 1);
                REQUIRE(m.at(0, 1) == 1);
                REQUIRE(m.at(1, 0) == 1);
                REQUIRE(m.at(1, 1) == 1);
            }

            SECTION("Matrix Multiplication") {
                auto v1 = Vector<int, 2>{1, 2};
                auto m = Matrix<int, 2, 2>{{1, 2}, {3, 4}};
                auto v2 = v1 * m;
                REQUIRE(v2.at(0) == 7);
                REQUIRE(v2.at(1) == 10);

                auto u1 = Vector<int, 1>{1};
                auto m2 = Matrix<int, 1, 1>{{2}};
                auto u2 = u1 * m2;
                REQUIRE(u2.at(0) == 2);

                auto v3 = m * v1;
                REQUIRE(v3.at(0) == 5);
                REQUIRE(v3.at(1) == 11);
            }

            SECTION("Scalar multiplication") {
                auto v1 = Vector<int, 2>{1, 2};
                auto v2 = v1 * 2;
                REQUIRE(v2.at(0) == 2);
                REQUIRE(v2.at(1) == 4);

                auto u1 = Vector<int, 1>{1};
                auto u2 = u1 * 2;
                REQUIRE(u2.at(0) == 2);

                // Inplace multiplication
                auto v3 = Vector<int, 2>{1, 2};
                v3 *= 2;
                REQUIRE(v3.at(0) == 2);
                REQUIRE(v3.at(1) == 4);
            }

            SECTION("Scalar division") {
                auto v1 = Vector<int, 2>{1, 2};
                auto v2 = v1 / 2;
                REQUIRE(v2.at(0) == 0);
                REQUIRE(v2.at(1) == 1);

                auto u1 = Vector<int, 1>{1};
                auto u2 = u1 / 2;
                REQUIRE(u2.at(0) == 0);

                // Inplace division
                auto v3 = Vector<int, 2>{1, 2};
                v3 /= 2;
                REQUIRE(v3.at(0) == 0);
                REQUIRE(v3.at(1) == 1);
            }

            SECTION("Application") {
                SECTION("Unary apply (in‑place transformation)") {
                    Vector<double, 3> v{1.0, 2.0, 3.0};
                    Vector<double, 3> &ref = v.apply([](double &x) { x *= 2; });
                    REQUIRE(&ref == &v);
                    REQUIRE(fuzzyCompare(v[0], 2.0));
                    REQUIRE(fuzzyCompare(v[1], 4.0));
                    REQUIRE(fuzzyCompare(v[2], 6.0));
                }

                SECTION("Binary apply (combine two vectors)") {
                    Vector<double, 3> a{1.0, 2.0, 3.0};
                    Vector<double, 3> b{4.0, 5.0, 6.0};
                    auto &r = a.apply(b, [](double &x, const double &y) { x += y; });
                    REQUIRE(&r == &a);
                    REQUIRE(fuzzyCompare(a[0], 5.0));
                    REQUIRE(fuzzyCompare(a[1], 7.0));
                    REQUIRE(fuzzyCompare(a[2], 9.0));
                }

                SECTION("Chaining unary and binary applies") {
                    Vector<double, 3> a{1.0, 2.0, 3.0};
                    Vector<double, 3> b{10.0, 20.0, 30.0};
                    // First double every element, then add from b, then subtract 5
                    a.apply([](double &x) { x *= 2; })
                        .apply(b, [](double &x, const double &y) { x += y; })
                        .apply([](double &x) { x -= 5; });

                    // Expected: (([1,2,3]*2) + [10,20,30]) - 5 = [ (2+10)-5, (4+20)-5, (6+30)-5 ] = [7,19,31]
                    REQUIRE(fuzzyCompare(a[0], 7.0));
                    REQUIRE(fuzzyCompare(a[1], 19.0));
                    REQUIRE(fuzzyCompare(a[2], 31.0));
                }
            }

            SECTION("Dot product") {
                auto v1 = Vector<int, 2>{1, 2};
                auto v2 = Vector<int, 2>{3, 4};
                auto dot = v1.dot(v2);
                REQUIRE(dot == Approx(11));

                auto u1 = Vector<int, 1>{1};
                auto u2 = Vector<int, 1>{2};
                auto dot2 = u1.dot(u2);
                REQUIRE(dot2 == Approx(2));
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

            SECTION("Norms") {
                SECTION("L1 Norm") {
                    auto v = Vector<int, 2>{1, 2};
                    auto norm = v.l1();
                    REQUIRE(norm == Approx(3));
                }

                SECTION("L2 Norm") {
                    auto v = Vector<int, 2>{1, 2};
                    auto norm = v.euclid();
                    REQUIRE(norm == Approx(2.23606797749979));
                }

                SECTION("Weighted L2 Norm") {
                    auto v = Vector<int, 2>{1, 2};
                    auto w = Vector<int, 2>{3, 4};
                    auto norm = v.weightedL2(w);
                    REQUIRE(norm == Approx(4.3588989435));
                }
            }

            SECTION("Transpose") {
                auto v = Vector<int, 2>{1, 2};
                auto v2 = v.T();
                REQUIRE(v2.at(0).at(0) == 1);
                REQUIRE(v2.at(0).at(1) == 2);

                auto u = Vector<int, 1>{1};
                auto u2 = u.T();
                REQUIRE(u2.at(0).at(0) == 1);

                auto vTT = helpers::extractVectorFromTranspose(v2.T());
                REQUIRE(vTT.at(0) == 1);
                REQUIRE(vTT.at(1) == 2);
            }

            SECTION("Iteration") {
                auto v = Vector<int, 2>{1, 2};
                auto it = v.begin();
                REQUIRE(*it == 1);
                ++it;
                REQUIRE(*it == 2);
                ++it;
                REQUIRE(it == v.end());
            }

            SECTION("Indexing") {
                auto v = Vector<int, 2>{1, 2};
                REQUIRE(v[0] == 1);
                REQUIRE(v[1] == 2);
                REQUIRE(v.at(0) == 1);
                REQUIRE(v.at(1) == 2);
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
                auto v1 = Vector<int, Dynamic>{1, 2};
                auto v2 = Vector<int, Dynamic>{3, 4};
                auto v3 = v1 + v2;
                REQUIRE(v3.at(0) == 4);
                REQUIRE(v3.at(1) == 6);

                auto u1 = Vector<int, Dynamic>{1};
                auto u2 = Vector<int, Dynamic>{2};
                auto u3 = u1 + u2;
                REQUIRE(u3.at(0) == 3);

                // Inplace addition
                auto v4 = Vector<int, Dynamic>{1, 2};
                auto v5 = Vector<int, Dynamic>{3, 4};
                v4 += v5;
                REQUIRE(v4.at(0) == 4);
                REQUIRE(v4.at(1) == 6);
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

                // Inplace Subtraction
                auto v4 = Vector<int, Dynamic>{1, 2};
                auto v5 = Vector<int, Dynamic>{3, 4};
                v4 -= v5;
                REQUIRE(v4.at(0) == -2);
                REQUIRE(v4.at(1) == -2);
            }

            SECTION("Vector Multiplication") {
                auto v1 = Vector<int, Dynamic>{1, 2};
                auto v2 = Vector<int, Dynamic>{3, 4};
                auto dot1 = v1 * v2;
                REQUIRE(dot1 == 11);

                auto u1 = Vector<int, Dynamic>{1};
                auto u2 = Vector<int, Dynamic>{2};
                auto dot2 = u1 * u2;
                REQUIRE(dot2 == 2);

                auto w1 = Vector<int, Dynamic>{1, 1};
                auto m = w1 * w1.T();
                auto m1 = helpers::extractMatrixFromTranspose(m);
                REQUIRE(m1.at(0, 0) == 1);
                REQUIRE(m1.at(0, 1) == 1);
                REQUIRE(m1.at(1, 0) == 1);
                REQUIRE(m1.at(1, 1) == 1);
            }

            SECTION("Matrix Multiplication") {
                auto v1 = Vector<int, Dynamic>{1, 2};
                auto m = Matrix<int, Dynamic, Dynamic>{{1, 2}, {3, 4}};
                auto v2 = helpers::extractVectorFromTranspose(v1 * m);
                REQUIRE(v2.at(0) == 7);
                REQUIRE(v2.at(1) == 10);

                auto u1 = Vector<int, Dynamic>{1};
                auto m2 = Matrix<int, Dynamic, Dynamic>{{2}};
                auto u2 = helpers::extractMatrixFromTranspose(u1 * m2);
                REQUIRE(u2.at(0, 0) == 2);

                auto v3 = m * v1;
                REQUIRE(v3.at(0) == 5);
                REQUIRE(v3.at(1) == 11);
            }

            SECTION("Scalar multiplication") {
                auto v1 = Vector<int, Dynamic>{1, 2};
                auto v2 = v1 * 2;
                REQUIRE(v2.at(0) == 2);
                REQUIRE(v2.at(1) == 4);

                auto u1 = Vector<int, Dynamic>{1};
                auto u2 = u1 * 2;
                REQUIRE(u2.at(0) == 2);

                // Inplace Multiplication
                auto v3 = Vector<int, Dynamic>{1, 2};
                v3 *= 2;
                REQUIRE(v3.at(0) == 2);
                REQUIRE(v3.at(1) == 4);
            }

            SECTION("Scalar division") {
                auto v1 = Vector<int, Dynamic>{1, 2};
                auto v2 = v1 / 2;
                REQUIRE(v2.at(0) == 0);
                REQUIRE(v2.at(1) == 1);

                auto u1 = Vector<int, Dynamic>{1};
                auto u2 = u1 / 2;
                REQUIRE(u2.at(0) == 0);

                // Inplace Division
                auto v3 = Vector<int, Dynamic>{1, 2};
                v3 /= 2;
                REQUIRE(v3.at(0) == 0);
                REQUIRE(v3.at(1) == 1);
            }

            SECTION("Application") {
                SECTION("Unary apply (in‑place transformation)") {
                    Vector<double, 3> v{1.0, 2.0, 3.0};
                    Vector<double, 3> &ref = v.apply([](double &x) { x *= 2; });
                    REQUIRE(&ref == &v);
                    REQUIRE(fuzzyCompare(v[0], 2.0));
                    REQUIRE(fuzzyCompare(v[1], 4.0));
                    REQUIRE(fuzzyCompare(v[2], 6.0));
                }

                SECTION("Binary apply (combine two vectors)") {
                    Vector<double, 3> a{1.0, 2.0, 3.0};
                    Vector<double, 3> b{4.0, 5.0, 6.0};
                    auto &r = a.apply(b, [](double &x, const double &y) { x += y; });
                    REQUIRE(&r == &a);
                    REQUIRE(fuzzyCompare(a[0], 5.0));
                    REQUIRE(fuzzyCompare(a[1], 7.0));
                    REQUIRE(fuzzyCompare(a[2], 9.0));
                }

                SECTION("Chaining unary and binary applies") {
                    Vector<double, 3> a{1.0, 2.0, 3.0};
                    Vector<double, 3> b{10.0, 20.0, 30.0};
                    // First double every element, then add from b, then subtract 5
                    a.apply([](double &x) { x *= 2; })
                        .apply(b, [](double &x, const double &y) { x += y; })
                        .apply([](double &x) { x -= 5; });

                    // Expected: (([1,2,3]*2) + [10,20,30]) - 5 = [ (2+10)-5, (4+20)-5, (6+30)-5 ] = [7,19,31]
                    REQUIRE(fuzzyCompare(a[0], 7.0));
                    REQUIRE(fuzzyCompare(a[1], 19.0));
                    REQUIRE(fuzzyCompare(a[2], 31.0));
                }
            }

            SECTION("Dot product") {
                auto v1 = Vector<int, Dynamic>{1, 2};
                auto v2 = Vector<int, Dynamic>{3, 4};
                auto dot = v1.dot(v2);
                REQUIRE(dot == Approx(11));

                auto u1 = Vector<int, Dynamic>{1};
                auto u2 = Vector<int, Dynamic>{2};
                auto dot2 = u1.dot(u2);
                REQUIRE(dot2 == Approx(2));
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

            SECTION("Norms") {
                SECTION("L1 Norm") {
                    auto v = Vector<int, Dynamic>{1, 2};
                    auto norm = v.l1();
                    REQUIRE(norm == Approx(3));
                }

                SECTION("L2 Norm") {
                    auto v = Vector<int, Dynamic>{1, 2};
                    auto norm = v.euclid();
                    REQUIRE(norm == Approx(2.23606797749979));
                }

                SECTION("Weighted L2 Norm") {
                    auto v = Vector<int, Dynamic>{1, 2};
                    auto w = Vector<int, Dynamic>{3, 4};
                    auto norm = v.weightedL2(w);
                    REQUIRE(norm == Approx(4.3588989435));
                }
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

            SECTION("Indexing") {
                auto v = Vector<int, 2>{1, 2};
                REQUIRE(v[0] == 1);
                REQUIRE(v[1] == 2);
                REQUIRE(v.at(0) == 1);
                REQUIRE(v.at(1) == 2);
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
                auto dot = v1 * v2;
                REQUIRE(dot == 11);
            }

            SECTION("Dot product") {
                auto v1 = Vector<int, Dynamic>{1, 2};
                auto v2 = Vector<int, 2>{3, 4};
                auto dot = v1.dot(v2);
                REQUIRE(dot == Approx(11));
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
                REQUIRE(len == Approx(0));
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

                for (size_t i = 0; i < large_size; i++) {
                    v1.at(static_cast<int>(i)) = i;
                    v2.at(static_cast<int>(i)) = i;
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
                REQUIRE(len == Approx(0));
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
                    v1.at(static_cast<int>(i)) = i;
                    v2.at(static_cast<int>(i)) = i;
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
