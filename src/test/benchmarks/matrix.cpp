#include <MLinalg.hpp>
#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <random>

#include "helpers.hpp"
#include "operations/Builders.hpp"

using namespace mlinalg;

TEST_CASE("Benchmarks") {
    constexpr size_t start{5};
    constexpr size_t end{10};
    SECTION("Matrix") {
        SECTION("Compile Time") {
            constexpr size_t range{end - start};
            SECTION("Matrix-Matrix Multiplication") {
                SECTION("Square") {
                    for_(
                        [&](auto n) {
                            constexpr size_t m{ipow(2, n.value + start)};
                            auto A{matrixRandom<double, m, m>()};
                            auto B{matrixRandom<double, m, m>()};
                            BENCHMARK(std::format("A * B ({}x{})", m, m)) { return A * B; };
                        },
                        std::make_index_sequence<range>());
                }

                SECTION("Rectangle") {
                    for_(
                        [&](auto i) {
                            constexpr size_t m{ipow(2, i.value + start)};
                            constexpr size_t n{m / 2};
                            auto A{matrixRandom<double, m, n>()};
                            auto B{matrixRandom<double, n, m>()};
                            BENCHMARK(std::format("A * B ({}x{} x {}x{})", m, n, n, m)) { return A * B; };
                        },
                        std::make_index_sequence<range>());
                }
            }
        }

        SECTION("Dynamic") {
            SECTION("Matrix-Matrix Multiplication") {
                SECTION("Square") {
                    for (size_t i{start}; i < end; ++i) {
                        size_t m{static_cast<size_t>(std::pow(2, i))};
                        auto A{matrixRandom<double, Dynamic, Dynamic>(m, m, 0, 100)};
                        auto B{matrixRandom<double, Dynamic, Dynamic>(m, m, 0, 100)};
                        BENCHMARK(std::format("A * B ({}x{})", m, m)) { return A * B; };
                    }
                }

                SECTION("Rectangle") {
                    for (size_t i{start}; i < end; ++i) {
                        size_t m{static_cast<size_t>(std::pow(2, i))};
                        size_t n{m / 2};
                        auto A{matrixRandom<double, Dynamic, Dynamic>(m, n, 0, 100)};
                        auto B{matrixRandom<double, Dynamic, Dynamic>(n, m, 0, 100)};
                        BENCHMARK(std::format("A * B ({}x{} x {}x{})", m, n, n, m)) { return A * B; };
                    }
                }
            }
        }
    }
}
