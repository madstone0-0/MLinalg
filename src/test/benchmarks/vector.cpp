#include "structures/Vector.hpp"

#include <MLinalg.hpp>
#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <random>
#include <utility>

#include "helpers.hpp"
#include "operations/Builders.hpp"

using namespace mlinalg;

TEST_CASE("Benchmarks") {
    constexpr size_t start{15};
    constexpr size_t end{20};
    SECTION("Vector") {
        SECTION("Compile Time") {
            constexpr size_t range{end - start};
            SECTION("Vector-Vector Multiplication") {
                for_(
                    [&](auto n) {
                        constexpr size_t m{ipow(2, n.value + start)};
                        auto v{vectorRandom<double, m>()};
                        auto w{vectorRandom<double, m>()};
                        BENCHMARK(std::format("v * w ({}x{})", m, m)) { return v * w; };
                    },
                    std::make_index_sequence<range>());
            }

            SECTION("Vector-Vector Addition") {
                for_(
                    [&](auto n) {
                        constexpr size_t m{ipow(2, n.value + start)};
                        auto v{vectorRandom<double, m>()};
                        auto w{vectorRandom<double, m>()};
                        BENCHMARK(std::format("v + w ({}x{})", m, m)) { return v + w; };
                    },
                    std::make_index_sequence<range>());
            }

            SECTION("Vector-Vector Inline Addition") {
                for_(
                    [&](auto n) {
                        constexpr size_t m{ipow(2, n.value + start)};
                        auto v{vectorRandom<double, m>()};
                        auto w{vectorRandom<double, m>()};
                        BENCHMARK(std::format("v += w ({}x{})", m, m)) { return v += w; };
                    },
                    std::make_index_sequence<range>());
            }

            SECTION("Vector-Vector Subtraction") {
                for_(
                    [&](auto n) {
                        constexpr size_t m{ipow(2, n.value + start)};
                        auto v{vectorRandom<double, m>()};
                        auto w{vectorRandom<double, m>()};
                        BENCHMARK(std::format("v - w ({}x{})", m, m)) { return v - w; };
                    },
                    std::make_index_sequence<range>());
            }

            SECTION("Vector-Vector Inline Subtraction") {
                for_(
                    [&](auto n) {
                        constexpr size_t m{ipow(2, n.value + start)};
                        auto v{vectorRandom<double, m>()};
                        auto w{vectorRandom<double, m>()};
                        BENCHMARK(std::format("v -= w ({}x{})", m, m)) { return v -= w; };
                    },
                    std::make_index_sequence<range>());
            }

            SECTION("Vector-Scalar Multiplication") {
                for_(
                    [&](auto n) {
                        constexpr size_t m{ipow(2, n.value + start)};
                        auto v{vectorRandom<double, m>()};
                        double scalar{5.0};
                        BENCHMARK(std::format("v * {} ({}x1)", scalar, m)) { return v * scalar; };
                    },
                    std::make_index_sequence<range>());
            }

            SECTION("Vector-Scalar Inline Multiplication") {
                for_(
                    [&](auto n) {
                        constexpr size_t m{ipow(2, n.value + start)};
                        auto v{vectorRandom<double, m>()};
                        double scalar{5.0};
                        BENCHMARK(std::format("v *= {} ({}x1)", scalar, m)) { return v *= scalar; };
                    },
                    std::make_index_sequence<range>());
            }

            SECTION("Vector-Scalar Division") {
                for_(
                    [&](auto n) {
                        constexpr size_t m{ipow(2, n.value + start)};
                        auto v{vectorRandom<double, m>()};
                        double scalar{5.0};
                        BENCHMARK(std::format("v / {} ({}x1)", scalar, m)) { return v / scalar; };
                    },
                    std::make_index_sequence<range>());
            }

            SECTION("Vector-Scalar Inline Division") {
                for_(
                    [&](auto n) {
                        constexpr size_t m{ipow(2, n.value + start)};
                        auto v{vectorRandom<double, m>()};
                        double scalar{5.0};
                        BENCHMARK(std::format("v /= {} ({}x1)", scalar, m)) { return v /= scalar; };
                    },
                    std::make_index_sequence<range>());
            }
        }

        SECTION("Dynamic") {
            SECTION("Vector-Vector Multiplication") {
                for (size_t i{start}; i < end; ++i) {
                    size_t m{static_cast<size_t>(std::pow(2, i))};
                    auto v{vectorRandom<double, Dynamic>(m, 0, 100)};
                    auto w{vectorRandom<double, Dynamic>(m, 0, 100)};
                    BENCHMARK(std::format("v * w ({}x{})", m, m)) { return v * w; };
                }
            }

            SECTION("Vector-Vector Addition") {
                for (size_t i{start}; i < end; ++i) {
                    size_t m{static_cast<size_t>(std::pow(2, i))};
                    auto v{vectorRandom<double, Dynamic>(m, 0, 100)};
                    auto w{vectorRandom<double, Dynamic>(m, 0, 100)};
                    BENCHMARK(std::format("v + w ({}x{})", m, m)) { return v + w; };
                }
            }

            SECTION("Vector-Vector Inline Addition") {
                for (size_t i{start}; i < end; ++i) {
                    size_t m{static_cast<size_t>(std::pow(2, i))};
                    auto v{vectorRandom<double, Dynamic>(m, 0, 100)};
                    auto w{vectorRandom<double, Dynamic>(m, 0, 100)};
                    BENCHMARK(std::format("v += w ({}x{})", m, m)) { return v += w; };
                }
            }

            SECTION("Vector-Vector Subtraction") {
                for (size_t i{start}; i < end; ++i) {
                    size_t m{static_cast<size_t>(std::pow(2, i))};
                    auto v{vectorRandom<double, Dynamic>(m, 0, 100)};
                    auto w{vectorRandom<double, Dynamic>(m, 0, 100)};
                    BENCHMARK(std::format("v - w ({}x{})", m, m)) { return v - w; };
                }
            }

            SECTION("Vector-Vector Inline Subtraction") {
                for (size_t i{start}; i < end; ++i) {
                    size_t m{static_cast<size_t>(std::pow(2, i))};
                    auto v{vectorRandom<double, Dynamic>(m, 0, 100)};
                    auto w{vectorRandom<double, Dynamic>(m, 0, 100)};
                    BENCHMARK(std::format("v -= w ({}x{})", m, m)) { return v -= w; };
                }
            }

            SECTION("Vector-Scalar Multiplication") {
                for (size_t i{start}; i < end; ++i) {
                    size_t m{static_cast<size_t>(std::pow(2, i))};
                    auto v{vectorRandom<double, Dynamic>(m, 0, 100)};
                    double scalar{5.0};
                    BENCHMARK(std::format("v * {} ({}x1)", scalar, m)) { return v * scalar; };
                }
            }

            SECTION("Vector-Scalar Inline Multiplication") {
                for (size_t i{start}; i < end; ++i) {
                    size_t m{static_cast<size_t>(std::pow(2, i))};
                    auto v{vectorRandom<double, Dynamic>(m, 0, 100)};
                    double scalar{5.0};
                    BENCHMARK(std::format("v *= {} ({}x1)", scalar, m)) { return v *= scalar; };
                }
            }

            SECTION("Vector-Scalar Division") {
                for (size_t i{start}; i < end; ++i) {
                    size_t m{static_cast<size_t>(std::pow(2, i))};
                    auto v{vectorRandom<double, Dynamic>(m, 0, 100)};
                    double scalar{5.0};
                    BENCHMARK(std::format("v / {} ({}x1)", scalar, m)) { return v / scalar; };
                }
            }

            SECTION("Vector-Scalar Inline Division") {
                for (size_t i{start}; i < end; ++i) {
                    size_t m{static_cast<size_t>(std::pow(2, i))};
                    auto v{vectorRandom<double, Dynamic>(m, 0, 100)};
                    double scalar{5.0};
                    BENCHMARK(std::format("v /= {} ({}x1)", scalar, m)) { return v /= scalar; };
                }
            }
        }
    }
}
