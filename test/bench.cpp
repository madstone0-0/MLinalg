#include <MLinalg.hpp>
#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <iostream>
#include <random>

#include "structures/Vector.hpp"

double rng() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<> dis(0, 1);
    return dis(gen);
}

using namespace mlinalg;

template <size_t m, size_t n>
Matrix<double, m, n> genMartix() {
    Matrix<double, m, n> A;
    for (size_t i{}; i < m; i++) {
        for (size_t j{}; j < m; j++) {
            A.at(i, j) = rng();
        }
    }
    return A;
}

Matrix<double, Dynamic, Dynamic> genMartix(size_t m, size_t n) {
    Matrix<double, Dynamic, Dynamic> A{(int)m, (int)n};
    for (size_t i{}; i < m; i++) {
        for (size_t j{}; j < m; j++) {
            A.at(i, j) = rng();
        }
    }
    return A;
}

TEST_CASE("Benchmarks") {
    SECTION("Matrix") {
        SECTION("Compile Time") {
            {
                constexpr size_t m{100};
                auto A{genMartix<m, m>()};
                auto B{genMartix<m, m>()};
                BENCHMARK("Beeg A * B (100x100)") {
                    // Biiig square matrix
                    return A * B;
                };
            }

            {
                constexpr size_t m{127};
                auto A{genMartix<m, m>()};
                auto B{genMartix<m, m>()};
                BENCHMARK("Beeg A * B (127x127)") { return A * B; };
            }

            {
                constexpr size_t m{128};
                auto A{genMartix<m, m>()};
                auto B{genMartix<m, m>()};
                BENCHMARK("Beeg A * B (128x128)") { return A * B; };
            }

            // FIX: Too slow fix
            // BENCHMARK("Beeg A * B (1023x1023)") {
            //     constexpr size_t m{1023};
            //     auto A{genMartix<m, m>()};
            //     auto B{genMartix<m, m>()};
            //     return A * B;
            // };

            // FIX: Too slow fix
            // BENCHMARK("Beeg A * B (1024x1024)") {
            //     constexpr size_t m{1024};
            //     auto A{genMartix<m, m>()};
            //     auto B{genMartix<m, m>()};
            //     return A * B;
            // };
        }

        SECTION("Dynamic") {
            {
                constexpr size_t m{100};
                auto A{genMartix(m, m)};
                auto B{genMartix(m, m)};
                BENCHMARK("Beeg A * B (100x100)") {
                    // Biiig square matrix
                    return A * B;
                };
            }

            {
                constexpr size_t m{127};
                auto A{genMartix(m, m)};
                auto B{genMartix(m, m)};
                BENCHMARK("Beeg A * B (127x127)") { return A * B; };
            }

            {
                constexpr size_t m{128};
                auto A{genMartix(m, m)};
                auto B{genMartix(m, m)};
                BENCHMARK("Beeg A * B (128x128)") { return A * B; };
            }

            // FIX: Too slow fix
            // BENCHMARK("Beeg A * B (1023x1023)") {
            //     constexpr size_t m{1023};
            //     auto A{genMartix<m, m>()};
            //     auto B{genMartix<m, m>()};
            //     return A * B;
            // };

            // FIX: Too slow fix
            // BENCHMARK("Beeg A * B (1024x1024)") {
            //     constexpr size_t m{1024};
            //     auto A{genMartix<m, m>()};
            //     auto B{genMartix<m, m>()};
            //     return A * B;
            // };
        }
    }
}
