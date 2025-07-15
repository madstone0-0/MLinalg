#include <MLinalg.hpp>
#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <random>

#include "operations/Builders.hpp"
#include "structures/Vector.hpp"

using namespace mlinalg;

TEST_CASE("Benchmarks") {
    SECTION("Matrix") {
        SECTION("Compile Time") {
            SECTION("Square") {
                {
                    constexpr size_t m{100};
                    auto A{matrixRandom<double, m, m>()};
                    auto B{matrixRandom<double, m, m>()};
                    BENCHMARK("Beeg A * B (100x100)") {
                        // Biiig square matrix
                        return A * B;
                    };
                }

                {
                    constexpr size_t m{127};
                    auto A{matrixRandom<double, m, m>()};
                    auto B{matrixRandom<double, m, m>()};
                    BENCHMARK("Beeg A * B (127x127)") { return A * B; };
                }

                {
                    constexpr size_t m{128};
                    auto A{matrixRandom<double, m, m>()};
                    auto B{matrixRandom<double, m, m>()};
                    BENCHMARK("Beeg A * B (128x128)") { return A * B; };
                }

                {
                    constexpr size_t m{255};
                    auto A{matrixRandom<double, m, m>()};
                    auto B{matrixRandom<double, m, m>()};
                    BENCHMARK("Beeg A * B (255x255)") { return A * B; };
                }

                {
                    constexpr size_t m{256};
                    auto A{matrixRandom<double, m, m>()};
                    auto B{matrixRandom<double, m, m>()};
                    BENCHMARK("Beeg A * B (256x256)") { return A * B; };
                }

                {
                    constexpr size_t m{511};
                    auto A{matrixRandom<double, m, m>()};
                    auto B{matrixRandom<double, m, m>()};
                    BENCHMARK("Beeg A * B (511x511)") { return A * B; };
                }

                {
                    constexpr size_t m{512};
                    auto A{matrixRandom<double, m, m>()};
                    auto B{matrixRandom<double, m, m>()};
                    BENCHMARK("Beeg A * B (512x512)") { return A * B; };
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

            SECTION("Rectangle") {
                {
                    constexpr size_t m{100};
                    constexpr size_t n{50};
                    auto A{matrixRandom<double, m, n>()};
                    auto B{matrixRandom<double, n, m>()};
                    BENCHMARK("Beeg A * B (100x50 x 50x100)") { return A * B; };
                }

                {
                    constexpr size_t m{127};
                    constexpr size_t n{100};
                    auto A{matrixRandom<double, m, n>()};
                    auto B{matrixRandom<double, n, m>()};
                    BENCHMARK("Beeg A * B (127x100 x 100x127)") { return A * B; };
                }

                {
                    constexpr size_t m{128};
                    constexpr size_t n{100};
                    auto A{matrixRandom<double, m, n>()};
                    auto B{matrixRandom<double, n, m>()};
                    BENCHMARK("Beeg A * B (128x100 x 100x128)") { return A * B; };
                }

                {
                    constexpr size_t m{255};
                    constexpr size_t n{128};
                    auto A{matrixRandom<double, m, n>()};
                    auto B{matrixRandom<double, n, m>()};
                    BENCHMARK("Beeg A * B (255x128 x 128x255)") { return A * B; };
                }

                {
                    constexpr size_t m{256};
                    constexpr size_t n{128};
                    auto A{matrixRandom<double, m, n>()};
                    auto B{matrixRandom<double, n, m>()};
                    BENCHMARK("Beeg A * B (256x128 x 128x256)") { return A * B; };
                }

                {
                    constexpr size_t m{511};
                    constexpr size_t n{256};
                    auto A{matrixRandom<double, m, n>()};
                    auto B{matrixRandom<double, n, m>()};
                    BENCHMARK("Beeg A * B (511x256 x 256x511)") { return A * B; };
                }

                {
                    constexpr size_t m{512};
                    constexpr size_t n{256};
                    auto A{matrixRandom<double, m, n>()};
                    auto B{matrixRandom<double, n, m>()};
                    BENCHMARK("Beeg A * B (512x256 x 256x512)") { return A * B; };
                }
            }
        }

        SECTION("Dynamic") {
            SECTION("Square") {
                {
                    constexpr size_t m{100};
                    auto A{matrixRandom<double, Dynamic, Dynamic>(m, m, 0, 100)};
                    auto B{matrixRandom<double, Dynamic, Dynamic>(m, m, 0, 100)};
                    BENCHMARK("Beeg A * B (100x100)") {
                        // Biiig square matrix
                        return A * B;
                    };
                }

                {
                    constexpr size_t m{127};
                    auto A{matrixRandom<double, Dynamic, Dynamic>(m, m, 0, 100)};
                    auto B{matrixRandom<double, Dynamic, Dynamic>(m, m, 0, 100)};
                    BENCHMARK("Beeg A * B (127x127)") { return A * B; };
                }

                {
                    constexpr size_t m{128};
                    auto A{matrixRandom<double, Dynamic, Dynamic>(m, m, 0, 100)};
                    auto B{matrixRandom<double, Dynamic, Dynamic>(m, m, 0, 100)};
                    BENCHMARK("Beeg A * B (128x128)") { return A * B; };
                }

                {
                    constexpr size_t m{255};
                    auto A{matrixRandom<double, Dynamic, Dynamic>(m, m, 0, 100)};
                    auto B{matrixRandom<double, Dynamic, Dynamic>(m, m, 0, 100)};
                    BENCHMARK("Beeg A * B (255x255)") { return A * B; };
                }

                {
                    constexpr size_t m{256};
                    auto A{matrixRandom<double, Dynamic, Dynamic>(m, m, 0, 100)};
                    auto B{matrixRandom<double, Dynamic, Dynamic>(m, m, 0, 100)};
                    BENCHMARK("Beeg A * B (256x256)") { return A * B; };
                }

                {
                    constexpr size_t m{511};
                    auto A{matrixRandom<double, Dynamic, Dynamic>(m, m, 0, 100)};
                    auto B{matrixRandom<double, Dynamic, Dynamic>(m, m, 0, 100)};
                    BENCHMARK("Beeg A * B (511x511)") { return A * B; };
                }

                {
                    constexpr size_t m{512};
                    auto A{matrixRandom<double, Dynamic, Dynamic>(m, m, 0, 100)};
                    auto B{matrixRandom<double, Dynamic, Dynamic>(m, m, 0, 100)};
                    BENCHMARK("Beeg A * B (512x512)") { return A * B; };
                }

                // FIX: Too slow fix
                // {
                //     constexpr size_t m{1023};
                //     auto A{genMartix(m, m)};
                //     auto B{genMartix(m, m)};
                //     BENCHMARK("Beeg A * B (1023x1023)") { return A * B; };
                // }

                // FIX: Too slow fix
                // {
                //     constexpr size_t m{1024};
                //     auto A{genMartix<m, m>()};
                //     auto B{genMartix<m, m>()};
                //     BENCHMARK("Beeg A * B (1024x1024)") { return A * B; };
                // }
            }

            SECTION("Rectangle") {
                {
                    constexpr size_t m{100};
                    constexpr size_t n{50};
                    auto A{matrixRandom<double, Dynamic, Dynamic>(m, n, 0, 100)};
                    auto B{matrixRandom<double, Dynamic, Dynamic>(n, m, 0, 100)};
                    BENCHMARK("Beeg A * B (100x50 x 50x100)") { return A * B; };
                }

                {
                    constexpr size_t m{127};
                    constexpr size_t n{100};
                    auto A{matrixRandom<double, Dynamic, Dynamic>(m, n, 0, 100)};
                    auto B{matrixRandom<double, Dynamic, Dynamic>(n, m, 0, 100)};
                    BENCHMARK("Beeg A * B (127x100 x 100x127)") { return A * B; };
                }

                {
                    constexpr size_t m{128};
                    constexpr size_t n{100};
                    auto A{matrixRandom<double, Dynamic, Dynamic>(m, n, 0, 100)};
                    auto B{matrixRandom<double, Dynamic, Dynamic>(n, m, 0, 100)};
                    BENCHMARK("Beeg A * B (128x100 x 100x128)") { return A * B; };
                }

                {
                    constexpr size_t m{255};
                    constexpr size_t n{128};
                    auto A{matrixRandom<double, Dynamic, Dynamic>(m, n, 0, 100)};
                    auto B{matrixRandom<double, Dynamic, Dynamic>(n, m, 0, 100)};
                    BENCHMARK("Beeg A * B (255x128 x 128x255)") { return A * B; };
                }

                {
                    constexpr size_t m{256};
                    constexpr size_t n{128};
                    auto A{matrixRandom<double, Dynamic, Dynamic>(m, n, 0, 100)};
                    auto B{matrixRandom<double, Dynamic, Dynamic>(n, m, 0, 100)};
                    BENCHMARK("Beeg A * B (256x128 x 128x256)") { return A * B; };
                }

                {
                    constexpr size_t m{511};
                    constexpr size_t n{256};
                    auto A{matrixRandom<double, Dynamic, Dynamic>(m, n, 0, 100)};
                    auto B{matrixRandom<double, Dynamic, Dynamic>(n, m, 0, 100)};
                    BENCHMARK("Beeg A * B (511x256 x 256x511)") { return A * B; };
                }

                {
                    constexpr size_t m{512};
                    constexpr size_t n{256};
                    auto A{matrixRandom<double, Dynamic, Dynamic>(m, n, 0, 100)};
                    auto B{matrixRandom<double, Dynamic, Dynamic>(n, m, 0, 100)};
                    BENCHMARK("Beeg A * B (512x256 x 256x512)") { return A * B; };
                }
            }
        }
    }
}
