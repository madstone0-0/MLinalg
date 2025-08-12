#include <MLinalg.hpp>
#include <cmath>
#include <iostream>
#include <print>
#include <random>

#include "operations/Decomposition.hpp"
#include "structures/Aliases.hpp"

using namespace mlinalg;

template <Number number, int m, int n>
void roundMatrix(Matrix<number, m, n>& A) {
    A.apply([](auto& x) { x = std::round(x); });
}

int main() {
    constexpr size_t m{512};
    using num = double;
    // {
    //     auto A{matrixRandom<num, m, m>()};
    //     auto B{matrixRandom<num, m, m>()};
    //     A* B;
    // }
    // {
    //     auto A{matrixRandom<num, Dynamic, Dynamic>(m, m)};
    //     auto B{matrixRandom<num, Dynamic, Dynamic>(m, m)};
    //     A* B;
    // }
    {
        constexpr auto r = m / 16;
        auto A{matrixRandom<num, r, r>()};
        const auto& [U, S, VT] = svd(A);
        auto SVD = U * S * VT;
        std::println("A == U*S*VT -> {}", A == SVD);
        if constexpr (r <= 30) {
            roundMatrix(A);
            roundMatrix(SVD);
            std::println("A ->\n{}\nU*S*VT ->\n{}", string(A), string(SVD));
        }
    }
    return 0;
}
