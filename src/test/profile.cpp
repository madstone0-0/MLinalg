#include <MLinalg.hpp>
#include <cmath>
#include <iostream>
#include <random>

#include "operations/Decomposition.hpp"
#include "structures/Vector.hpp"

double rng() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<> dis(0, 1);
    return dis(gen);
}

using namespace mlinalg;

template <size_t m, size_t n>
constexpr Matrix<double, m, n> genMatrix() {
    Matrix<double, m, n> A;
    for (size_t i{}; i < m; i++) {
        for (size_t j{}; j < n; j++) {
            A(i, j) = rng();
        }
    }
    return A;
}

Matrix<double, Dynamic, Dynamic> genMatrix(size_t m, size_t n) {
    Matrix<double, Dynamic, Dynamic> A{m, n};
    for (size_t i{}; i < m; i++) {
        for (size_t j{}; j < n; j++) {
            A(i, j) = rng();
        }
    }
    return A;
}

int main() {
    constexpr size_t m{512};
    {
        auto A{genMatrix<m, m>()};
        auto B{genMatrix<m, m>()};
        A* B;
    }
    {
        auto A{genMatrix(m, m)};
        auto B{genMatrix(m, m)};
        A* B;
    }
    {
        auto A{genMatrix<m / 16, m / 16>()};
        const auto& [U, S, VT] = svd(A);
    }
    return 0;
}
