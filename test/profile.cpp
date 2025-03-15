#include <MLinalg.hpp>
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

int main() {
    {
        constexpr size_t m{128};
        auto A{genMartix<m, m>()};
        auto B{genMartix<m, m>()};
        A* B;
    }
    {
        constexpr size_t m{128};
        auto A{genMartix(m, m)};
        auto B{genMartix(m, m)};
        A* B;
    }
    return 0;
}
