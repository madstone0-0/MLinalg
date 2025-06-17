#include <MLinalg.hpp>

int main() {
    auto v = mlinalg::structures::Vector<int, 2>{1, 2};
    auto u = mlinalg::structures::Vector<int, 2>{1, 2};
    v* u;
    return 0;
}
