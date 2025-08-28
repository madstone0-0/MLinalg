#include <numeric>
#include <print>

#include "utils.hpp"

int main() {
    using namespace mlinalg;
    using namespace std;
    using namespace utils;

    constexpr size_t dim{2};
    Seed seed{42};
    auto X{matrixRandom<double, Dynamic, Dynamic>(10, 100, 0, 1, seed)};
    println("4x4 sample from 10x100 matrix ->\n{}", string(X.view({.start = 0, .end = 4}, {.start = 0, .end = 4})));

    try {
        const auto& centered = center(X);
        const auto& [U, S, VT] = svd(centered);

        auto vt = VT.rowToVectorSet();
        vt.erase(vt.begin() + dim, vt.end());

        const auto& W = fromColVectorSet<double, Dynamic, Dynamic>(vt);
        const auto& Xdim = centered * W;
        println("Reduced X from {} features to {}", X.numCols(), Xdim.numCols());
        println("SVD PCA({}) ->\n{}", dim, string(Xdim));

        // Check if the principal components are orthogonal
        println("Are the first {} principal components orthogonal? -> {}", dim, isOrthogonal(W));

        // Check explained variance ratios
        const auto& explainedVar =
            diag(S).view(0, 2).toVector().apply([](auto& x) { x *= x; }) / (double)(X.numRows() - 1);

        const auto& allExplainedVar = diag(S).apply([](auto& x) { x *= x; }) / (double)(X.numRows() - 1);
        const auto& totalOriginalVar = accumulate(allExplainedVar.begin(), allExplainedVar.end(), 0.0);

        // Ratios relative to total original variance
        auto explainedVarRatio = explainedVar / totalOriginalVar;
        const auto& totalExplainedVar = accumulate(explainedVarRatio.begin(), explainedVarRatio.end(), 0.0);

        println("Explained variance -> {}", explainedVar);
        println("Total explained variance -> {}", totalExplainedVar);
        println("Explained variance ratios -> {}", explainedVarRatio);

    } catch (const std::exception& e) {
        println("Error -> {}", e.what());
    }

    return 0;
}
