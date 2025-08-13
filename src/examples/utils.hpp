#pragma once
#include <mlinalg/MLinalg.hpp>

// Utility functions not found in MLinalg
namespace utils {
    using namespace std;
    using namespace mlinalg;
    namespace rg = std::ranges;

    inline double mean(const VD<double>& v) {
        double res{};
        v.apply([&](const auto x) { res += x; });
        return res / (double)v.size();
    }

    inline VD<double> mean(const MD<double>& X, bool axis = true) {
        auto [nR, nC] = X.shape();
        if (axis) {
            VD<double> res(nC);
            for (size_t i{}; i < nC; ++i) {
                auto col = X.col(i).toVector();
                auto m = mean(col);
                res[i] = m;
            }
            return res;
        } else {
            VD<double> res(nR);
            size_t i{};
            X.applyRow([&](const auto& row) { res[++i] = mean(row); });
            return res;
        }
    }

    inline double var(const VD<double>& v) {
        double res{};
        const auto& m = mean(v);
        v.apply([&](const auto& x) {
            auto val = x - m;
            res += val * val;
        });
        return res / ((double)v.size() - 1);
    }

    inline VD<double> var(const MD<double>& X, bool axis = true) {
        auto [nR, nC] = X.shape();
        auto m = mean(X, axis);
        if (axis) {
            VD<double> res(nC);
            for (size_t i{}; i < nC; i++) {
                auto ones{vectorOnes<double, Dynamic>(nR)};
                auto xMean = X.col(i) - ones * m[i];
                xMean.apply([](auto& x) { x *= x; });
                double sum{};
                xMean.apply([&sum](const auto& x) { sum += x; });
                res[i] = sum / (double)(nR);
            }
            return res;
        } else {
            VD<double> res(nR);
            size_t i{};
            X.applyRow([&](auto row) {
                auto ones{vectorOnes<double, Dynamic>(nC)};
                auto xMean = row - ones * m[i];
                xMean.apply([](auto& x) { x *= x; });
                double sum{};
                xMean.apply([&sum](const auto& x) { sum += x; });
                res[++i] = sum / (double)(nC);
            });
            return res;
        }
    }

    inline VD<double> std(const MD<double>& X, bool axis = true) {
        auto std = var(X, axis);
        std.apply([&](auto& x) { x = std::sqrt(x); });
        return std;
    }

    inline MD<double> normalize(const MD<double>& X, bool axis = true) {
        auto [nR, nC] = X.shape();
        auto res{X};
        auto m = mean(res, axis);
        auto s = std(res, axis);

        if (axis) {
            for (size_t j{}; j < nC; ++j)
                for (size_t i{}; i < nR; ++i) {
                    res(i, j) = (res(i, j) - m[j]) / (fuzzyCompare(s[j], 0.0) ? 1e-15 : s[j]);
                }
        } else {
            for (size_t i{}; i < nR; ++i)
                for (size_t j{}; j < nC; ++j) {
                    res(i, j) = (res(i, j) - m[i]) / (fuzzyCompare(s[i], 0.0) ? 1e-15 : s[j]);
                }
        }

        return res;
    }

    // Calculate R-squared for model evaluation
    inline auto r2_score(const VD<double>& y, const VD<double>& yp) {
        double ss_tot{};

        // Calculate mean of y_true
        double yM{};
        y.apply([&yM](auto& val) { yM += val; });
        yM /= y.size();

        auto diff = y - yp;
        double ss_res = diff.dot(diff);
        diff = y - yM * vectorOnes<double, Dynamic>(y.size());
        ss_tot = diff.dot(diff);

        return 1.0 - (ss_res / ss_tot);
    }

    // Calculate Mean Squared Error
    inline auto calculateMSE(const VD<double>& y_true, const VD<double>& y_pred) {
        auto diff = y_true - y_pred;
        double mse = diff.dot(diff);
        return mse / (double)y_true.size();
    }

    struct TrainTestSplit {
        MD<double> XTrain, XTest;
        VD<double> yTrain, yTest;
        vector<size_t> trainIdx, testIdx;
    };

    inline TrainTestSplit trainTestSplit(const MD<double>& X, const VD<double>& y, double testSize = 0.2,
                                         Seed seed = Seed{42}) {
        if (X.numRows() != y.size()) throw std::invalid_argument{"X and y must have the ssame number of samples"};

        if (testSize <= 0.0 || testSize >= 1.0) throw std::invalid_argument{"testSize must be between 0.0 and 1.0"};

        size_t samples = y.size();
        auto testSample = static_cast<size_t>(samples * testSize);
        size_t trainSample = samples - testSample;

        vector<size_t> idx(samples);
        rg::iota(idx, 0);

        mt19937 gen(seed.value());
        rg::shuffle(idx, gen);

        vector<size_t> trainIdx{idx.begin(), idx.begin() + trainSample};
        vector<size_t> testIdx{idx.begin() + trainSample, idx.end()};

        MD<double> XTrain(trainSample, X.numCols());
        MD<double> XTest(testSample, X.numCols());
        VD<double> yTrain(trainSample);
        VD<double> yTest(testSample);

        for (size_t i{}; i < trainSample; ++i) {
            auto idx{trainIdx[i]};
            XTrain[i] = X[idx];
            yTrain[i] = y[idx];
        }

        for (size_t i{}; i < testSample; ++i) {
            auto idx{testIdx[i]};
            XTest[i] = X[idx];
            yTest[i] = y[idx];
        }

        return {.XTrain = XTrain,
                .XTest = std::move(XTest),
                .yTrain = std::move(yTrain),
                .yTest = std::move(yTest),
                .trainIdx = std::move(trainIdx),
                .testIdx = std::move(testIdx)};
    }

    struct StandardScaler {
        void fit(const MD<double>& X) {
            means_.resize(X.numCols());
            stds_.resize(X.numCols());
            means_ = mean(X, true);
            stds_ = std(X, true);
            fitted_ = true;
        }

        MD<double> transform(const MD<double>& X) {
            if (!fitted_) throw runtime_error{"Scaler must be fitted before transform"};
            auto [nR, nC] = X.shape();
            auto res{X};

            for (size_t j{}; j < nC; ++j)
                for (size_t i{}; i < nR; ++i) {
                    res(i, j) = (res(i, j) - means_[j]) / (fuzzyCompare(stds_[j], 0.0) ? 1e-15 : stds_[j]);
                }

            return res;
        }

        MD<double> fitTransform(const MD<double>& X) {
            fit(X);
            return transform(X);
        }

        [[nodiscard]] auto means() const { return means_; }
        [[nodiscard]] auto stds() const { return stds_; }

       private:
        bool fitted_{};
        VD<double> means_{0};
        VD<double> stds_{0};
    };

    inline auto center(const MD<double>& X, bool axis = true) {
        auto [nR, nC] = X.shape();
        auto res{X};
        auto m = mean(res, axis);

        if (axis) {
            for (size_t j{}; j < nC; ++j)
                for (size_t i{}; i < nR; ++i) {
                    res(i, j) = (res(i, j) - m[j]);
                }
        } else {
            for (size_t i{}; i < nR; ++i)
                for (size_t j{}; j < nC; ++j) {
                    res(i, j) = (res(i, j) - m[i]);
                }
        }

        return res;
    }

    inline auto covariance(const VD<double>& v, const VD<double>& w) {
        if (v.size() != w.size()) throw runtime_error{"Vectors must be of the same size to calculate their covariance"};
        double res{};
        const auto& vM = mean(v);
        const auto& wM = mean(w);
        v.apply(w, [&](const auto& x, const auto& y) { res += (x - vM) * (y - wM); });

        return res / ((double)v.size() - 1);
    }

    struct Position {
        size_t i{};
        size_t j{};

        auto operator<=>(const Position& other) const {
            if (auto cmp = i <=> other.i; cmp != 0) return cmp;
            return j <=> other.j;
        }

        bool operator==(const Position& other) const {
            return ((other.i == i) && (other.j == j)) || ((other.i == j) && (other.j == i));
        }
    };

    inline auto cov(const MD<double>& X) {
        const auto [nR, nC] = X.shape();
        auto res{matrixZeros<double, Dynamic, Dynamic>(nC, nC)};
        std::map<Position, double> covmap{};

        for (size_t i{}; i < nC; ++i) {
            const auto& colI = X.col(i).toVector();
            var(X.col(i).toVector());
            for (size_t j{}; j < nC; ++j) {
                if (i == j) continue;
                const auto& colJ = X.col(j).toVector();
                Position pos{.i = i, .j = j};
                if (covmap.contains(pos))
                    res(i, j) = covmap[pos];
                else {
                    auto cv = covariance(colI, colJ);
                    covmap[pos] = cv;
                    res(i, j) = cv;
                }
            }
        }

        return res;
    }
}  // namespace utils
