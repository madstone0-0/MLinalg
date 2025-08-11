#include <algorithm>
#include <numeric>
#include <print>
#include <random>

#include "MLinalg.hpp"
#include "Numeric.hpp"
#include "operations/Builders.hpp"
#include "structures/Aliases.hpp"

using namespace std;
using namespace mlinalg;
using namespace mlinalg::structures;
namespace rg = std::ranges;

// Helper functions not found in MLinalg
namespace {
    double mean(const VD<double>& v) {
        double res{};
        v.apply([&](const auto x) { res += x; });
        return res / (double)v.size();
    }

    VD<double> mean(const MD<double>& X, bool axis = true) {
        auto [nR, nC] = X.shape();
        if (axis) {
            VD<double> res(nC);
            for (size_t i{}; i < nC; ++i) {
                auto col = X.col(i).toVector();
                auto m = mean(col);
                res.emplaceBack(m);
            }
            return res;
        } else {
            VD<double> res(nR);
            X.applyRow([&](const auto& row) { res.emplaceBack(mean(row)); });
            return res;
        }
    }

    double var(const VD<double>& v) {
        double res{};
        const auto& m = mean(v);
        v.apply([&](const auto& x) {
            auto val = x - m;
            res += val * val;
        });
        return res / ((double)v.size() - 1);
    }

    VD<double> var(const MD<double>& X, bool axis = true) {
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
                res.emplaceBack(sum / (double)(nC));
                i++;
            });
            return res;
        }
    }

    VD<double> std(const MD<double>& X, bool axis = true) {
        auto std = var(X, axis);
        std.apply([&](auto& x) { x = sqrt(x); });
        return std;
    }

    MD<double> normalize(const MD<double>& X, bool axis = true) {
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
    auto r2_score(const VD<double>& y, const VD<double>& yp) {
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
    auto calculateMSE(const VD<double>& y_true, const VD<double>& y_pred) {
        auto diff = y_true - y_pred;
        double mse = diff.dot(diff);
        return mse / (double)y_true.size();
    }

    struct TrainTestSplit {
        MD<double> XTrain, XTest;
        VD<double> yTrain, yTest;
        vector<size_t> trainIdx, testIdx;
    };

    TrainTestSplit trainTestSplit(const MD<double>& X, const VD<double>& y, double testSize = 0.2,
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

}  // namespace

int main() {
    Seed seed{42};  // Set a seed for reproducibility
    // First 50 rows of the California housing dataset
    MD<double> X{
        {-122.23, 37.88, 41.0, 880.0, 129.0, 322.0, 126.0, 8.3252, 452600.0, 0.0},
        {-122.22, 37.86, 21.0, 7099.0, 1106.0, 2401.0, 1138.0, 8.3014, 358500.0, 0.0},
        {-122.24, 37.85, 52.0, 1467.0, 190.0, 496.0, 177.0, 7.2574, 352100.0, 0.0},
        {-122.25, 37.85, 52.0, 1274.0, 235.0, 558.0, 219.0, 5.6431, 341300.0, 0.0},
        {-122.25, 37.85, 52.0, 1627.0, 280.0, 565.0, 259.0, 3.8462, 342200.0, 0.0},
        {-122.25, 37.85, 52.0, 919.0, 213.0, 413.0, 193.0, 4.0368, 269700.0, 0.0},
        {-122.25, 37.84, 52.0, 2535.0, 489.0, 1094.0, 514.0, 3.6591, 299200.0, 0.0},
        {-122.25, 37.84, 52.0, 3104.0, 687.0, 1157.0, 647.0, 3.12, 241400.0, 0.0},
        {-122.26, 37.84, 42.0, 2555.0, 665.0, 1206.0, 595.0, 2.0804, 226700.0, 0.0},
        {-122.25, 37.84, 52.0, 3549.0, 707.0, 1551.0, 714.0, 3.6912, 261100.0, 0.0},
        {-122.26, 37.85, 52.0, 2202.0, 434.0, 910.0, 402.0, 3.2031, 281500.0, 0.0},
        {-122.26, 37.85, 52.0, 3503.0, 752.0, 1504.0, 734.0, 3.2705, 241800.0, 0.0},
        {-122.26, 37.85, 52.0, 2491.0, 474.0, 1098.0, 468.0, 3.075, 213500.0, 0.0},
        {-122.26, 37.84, 52.0, 696.0, 191.0, 345.0, 174.0, 2.6736, 191300.0, 0.0},
        {-122.26, 37.85, 52.0, 2643.0, 626.0, 1212.0, 620.0, 1.9167, 159200.0, 0.0},
        {-122.26, 37.85, 50.0, 1120.0, 283.0, 697.0, 264.0, 2.125, 140000.0, 0.0},
        {-122.27, 37.85, 52.0, 1966.0, 347.0, 793.0, 331.0, 2.775, 152500.0, 0.0},
        {-122.27, 37.85, 52.0, 1228.0, 293.0, 648.0, 303.0, 2.1202, 155500.0, 0.0},
        {-122.26, 37.84, 50.0, 2239.0, 455.0, 990.0, 419.0, 1.9911, 158700.0, 0.0},
        {-122.27, 37.84, 52.0, 1503.0, 298.0, 690.0, 275.0, 2.6033, 162900.0, 0.0},
        {-122.27, 37.85, 40.0, 751.0, 184.0, 409.0, 166.0, 1.3578, 147500.0, 0.0},
        {-122.27, 37.85, 42.0, 1639.0, 367.0, 929.0, 366.0, 1.7135, 159800.0, 0.0},
        {-122.27, 37.84, 52.0, 2436.0, 541.0, 1015.0, 478.0, 1.725, 113900.0, 0.0},
        {-122.27, 37.84, 52.0, 1688.0, 337.0, 853.0, 325.0, 2.1806, 99700.0, 0.0},
        {-122.27, 37.84, 52.0, 2224.0, 437.0, 1006.0, 422.0, 2.6, 132600.0, 0.0},
        {-122.28, 37.85, 41.0, 535.0, 123.0, 317.0, 119.0, 2.4038, 107500.0, 0.0},
        {-122.28, 37.85, 49.0, 1130.0, 244.0, 607.0, 239.0, 2.4597, 93800.0, 0.0},
        {-122.28, 37.85, 52.0, 1898.0, 421.0, 1102.0, 397.0, 1.808, 105500.0, 0.0},
        {-122.28, 37.84, 50.0, 2082.0, 492.0, 1131.0, 473.0, 1.6424, 108900.0, 0.0},
        {-122.28, 37.84, 52.0, 729.0, 160.0, 395.0, 155.0, 1.6875, 132000.0, 0.0},
        {-122.28, 37.84, 49.0, 1916.0, 447.0, 863.0, 378.0, 1.9274, 122300.0, 0.0},
        {-122.28, 37.84, 52.0, 2153.0, 481.0, 1168.0, 441.0, 1.9615, 115200.0, 0.0},
        {-122.27, 37.84, 48.0, 1922.0, 409.0, 1026.0, 335.0, 1.7969, 110400.0, 0.0},
        {-122.27, 37.83, 49.0, 1655.0, 366.0, 754.0, 329.0, 1.375, 104900.0, 0.0},
        {-122.27, 37.83, 51.0, 2665.0, 574.0, 1258.0, 536.0, 2.7303, 109700.0, 0.0},
        {-122.27, 37.83, 49.0, 1215.0, 282.0, 570.0, 264.0, 1.4861, 97200.0, 0.0},
        {-122.27, 37.83, 48.0, 1798.0, 432.0, 987.0, 374.0, 1.0972, 104500.0, 0.0},
        {-122.28, 37.83, 52.0, 1511.0, 390.0, 901.0, 403.0, 1.4103, 103900.0, 0.0},
        {-122.26, 37.83, 52.0, 1470.0, 330.0, 689.0, 309.0, 3.48, 191400.0, 0.0},
        {-122.26, 37.83, 52.0, 2432.0, 715.0, 1377.0, 696.0, 2.5898, 176000.0, 0.0},
        {-122.26, 37.83, 52.0, 1665.0, 419.0, 946.0, 395.0, 2.0978, 155400.0, 0.0},
        {-122.26, 37.83, 51.0, 936.0, 311.0, 517.0, 249.0, 1.2852, 150000.0, 0.0},
        {-122.26, 37.84, 49.0, 713.0, 202.0, 462.0, 189.0, 1.025, 118800.0, 0.0},
        {-122.26, 37.84, 52.0, 950.0, 202.0, 467.0, 198.0, 3.9643, 188800.0, 0.0},
        {-122.26, 37.83, 52.0, 1443.0, 311.0, 660.0, 292.0, 3.0125, 184400.0, 0.0},
        {-122.26, 37.83, 52.0, 1656.0, 420.0, 718.0, 382.0, 2.6768, 182300.0, 0.0},
        {-122.26, 37.83, 50.0, 1125.0, 322.0, 616.0, 304.0, 2.026, 142500.0, 0.0},
        {-122.27, 37.82, 43.0, 1007.0, 312.0, 558.0, 253.0, 1.7348, 137500.0, 0.0},
        {-122.26, 37.82, 40.0, 624.0, 195.0, 423.0, 160.0, 0.9506, 187500.0, 0.0},
    };

    // Separate the target variable (median_house_value) from the features
    auto y = X.removeCol(8);
    println("Dataset shape: {} rows, {} features", X.numRows(), X.numCols());
    const auto& [XTrain, XTest, yTrain, yTest, trainIdx, testIdx] = trainTestSplit(X, y);

    // Normalize the features
    StandardScaler scale{};
    auto XNorm = scale.fitTransform(XTrain);
    println("Normalized features:\n{}", string(XNorm));

    // Linear Regression
    struct LinearRegression {
        LinearRegression() = default;
        explicit LinearRegression(double alpha, Seed seed = Seed{42}) : alpha{alpha}, seed{seed} {}

        [[nodiscard]] auto predict(const Matrix<double, Dynamic, Dynamic>& X) const {
            // Check if bias term is already included
            if (X.numCols() != theta.size()) {
                auto Xb = matrixOnes<double, Dynamic, Dynamic>(X.numRows(), 1);
                Xb = Xb.augment(X);
                return Xb * theta;
            }
            return X * theta;
        }

        void train(const Matrix<double, Dynamic, Dynamic>& X, const Vector<double, Dynamic>& y, size_t iterations,
                   bool verbose = false) {
            const auto [nR, nC] = X.shape();

            // Add bias term (column of ones)
            auto Xb = matrixOnes<double, Dynamic, Dynamic>(nR, 1);
            Xb = Xb.augment(X);

            // Initialize theta with smaller random values for better convergence
            theta = vectorRandom<double, Dynamic>(Xb.numCols(), -0.1, 0.1, seed);
            auto m = (double)y.size();

            // Track cost history for monitoring convergence
            vector<double> costHistory;
            costHistory.reserve(iterations);

            for (size_t i{}; i < iterations; ++i) {
                auto predictions = predict(Xb);
                auto error = predictions - y;

                // Calculate cost (MSE)
                double cost = calculateMSE(y, predictions);
                costHistory.emplace_back(cost);

                // Gradient descent update
                auto gradients = (1.0 / m) * ((helpers::extractMatrixFromTranspose(Xb.T())) * error);
                theta -= alpha * gradients;

                // Print progress every 5000 iterations
                if (verbose && i % 5000 == 0) {
                    println("Iteration {}, Cost: {:.4f}", i, cost);
                }
            }

            if (verbose) {
                println("Training completed after {} iterations", iterations);
                println("Final cost: {:.4f}", costHistory.back());
                println("Cost reduction: {:.4f}", costHistory[0] - costHistory.back());
            }
        }

        double alpha{0.01};
        VD<double> theta{0};
        Seed seed;  // Seed for reproducibility
    };

    try {
        // Training params
        size_t iterations{100'000};
        double learningRate{0.05};

        println("Training Linear Regression Model...");
        println("Learning rate: {}, Iterations: {}", learningRate, iterations);
        LinearRegression linReg{learningRate, seed};
        linReg.train(XNorm, yTrain, iterations, true);

        println("Model parameters (theta):\n{}", linReg.theta);
        println("\n");

        // Make predictions on the test data
        VD<double> preds{linReg.predict(XNorm)};

        // Train performance
        double r2 = r2_score(yTrain, preds);
        double mse = calculateMSE(yTrain, preds);
        double rmse = sqrt(mse);

        println("=== Train Performance ===");
        println("R-squared (R²): {:.4f}", r2);
        println("Mean Squared Error (MSE): {:.4f}", mse);
        println("Root Mean Squared Error (RMSE): {:.4f}", rmse);
        println("\n");

        // Make predictions on the test data
        const auto& XTestNorm = scale.transform(XTest);
        VD<double> predsTest(linReg.predict(XTestNorm));

        // Test performance
        double r2T = r2_score(yTest, predsTest);
        double mseT = calculateMSE(yTest, predsTest);
        double rmseT = sqrt(mseT);

        println("=== Test Performance ===");
        println("R-squared (R²): {:.4f}", r2T);
        println("Mean Squared Error (MSE): {:.4f}", mseT);
        println("Root Mean Squared Error (RMSE): {:.4f}", rmseT);
        println("\n");

        // Prediction vs Actual
        println("=== Test Predictions vs Actual (First 10 samples) ===");
        println("Predicted\t\tActual\t\t\tError");
        println("--------\t\t------\t\t\t-----");
        for (size_t i{}; i < min(10UL, yTest.size()); i++) {
            double pred = predsTest[i];
            double actual = yTest[i];
            double error = abs(pred - actual);
            println("{:.0f}\t\t\t{:.0f}\t\t\t{:.0f}", pred, actual, error);
        }
    } catch (const std::exception& e) {
        println("Error during training or evaluation: {}", e.what());
        return 1;
    }

    return 0;
}
