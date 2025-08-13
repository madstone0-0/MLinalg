#include <algorithm>
#include <cmath>
#include <mlinalg/MLinalg.hpp>
#include <print>

#include "utils.hpp"

using namespace std;
using namespace mlinalg;
using namespace mlinalg::structures;

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
            double cost = utils::calculateMSE(y, predictions);
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
    const auto& [XTrain, XTest, yTrain, yTest, trainIdx, testIdx] = utils::trainTestSplit(X, y);

    // Normalize the features
    utils::StandardScaler scale{};
    auto XNorm = scale.fitTransform(XTrain);
    println("Normalized features:\n{}", string(XNorm));

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
        double r2 = utils::r2_score(yTrain, preds);
        double mse = utils::calculateMSE(yTrain, preds);
        double rmse = std::sqrt(mse);

        println("=== Train Performance ===");
        println("R-squared (R²): {:.4f}", r2);
        println("Mean Squared Error (MSE): {:.4f}", mse);
        println("Root Mean Squared Error (RMSE): {:.4f}", rmse);
        println("\n");

        // Make predictions on the test data
        const auto& XTestNorm = scale.transform(XTest);
        VD<double> predsTest(linReg.predict(XTestNorm));

        // Test performance
        double r2T = utils::r2_score(yTest, predsTest);
        double mseT = utils::calculateMSE(yTest, predsTest);
        double rmseT = std::sqrt(mseT);

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
