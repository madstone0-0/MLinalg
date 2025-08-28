#include <algorithm>
#include <cmath>
#include <print>

#include "../pub/MLinalg.hpp"
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
               bool verbose = false, float tol = 1e-7) {
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

            // Early stop
            if (i > 0 && std::abs(costHistory[i] - costHistory[i - 1]) < tol) {
                if (verbose) {
                    println("Early stopping at iteration {} due to minimal cost change.", i);
                }
                break;
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

    using HousingRowType = tuple<double, double, double, double, double, double, double, double, string, double>;
    using HousingHeaderType = tuple<  //
        string,                       // Longitude
        string,                       // Latitude
        string,                       // Housing Median Age
        string,                       // Total Rooms
        string,                       // Total Bedrooms
        string,                       // Population
        string,                       // Households
        string,                       // Median Income
        string,                       // Ocean Proximity
        string                        // Median House Value
        >;
    vector<string> catCols{};
    vector<double> values{};
    HousingRowType row;

    utils::Dataset dataset;
    try {
        auto headersOpt = utils::readCSV<10, true, HousingHeaderType, HousingRowType>(
            "./src/examples/datasets/housing.csv", row, values, catCols);
        dataset = utils::fillDatasetMatrix<9, 1>(values, catCols);
    } catch (const std::exception& e) {
        println("Error reading CSV file: {}", e.what());
        return 1;
    }

    auto& X = dataset.values;
    // Separate the target variable (median_house_value) from the features
    auto y = X.removeCol(X.numCols() - 1);
    println("Dataset shape: {} rows, {} features", X.numRows(), X.numCols());
    const auto& [XTrain, XTest, yTrain, yTest, trainIdx, testIdx] = utils::trainTestSplit(X, y);

    // Normalize the features
    utils::StandardScaler scale{};
    auto XNorm = scale.fitTransform(XTrain);
    println("Normalized features:\n{}", string(XNorm.view({.start = 0, .end = std::min((int)XNorm.numRows(), 4)},
                                                          {.start = 0, .end = std::min((int)XNorm.numCols(), 4)})));

    try {
        // Training params
        size_t iterations{100'000};
        double learningRate{0.05};

        println("Training Linear Regression Model...");
        println("Learning rate: {}, Iterations: {}", learningRate, iterations);
        LinearRegression linReg{learningRate, seed};
        linReg.train(XNorm, yTrain, iterations, true);

        println("First 3 Model parameters (theta):\n{}", string(linReg.theta.view(0, 3)));
        println("\n");

        // Make predictions on the train data
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
