#include <array>
#include <chrono>
#include <cmath>
#include <map>
#include <print>

#include "utils.hpp"

using namespace std;
using namespace mlinalg;
using namespace mlinalg::structures;
using namespace utils;

namespace rg = std::ranges;

using M = MD<double>;
using V = VD<double>;

namespace {

    auto exp(const auto& A) {
        auto res{A};
        return res.apply([](auto& x) { x = std::exp(x); });
    }

    auto sigmoid(const V& w, bool derivative = false) {
        auto v{w};
        auto expV = exp(-v);
        auto expVP1{expV};
        expVP1.apply([](auto& x) { x += 1; });

        if (derivative) {
            return expV / (expVP1 * expVP1);
        }
        auto result = expVP1;
        result.apply([](auto& x) { x = 1.0 / x; });
        return result;
    }

    auto softmax(const V& v) {
        auto exps = exp(v - utils::max(v));
        auto sumExps = sum(exps);
        return exps.apply([&](auto& x) { x /= sumExps; });
    }

    auto argmax(const V& v) {
        size_t maxI{0};
        for (size_t i{}; i < v.size(); i++) {
            if (v[i] > v[maxI]) maxI = i;
        }
        return maxI;
    }

    M outerProduct(const V& a, const V& b) {
        if (a.size() == 0 || b.size() == 0) {
            throw std::invalid_argument("Vectors cannot be empty");
        }

        M result(a.size(), b.size());
        for (size_t i = 0; i < a.size(); ++i) {
            for (size_t j = 0; j < b.size(); ++j) {
                result.at(i, j) = a.at(i) * b.at(j);
            }
        }
        return result;
    }

    auto toCategorical(const V& y) {
        auto nClasses = rg::max(y) + 1;
        auto cat = matrixZeros<double, Dynamic, Dynamic>(y.size(), (size_t)nClasses);
        for (size_t i{}; i < y.size(); ++i) {
            cat[i][(size_t)y[i]] = 1;
        }
        return cat;
    }
}  // namespace

// Simple Feed Forward Neural Network with Mlinalg
class NeuralNetwork {
   public:
    using Sizes = array<size_t, 4>;
    using Weights = map<string, M>;
    using Activations = map<string, V>;

    explicit NeuralNetwork(Sizes sizes, size_t epochs = 10, double lr = 0.001, Seed seed = 42)
        : sizes_{sizes}, Ws_{init(seed)}, epochs_{epochs}, lr_{lr} {}

    [[nodiscard]] Weights Ws() const { return Ws_; }

    auto forwardPass(const V& XTrain) {
        if (XTrain.size() != Ws_["W1"].numCols())
            throw std::invalid_argument("Input size does not match the first layer size.");

        acts_["A0"] = XTrain;

        acts_["Z1"] = Ws_["W1"] * acts_["A0"];
        acts_["A1"] = sigmoid(acts_["Z1"]);

        acts_["Z2"] = Ws_["W2"] * acts_["A1"];
        acts_["A2"] = sigmoid(acts_["Z2"]);

        acts_["Z3"] = Ws_["W3"] * acts_["A2"];
        acts_["A3"] = softmax(acts_["Z3"]);

        return acts_["A3"];
    }

    auto backwardsPass(const V& yTrain, const V& output) {
        Weights changeW{};

        const auto& error3 = output - yTrain;
        changeW["W3"] = outerProduct(error3, acts_["A2"]);

        const auto elemMult = [&](auto& x, const auto& y) { x *= y; };

        const auto& sigActsZ2 = sigmoid(acts_["Z2"], true);
        auto err2Applicand = extractMatrixFromTranspose(Ws_["W3"].T()) * error3;
        const auto& error2 = err2Applicand.apply(sigActsZ2, elemMult);
        changeW["W2"] = outerProduct(error2, acts_["A1"]);

        const auto& sigActsZ1 = sigmoid(acts_["Z1"], true);
        auto err1Applicand = extractMatrixFromTranspose(Ws_["W2"].T()) * error2;
        const auto& error1 = err1Applicand.apply(sigActsZ1, elemMult);
        changeW["W1"] = outerProduct(error1, acts_["A0"]);

        return changeW;
    }

    double computeAccuracy(const M& XVal, const M& yVal) {
        V preds;
        preds.reserve(yVal.numRows());
        auto XItr = XVal.begin();
        auto yItr = yVal.begin();
        for (; XItr != XVal.end() && yItr != yVal.end(); ++XItr, ++yItr) {
            const auto& output = forwardPass(*XItr);
            auto pred = argmax(output);
            auto actual = argmax(*yItr);
            preds.pushBack(static_cast<double>(pred == actual));
        }
        return sum(preds) / preds.size();
    }

    void updateParams(const Weights& wChanges) {
        for (const auto& [k, changeMatrix] : wChanges) {
            Ws_[k] = Ws_[k] - (changeMatrix * lr_);
        }
    }

    auto train(const M& XTrain, const M& yTrain, const M& XVal, const M& yVal) {
        auto t0 = chrono::steady_clock::now();
        auto printInterval = std::min(1000UZ, epochs_ / 10);
        bool needsNewline{};
        for (size_t i{}; i < epochs_; ++i) {
            auto XItr = XTrain.begin();
            auto yItr = yTrain.begin();
            for (; XItr != XTrain.end() && yItr != yTrain.end(); ++XItr, ++yItr) {
                const auto& x = *XItr;
                const auto& y = *yItr;
                const auto& output = forwardPass(x);
                const auto& wChanges = backwardsPass(y, output);
                updateParams(wChanges);
            }
            auto accuracy = computeAccuracy(XVal, yVal);
            auto elapsed = chrono::duration_cast<chrono::seconds>(chrono::steady_clock::now() - t0);

            if (i % printInterval == 0) {
                if (needsNewline) {
                    println();  // Finish the carriage return line first
                }
                println("Epoch: {:6}/{}, Time: {:3}s, Accuracy: {:.5f}", i + 1, epochs_, elapsed.count(), accuracy);
                needsNewline = false;
            } else {
                print("\rEpoch: {:6}/{}, Time: {:3}s, Accuracy: {:.5f}", i + 1, epochs_, elapsed.count(), accuracy);
                cout.flush();
                needsNewline = true;
            }
        }
        if (needsNewline) println();
    }

    auto predict(const M& X) {
        V preds;
        preds.reserve(X.numRows());
        for (const auto& inputs : X) {
            const auto& output = forwardPass(inputs);
            auto pred = argmax(output);
            preds.pushBack(pred);
        }
        return preds;
    }

   private:
    Weights init(Seed seed) {
        auto inputLayer = sizes_.at(0);
        auto hidden1 = sizes_.at(1);
        auto hidden2 = sizes_.at(2);
        auto outputLayer = sizes_.at(3);

        Weights params{
            {"W1", matrixRandom<double, M::rows, M::cols>(hidden1, inputLayer, 0, 1, seed) * sqrt(1. / hidden1)},
            {"W2", matrixRandom<double, M::rows, M::cols>(hidden2, hidden1, 0, 1, seed) * sqrt(1. / hidden2)},
            {"W3", matrixRandom<double, M::rows, M::cols>(outputLayer, hidden2, 0, 1, seed) * sqrt(1. / outputLayer)},
        };
        return params;
    }

    Sizes sizes_{};
    Weights Ws_;
    Activations acts_;
    size_t epochs_;
    double lr_;
};

int main() {
    Seed seed{42};

    try {
        using IrisRowType = tuple<double, double, double, double, string>;
        using IrisHeaderType = tuple<string, string, string, string, string>;
        vector<string> catCols{};
        vector<double> values{};
        IrisRowType row;

        utils::Dataset dataset;
        try {
            auto headersOpt = utils::readCSV<5, false, IrisHeaderType, IrisRowType>("./src/examples/datasets/iris.csv",
                                                                                    row, values, catCols);
            dataset = utils::fillDatasetMatrix<4, 1>(values, catCols);
        } catch (const std::exception& e) {
            println("Error reading CSV file: {}", e.what());
            return 1;
        }

        auto& X = dataset.values;
        auto y = toCategorical(X.removeCol(X.numCols() - 1));
        println("Dataset shape: {} rows, {} features", X.numRows(), X.numCols());
        const auto& [XTrain, XVal, yTrain, yVal, trainIdx, valIdx] = utils::trainTestSplit(X, y);

        size_t iterations{5'000};
        double learningRate{0.1};
        array<size_t, 4> layerSizes{X.numCols(), 12, 10, 3};
        println("Training Neural Network...");
        println("Learning rate: {}, Iterations: {}, Layer sizes: {}", learningRate, iterations, layerSizes);
        NeuralNetwork nn{layerSizes, iterations, learningRate, seed};
        nn.train(XTrain, yTrain, XVal, yVal);

        println("\n");

        // Make predictions on the train data
        auto preds{nn.predict(XTrain)};

        // Train performance
        auto accuracy = nn.computeAccuracy(XTrain, yTrain) * 100;

        println("=== Train Performance ===");
        println("Accuracy: {:.4f}", accuracy);
        println("\n");

        // Make predictions on the test data
        auto predsTest(nn.predict(XVal));

        // Test performance
        auto accuracyT = nn.computeAccuracy(XVal, yVal) * 100;

        println("=== Test Performance ===");
        println("Accuracy: {:.4f}", accuracyT);
        println("\n");

        // Prediction vs Actual
        println("=== Test Predictions vs Actual (First 10 samples) ===");
        println("Predicted\t\tActual");
        println("--------\t\t------");
        for (size_t i{}; i < min(10UL, yVal.size()); i++) {
            double pred = predsTest[i];
            double actual = argmax(yVal[i]);
            println("{:.0f}\t\t\t{:.0f}", pred, actual);
        }
    } catch (const std::exception& e) {
        println("An error occurred: {}", e.what());
        return 1;
    }

    return 0;
}
