#pragma once
#include <mlinalg/MLinalg.hpp>
#include <mlinalg/structures/Aliases.hpp>
#include <optional>
#include <string>

#include "third_party/csv.h"

// Utility functions not found in MLinalg
namespace utils {
    using namespace std;
    using namespace mlinalg;
    namespace rg = std::ranges;

    /* Taken from Hands On Machine Learning with C++, Kolodiazhnyi, 2020, Section 1.2.2 */

    using DefaultRowType = tuple<double, double, double, double, double, double>;
    using DefaultHeaderType = tuple<string, string, string, string, string, string>;

    template <size_t... Idx, typename T, typename R>
    bool readRowHelp(index_sequence<Idx...> /*idx*/, T& row, R& r) {
        return r.read_row(get<Idx>(row)...);
    }

    template <typename T>
    void processElement(const T& element, vector<double>& data, vector<string>& catCols) {
        if constexpr (std::is_same_v<T, double>) {
            data.push_back(element);
        } else if constexpr (std::is_same_v<T, string>) {
            catCols.push_back(element);
        }
    }

    // Unpack all elements of the tuple and process each one.
    template <size_t... Idx, typename T>
    void fillValues(index_sequence<Idx...> /*unused*/, T& row, vector<double>& data, vector<std::string>& catCols) {
        (processElement(get<Idx>(row), data, catCols), ...);
    }

    template <size_t nCols, bool hasHeader = true, typename Header = DefaultHeaderType, typename Row = DefaultRowType,
              typename T, typename Quant, typename Cat>
    optional<Header> readCSV(const string& path, T& row, Quant& values, Cat& catCols) {
        try {
            io::CSVReader<nCols> reader(path);
            Header header;
            if constexpr (hasHeader) {
                // Skip header
                readRowHelp(make_index_sequence<tuple_size<Header>::value>{}, header, reader);

                // array<string, nCols> header;
                // for (size_t i{}; i < nCols; i++) header.at(i) = "C" + to_string(i);
                // apply([&reader](auto&&... args) { reader.read_header(io::ignore_missing_column, args...); }, header);
            }

            bool done = false;
            while (!done) {
                done = !readRowHelp(make_index_sequence<tuple_size<Row>::value>{}, row, reader);
                if (!done) {
                    fillValues(make_index_sequence<nCols>{}, row, values, catCols);
                }
            }

            if (hasHeader)
                return header;
            else
                return nullopt;
        } catch (const io::error::no_digit& err) {
            cerr << err.what() << '\n';
        }
        return nullopt;
    }
    /* Taken from Hands On Machine Learning with C++, Kolodiazhnyi, 2020, Section 1.2.2 */

    using Labels = unordered_map<string, size_t>;

    inline pair<Labels, VD<double>> labelEncode(const vector<std::string>& labels) {
        using namespace std;
        Labels labelMap{};
        double i{};
        for (const auto& label : labels) {
            if (labelMap.contains(label)) continue;
            labelMap.insert({label, i++});
        }
        VD<double> encodedLabels{};
        encodedLabels.reserve(labels.size());
        for (const auto& label : labels) {
            encodedLabels.pushBack(labelMap[label]);
        }
        return {labelMap, encodedLabels};
    }

    struct Dataset {
        MD<double> values;
        map<string, Labels> labelMap;
    };

    template <size_t QuantCols, size_t CatCols, typename Header = DefaultHeaderType, typename Quant, typename Cat>
    inline Dataset fillDatasetMatrix(const Quant& quant, const Cat& cat) {
        constexpr auto TotalCols = QuantCols + CatCols;
        const auto totalSize = quant.size() + cat.size();
        const auto totalRows = totalSize / TotalCols;
        Dataset res{};
        // res.values.reserve(totalRows, TotalCols);

        optional<MD<double>> allEncodedLabelsOpt{nullopt};
        if constexpr (CatCols > 0) {
            vector<vector<string>> catRows{};
            for (size_t i{}; i < cat.size(); i += CatCols) {
                catRows.emplace_back(cat.begin() + i, cat.begin() + (i + CatCols));
            }

            const auto catCols{catRows[0].size()};
            MD<double> allEncodedLabels(catRows.size(), catCols);

            for (size_t i{}; i < catCols; i++) {
                vector<string> labels{};
                labels.reserve(catCols);
                for (size_t j{}; j < catRows.size(); j++) {
                    labels.push_back(catRows.at(j).at(i));
                }
                const auto& [labelMap, encodedLabels] = labelEncode(labels);
                res.labelMap[format("cat_{}", i + 1)] = labelMap;
                for (size_t j{}; j < catRows.size(); j++) {
                    allEncodedLabels(j, i) = encodedLabels[j];
                }
            }
            allEncodedLabelsOpt = allEncodedLabels;
        }

        MD<double> quantVals(quant.size() / QuantCols, QuantCols);
        for (size_t i{}; i < quantVals.numRows(); i++) {
            quantVals[i] = VD<double>(quant.begin() + (i * QuantCols), quant.begin() + ((i + 1) * QuantCols));
        }

        if (allEncodedLabelsOpt.has_value()) {
            res.values = allEncodedLabelsOpt.value().augment(quantVals);
        } else {
            res.values = quantVals;
        }
        return res;
    }

    inline VD<double> max(const VD<double>& v) {
        double max{-1};
        v.apply([&](const auto& x) { max = std::max(max, x); });
        VD<double> res(v.size());
        res.apply([&](auto& x) { x = max; });
        return res;
    }

    inline MD<double> max(const MD<double>& A) {
        const auto [nR, nC] = A.shape();
        double max{-1};
        A.apply([&](const auto& x) { max = std::max(max, x); });
        MD<double> res(nR, nC);
        res.apply([&](auto& x) { x = max; });
        return res;
    }

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

    inline double sum(const VD<double>& v) {
        double res{};
        v.apply([&](const auto& x) { res += x; });
        return res;
    }

    inline VD<double> sum(const MD<double>& X, bool axis = true) {
        auto [nR, nC] = X.shape();
        if (axis) {
            VD<double> res(nC);
            for (size_t i{}; i < nC; ++i) res[i] = sum(X.col(i).toVector());
            return res;
        } else {
            VD<double> res(nR);
            for (size_t i{}; i < nR; ++i) res[i] = sum(X[i]);
            return res;
        }
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

    using Indices = vector<size_t>;

    template <typename Features, typename Target>
    struct TrainTestSplit {
        Features XTrain, XTest;
        Target yTrain, yTest;
        Indices trainIdx, testIdx;
    };

    template <typename Features, typename Target>
    tuple<size_t, size_t, Indices, Indices> shuffleIndices(const Features& X, const Target& y, double testSize,
                                                           Seed seed) {
        size_t samples = y.size();
        auto testSample = static_cast<size_t>(samples * testSize);
        size_t trainSample = samples - testSample;

        vector<size_t> idx(samples);
        rg::iota(idx, 0);

        mt19937 gen(seed.value());
        rg::shuffle(idx, gen);

        Indices trainIdx{idx.begin(), idx.begin() + trainSample};
        Indices testIdx{idx.begin() + trainSample, idx.end()};
        return {trainSample, testSample, trainIdx, testIdx};
    }

    inline auto trainTestSplit(const MD<double>& X, const VD<double>& y, double testSize = 0.2, Seed seed = Seed{42})
        -> TrainTestSplit<MD<double>, VD<double>> {
        if (X.numRows() != y.size()) throw std::invalid_argument{"X and y must have the same number of samples"};

        if (testSize <= 0.0 || testSize >= 1.0) throw std::invalid_argument{"testSize must be between 0.0 and 1.0"};

        auto [trainSample, testSample, trainIdx, testIdx] = shuffleIndices(X, y, testSize, seed);

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

        return {.XTrain = std::move(XTrain),
                .XTest = std::move(XTest),
                .yTrain = std::move(yTrain),
                .yTest = std::move(yTest),
                .trainIdx = std::move(trainIdx),
                .testIdx = std::move(testIdx)};
    }

    inline auto trainTestSplit(const MD<double>& X, const MD<double>& y, double testSize = 0.2, Seed seed = Seed{42})
        -> TrainTestSplit<MD<double>, MD<double>> {
        if (X.numRows() != y.size()) throw std::invalid_argument{"X and y must have the same number of samples"};

        if (testSize <= 0.0 || testSize >= 1.0) throw std::invalid_argument{"testSize must be between 0.0 and 1.0"};

        auto [trainSample, testSample, trainIdx, testIdx] = shuffleIndices(X, y, testSize, seed);

        MD<double> XTrain(trainSample, X.numCols());
        MD<double> XTest(testSample, X.numCols());
        MD<double> yTrain(trainSample, y.numCols());
        MD<double> yTest(testSample, y.numCols());

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

    // Taken from https://stackoverflow.com/a/37094024
    template <size_t, class T>
    using T_ = T;

    template <class T, size_t... Is>
    auto gen(std::index_sequence<Is...>) {
        return std::tuple<T_<Is, T>...>{};
    }

    template <class T, size_t N>
    auto gen() {
        return gen<T>(std::make_index_sequence<N>{});
    }
}  // namespace utils
