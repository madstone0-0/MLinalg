/**
 * @file Helpers.hpp
 * @brief Helper functions for structures
 */

#pragma once
#define BOOST_STACKTRACE_USE_ADDR2LINE
#include <algorithm>
#include <boost/stacktrace.hpp>
#include <concepts>
#include <exception>
#include <functional>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

#include "Concepts.hpp"
#include "structures/Aliases.hpp"

namespace mlinalg::stacktrace {

    template <typename E>
    concept Exception = requires {
        std::derived_from<E, std::exception>;
        std::constructible_from<E, std::string>;
    };

    template <Exception E = std::runtime_error>
    class StackError : public E {
       public:
#ifdef DEBUG
        StackError(const std::string& msg, const boost::stacktrace::stacktrace& st = boost::stacktrace::stacktrace())
            : E(makeMsg(msg, st)) {}
#else
        StackError(const std::string& msg) : E(msg) {}
#endif  // DEBUG

       private:
        std::string makeMsg(const std::string& msg, const boost::stacktrace::stacktrace& st) {
            std::ostringstream oss;
            oss << msg << "\n" << st;
            return oss.str();
        }
    };
}  // namespace mlinalg::stacktrace

namespace mlinalg::structures::helpers {
    using namespace mlinalg::stacktrace;
    using std::vector;
    namespace rg = std::ranges;

    template <Number number, int m, int n>
    using TransposeVariant = std::variant<Vector<number, m>, Matrix<number, n, m>>;

    template <Number num>
    num rng(int min, int max, std::optional<size_t> seed = std::nullopt) {
        static std::mt19937 generator(seed.value_or(std::random_device{}()));
        if constexpr (std::is_integral_v<num>) {
            std::uniform_int_distribution<num> distribution(min, max);
            return distribution(generator);
        } else {
            std::uniform_real_distribution<num> distribution(min, max);
            return distribution(generator);
        }
    }

    /**
     * @brief Generate a matrix from a vector of column vectors
     *
     * @param vecSet Vector of column vectors
     * @return  Matrix<number, m, n>
     */
    template <Number num, int m, int n>
    Matrix<num, m, n> fromColVectorSet(const vector<Vector<num, m>>& vecSet) {
        if constexpr (m == Dynamic || n == Dynamic) {
            const size_t& nRows{vecSet.size()};
            const size_t& nCols{vecSet.at(0).size()};
            Matrix<num, Dynamic, Dynamic> res(nRows, nCols);
            for (size_t i{}; i < nCols; i++) {
                const auto& vec{vecSet.at(i)};
                for (size_t j{}; j < nRows; j++) {
                    res.at(j, i) = vec.at(j);
                }
            }
            return res;
        } else {
            Matrix<num, m, n> res;
            for (size_t i{}; i < n; i++) {
                const auto& vec{vecSet.at(i)};
                for (size_t j{}; j < m; j++) {
                    res.at(j, i) = vec.at(j);
                }
            }
            return res;
        }
    }

    /**
     * @brief Generate a matrix from a vector of row vectors
     *
     * @param vecSet  Vector of row vectors
     * @return  Matrix<number, m, n>
     */
    template <Number num, int m, int n>
    Matrix<num, m, n> fromRowVectorSet(const vector<Vector<num, n>>& vecSet) {
        Matrix<num, m, n> res;
        for (int i{}; i < m; i++) {
            res.at(i) = vecSet.at(i);
        }
        return res;
    }

    /**
     * @brief Extract a matrix from a TransposeVariant
     *
     * @param T  TransposeVariant
     * @return Matrix<number, n, m>
     */
    template <Number num, int m, int n>
    Matrix<num, n, m> extractMatrixFromTranspose(const TransposeVariant<num, m, n> T) {
        return std::get<Matrix<num, n, m>>(T);
    }

    /**
     * @brief Extract a vector from a TransposeVariant
     *
     * @param T TransposeVariant
     * @return Vector<number, m>
     */
    template <Number num, int m, int n>
    Vector<num, m> extractVectorFromTranspose(const TransposeVariant<num, m, n> T) {
        return std::get<Vector<num, m>>(T);
    }

    /**
     * @brief Convert a matrix to a dynamic matrix
     *
     * @param matrix Compile time matrix
     * @return Matrix<number, Dynamic, Dynamic>
     */
    template <Number num, int m, int n>
    Matrix<num, Dynamic, Dynamic> toDynamic(const Matrix<num, m, n> matrix) {
        return Matrix<num, Dynamic, Dynamic>{matrix};
    }

    template <Number to, Number from, int m, int n>
    Matrix<to, m, n> cast(const Matrix<from, m, n>& A) {
        if constexpr (!std::is_convertible_v<from, to>) {
            throw StackError<std::invalid_argument>{"Cannot convert to type"};
        }
        auto [nR, nC] = A.shape();
        Matrix<to, m, n> M(nR, nC);
        for (size_t i{}; i < nR; i++)
            for (size_t j{}; j < nC; j++) M(i, j) = to(A(i, j));
        return M;
    }

    template <Number to, Number from, int m>
    Vector<to, m> cast(const Vector<from, m>& v) {
        if constexpr (!std::is_convertible_v<from, to>) {
            throw StackError<std::invalid_argument>{"Cannot convert to type"};
        }
        auto size = v.size();
        Vector<to, m> w(size);
        for (size_t i{}; i < size; i++) w[i] = to(v[i]);
        return v;
    }

    // https://stackoverflow.com/a/17074810
    template <typename T, typename Compare = std::less<>>
    vector<size_t> sortPermutation(const vector<T>& vec, Compare cmp = Compare()) {
        vector<size_t> res(vec.size());
        std::iota(res.begin(), res.end(), 0);
        std::sort(res.begin(), res.end(), [&vec, &cmp](size_t i, size_t j) { return cmp(vec[i], vec[j]); });
        return res;
    }

    template <typename T>
    void applySortPermutation(vector<T>& vec, const vector<size_t>& permutation) {
        vector<bool> done(vec.size());
        for (size_t i{}; i < vec.size(); i++) {
            if (done[i]) continue;
            done[i] = true;
            size_t prevJ{i};
            size_t j{permutation[i]};
            while (i != j) {
                std::swap(vec[prevJ], vec[j]);
                done[j] = true;
                prevJ = j;
                j = permutation[j];
            }
        }
    }

}  // namespace mlinalg::structures::helpers
