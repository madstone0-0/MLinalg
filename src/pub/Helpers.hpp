/**
 * @file Helpers.hpp
 * @brief Helper functions for structures
 */

#pragma once
#include <iterator>
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
        StackError(const std::string& msg, const boost::stacktrace::stacktrace& st = boost::stacktrace::stacktrace())
            : E(
#ifdef DEBUG
                  makeMsg(msg, st)
#else
                  msg
#endif
              ) {
        }

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

    template <Number num>
    num rng(int min, int max, std::optional<size_t> seed = std::nullopt)
        requires(std::is_integral_v<num>)
    {
        static std::mt19937 generator(seed.value_or(std::random_device{}()));
        std::uniform_int_distribution<num> distribution(min, max);
        return distribution(generator);
    }

    template <Number num>
    num rng(double min, double max, std::optional<size_t> seed = std::nullopt)
        requires(!std::is_integral_v<num>)
    {
        static std::mt19937 generator(seed.value_or(std::random_device{}()));
        std::uniform_real_distribution<num> distribution(min, max);
        return distribution(generator);
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
    auto extractMatrixFromTranspose(const TransposeVariant<num, m, n> T) -> MatrixTransposeVariant<num, m, n> {
        return std::get<MatrixTransposeVariant<num, m, n>>(T);
    }

    /**
     * @brief Extract a vector from a TransposeVariant
     *
     * @param T TransposeVariant
     * @return Vector<number, m>
     */
    template <Number num, int m, int n>
    auto extractVectorFromTranspose(const TransposeVariant<num, m, n> T) -> VectorTransposeVariant<num, m, n> {
        return std::get<VectorTransposeVariant<num, m, n>>(T);
    }

    /**
     * @brief Check if a TransposeVariant contains a MatrixTransposeVariant
     *
     * @param T the transpose variant to check
     * @return true if it contains a VectorTransposeVariant, false otherwise
     */
    template <Number num, int m, int n>
    bool containsVectorVariant(const TransposeVariant<num, m, n>& T) {
        return std::holds_alternative<VectorTransposeVariant<num, m, n>>(T);
    }

    /**
     * @brief Convert a matrix to a dynamic matrix
     *
     * @param matrix Compile time matrix
     * @return Matrix<number, Dynamic, Dynamic>
     */
    template <Number num, int m, int n>
    Matrix<num, Dynamic, Dynamic> toDynamic(const Matrix<num, m, n> matrix) {
        const auto& dyna = Matrix<num, Dynamic, Dynamic>{matrix};
        return dyna;
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

    template <Number number, int m, int n>
    void copyVectorIntoMatrixCol(Matrix<number, m, n>& A, const Vector<number, m>& v, size_t j) {
        const auto& [nR, nC] = A.shape();
        if (j >= nC) throw StackError<std::out_of_range>{"Column index out of range"};

        for (size_t i{}; i < nR; ++i) A(i, j) = v[i];
    }

    template <Number number, int m, int n>
    void copyVectorIntoMatrixRow(Matrix<number, m, n>& A, const Vector<number, n>& v, size_t i) {
        const auto& [nR, nC] = A.shape();
        if (i >= nR) throw StackError<std::out_of_range>{"Row index out of range"};

        for (size_t j{}; j < nC; ++j) A(i, j) = v[j];
    }

    template <Number number, int m, int n>
    Matrix<number, std::max(m, n), std::max(m, n)> padMatrixToSquare(const Matrix<number, m, n>& A,
                                                                     number a = number(0)) {
        if constexpr (m == n) return A;

        const auto& [nR, nC] = A.shape();
        if constexpr (m == Dynamic || n == Dynamic) {
            if (nR == nC) return A;
        }

        constexpr auto nM = std::max(m, n);
        auto nD = std::max(nR, nC);

        Matrix<number, nM, nM> res(nM, nM);
        for (size_t i{}; i < nD; ++i) {
            for (size_t j{}; j < nD; ++j) {
                if (i < nR && j < nC) {
                    res(i, j) = A(i, j);
                } else {
                    res(i, j) = a;
                }
            }
        }
        return res;
    }

    // Adapted from https://stackoverflow.com/a/17074810
    template <typename Itr, typename Compare = std::less<>>
    vector<size_t> sortPermutation(Itr beg, Itr end, Compare cmp = Compare()) {
        vector<size_t> res(std::distance(beg, end));
        rg::iota(res, 0);
        std::sort(res.begin(), res.end(), [&](size_t i, size_t j) { return cmp(*(beg + i), *(beg + j)); });
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

    template <Number number, int m, int n>
    std::string formatMatrix(const Matrix<number, m, n>& A) {
        const auto& [nR, nC] = A.shape();
        return (std::stringstream{} << A).str();
    }

    template <Container T>
    auto unwrap(T&& c) {
        return [](auto&& x) -> decltype(auto) {
            if constexpr (std::is_pointer_v<std::decay_t<decltype(x)>>) {
                return *x;
            } else {
                return x;
            }
        }(std::forward<T>(c));
    }

}  // namespace mlinalg::structures::helpers
