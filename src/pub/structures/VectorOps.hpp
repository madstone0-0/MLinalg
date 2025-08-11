/**
 * @file VectorOps.hpp
 * @brief Header file for vector operations
 */

#pragma once

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

#include "../Concepts.hpp"
#include "../Helpers.hpp"
#include "../Numeric.hpp"
#include "Aliases.hpp"

namespace mlinalg::structures {
    using namespace mlinalg::stacktrace;

    template <typename D, Number number>
    class VectorBase;

    template <Number number, int n>
    class Vector;

    template <Number number, int m, int n>
    class Matrix;

    template <Container T, Container U>
    inline void checkOperandSize(const T& row, const U& otherRow) {
        if (row.size() != otherRow.size()) throw StackError<std::invalid_argument>("Vectors must be of the same size");
    }

    template <Container T, Container U>
    inline bool vectorEqual(const T& row, const U& otherRow) {
        checkOperandSize(row, otherRow);
        auto n = row.size();
        for (size_t i{}; i < n; i++)
            if (!fuzzyCompare(row[i], otherRow[i])) return false;
        return true;
    }

    template <Number number, Container T>
    inline number& vectorAt(T& row, size_t i) {
        return row.at(i);
    }

    template <Number number, Container T>
    inline const number& vectorConstAt(const T& row, size_t i) {
        return row.at(i);
    }

    template <typename F, Container T>
    inline void vectorApply(T& row, F f) {
        for (auto& x : row) f(x);
    }

    template <typename F, Container T>
    inline void vectorApply(const T& row, F f) {
        for (auto& x : row) f(x);
    }

    template <typename F, Container T, Container U, bool checkSizes = false>
    inline void vectorApply(T& row, const U& otherRow, F f) {
        const auto n = row.size();
        const auto otherN = otherRow.size();
        if constexpr (checkSizes) {
            if (n != otherN)
                throw StackError<std::invalid_argument>("Vectors must be of the same size for vectorApply");
        }
        auto i = row.begin();
        auto j = otherRow.begin();
        for (; i != row.end(); ++i, ++j) f(*i, *j);
    }

    template <typename F, Container T, Container U, bool checkSizes = false>
    inline void vectorApply(const T& row, const U& otherRow, F f) {
        const auto n = row.size();
        const auto otherN = otherRow.size();
        if constexpr (checkSizes) {
            if (n != otherN)
                throw StackError<std::invalid_argument>("Vectors must be of the same size for vectorApply");
        }
        auto i = row.begin();
        auto j = otherRow.begin();
        for (; i != row.end(); ++i, ++j) f(*i, *j);
    }

    template <Number number, int n, Container T>
    inline Vector<number, n> vectorNeg(const T& row) {
        constexpr int vSize = (n != Dynamic) ? n : Dynamic;
        auto size = row.size();
        Vector<number, vSize> res(size);
        for (size_t i{}; i < size; i++) res[i] = -row[i];
        return res;
    }

    template <Number number, Container T>
    inline void vectorNeg(T& row) {
        auto size = row.size();
        for (size_t i{}; i < size; i++) row[i] = -row[i];
    }

    template <Number number, Container T, Container U>
    inline void vectorAddI(T& row, const U& otherRow) {
        checkOperandSize(row, otherRow);
        vectorApply(row, otherRow, [&](auto& x, const auto& y) { x += y; });
    }

    template <Number number, Container T, Container U>
    inline void vectorSubI(T& row, const U& otherRow) {
        checkOperandSize(row, otherRow);
        vectorApply(row, otherRow, [&](auto& x, const auto& y) { x -= y; });
    }

    template <Number number, Container T>
    inline void vectorScalarMultI(T& row, const number& scalar) {
        vectorApply(row, [&](auto& x) { x *= scalar; });
    }

    template <Number number, Container T>
    inline void vectorScalarDivI(T& row, const number& scalar) {
        if (fuzzyCompare(scalar, number(0))) throw StackError<std::domain_error>("Division by zero");
        vectorApply(row, [&](auto& x) { x /= scalar; });
    }

    template <Number number, typename D, typename OtherD>
    inline number vectorVectorMult(const VectorBase<D, number>& vec, const VectorBase<OtherD, number>& otherVec) {
        if (vec.size() != otherVec.size()) throw StackError<std::invalid_argument>("Vectors must be of the same size");

        number sum{0};
        for (size_t i{}; i < vec.size(); i++) {
            sum += vec[i] * otherVec[i];
        }
        return sum;
    }

    template <Container T>
    inline std::string vectorStringRepr(const T& row) {
        const auto size = row.size();
        std::stringstream ss{};

        if (size == 1)
            ss << "[ " << row[0] << " ]\n";
        else {
            ss << '[';
            for (size_t i{}; i < size; i++) ss << row[i] << (i != (size - 1) ? ", " : "");
            ss << "]\n";
        }
        return ss.str();
    }

    template <Container T>
    inline std::ostream& vectorOptionalRepr(std::ostream& os, const T& row) {
        const auto& size = row.size();

        auto hasVal = [](auto rowVal) {
            std::stringstream val;
            if (rowVal.has_value())
                val << rowVal.value();
            else
                val << "None";
            return val.str();
        };

        if (size == 1) {
            os << "[ " << hasVal(row[0]) << " ]";
        } else {
            os << '[';
            for (size_t i{}; i < size; i++) os << hasVal(row[i]) << (i != (size - 1) ? ", " : "");
            os << "]\n";
        }

        return os;
    }

    template <Number number, int m, int n, Container T>
    inline Matrix<number, m, n> vectorTranspose(const T& row) {
        constexpr auto sizeP = (n == Dynamic) ? SizePair{Dynamic, Dynamic} : SizePair{1, n};
        const auto size = row.size();
        Matrix<number, sizeP.first, sizeP.second> res(1, size);
        for (size_t i{}; i < size; i++) res(0, i) = row[i];
        return res;
    }

    template <Number number, typename D, typename OtherD>
    inline number vectorDot(const VectorBase<D, number>& v, const VectorBase<OtherD, number>& w) {
        if (v.size() != w.size()) throw StackError<std::invalid_argument>("Vectors must be of the same size");
#ifdef BY_DEF
        // By the defintion of the dot product
        // v . w = v^T * w
        return (v.T() * w)[0];
#else
        number res{};
        for (size_t i{}; i < v.size(); ++i) res += v[i] * w[i];
        return res;
#endif  // BY_DEF
    }

    template <Number number, int n, int otherN>
    inline number vectorDist(const Vector<number, n>& v, const Vector<number, otherN>& w) {
        if (v.size() != w.size()) throw StackError<std::invalid_argument>("Vectors must be of the same size");
        auto diff = v - w;
        return std::sqrt(diff.dot(diff));
    }

    template <Number number, int n>
    inline Vector<number, n> vectorNormalize(const Vector<number, n>& v) {
        auto len = v.length();
        if (fuzzyCompare(len, number(0))) return v;
        return v / len;
    }

    template <Number number, int n>
    inline void vectorNormalizeI(Vector<number, n>& v) {
        auto len = v.length();
        if (fuzzyCompare(len, number(0))) return;
        v /= len;
    }

    template <Number number, int n>
    inline void vectorClear(Vector<number, n>& v) {
        for (size_t i{}; i < v.size(); i++) {
            v[i] = number{};
        }
    }

    // ===========
    // P-Norms
    // ===========

    /**
     * @brief L1-Norm of a vector
     *
     * @param row Vector to compute the norm of
     * @return L1-Norm of the vector
     */
    template <Number number, Container T>
    inline number L1Norm(const T& row) {
        number sum{};
        for (const auto& elem : row) {
            sum += std::abs(elem);
        }
        return sum;
    }

    /**
     * @brief Euclidean norm (L2-Norm) of a vector
     *
     * @param vec Vector to compute the norm of
     * @return Euclidean norm of the vector
     */
    template <Number number, typename D>
    inline number EuclideanNorm(const VectorBase<D, number>& row) {
        return std::sqrt(vectorDot(row, row));
    }

    // ================
    // Weighted P-Norms
    // ================

    /**
     * @brief Weighted L2-Norm of a vector
     *
     * @param vec Vector to compute the norm of
     * @return Weighted L2-Norm of the vector
     */
    template <Number number, Container T>
    inline number WeightedL2Norm(const T& row, const T& otherRow) {
        if (otherRow.size() != row.size())
            throw StackError<std::invalid_argument>("Matrix and vector must have the same size");
        number sum{};
        for (size_t i{}; i < otherRow.size(); i++) {
            const auto& val = otherRow[i] * (row[i] * row[i]);
            sum += std::abs(val);
        }
        return std::sqrt(sum);
    }

}  // namespace mlinalg::structures
