/**
 * @file VectorOps.hpp
 * @brief Header file for vector operations
 */

#pragma once

#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

#include "../Concepts.hpp"
#include "Aliases.hpp"

namespace mlinalg::structures {
    template <Number number, int n>
    class Vector;

    template <Number number, int m, int n>
    class Matrix;

    template <Container T, Container U>
    void checkOperandSize(const T& row, const U& otherRow) {
        if (row.size() != otherRow.size()) throw std::invalid_argument("Vectors must be of the same size");
    }

    template <Container T, Container U>
    bool vectorEqual(const T& row, const U& otherRow) {
        checkOperandSize(row, otherRow);
        auto n = row.size();
        for (size_t i{}; i < n; i++)
            if (!fuzzyCompare(row.at(i), otherRow.at(i))) return false;
        return true;
    }

    template <Number number, Container T>
    number& vectorAt(T& row, size_t i) {
        return row.at(i);
    }

    template <Number number, Container T>
    const number& vectorConstAt(const T& row, size_t i) {
        return row.at(i);
    }

    template <Number number, int n, Container T, Container U>
    Vector<number, n> vectorSub(const T& row, const U& otherRow) {
        checkOperandSize(row, otherRow);
        constexpr int vSize = (n != Dynamic) ? n : Dynamic;
        auto size = row.size();
        Vector<number, vSize> res(size);
        for (size_t i{}; i < size; i++) res.at(i) = row.at(i) - otherRow.at(i);
        return res;
    }

    template <Number number, int n, Container T>
    Vector<number, n> vectorNeg(const T& row) {
        constexpr int vSize = (n != Dynamic) ? n : Dynamic;
        auto size = row.size();
        Vector<number, vSize> res(size);
        for (size_t i{}; i < size; i++) res.at(i) = -row.at(i);
        return res;
    }

    template <Number number, Container T, Container U>
    void vectorSubI(T& row, const U& otherRow) {
        checkOperandSize(row, otherRow);
        for (size_t i{}; i < row.size(); i++) row.at(i) -= otherRow.at(i);
    }

    template <Number number, int n, Container T, Container U>
    Vector<number, n> vectorAdd(const T& row, const U& otherRow) {
        checkOperandSize(row, otherRow);
        constexpr int vSize = (n != Dynamic) ? n : Dynamic;
        auto size = row.size();
        Vector<number, vSize> res(size);
        for (size_t i{}; i < size; i++) res.at(i) = row.at(i) + otherRow.at(i);
        return res;
    }

    template <Number number, Container T, Container U>
    void vectorAddI(T& row, const U& otherRow) {
        checkOperandSize(row, otherRow);
        for (size_t i{}; i < row.size(); i++) row.at(i) += otherRow.at(i);
    }

    template <Number number, int n, Container T>
    Vector<number, n> vectorScalarMult(const T& row, const number& scalar) {
        constexpr int vSize = (n != Dynamic) ? n : Dynamic;
        auto size = row.size();
        Vector<number, vSize> res(size);
        for (size_t i{}; i < size; i++) res.at(i) = scalar * row[i];
        return res;
    }

    template <Number number, int n, Container T>
    Vector<number, n> vectorScalarDiv(const T& row, const number& scalar) {
        if (fuzzyCompare(scalar, number(0))) throw std::domain_error("Division by zero");
        constexpr int vSize = (n != Dynamic) ? n : Dynamic;
        auto size = row.size();
        Vector<number, vSize> res(size);
        for (size_t i{}; i < size; i++) res.at(i) = row.at(i) / scalar;
        return res;
    }

    template <Number number, Container T>
    void vectorScalarMultI(T& row, const number& scalar) {
        for (size_t i{}; i < row.size(); i++) row.at(i) *= scalar;
    }

    template <Number number, Container T>
    void vectorScalarDivI(T& row, const number& scalar) {
        for (size_t i{}; i < row.size(); i++) row.at(i) /= scalar;
    }

    template <Number number, int n, int otherN>
    number vectorVectorMult(const Vector<number, n>& vec, const Vector<number, otherN>& otherVec) {
        if (vec.size() != otherVec.size()) throw std::invalid_argument("Vectors must be of the same size");

        const auto res = vec.T() * otherVec;
        return res[0];
    }

    template <Container T>
    std::string vectorStringRepr(const T& row) {
        std::stringstream ss{};

        size_t maxWidth = 0;
        for (const auto& elem : row) {
            std::stringstream temp_ss;
            temp_ss << elem;
            maxWidth = std::max(maxWidth, temp_ss.str().length());
        }

        if (row.size() == 1)
            ss << "[ " << std::setw(static_cast<int>(maxWidth)) << row.at(0) << " ]\n";
        else
            for (size_t i{}; i < row.size(); i++)
                if (i == 0) {
                    ss << "⎡ " << std::setw(static_cast<int>(maxWidth)) << row.at(i) << " ⎤\n";
                } else if (i == row.size() - 1) {
                    ss << "⎣ " << std::setw(static_cast<int>(maxWidth)) << row.at(i) << " ⎦\n";
                } else {
                    ss << "| " << std::setw(static_cast<int>(maxWidth)) << row.at(i) << " |\n";
                }
        return ss.str();
    }

    template <Container T>
    std::ostream& vectorOptionalRepr(std::ostream& os, const T& row) {
        const auto& size = row.size();

        auto hasVal = [](auto rowVal) {
            std::stringstream val;
            if (rowVal.has_value())
                val << rowVal.value();
            else
                val << "None";
            return val.str();
        };

        size_t maxWidth = 0;
        for (const auto& elem : row) {
            std::stringstream temp_ss;
            temp_ss << hasVal(elem);
            maxWidth = std::max(maxWidth, temp_ss.str().length());
        }

        if (size == 1) {
            os << "[ ";
            if (row.at(0).has_value())
                os << row.at(0).value();
            else
                os << "None";
            os << " ]\n";
        } else
            for (size_t i{}; i < row.size(); i++) {
                if (i == 0) {
                    os << "⎡ " << std::setw(static_cast<int>(maxWidth)) << hasVal(row.at(i)) << " ⎤\n";
                } else if (i == row.size() - 1) {
                    os << "⎣ " << std::setw(static_cast<int>(maxWidth)) << hasVal(row.at(i)) << " ⎦\n";

                } else {
                    os << "| " << std::setw(static_cast<int>(maxWidth)) << hasVal(row.at(i)) << " |\n";
                }
            }
        return os;
    }

    template <Number number, int m, int n, Container T>
    Matrix<number, m, n> vectorTranspose(const T& row) {
        constexpr auto sizeP = (n == Dynamic) ? SizePair{Dynamic, Dynamic} : SizePair{1, n};
        const auto size = row.size();
        Matrix<number, sizeP.first, sizeP.second> res(1, size);
        for (size_t i{}; i < size; i++) res.at(0, i) = row.at(i);
        return res;
    }

}  // namespace mlinalg::structures
