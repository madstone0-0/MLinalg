/**
 * @file Helpers.hpp
 * @brief Helper functions for structures
 */

#pragma once
#include <variant>
#include <vector>

#include "Concepts.hpp"
#include "structures/Vector.hpp"

namespace mlinalg::structures::helpers {
    using std::vector;

    template <Number number, int m, int n>
    using TransposeVariant = std::variant<Vector<number, m>, Matrix<number, n, m>>;

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

    template <Number num, int m, int n>
    Matrix<num, Dynamic, Dynamic> toDynamic(const Matrix<num, m, n> matrix) {
        return Matrix<num, Dynamic, Dynamic>{matrix};
    }

}  // namespace mlinalg::structures::helpers
