#pragma once

#include "../Concepts.hpp"
#include "../structures/Matrix.hpp"

namespace mlinalg {
    using namespace structures;

    template <Number num, int n>
    Vector<num, n> extractSolutionVector(const Vector<optional<num>, n>& solutions) {
        if (rg::any_of(solutions, [](const auto& val) { return !val.has_value(); })) {
            throw StackError("Cannot extract solution vector from incomplete solutions");
        }
        const auto size = solutions.size();

        Vector<num, n> res(size);
        for (size_t i{}; i < size; i++) {
            res.at(i) = solutions.at(i).value();
        }
        return res;
    }

    /**
     * @brief Find the identity matrix of a square linear system
     * @return  The identity matrix of the system.
     */
    template <Number number, int m>
    Matrix<number, m, m> I() {
        Matrix<number, m, m> identity{};
        for (size_t i{}; i < m; i++) {
            identity.at(i).at(i) = 1;
        }
        return identity;
    }

    /**
     * @brief Find the identity matrix of a square linear system
     *
     * @return  The identity matrix of the system.
     */
    template <Number number, int m>
    Matrix<number, m, m> I(int nRows) {
        if constexpr (m == -1) {
            Matrix<number, m, m> identity(nRows, nRows);
            for (int i{}; i < nRows; i++) {
                identity.at(i, i) = 1;
            }
            return identity;
        } else {
            return I<number, m>();
        }
    }

    /**
     * @brief Creates a zero matrix of the given size.
     *
     * @tparam m The number of rows in the matrix.
     * @tparam n The number of columns in the matrix.
     * @return  The zero matrix of the given size.
     */
    template <Number number, int m, int n>
    Matrix<number, m, n> matrixZeros() {
        return Matrix<number, m, n>{};
    }

    /**
     * @brief Creates a zero matrix of the given size.
     *
     * @tparam m The number of rows in the matrix.
     * @tparam n The number of columns in the matrix.
     * @param nRows The number of rows in the matrix.
     * @param nCols The number of columns in the matrix.
     * @return  The zero matrix of the given size.
     */
    template <Number number, int m, int n>
    Matrix<number, m, n> matrixZeros(int nRows, int nCols) {
        if constexpr (m == Dynamic || n == Dynamic) {
            return Matrix<number, m, n>(nRows, nCols);
        } else {
            return matrixZeros<number, m, n>();
        }
    }

    /**
     * @brief Creates a zero vector of the given size.
     *
     * @tparam n The size of the vector.
     * @return The zero vector of the given size.
     */
    template <Number number, int n>
    Vector<number, n> vectorZeros() {
        return Vector<number, n>{};
    }

    /**
     * @brief Creates a zero vector of the given size.
     *
     * @tparam n The size of the vector.
     * @param size
     */
    template <Number number, int n>
    Vector<number, n> vectorZeros(int size) {
        if constexpr (n == Dynamic) {
            return Vector<number, n>(size);
        } else {
            return vectorZeros<number, n>();
        }
    }

    /**
     * @brief Creates a matrix of ones of the given size.
     *
     * @tparam m The number of rows in the matrix.
     * @tparam n The number of columns in the matrix.
     * @return The matrix of ones of the given size.
     */
    template <Number number, int m, int n>
    Matrix<number, m, n> matrixOnes() {
        Matrix<number, m, n> res{};
        for (size_t i{}; i < static_cast<size_t>(m); i++)
            for (size_t j{}; j < static_cast<size_t>(n); j++) res(i, j) = 1;
        return res;
    }

    /**
     * @brief Creates a matrix of ones of the given size.
     *
     * @tparam m The number of rows in the matrix.
     * @tparam n The number of columns in the matrix.
     * @param nRows The number of rows in the matrix.
     * @param nCols The number of columns in the matrix.
     * @return The matrix of ones of the given size.
     */
    template <Number number, int m, int n>
    Matrix<number, m, n> matrixOnes(int nRows, int nCols) {
        if constexpr (m == Dynamic || n == Dynamic) {
            Matrix<number, m, n> res(nRows, nCols);
            for (size_t i{}; i < static_cast<size_t>(nRows); i++)
                for (size_t j{}; j < static_cast<size_t>(nCols); j++) res(i, j) = 1;
            return res;
        } else {
            return matrixOnes<number, m, n>();
        }
    }

    /**
     * @brief Creates a vector of ones of the given size.
     *
     * @tparam n The size of the vector.
     * @return The vector of ones of the given size.
     */
    template <Number number, int n>
    Vector<number, n> vectorOnes() {
        Vector<number, n> res{};
        for (size_t i{}; i < static_cast<size_t>(n); i++) res[i] = 1;
        return res;
    }

    /**
     * @brief Creates a vector of ones of the given size.
     *
     * @tparam n The size of the vector.
     * @param size The size of the vector.
     */
    template <Number number, int n>
    Vector<number, n> vectorOnes(int size) {
        if constexpr (n == Dynamic) {
            Vector<number, n> res(size);
            for (int i{}; i < size; i++) res[i] = 1;
            return res;
        } else {
            return vectorOnes<number, n>();
        }
    }

    /**
     * @brief Creates a random vector of the given size.
     *
     * @tparam n The size of the vector.
     * @param min The minimum value of the random numbers.
     * @param max The maximum value of the random numbers.
     * @param seed The seed for the random number generator.
     * @return The random vector of the given size.
     */
    template <Number num, int n>
    Vector<num, n> vectorRandom(const int min = 0, const int max = 100, const Seed& seed = std::nullopt) {
        Vector<num, n> vec{};
        for (int i{}; i < n; i++) {
            vec.at(i) = helpers::rng<num>(min, max, seed);
        }
        return vec;
    }

    /**
     * @brief Creates a random vector of the given size.
     *
     * @tparam n The size of the vector.
     * @param min The minimum value of the random numbers.
     * @param max The maximum value of the random numbers.
     * @param seed The seed for the random number generator.
     * @param size The size of the vector.
     * @return The random vector of the given size.
     */
    template <Number number, int n>
    Vector<number, n> vectorRandom(const int size, const int min = 0, const int max = 100, const Seed& seed = std::nullopt) {
        if constexpr (n == Dynamic) {
            Vector<number, n> vec(size);
            for (int i{}; i < size; i++) {
                vec.at(i) = helpers::rng<number>(min, max, seed);
            }
            return vec;
        } else {
            return vectorRandom<number, n>(min, max, seed);
        }
    }

    /**
     * @brief Creates a random matrix of the given size.
     *
     * @tparam m The number of rows in the matrix.
     * @tparam n The number of columns in the matrix.
     * @param min The minimum value of the random numbers.
     * @param max The maximum value of the random numbers.
     * @param seed The seed for the random number generator.
     * @return The random matrix of the given size.
     */
    template <Number num, int m, int n>
    Matrix<num, m, n> matrixRandom(int min = 0, int max = 100, Seed seed = std::nullopt) {
        Matrix<num, m, n> res{};
        for (int i{}; i < m; i++) {
            for (int j{}; j < n; j++) res(i, j) = helpers::rng<num>(min, max, seed);
        }
        return res;
    }

    /**
     * @brief Creates a random matrix of the given size.
     *
     * @tparam m The number of rows in the matrix.
     * @tparam n The number of columns in the matrix.
     * @param numRows The number of rows in the matrix.
     * @param numCols The number of columns in the matrix.
     * @param min The minimum value of the random numbers.
     * @param max The maximum value of the random numbers.
     * @param seed The seed for the random number generator.
     * @return The random matrix of the given size.
     */
    template <Number number, int m, int n>
    Matrix<number, m, n> matrixRandom(const int numRows, const int numCols, const int min = 0, const int max = 100, const Seed& seed = std::nullopt) {
        if constexpr (n == Dynamic || m == Dynamic) {
            Matrix<number, m, n> res(numRows, numCols);
            for (int i{}; i < m; i++) {
                for (int j{}; j < n; j++) res(i, j) = helpers::rng<number>(min, max, seed);
            }
            return res;
        } else {
            return matrixRandom<number, m, n>(min, max, seed);
        }
    }

    /**
     * @brief Create a diagonal matrix with the given entries on the diagonal.
     *
     * @param a The value to fill the diagonal with.
     * @return A diagonal matrix with the given value on the diagonal.
     */
    template <int n, Number number>
    Matrix<number, n, n> diagonal(number a) {
        Matrix<number, n, n> res{n, n};
        size_t i{};
        while (i < static_cast<size_t>(n)) {
            res(i, i) = a;
            i++;
        }
        return res;
    }

    /**
     * @brief Create a diagonal matrix with the given entries on the diagonal
     * only for dynamic matrices
     *
     * @param a The value to fill the diagonal with.
     * @param size The size of the diagonal matrix
     * @return A diagonal matrix of size (size) with the given value on the diagonal
     */
    template <int n, Number number>
    Matrix<number, n, n> diagonal(number a, size_t size)
        requires(n == Dynamic)
    {
        Matrix<number, n, n> res(size, size);
        size_t i{};
        while (i < size) {
            res(i, i) = a;
            i++;
        }
        return res;
    }

    /**
     * @brief Create a diagonal matrix with the given entries on the diagonal.
     *
     * @param entries The entries to fill the diagonal with.
     * @return A diagonal matrix with the given entries on the diagonal.
     */
    template <int n, Number number>
    Matrix<number, n, n> diagonal(const std::initializer_list<number>& entries) {
        Matrix<number, n, n> res{n, n};
        size_t i{};
        for (const auto& entry : entries) {
            if (i >= static_cast<size_t>(n)) throw StackError<std::out_of_range>{"Too many entries for diagonal matrix"};
            res(i, i) = entry;
            i++;
        }
        return res;
    }

    /**
     * @brief Create a diagonal matrix with the given entries on the diagonal.
     *
     * @param entries The entries to fill the diagonal with.
     * @return A diagonal matrix with the given entries on the diagonal.
     */
    template <int n, Number number>
    Matrix<number, n, n> diagonal(const std::initializer_list<number>& entries, size_t size)
        requires(n == Dynamic)
    {
        Matrix<number, n, n> res(size, size);
        size_t i{};
        for (const auto& entry : entries) {
            if (i >= size) throw StackError<std::out_of_range>{"Too many entries for diagonal matrix"};
            res(i, i) = entry;
            i++;
        }
        return res;
    }

    /**
     * @brief Create a diagonal matrix with the given entries on the diagonal.
     *
     * @tparam Itr The iterator type for the entries.
     * @param begin The beginning iterator for the entries.
     * @param end  The ending iterator for the entries.
     * @return A diagonal matrix with the given entries on the diagonal.
     */
    template <int n, Number number, typename Itr>
    Matrix<number, n, n> diagonal(Itr begin, Itr end) {
        auto dist = std::distance(begin, end);

        if constexpr (n == Dynamic) {
            Matrix<number, n, n> res(dist, dist);
            size_t i{};
            for (auto itr{begin}; itr != end; ++itr) {
                res(i, i) = *itr;
                i++;
            }
            return res;
        } else {
            if (n != dist) throw StackError<std::out_of_range>{"Too many entries for diagonal matrix"};
            Matrix<number, n, n> res(n, n);
            size_t i{};
            for (auto itr{begin}; itr != end; ++itr) {
                res(i, i) = *itr;
                i++;
            }
            return res;
        }
    }

    /**
     * @brief Create a diagonal matrix with the given entries on the diagonal.
     *
     * @param entries The entries to fill the diagonal with.
     * @return A diagonal matrix with the given entries on the diagonal.
     */
    template <int n, Number number>
    Matrix<number, n, n> diagonal(const array<number, n>& entries) {
        Matrix<number, n, n> res{n, n};
        for (size_t i{}; i < static_cast<size_t>(n); i++) {
            res(i, i) = entries[i];
        }
        return res;
    }

}  // namespace mlinalg
