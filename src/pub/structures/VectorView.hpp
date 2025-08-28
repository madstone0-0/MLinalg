/**
 * @file VectorView.hpp
 * @brief Header file for VectorView and ColumnView classes
 */

#pragma once

#include <iterator>
#include <type_traits>

#include "../Concepts.hpp"
#include "../Helpers.hpp"
#include "Aliases.hpp"
#include "Container.hpp"

namespace mlinalg::structures {

    template <typename D, Number number>
    class VectorBase;

    template <Number number, Dim n>
    using VectorViewBaseType = helpers::IsDynamicT<n, n, VectorRowDynamic<number>, VectorRow<number, n>>;

    /**
     * @brief Base CRTP class for VectorView operations
     *
     * @tparam D The derived class type, used for CRTP
     * @tparam number The type of the number in the vector, which can be any numeric type
     */
    template <typename D, Number number>
    class VectorViewBase : public VectorBase<D, number> {
       protected:
        // CRTP helpers
        D& d() { return static_cast<D&>(*this); }
        constexpr const D& d() const { return static_cast<const D&>(*this); }

       public:
        using value_type = number;
        using size_type = size_t;
        using ref = number&;
        using const_ref = const number&;

        // ======================
        // Arithmetic Operations
        // ======================

        template <typename OtherD>
        friend auto operator+(const VectorViewBase<D, number>& vec, const VectorBase<OtherD, number>& other) {
            auto res = vec.d().toVector();
            res += other;
            return res;
        }

        template <typename OtherD>
        friend auto operator-(const VectorViewBase<D, number>& vec, const VectorBase<OtherD, number>& other) {
            auto res = vec.d().toVector();
            res -= other;
            return res;
        }

        template <typename OtherD>
        friend auto operator*(const VectorViewBase<D, number>& vec, const VectorBase<OtherD, number>& other) {
            return vectorVectorMult(vec.d().toVector(), static_cast<const OtherD&>(other));
        }

        friend auto operator*(const VectorViewBase<D, number>& vec, const number& scalar) {
            auto res = vec.d().toVector();
            res *= scalar;
            return res;
        }

        friend auto operator*(const number& scalar, const VectorViewBase<D, number>& vec) { return vec.d() * scalar; }

        friend auto operator/(const VectorViewBase<D, number>& vec, const number& scalar) {
            auto res = vec.d().toVector();
            res /= scalar;
            return res;
        }
    };

    /**
     * @brief A memory view of a vector, allowing for efficient access to a subset of the vector
     *
     * It supports all the operations of a vector, but does not own the data. As such each
     * out-of-place operation (like addition, subtraction, multiplication, and division) will return a new vector
     * that is a copy of the original vector with the operation applied, however, in-place operations
     * (like +=, -=, *=, /=) will modify the original vector.
     *
     * @tparam number The type of the elements in the vector
     * @tparam n The number of elements in the vector
     * @tparam newSize The size of the vector view, defaults to n, used for converting to a concrete vector
     * @tparam T The type of the container that holds the vector data
     * @param v A reference to the vector data, which can be anything that satisfies the Container concept
     * @param start The starting index of the view, defaults to 0
     * @param end The ending index of the view, defaults to -1 (which means to the end of the vector)
     * @param stride The stride for the view, defaults to 1, which means to access every element in the range
     * @return A VectorView object that provides a view of the vector data
     */
    template <Number number, Dim n, Dim newSize = n>
    class VectorView : public VectorViewBase<VectorView<number, n, newSize>, number> {
        using VectorRef = VectorViewBaseType<number, n>&;
        using Base = VectorView<number, n, newSize>;

       public:
        using difference_type = std::ptrdiff_t;
        constexpr static auto elements = newSize;

        explicit VectorView(VectorRef v, difference_type start = 0, difference_type end = -1,
                            difference_type stride = 1)
            : row{v, start, end, stride}, s{start}, e{end}, stride{stride} {}

        // ======================
        // Miscellaneous Operations
        // ======================

        /**
         * @brief Convert the VectorView to a concrete Vector type.
         *
         * @return A Vector object of type Vector<number, newSize> containing the data from the view.
         */
        auto toVector() const {
            Vector<number, newSize> res(row.size());
            for (size_t i{}; i < row.size(); ++i) {
                res[i] = row[i];
            }
            return res;
        }

        /**
         * @brief Get the size of the vector view.
         *
         * @return The size of the vector view, which is the number of elements in the view.
         */
        [[nodiscard]] size_t size() const override { return static_cast<size_t>(row.size()); }

        friend class Vector<number, n>;

        friend class VectorBase<VectorView<number, n, newSize>, number>;

        friend class VectorViewBase<VectorView<number, n, newSize>, number>;

       private:
        container::StrideContainer<std::remove_reference_t<VectorRef>> row;
        // Start index
        difference_type s{};
        // End index, -1 means to the end of the vector
        difference_type e{-1};
        difference_type stride{1};
    };

    /**
     * @brief A memory view of a column in a matrix, allowing for efficient access to the columns of a matrix.
     *
     * It supports all the operations of a vector, but does not own the data. As such each
     * out-of-place operation (like addition, subtraction, multiplication) will return a new vector
     * that is a copy of the original vector with the operation applied, however, in-place operations
     * (like +=, -=, *=, /=) will modify the original vector.
     *
     * @tparam number The type of the elements in the matrix
     * @tparam m The number of rows in the matrix
     * @tparam n The number of columns in the matrix
     * @param matrix A reference to the matrix data, which can be anything that satisfies the Container concept
     * @param colIdx The index of the column to view
     * @return A ColumnView object that provides a view of the column data
     */
    template <Number number, Dim m, Dim n, Container T>
    class ColumnView : public VectorViewBase<ColumnView<number, m, n, T>, number> {
       public:
        using MatrixType = T;
        using MatrixRef = MatrixType&;
        using MatrixPtr = MatrixType*;
        using difference_type = std::ptrdiff_t;
        using value_type = number;
        using size_type = size_t;
        static constexpr auto elements = m;

        ColumnView(MatrixPtr matrix, size_type colIdx) : row{matrix, colIdx} {
            if (colIdx < 0 || colIdx >= matrix->numCols()) {
                throw mlinalg::stacktrace::StackError<std::out_of_range>("Column index out of range");
            }
        }

        /**
         * @class Iterator
         * @brief An iterator for the ColumnView class that allows iteration over the elements of a column in a matrix.
         *
         */
        struct Iterator {
            using iterator_category = std::random_access_iterator_tag;
            using size_type = size_type;
            using difference_type = std::ptrdiff_t;
            using value_type = value_type;
            using pointer = value_type*;
            using reference = value_type&;

            Iterator(MatrixPtr A, size_type colIdx, size_type idx, size_type end)
                : A{A}, colIdx{colIdx}, idx{idx}, end{end} {}

            reference operator*() const {
                if (idx >= end) throw std::out_of_range{"Iterator out of range"};
                return A->operator()(idx, colIdx);
            }

            pointer operator->() {
                if (idx >= end) throw std::out_of_range{"Iterator out of range"};
                return &A->operator()(idx, colIdx);
            }

            Iterator& operator++() {
                if (idx++ >= end) {
                    throw mlinalg::stacktrace::StackError<std::out_of_range>("Iterator out of range");
                }
                return *this;
            }

            Iterator operator++(int) {
                Iterator tmp = *this;
                ++(*this);
                return tmp;
            }

            Iterator& operator--() {
                if (idx-- < 0) {
                    throw mlinalg::stacktrace::StackError<std::out_of_range>("Iterator out of range");
                }
                return *this;
            }

            Iterator operator--(int) {
                Iterator tmp = *this;
                --(*this);
                return tmp;
            }

            friend bool operator==(const Iterator& a, const Iterator& b) {
                return a.A == b.A && a.idx == b.idx && a.colIdx == b.colIdx && a.end == b.end;
            };
            friend bool operator!=(const Iterator& a, const Iterator& b) { return !(a == b); };

           private:
            MatrixPtr A;
            size_type end{};
            size_type idx{};
            size_type colIdx{};
        };

        using iterator = Iterator;

        // ======================
        // Miscellaneous Operations
        // ======================

        [[nodiscard]] size_t size() const override { return row.size(); }

        auto toVector() const {
            Vector<number, m> res(size());
            for (size_t i{}; i < size(); ++i) {
                res[i] = row[i];
            }
            return res;
        }

       private:
        /**
         * @class Backing
         * @brief A helper struct that provides access to the elements of a column in a matrix.
         *
         */
        struct Backing {
            using value_type = number;
            using size_type = size_t;
            using iterator = Iterator;

            MatrixPtr matrix;
            size_type idx{};

            // ======================
            // Indexing and Accessors
            // ======================
            value_type& operator[](size_type i) { return matrix->operator[](i)[idx]; }

            value_type& operator[](size_type i) const { return matrix->operator[](i)[idx]; }

            value_type& at(size_t i) { return matrix->at(i).at(idx); }

            value_type& at(size_t i) const { return matrix->at(i).at(idx); }

            auto begin() { return Iterator(matrix, idx, 0, matrix->size()); }

            auto end() { return Iterator(matrix, idx, matrix->size(), matrix->size()); }

            auto begin() const { return Iterator(matrix, idx, 0, matrix->size()); }

            auto end() const { return Iterator(matrix, idx, matrix->size(), matrix->size()); }

            [[nodiscard]] size_t size() const { return static_cast<size_t>(matrix->size()); }
        };

        Backing row;

        friend class Vector<number, m>;
        friend class Matrix<number, m, n>;
        friend class VectorBase<ColumnView<number, m, n, T>, number>;
        friend class VectorViewBase<ColumnView<number, m, n, T>, number>;
    };

    template <Number number, Dim n, Dim newSize = n, Container T>
    inline auto View(T& v, long start = 0, long end = -1, long stride = 1) {

        if (start >= end || start < 0) throw mlinalg::stacktrace::StackError<std::out_of_range>("Offset out of range");
        if (stride == 0) throw mlinalg::stacktrace::StackError<std::invalid_argument>("Stride cannot be zero");
        return VectorView<number, n, newSize>{v, start, end, stride};
    }

}  // namespace mlinalg::structures
