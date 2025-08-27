#pragma once
#include <cassert>

#include "../Concepts.hpp"
#include "../Stacktrace.hpp"

namespace mlinalg::structures::container {

    /**
     * @brief A wrapper around std::array to provide the same interface as std::vector
     *
     * Intended to be used in small vector optimization, and used interchangeably with std::vector
     * @tparam T The type of the elements in the container
     * @tparam N The number of elements in the container
     * @param count The number of elements in the container, must be equal to N
     * @param value The value to initialize the elements with
     */
    template <typename T, int N>
    struct StaticContainer {
        using value_type = T;
        using iterator = T*;
        using size_type = size_t;

        std::array<T, N> data_;

        explicit StaticContainer(size_type count = N, const T& value = T{}) {
            assert(count == N && "StaticVector must be full-size");
            data_.fill(value);
        }

        [[nodiscard]] constexpr size_t size() const noexcept { return N; }

        constexpr value_type* data() noexcept { return data_.data(); }

        constexpr const T* data() const noexcept { return data_.data(); }

        constexpr auto begin() noexcept { return data_.begin(); }

        constexpr auto end() noexcept { return data_.end(); }

        constexpr auto rbegin() noexcept { return data_.rbegin(); }

        constexpr auto rend() noexcept { return data_.rend(); }

        constexpr T& back() noexcept { return data_.back(); }

        T& at(size_type i) { return data_.at(i); }

        const T& at(size_type i) const { return data_.at(i); }

        constexpr const T& back() const noexcept { return data_.back(); }

        constexpr auto begin() const noexcept { return data_.begin(); }

        constexpr auto end() const noexcept { return data_.end(); }

        constexpr auto rbegin() const noexcept { return data_.rbegin(); }

        constexpr auto rend() const noexcept { return data_.rend(); }

        constexpr auto cbegin() const noexcept { return data_.cbegin(); }

        constexpr auto cend() const noexcept { return data_.cend(); }

        T& operator[](size_type i) noexcept { return data_[i]; }

        const T& operator[](size_type i) const noexcept { return data_[i]; }
    };

    /**
     * @brief A container that provides a view of a subset of a container with a specified stride.
     *
     * @tparam T The type of the container, which must satisfy the Container concept.
     * @param container The type of the container, which must satisfy the Container concept.
     * @param start The starting index of the view, defaults to 0.
     * @param end The ending index of the view, defaults to -1 (which means to the end of the container).
     * @param stride The stride for the view, defaults to 1, which means to access every element in the range.
     * @param startIdx The starting index for the iterator, defaults to 0.
     */
    template <Container T>
    class StrideContainer {
       public:
        using difference_type = std::ptrdiff_t;
        using value_type = ContainerTraits<T>::value_type;
        using size_type = ContainerTraits<T>::size_type;

        /**
         * @class Iterator
         * @brief An iterator for the StrideContainer that allows for random access iteration bounded
         * by the specified start, end, and stride.
         *
         */
        struct Iterator {
            using iterator_category = std::random_access_iterator_tag;
            using difference_type = std::ptrdiff_t;
            using value_type = value_type;
            using pointer = value_type*;
            using reference = value_type&;

            Iterator(T& container, difference_type start = 0, difference_type end = -1, difference_type stride = 1,
                     difference_type startIdx = 0)
                : container{container}, start{start}, end{end}, stride{stride}, idx{startIdx} {
                if (end == -1) end = container.size();
                if (end <= start) throw std::out_of_range{"End index must be greater than start index"};
            }

            reference operator*() const {
                if (idx >= maxIdx) throw std::out_of_range{"Iterator out of range"};
                return container[start + (idx * stride)];
            }

            pointer operator->() {
                if (idx >= maxIdx) throw std::out_of_range{"Iterator out of range"};
                return &container[start + (idx * stride)];
            }

            // Prefix increment
            Iterator& operator++() {
                if (idx++ >= maxIdx) {
                    throw mlinalg::stacktrace::StackError<std::out_of_range>("Iterator out of range");
                }
                return *this;
            }

            // Postfix increment
            Iterator operator++(int) {
                Iterator tmp = *this;
                ++(*this);
                return tmp;
            }

            // Prefix decrement
            Iterator& operator--() {
                if (idx-- < 0) {
                    throw mlinalg::stacktrace::StackError<std::out_of_range>("Iterator out of range");
                }
                return *this;
            }

            // Postfix decrement
            Iterator operator--(int) {
                Iterator tmp = *this;
                --(*this);
                return tmp;
            }

            friend bool operator==(const Iterator& a, const Iterator& b) {
                return a.container == b.container && a.start == b.start && a.stride == b.stride && a.idx == b.idx;
            };
            friend bool operator!=(const Iterator& a, const Iterator& b) { return !(a == b); };

           private:
            T& container;
            difference_type start{};
            difference_type end{};
            difference_type idx{};
            difference_type stride{1};
            difference_type maxIdx{(end - start + stride - 1) / stride};
        };

        using iterator = Iterator;

        StrideContainer(T& container, difference_type start = 0, difference_type end = -1, difference_type stride = 1)
            : container{container}, s{start}, e{end}, stride{stride} {
            if (end == -1) e = container.size();
            if (e <= s) throw std::out_of_range{"End index must be greater than start index"};
            if (e > container.size()) e = container.size();
            if (s < 0 || s >= container.size()) throw std::out_of_range{"Start index out of range"};
        }

        value_type& operator[](size_type i) { return container[s + (i * stride)]; }

        value_type& operator[](size_type i) const { return container[s + (i * stride)]; }

        value_type& at(size_t i) { return container[s + (i * stride)]; }

        value_type& at(size_t i) const { return container[s + (i * stride)]; }

        size_type size() const { return (e - s + stride - 1) / stride; }

        Iterator begin() { return Iterator(container, s, e, stride); }

        Iterator end() { return Iterator(container, s, e, stride, size()); }

        auto begin() const { return Iterator(container, s, e, stride); }

        auto end() const { return Iterator(container, s, e, stride, size()); }

       private:
        T& container;
        difference_type s{};
        difference_type e{-1};
        difference_type stride{1};
    };

}  // namespace mlinalg::structures::container
