#pragma once

#include <bitset>
#include <cstddef>
#include <cstdlib>

namespace mlinalg::allocator {
    /**
     * @brief Temporary (hopefully) allocator that provides aligned memory allocation for backing structures.
     *
     * @tparam T The type of elements to allocate.
     */
    template <typename T>
    class BootlegAllocator {
       public:
        using value_type = T;
        using size_type = size_t;
        using difference_type = std::ptrdiff_t;

        template <typename U>
        struct rebind {
            using other = BootlegAllocator<U>;
        };

        BootlegAllocator() = default;

        template <typename U>
        BootlegAllocator(const BootlegAllocator<U>&) noexcept {}

        constexpr T* allocate(size_type n) {
            if (n == 0) return nullptr;

            // Aligned memory allocation for T
            return static_cast<T*>(std::aligned_alloc(alignof(T), n * sizeof(T)));
        }

        constexpr void deallocate(T* ptr, size_type n) noexcept {
            if (!ptr) return;

            std::free(ptr);
        }
    };

    template <typename T, typename U>
    bool operator==(const BootlegAllocator<T>&, const BootlegAllocator<U>&) noexcept {
        return true;
    }

    template <typename T, typename U>
    bool operator!=(const BootlegAllocator<T>&, const BootlegAllocator<U>&) noexcept {
        return false;
    }

}  // namespace mlinalg::allocator
