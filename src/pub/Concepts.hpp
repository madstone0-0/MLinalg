/**
 * @file Concepts.hpp
 * @brief Header file for the Concepts
 */

#pragma once
#include <memory>
#include <string>
#include <type_traits>

/**
 * @brief Concept for a number type
 *
 * @tparam T Type to check
 */
template <typename T>
concept Number = requires {
    std::is_integral_v<T> || std::is_floating_point_v<T>;
    std::is_convertible_v<T, std::string>;
};

template <typename T, typename = void>
struct ContainerTraits {
    using CleanT = std::remove_cvref_t<T>;
    using value_type = typename CleanT::value_type;
    using size_type = typename CleanT::size_type;
    using iterator = typename CleanT::iterator;
};

template <typename T>
struct ContainerTraits<
    T, std::enable_if_t<
           std::is_same_v<std::remove_cvref_t<T>, std::unique_ptr<typename std::remove_cvref_t<T>::element_type>>>> {
    using ElementType = typename std::remove_cvref_t<T>::element_type;
    using value_type = typename ElementType::value_type;
    using size_type = typename ElementType::size_type;
    using iterator = typename ElementType::iterator;
};

/**
 * @brief Concept for a container type
 *
 * @tparam T Type to check
 */
template <typename T>
concept Container = requires {
    typename ContainerTraits<T>::value_type;
    typename ContainerTraits<T>::size_type;
    typename ContainerTraits<T>::iterator;
};
