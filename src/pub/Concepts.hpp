/**
 * @file Concepts.hpp
 * @brief Header file for the Concepts
 */

#pragma once
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

/**
 * @brief Concept for a container type
 *
 * @tparam T Type to check
 */
template <typename T>
concept Container = requires {
    typename T::value_type;
    typename T::size_type;
    typename T::iterator;
};
