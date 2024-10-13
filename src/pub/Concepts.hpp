/**
 * @brief Concept for a number type
 *
 * @tparam T Type to check
 */
#pragma once
#include <string>
#include <type_traits>
template <typename T>
concept Number = requires {
    std::is_integral_v<T> || std::is_floating_point_v<T>;
    std::is_convertible_v<T, std::string>;
};
