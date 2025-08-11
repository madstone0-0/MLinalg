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

template <typename type_t, class orig_t>
struct unwrap_impl {
    using type = orig_t;
};

template <typename type_t, class V>
struct unwrap_impl<std::reference_wrapper<type_t>, V> {
    using type = type_t;
};

template <class T>
struct unwrap {
    using type = typename unwrap_impl<std::decay_t<T>, T>::type;
};

template <typename type_t>
using unwrap_t = typename unwrap<type_t>::type;

/**
 * @brief A utility to clean up types by removing cv-qualifiers and references.
 *
 * @tparam T The type to clean
 */
template <typename T>
struct Clean {
    using type = unwrap_t<std::remove_cv_t<std::remove_reference_t<T>>>;
};

/**
 * @brief A utility to clean up types by removing cv-qualifiers and references.
 *
 * @tparam T The type to clean
 */
template <typename T>
using CleanT = typename Clean<T>::type;

/**
 * @brief Empty default trait to prevent errors when a type is not fully specialized, i.e
 * when it's foward declared
 */
template <typename, typename = void>
struct ContainerTraits {};

/**
 * @brief Traits for container types that have value_type, size_type, and iterator types.
 *
 * @tparam T The container type
 */
template <typename T>
struct ContainerTraits<T, std::void_t<typename CleanT<T>::value_type,  //
                                      typename CleanT<T>::size_type,   //
                                      typename CleanT<T>::iterator>    //
                       > {
    using value_type = typename CleanT<T>::value_type;
    using size_type = typename CleanT<T>::size_type;
    using iterator = typename CleanT<T>::iterator;
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
