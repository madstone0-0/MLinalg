#pragma once

// Taken from https://stackoverflow.com/a/47563100
#include <cstddef>
#include <utility>
template <std::size_t N>
struct num {
    static const constexpr auto value = N;
};

template <class F, std::size_t... Is>
inline void for_(F func, std::index_sequence<Is...>) {
    (func(num<Is>{}), ...);
}

// Taken from https://prosepoetrycode.potterpcs.net/2015/07/a-simple-constexpr-power-function-c/
template <typename T>
constexpr T ipow(T num, unsigned int pow) {
    return (pow >= sizeof(unsigned int) * 8) ? 0 : pow == 0 ? 1 : num * ipow(num, pow - 1);
}
