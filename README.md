# MLinalg

MLinalg is an educational project designed for learning and practicing numerical linear algebra using modern C++. This project is ideal for any one who wishes to gain hands-on experience with numerical methods and modern C++ techniques.

## Overview

MLinalg serves as both a learning tool and a proof-of-concept for building numerical linear algebra libraries. It demonstrates:
- **Self-Initiative & Learning:** A project built from scratch to understand and implement core linear algebra concepts.
- **Technical Breadth:** Implementation of matrices and vectors (both static and dynamic), along with standard operations such as addition, subtraction, multiplication, transposition, and determinant calculation.
- **Advanced Algorithms:** Includes efficient methods like Strassen's algorithm for matrix multiplication.
- **Modern C++ Practices:** Uses C++23 features, templates, concepts, and compile-time optimizations to ensure type safety and performance.
- **Robustness:** Incorporates fuzzy comparisons for floating point arithmetic to handle numerical precision issues, with extensive unit tests (using Catch2) to verify correctness.

## Features

- **Data Structures:**  
  - Matrix and vector classes with support for both static and dynamic sizing.
  - 2D and 3D vector implementations.
  
- **Basic Operations:**  
  - Addition, subtraction, scalar multiplication, and division.
  - Matrix multiplication (both matrix–matrix and matrix–vector).
  - Transposition of matrices and vectors.
  
- **Advanced Operations:**  
  - Determinant calculation.
  - Matrix augmentation with other matrices or vectors.
  - Efficient multiplication via Strassen's algorithm.
  
- **Numerical Robustness:**  
  - Fuzzy comparisons for floating point operations to mitigate precision issues.
  - Comprehensive test suite to ensure stability and correctness.

## Project Structure

- **CMakeLists.txt**  
  CMake build configuration file.

- **src/**  
  Source code directory:
  - **pub/structures/**:  
    - `Matrix.hpp`: Implementation of the Matrix class and related functions.
    - `Vector.hpp`: Implementation of the Vector class and related functions.
  - **pub/Numeric.hpp**:  
    Defines fuzzy comparisons and numerical constants.
  - **Concepts.hpp**:  
    Contains C++ concept definitions used in the project.
  - **Helpers.hpp**:  
    Contains utility functions to support the core implementations.

- **test/**  
  Unit tests (using Catch2) to verify the correctness of all implemented functions.
  - `examples.cpp`: Contains examples of using the library.

## Requirements

- **CMake:** Version 3.20 or later.
- **C++ Compiler:** A compiler that supports C++23 (e.g., GCC 11+, Clang 12+, or the latest MSVC).

## Building the Project

To build the project, follow these steps:

1. Clone the repository:
```sh
git clone https://github.com/madstone0-0/MLinalg.git
cd MLinalg
```

2. Run `build.sh`

### Debug build
```sh
./build.sh
```

### Release build
```sh
./build.sh release
```

## Usage

Once the project is built, you can install an link the library to your project using the following commands:

### Install
#### Requirements
- Ninja / Make
```sh
ninja -C ./build-release install
```

### Linking

The library can be linked using CMake's `find_package` command. Add the following lines to your `CMakeLists.txt` file:

```cmake
find_package(mlinalg REQUIRED)

add_executable(exe exe.cpp)

target_link_libraries(exe PRIVATE mlinalg::mlinalg)
```

The project includes test cases using Catch2 to verify the correctness of the implemented functions.

