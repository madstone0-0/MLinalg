# MLinalg

MLinalg is a modern C++23 library for numerical linear algebra, designed to facilitate learning and experimentation with
numerical linear algebra algorithms. It provides a comprehensive set of tools for matrix and vector operations, advanced decompositions, and robust numerical methods. The code is optimized for readability and clarity, instead of raw performance.

---

## Project Highlights

- **Modern C++23:** Utilizes templates, concepts, and compile-time optimizations for type safety and performance.
- **Advanced Algorithms:** Implements SVD, QR, LU, Schur decompositions, Strassen's multiplication, and more.
- **Robust Design:** Fuzzy comparisons for floating point arithmetic, strong error handling, and comprehensive unit tests.
- **Modular & Extensible:** Clean architecture for easy extension and integration into other projects.

---

## Features

- **Data Structures:**
  - Matrix and vector classes (static/dynamic sizing), 2D/3D vector support.
- **Basic Operations:**
  - Addition, subtraction, scalar multiplication/division, matrix multiplication, transposition.
- **Advanced Operations:**
  - Determinant calculation, matrix augmentation, Strassen's algorithm.
- **Matrix Decomposition & Linear Algebra Tools:**
  - **SVD (Singular Value Decomposition)**
  - **Eigenvalue/Eigenvector Computation**
  - **QR Decomposition** (Gram-Schmidt & Householder)
  - **LU Decomposition**
  - **Schur Decomposition**
  - **Exact & Least Squares Linear System Solving**
  - **Moore-Penrose Pseudoinverse**
  - **Customizable QR Methods**
  - **Robust Error Handling**
- **Numerical Robustness:**
  - Fuzzy comparisons, comprehensive test suite.

---

## Example Usage

Below is a minimal example. For more complete and practical usage, see the [`src/examples`](src/examples) directory.

```cpp
#include <MLinalg.hpp>
using namespace mlinalg;

// Create a matrix and vector
Matrix<double, 3, 3> A = {/* ... */};
Vector<double, 3> b = {/* ... */};

// Solve Ax = b exactly (if possible)
auto result = solveExact(A, b);

// Compute SVD
auto [U, Sigma, VT] = svd(A);
```

---

## Why MLinalg?

- **For Learners:** Understand how advanced linear algebra algorithms are implemented in modern C++.
- **For Researchers:** Use as a foundation for further numerical experiments or algorithm development.

---

## Project Structure

- **src/**: Source code (core algorithms, data structures, utilities)
  - **test/**: Unit tests and usage examples (Catch2)
  - **pub/structures/**: Matrix and vector implementations
  - **pub/operations/**: Decompositions, solvers, pseudoinverse, etc.
  - **examples/**: Complete usage examples and practical demonstrations
- **docs/**: Documentation (generated in release builds)

---

## Requirements

- **CMake:** Version 3.20+
- **C++ Compiler:** C++23 support (GCC 11+, Clang 12+, MSVC latest)

---

## Building & Installation

```sh
git clone https://github.com/madstone0-0/MLinalg.git
cd MLinalg
./build.sh        # Debug build
./build.sh release # Release build
ninja -C ./build-release install
```

---

## Linking

Add to your `CMakeLists.txt`:

```cmake
find_package(mlinalg REQUIRED)
add_executable(exe exe.cpp)
target_link_libraries(exe PRIVATE mlinalg::mlinalg)
```

---

## Documentation

Documentation is generated during release build but documentation for the current release can be found [here](https://madstone0-0.github.io/MLinalg/)

--- 

## Author & Contact

GitHub: [madstone0-0](https://github.com/madstone0-0)

Email: [mhquansah@gmail.com](mailto:mhquansah@gmail.com)

---

## License

MIT License
