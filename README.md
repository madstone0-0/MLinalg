# MLinalg

MLinalg is a toy project designed for learning numerical linear algebra using C++. This project implements various data structures and algorithms related to linear algebra, including matrices and vectors, along with common operations such as addition, subtraction, multiplication, and transposition.
## Features

- Matrix and vector data structures
- Support for both static and dynamic matrices and vectors
- Basic linear algebra operations: addition, subtraction, scalar multiplication, and division
- Matrix multiplication by another matrix or a vector
- Transposition of matrices and vectors
- Determinant calculation for matrices
- Augmentation of matrices with other matrices or vectors
- Support for 2D and 3D vectors

## Project Structure

The project is organized as follows:

- CMakeLists.txt: CMake build configuration file
- src/: Source code directory
  - pub/structures/: Contains the implementation of Matrix and Vector classes
     - Matrix.hpp: Implementation of the Matrix class and related functions
     - Vector.hpp: Implementation of the Vector class and related functions
  - Concepts.hpp: Contains concept definitions used in the project
  - Helpers.hpp: Contains helper functions used in the implementation

## Requirements

- CMake 3.20 or later
- A C++ compiler that supports C++23

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

