#include <MLinalg.hpp>
#include <boost/rational.hpp>
#include <chrono>
#include <iostream>
#include <random>

#include "Operations.hpp"

using namespace std;
using namespace mlinalg::structures;

using namespace mlinalg::structures;
using namespace mlinalg;
using namespace std;

// static int allocs{};
//
// void* operator new(size_t size) {
//     allocs++;
//     return malloc(size);
// }
//
// void operator delete(void* p) { free(p); }

int main() {
    // {
    //     Vector<double, Dynamic> e1{1, 0};
    //     Vector<double, Dynamic> e2{0, 1};
    //     auto A = Matrix<double, Dynamic, Dynamic>{{1, -2, 1}, {0, 2, -8}, {5, 0, -5}};
    //     auto b = Vector<double, Dynamic>{1, 0, -1};
    //     cout << "Vectors\n\n";
    //     cout << e1 + e2 << '\n';
    //     cout << e2 + e1 << '\n';
    //     cout << A << '\n';
    //     cout << (A * b).T() << '\n';
    //     cout << e1.T() * e2 << '\n';
    //     cout << e1 * e2 << '\n';
    //     cout << e2 - e1 << '\n';
    //     cout << e1.T() << '\n';
    //     cout << e2.T() << '\n';
    //     e1 = e2.T();
    //     cout << e1 << '\n';
    //     cout << e1.dist(e2) << '\n';
    //     cout << e1.dot(e1) << '\n';
    //     cout << (e1.T() * e1).at(0) << '\n';
    // }
    // {
    //     vector<int> vec{1, 2, 3, 4, 5};
    //     Vector<int, Dynamic> v{vec.begin(), vec.end()};
    //     // for (int i{}; i < vec.size(); i++) v.at(i) = vec.at(i);
    //     Vector<int, Dynamic> v2{v * 2};
    //     Matrix<int, Dynamic, Dynamic> A{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    //     auto B = Matrix<double, Dynamic, Dynamic>{{1, -2, 1}, {0, 2, -8}, {5, 0, -5}};
    //     cout << v << '\n';
    //     cout << v2 << '\n';
    //     auto inv = inverse(A);
    //     auto inv2 = inverse(B);
    //     if (inv2.has_value()) {
    //         cout << inv2.value() << '\n';
    //     } else {
    //         cout << "Matrix is not invertible\n";
    //     }
    //
    //     if (inv.has_value()) {
    //         cout << inv.value() << '\n';
    //     } else {
    //         cout << "Matrix is not invertible\n";
    //     }
    // }
    // {
    //     int size{};
    //     cout << "Enter number of elements: ";
    //     cin >> size;
    //     Vector<double, Dynamic> v(size);
    //     for (int i{}; i < size; i++) {
    //         cout << "v[" << i << "]: ";
    //         cin >> v.at(i);
    //     }
    //     cout << v << '\n';
    // }
    // {
    //     auto A = Matrix<int, Dynamic, Dynamic>{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    //     cout << 2 * A << '\n';
    //     auto v = Vector<int, Dynamic>{1, 2, 3};
    //     cout << A * v << '\n';
    //     cout << v.T() << '\n';
    // }
    // {
    //     Vector<double, Dynamic> e1{1, 0};
    //     Vector<double, 2> e2{0, 1};
    //     cout << e1 - e2 << '\n';
    //     cout << e1 + e2 << '\n';
    //     // cout << e1 * e2 << '\n';
    // }
    {
        // auto A = Matrix<int, Dynamic, Dynamic>{
        //     {0, 1, 1, 1},  //
        //     {1, 0, 0, 0},  //
        //     {1, 0, 0, 1},  //
        //     {1, 0, 1, 0},  //
        // };
        // auto A3 = A * A * A;
        // cout << A3 << '\n';
        // auto m1 = Matrix<int, 3, 3>{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        // auto sliced = m1.slice<0, 2, 0, 2>();
        // cout << sliced << '\n';

        auto m2 = Matrix<int, 4, 4>{
            {1, 2, 5, 6},      //
            {3, 4, 7, 8},      //
            {9, 10, 13, 14},   //
            {11, 12, 15, 16},  //
        };
        constexpr int i1{m2.numRows() / 2};
        constexpr int j1{m2.numCols() / 2};
        auto sliced2 = m2.slice<0, i1, 0, j1>();
        cout << sliced2.at(1, 0) << '\n';
        cout << sliced2 << '\n';
        cout << m2 * m2 << '\n';
    }

    {
        const int N = 100;  // Matrix multiplication is more expensive, smaller N
        Matrix<long long, N, N> m1{};
        Matrix<long long, N, N> m2{};

        // Initialize matrices with large data
        for (int i{}; i < N; i++) {
            for (int j{}; j < N; j++) {
                m1.at(i, j) = i + j;
                m2.at(i, j) = i - j;
            }
        }

        // cout << m1 << '\n';
        // cout << m2 << '\n';
        auto res{m1 * m2};
        // cout << res << '\n';
    }

    // {
    //     auto A = Matrix<boost::rational<double>, 4, 4>{
    //         // auto A = Matrix<double, 4, 4>{
    //         {5., 6., 6., 8.},  //
    //         {2., 2., 2., 8.},  //
    //         {6., 6., 2., 8.},  //
    //         {2., 3., 6., 7.},  //
    //     };
    //     cout << A << '\n';
    //     auto AInv = inverse(A);
    //     if (AInv.has_value()) {
    //         cout << AInv.value() << '\n';
    //         cout << A * AInv.value() << '\n';
    //     } else {
    //         cout << "Matrix has no inverse\n";
    //     }
    // }
    // cout << allocs << " allocations\n";

    // {
    //     LinearSystem<double, 4, 6> system5{
    //         {1, -3, 0, -1, 0, -2},
    //         {0, 1, 0, 0, -4, 1},
    //         {0, 0, 0, 1, 9, 4},
    //         {0, 0, 0, 0, 0, 10},
    //     };
    //     cout << system5 << '\n';
    //     isInconsistent(system5);
    // }

    {
        LinearSystem<double, 3, 4> system1{{
            {1, 2, 3, 4},
            {0, 0, 2, 3},
            {0, 0, 0, 0},
        }};

        cout << system1 << '\n';
        auto res = getPivots(system1);
        for (const auto& val : res) {
            if (val.has_value())
                cout << val.value() << ' ';
            else
                cout << "None ";
        }
        cout << '\n';
        cout << res.size() << '\n';
    }

    {
        LinearSystem<double, 3, 5> system4{
            {1, -7, 0, 6, 5},
            {0, 0, 1, -2, -3},
            {-1, 7, -4, 2, 7},
        };

        cout << system4 << '\n';
        auto res = getPivots(system4);
        for (const auto& val : res) {
            if (val.has_value())
                cout << val.value() << ' ';
            else
                cout << "None ";
        }
        cout << '\n';
        cout << res.size() << '\n';
    }
    // {
    //     LinearSystem<double, 2, 2> squareSystem({
    //         {3, 1},
    //         {1, 2},
    //     });
    //     cout << squareSystem << '\n';
    //     auto res = getPivots(squareSystem);
    //     for (const auto& val : res) {
    //         if (val.has_value())
    //             cout << val.value() << ' ';
    //         else
    //             cout << "None ";
    //     }
    //     cout << '\n';
    //     cout << res.size() << '\n';
    // }
    {
        // Create random 1000x300 matrix
        std::mt19937 gen{std::random_device{}()};
        std::uniform_int_distribution<int> dist{-100, 100};
        constexpr int N = 512;
        Matrix<int, N, N> m1{};
        for (int i{}; i < N; i++) {
            for (int j{}; j < N; j++) {
                m1.at(i, j) = dist(gen);
            }
        }

        // auto start = std::chrono::high_resolution_clock::now();
        // auto mult = m1 * m1;
        // auto end = std::chrono::high_resolution_clock::now() - start;
        // cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end).count() << "ms\n";
    }
    return 0;
}
