#include <MLinalg.hpp>
#include <boost/rational.hpp>
#include <iostream>

using namespace std;
using namespace mlinalg::structures;

using namespace mlinalg::structures;
using namespace mlinalg;
using namespace std;

static int allocs{};

void* operator new(size_t size) {
    allocs++;
    return malloc(size);
}

void operator delete(void* p) { free(p); }

int main() {
    {
        Vector<double, Dynamic> e1{1, 0};
        Vector<double, Dynamic> e2{0, 1};
        auto A = Matrix<double, Dynamic, Dynamic>{{1, -2, 1}, {0, 2, -8}, {5, 0, -5}};
        auto b = Vector<double, Dynamic>{1, 0, -1};
        cout << "Vectors\n\n";
        cout << e1 + e2 << '\n';
        cout << e2 + e1 << '\n';
        cout << A << '\n';
        cout << (A * b).T() << '\n';
        cout << e1.T() * e2 << '\n';
        cout << e1 * e2 << '\n';
        cout << e2 - e1 << '\n';
        cout << e1.T() << '\n';
        cout << e2.T() << '\n';
        e1 = e2.T();
        cout << e1 << '\n';
        cout << e1.dist(e2) << '\n';
        cout << e1.dot(e1) << '\n';
        cout << (e1.T() * e1).at(0) << '\n';
    }
    {
        vector<int> vec{1, 2, 3, 4, 5};
        Vector<int, Dynamic> v{vec.begin(), vec.end()};
        // for (int i{}; i < vec.size(); i++) v.at(i) = vec.at(i);
        Vector<int, Dynamic> v2{v * 2};
        Matrix<int, Dynamic, Dynamic> A{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        auto B = Matrix<double, Dynamic, Dynamic>{{1, -2, 1}, {0, 2, -8}, {5, 0, -5}};
        cout << v << '\n';
        cout << v2 << '\n';
        auto inv = inverse(A);
        auto inv2 = inverse(B);
        if (inv2.has_value()) {
            cout << inv2.value() << '\n';
        } else {
            cout << "Matrix is not invertible\n";
        }

        if (inv.has_value()) {
            cout << inv.value() << '\n';
        } else {
            cout << "Matrix is not invertible\n";
        }
    }
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
    cout << allocs << " allocations\n";
    return 0;
}
