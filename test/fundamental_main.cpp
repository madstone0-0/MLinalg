#include <MLinalg.hpp>
#include <boost/rational.hpp>
#include <iostream>

using namespace std;
using namespace mlinalg::structures;

using namespace mlinalg::structures;
using namespace std;

static int allocs{};

void* operator new(size_t size) {
    allocs++;
    return malloc(size);
}

void operator delete(void* p) { free(p); }

int main() {
    {
        Vector<double, 2> e1{1, 0};
        Vector<double, 2> e2{0, 1};
        auto A = Matrix<double, 3, 3>{{1, -2, 1}, {0, 2, -8}, {5, 0, -5}};
        auto b = Vector<double, 3>{1, 0, -1};
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
    cout << allocs << " allocations\n";
    return 0;
}
