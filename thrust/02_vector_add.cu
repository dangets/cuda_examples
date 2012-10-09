/**
 * CUDA Thrust example showing addition of 2 vectors
 * (compare with that of raw CUDA)
 *
 *  Danny George 2012
 */

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

#include <iostream>


template <class T>
void print_vector_naive(T& v) {
    if (v.size() > 50) {
        // print first and last bits of vector
        for (int i=0; i<10; i++) {
            std::cout << "[" << i << "] = " << v[i] << std::endl;
        }
        std::cout << " ....... " << std::endl;
        for (int i=v.size()-10; i<v.size(); i++) {
            std::cout << "[" << i << "] = " << v[i] << std::endl;
        }
    } else {
        // print the whole thing...
        for (int i=0; i<v.size(); i++) {
            std::cout << "[" << i << "] = " << v[i] << std::endl;
        }
    }
}



int main(int argc, char *argv[])
{
    const int N = 10000;

    thrust::host_vector<int> h1(N);
    thrust::host_vector<int> h2(N);

    // initialize host vectors (would be more efficient to do this on the device)
    thrust::fill(h1.begin(), h1.end(), 111);
    thrust::fill(h2.begin(), h2.end(), 222);

    // init & copy host data to device vectors
    thrust::device_vector<int> d1(h1);
    thrust::device_vector<int> d2(h2);
    thrust::device_vector<int> dr(N);

    thrust::plus<int> binary_op;

    // transform is essentially a 'map' operation (std::map is used for a container)
    //   perform 'binary_op' on each pairwise element from d1 to d2 and store in dr
    thrust::transform(d1.begin(), d1.end(), d2.begin(), dr.begin(), binary_op);
    // NOTE: you MUST ensure that 'd2' and 'dr' have room for 
    //  at least d1.end() - d1.begin() elements

    // copy result back to host
    h1 = dr;

    print_vector_naive(h1);

    return 0;
}
