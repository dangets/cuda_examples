/**
 * An introduction to programming with CUDA Thrust
 *
 * Officially supported library distributed with CUDA since v4.0
 * Abstracts the low-level memory and launch dimensions concerns in raw CUDA
 * Provides containers and many algorithms for common problems to speed
 *      development on GPU
 * Modeled after C++ STL and standard library
 *
 *
 *  Danny George 2012
 */

#include <stdio.h>

#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/reduce.h>



int main(int argc, char const *argv[])
{
    // create a vector on the host
    thrust::host_vector<int> h_vec(50);

    // fill incrementing with simple for loop
    for (int i=0; i<h_vec.size(); ++i) {
        h_vec[i] = i;
    }

    // --- create vector on device - many ways ----
    // copy entire array
    thrust::device_vector<int> d_vec1(h_vec);
    // copy first 50 elements (h_vec must be at least 50 big)
    thrust::device_vector<int> d_vec2(h_vec.begin(), h_vec.begin() + 50);


    // --- examples of cudaMemcpy abstractions ----
    d_vec2 = h_vec;     // copy host to device
    d_vec1 = d_vec2;    // copy device to device
    h_vec = d_vec2;     // copy device to host

    for (int i=0; i<d_vec1.size(); ++i) {
        // direct access to vector elements
        std::cout << "d_vec1[" << i << "] = " << d_vec1[i] << std::endl;
        d_vec1[i] *= 2;

        // NOTE: this is useful for debugging, but is very inefficient!
        //  it calls cudaMemcpy once for each element in d_vec1
        //  usually better to copy the entire vector or large chunks of it
    }

    // basic intro to builtin thrust algorithms
    //  because of templates - works on host_vectors and device_vectors
    //  algorithms executed on device will run in parallel

    // set the elements to 0, 1, 2, 3, ...
    thrust::sequence(h_vec.begin(), h_vec.end());
    thrust::sequence(d_vec1.begin(), d_vec1.end());
    // set elements to decrementing sequence
    thrust::sequence(d_vec2.begin(), d_vec2.end(), (int)d_vec2.size(), -1);

    // fill to a constant number
    thrust::fill(d_vec1.begin(), d_vec1.end(), 10);
    // reduce (sum operation by default)
    int sum = thrust::reduce(d_vec1.begin(), d_vec1.end());
    std::cout << "sum: " << sum << std::endl;

    for (int i=0; i<d_vec2.size(); ++i) {
        std::cout << "d_vec2[" << i << "] = " << d_vec2[i] << std::endl;
    }

    return 0;
}
