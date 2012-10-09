/**
 * Thrust intro to Functors (function objects)
 *
 * Think similar to C's qsort and the cmp function pointer (except better)
 * Functors are objects that can act like a function, but
 * they can also store per-instance state and other niceties
 *
 *  Danny George 2012
 */

#include <stdio.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>


#include "CUDATimer.cuh"



struct AddFunctor {
    // _val is PER instance state - can create multiple instances with different numbers
    const float _val;

    // constructor
    AddFunctor(const float val) : _val(val) { }

    __host__ __device__
    float operator()(const float &x) const {
        return x + _val;
    }
};


void add_demo(const int N)
{
    thrust::device_vector<int> d1(N, 20);    // fill with N elements initialized to 20
    thrust::device_vector<int> dresult(N);

    AddFunctor add5(5.0f);
    AddFunctor add20(20.0f);


    // add  5 to every number in d1
    thrust::transform(d1.begin(), d1.end(), dresult.begin(), add5);

    // add 20 to every number in d1
    thrust::transform(d1.begin(), d1.end(), dresult.begin(), add20);

    // do whatever with dresult ...
}






struct SaxpyFunctor {
    const float _a;

    // constructor
    SaxpyFunctor(const float a) : _a(a) { }

    __host__ __device__
    float operator()(const float &x, const float &y) const {
        return _a * x + y;
    }
};


void saxpy_fast(float A, thrust::device_vector<float>& X, thrust::device_vector<float>& Y)
{
    // Y <- A * X + Y
    SaxpyFunctor mysaxpy(A);
    thrust::transform(X.begin(), X.end(), Y.begin(), Y.begin(), mysaxpy);
}


void saxpy_slow(float A, thrust::device_vector<float>& X, thrust::device_vector<float>& Y)
{
    thrust::device_vector<float> temp(X.size());

    // temp <- A
    thrust::fill(temp.begin(), temp.end(), A);
    // temp <- A * X
    thrust::transform(temp.begin(), temp.end(), X.begin(), temp.begin(), thrust::multiplies<float>());
    // Y <- A * X + Y
    thrust::transform(temp.begin(), temp.end(), Y.begin(), Y.begin(), thrust::plus<float>());

    // ASIDE:
    //  thrust::plus & thrust::multiplies are functors!
}


void saxpy_demo(int N)
{
    CUDATimer timer;
    float saxpy_fast_time;
    float saxpy_slow_time;

    float a = 5;
    thrust::device_vector<float> x(N);
    thrust::device_vector<float> y(N);


    // initialize device vectors
    thrust::fill(x.begin(), x.end(), 1234);
    thrust::fill(y.begin(), y.end(), 6789);

    timer.start();
    saxpy_fast(a, x, y);
    saxpy_fast_time = timer.get_elapsed_time_sync();


    // initialize device vectors
    thrust::fill(x.begin(), x.end(), 1234);
    thrust::fill(y.begin(), y.end(), 6789);

    timer.start();
    saxpy_slow(a, x, y);
    saxpy_slow_time = timer.get_elapsed_time_sync();

    // can you guess which is faster?
    printf(" ===== saxpy_demo ====== \n");
    printf("saxpy_fast: %g ms\n", saxpy_fast_time);
    printf("saxpy_slow: %g ms\n", saxpy_slow_time);


    // Now why is it faster? - reduced number of kernel launches
    //  saxpy_slow performs 1 vector alloc, 1 fill, and 2 transforms
    //  saxpy_fast cudaMemcpys a SaxpyFunctor struct to device (basically sizeof(float))
    //          and does 1 transform op.

    // Functors can also help reduce the number of memory transferred across the bus
    //  in some situations
}





int main(int argc, char *argv[])
{
    int N = 10000;

    if (argc > 1) {
        N = atoi(argv[1]);
    }

    printf("N = %d\n", N);

    add_demo(N);

    saxpy_demo(N);


    return 0;
}
