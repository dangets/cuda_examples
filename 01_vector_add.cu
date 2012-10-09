#include <stdio.h>

/*
 * A very simple CUDA example adding two arrays of ints together.
 *
 * Shows common paradigm of copy input data to device, copy results back to host
 * Introduces many CUDA concepts
 *  - device pointers with cudaMalloc, cudaMemcpy
 *  - writing GPU kernel code (and getting threadIdx)
 *  - launching kernel
 *
 *  Danny George 2012
 */


// NOTE: keep this value to a low number for this example
//  due to block dimension limitations and the way this
//  example was written. (I received failure with > 1024)
const int N = 512;


// initialize an array with a counting sequence
void fill_array_count(int *arr, const size_t n)
{
    for (size_t i=0; i<n; ++i) {
        arr[i] = (int)i;
    }
}

// initialize an array with a constant number
void fill_array_const(int *arr, const size_t n, const int val)
{
    for (size_t i=0; i<n; ++i) {
        arr[i] = val;
    }
}


// a CUDA kernel function
//      the CUDA runtime spawns many parallel threads to execute it
//      the executing thread id can be found through the threadIdx.[xyz] and blockIdx.[xyz] variables
//      (this example doesn't spawn more than one block)
// the __global__ attribute tells the compiler that this is
//      code that is called by the host and run on the device
__global__
void vector_add(int *a, int *b, int *r, const size_t n)
{
    int tid = threadIdx.x;
    if (tid >= N)
        return;

    r[tid] = a[tid] + b[tid];
}


int main(int argc, char const *argv[])
{
    int host_a[N];
    int host_b[N];

    int *dev_a;
    int *dev_b;
    int *dev_r;

    // NOTE: this example does no error checking!
    cudaError_t err;

    // ---- ALLOCATE MEMORY ON DEVICE ---------
    // cudaMalloc(void **dev_ptr, size_t count)
    err = cudaMalloc(&dev_a, sizeof(int) * N);
    err = cudaMalloc(&dev_b, sizeof(int) * N);
    err = cudaMalloc(&dev_r, sizeof(int) * N);

    // ---- INITIALIZE DATA ON HOST -----------
    fill_array_count(host_a, N);
    fill_array_const(host_b, N, 10);

    // ---- COPY DATA OVER TO DEVICE ----------
    // cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind)
    err = cudaMemcpy(dev_a, host_a, sizeof(int) * N, cudaMemcpyHostToDevice);
    err = cudaMemcpy(dev_b, host_b, sizeof(int) * N, cudaMemcpyHostToDevice);

    // ---- PERFORM COMPUTATION ON DEVICE -----
    vector_add<<<128, N/128>>>(dev_a, dev_b, dev_r, N);
    // the <<<dim3 gridDim, dim3 blockDim>>> is a CUDA extension to launch kernels
    //      grids are made up of blocks
    //      blocks are made up of threads
    // this example only launches 1 block (gridDim of (1, 0, 0)), and N threads (blockDim of (N, 0, 0))

    // ---- COPY RESULT DATA BACK TO HOST ----
    int host_r[N];
    err = cudaMemcpy(host_r, dev_r, sizeof(int) * N, cudaMemcpyDeviceToHost);

    // verify results
    bool success = true;
    for (size_t i=0; i<N; ++i) {
        if (host_r[i] != host_a[i] + host_b[i]) {
            fprintf(stderr, "ERROR (%u): %d != %d + %d", i, host_r[i], host_a[i], host_b[i]);
            success = false;
            break;
        }
    }

    // free memory on device
    err = cudaFree(dev_a);
    err = cudaFree(dev_b);
    err = cudaFree(dev_r);

    if (success)
        printf("It worked!\n");
    else
        return 1;

    return 0;
}

