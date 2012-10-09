/*
 * A very simple CUDA example adding two arrays of ints together.
 *
 * Shows common paradigm of copy input data to device, copy results back to host
 * Introduces many CUDA concepts
 *  - device pointers with cudaMalloc, cudaMemcpy
 *  - writing GPU kernel code (and getting threadIdx)
 *  - launching kernel
 *  - kernel launch dimensions, threads, blocks, grid
 *
 *  Danny George 2012
 */


#include <stdio.h>

void do_the_add(int *a, int *b, int *r, int i);



const int N = 512 * 1024;


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
    // convert from 2D launch to 1D array index
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= N)
        return;

    r[tid] = a[tid] + b[tid];

    // you can call __device__ functions from __global__ functions
    //do_the_add(a, b, r, tid);
}



// __device__ tells the compiler this function is called by the device and runs on the device
// __host__   tells the compiler to make another version to run on the host (normal function)
__device__ __host__
void do_the_add(int *a, int *b, int *r, int i)
{
    r[i] = a[i] + b[i];
}




int main(int argc, char const *argv[])
{
    int *host_a;
    int *host_b;
    int *host_r;

    int *dev_a;
    int *dev_b;
    int *dev_r;

    // NOTE: this example does no error checking!
    cudaError_t err;

    // ---- ALLOCATE MEMORY ON HOST -----------
    host_a = (int *)malloc(sizeof(int) * N);
    host_b = (int *)malloc(sizeof(int) * N);
    host_r = (int *)malloc(sizeof(int) * N);
    if (host_a == NULL || host_b == NULL || host_r == NULL) {
        fprintf(stderr, "malloc error on host\n");
        exit(1);
    }

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
    int threads_per_block = 128;
    int blocks_per_grid = ((N + threads_per_block - 1) / threads_per_block);    // integer div, ensures at least 1 block
    vector_add<<<blocks_per_grid, threads_per_block>>>(dev_a, dev_b, dev_r, N);
    // the <<<dim3 gridDim, dim3 blockDim>>> is a CUDA extension to launch kernels
    //      grids are made up of blocks
    //      blocks are made up of threads

    // ---- COPY RESULT DATA BACK TO HOST ----
    err = cudaMemcpy(host_r, dev_r, sizeof(int) * N, cudaMemcpyDeviceToHost);

    // verify results
    bool success = true;
    for (size_t i=0; i<N; ++i) {
        if (host_r[i] != host_a[i] + host_b[i]) {
            fprintf(stderr, "ERROR [index %u]: %d != %d + %d", i, host_r[i], host_a[i], host_b[i]);
            success = false;
            break;
        }
    }

    // ---- CLEANUP -------------------------
    // free memory on host
    free(host_a);
    free(host_b);
    free(host_r);

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

