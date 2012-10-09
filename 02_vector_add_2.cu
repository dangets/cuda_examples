/**
 * An extension of 01_vector_add.cu
 *
 * Shows common solution of limited block dimension launch parameters
 * Shows multidimensional kernel launches and code
 * Shows time measurement of CUDA code
 *
 * Emphasize awareness of memory transfers -
 *    initialize data on GPU when possible
 *    GPU compute speedup must outweigh data transfer times
 *
 * Danny George 2012
 */


#include <stdio.h>

#include "CUDATimer.cuh"


#define imin(a,b)          (a<b?a:b)

// (feel free to increase this number, but remember 3x this will be allocated on host)
const size_t N = 64 * 1024 * 1024;
const size_t threadsPerBlock = 256;
const size_t blocksPerGrid = imin(32, (N+threadsPerBlock-1) / threadsPerBlock);

// Max launch sizes on current GPUs
//  max threadsPerBlock is 1024 (across ALL dimensions) though using 256 is common
//  max blocksPerGrid is 65535 PER dimension
//      (2 dimensions possible on compute capability 1.x & 2.x cards)


/// ------------------ HOST INITIALIZATION FUNCTIONS ---------------------------

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


/// ------------------ DEVICE INITIALIZATION FUNCTIONS -------------------------

__global__
void dev_fill_array_count(int *arr, const size_t n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < n) {
        arr[tid] = tid;
        tid += blockDim.x * gridDim.x;
    }
}


__global__
void dev_fill_array_const(int *arr, const size_t n, const int val)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < n) {
        arr[tid] = val;
        tid += blockDim.x * gridDim.x;
    }
}


/// ------------------ DEVICE ADD FUNCTION -------------------------------------

__global__
void vector_add(int *a, int *b, int *r, const size_t n)
{
    // use threadIdx AND blockIdx for unique thread index
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // each thread is responsible for MORE than one output array index
    while (tid < n) {
        r[tid] = a[tid] + b[tid];
        tid += blockDim.x * gridDim.x;
    }
}



int main(int argc, char const *argv[])
{
    int *host_a = (int *)malloc(sizeof(int) * N);
    int *host_b = (int *)malloc(sizeof(int) * N);
    int *host_r = (int *)malloc(sizeof(int) * N);

    if (host_a == NULL || host_b == NULL || host_r == NULL) {
        fprintf(stderr, "malloc error on host\n");
        exit(1);
    }

    int *dev_a;
    int *dev_b;
    int *dev_r;

    CUDATimer timer;
    float init_time, memcpy_time, compute_time;

    // NOTE: this example does no error checking!
    cudaError_t err;

    // ---- ALLOCATE MEMORY ON DEVICE ---------
    err = cudaMalloc(&dev_a, sizeof(int) * N);
    err = cudaMalloc(&dev_b, sizeof(int) * N);
    err = cudaMalloc(&dev_r, sizeof(int) * N);

    printf("N = %d\n", N);
    printf("threadsPerBlock = %d\n", threadsPerBlock);
    printf("blocksPerGrid   = %d\n", blocksPerGrid);
    printf("\n");


    // ---- comparison of initialization on host and memcpy vs init on device -----
    printf("Compare host init & memcpy vs. init on device\n");
    printf("=============================================\n");
    timer.start();
    fill_array_count(host_a, N);
    init_time = timer.get_elapsed_time_sync();

    timer.start();
    err = cudaMemcpy(dev_a, host_a, sizeof(int) * N, cudaMemcpyHostToDevice);
    memcpy_time = timer.get_elapsed_time_sync();

    printf("host init time: %g ms\n", init_time);
    printf("upload time:    %g ms\n", memcpy_time);
    printf("total:          %g ms\n", init_time + memcpy_time);
    printf("---------------------------\n");

    // same initialization but done on device
    timer.start();
    dev_fill_array_count<<<blocksPerGrid, threadsPerBlock>>>(dev_a, N);
    init_time = timer.get_elapsed_time_sync();
    printf("dev init time:  %g ms\n", init_time);
    printf("\n");

    // make sure they are equal (host_a is still initialized)
    err = cudaMemcpy(host_r, dev_a, sizeof(int) * N, cudaMemcpyDeviceToHost);
    for (size_t i=0; i<N; ++i) {
        if (host_r[i] != host_a[i]) {
            fprintf(stderr, "ERROR - dev_init != host_init\n");
            exit(1);
        }
    }


    // ---- INITIALIZE DATA ON DEVICE ---------
    dev_fill_array_count<<<blocksPerGrid, threadsPerBlock>>>(dev_a, N);
    dev_fill_array_const<<<blocksPerGrid, threadsPerBlock>>>(dev_b, N, 10);

    // ---- PERFORM COMPUTATION ON DEVICE -----
    printf("Computation\n");
    printf("=============================================\n");
    timer.start();
    vector_add<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_r, N);
    compute_time = timer.get_elapsed_time_sync();

    // ---- COPY RESULT DATA BACK TO HOST ----
    timer.start();
    err = cudaMemcpy(host_r, dev_r, sizeof(int) * N, cudaMemcpyDeviceToHost);
    memcpy_time = timer.get_elapsed_time_sync();

    printf("Compute time:  %g ms\n", compute_time);
    printf("Download time: %g ms\n", memcpy_time);


    // ... verify results omitted ...


    free(host_a);
    free(host_b);
    free(host_r);

    // free memory on device
    err = cudaFree(dev_a);
    err = cudaFree(dev_b);
    err = cudaFree(dev_r);

    return 0;
}

