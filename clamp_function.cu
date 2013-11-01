#include <stdio.h>


template <typename T>
inline __device__
T clamp(T val, T vMin, T vMax) {
    return min(max(val, vMin), vMax);
}



__global__
void kernel(int* input, int* output, int length) {
    int tID = threadIdx.x;
    if (tID >= length) {
        return;
    }

    //output[tID] = min(max(input[tID], 100), 130);
    output[tID] = clamp<int>(input[tID], 100, 130);
}


int main(int argc, char const *argv[])
{
    int *d_input;
    int *d_output;

    int input[10] = { 10, 50, 100, 150, 131, 99, 155, 10000, 0, 100 };
    int output[10];

    cudaMalloc(&d_input,  sizeof(int) * 10);
    cudaMalloc(&d_output, sizeof(int) * 10);

    cudaMemcpy(d_input, input, sizeof(int) * 10, cudaMemcpyHostToDevice);

    kernel<<<1, 256>>>(d_input, d_output, 10);

    cudaMemcpy(output, d_output, sizeof(int) * 10, cudaMemcpyDeviceToHost);

    for (int i=0; i<10; ++i) {
        printf("%d: %5d -> %5d\n", i, input[i], output[i]);
    }

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
