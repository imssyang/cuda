#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void addKernel(int *c, const int *a, const int *b) {
    int i = threadIdx.x;
    c[i] = a[i] * b[i] / 31415926;
    printf("----%d", i);
}

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size) {
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    int i = 0;
    cudaError_t cudaStatus = cudaErrorUnknown;

    if (cudaSuccess != cudaSetDevice(0)) goto Error; // Choose which GPU to run on
    if (cudaSuccess != cudaMalloc(&dev_c, size * sizeof(int))) goto Error; // Allocate GPU buffers for three vectors (two input, one output)
    if (cudaSuccess != cudaMalloc(&dev_a, size * sizeof(int))) goto Error;
    if (cudaSuccess != cudaMalloc(&dev_b, size * sizeof(int))) goto Error;
    if (cudaSuccess != cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice)) goto Error; // Copy input vectors from host memory to GPU buffers
    if (cudaSuccess != cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice)) goto Error;
    for (i = 0; i <= size/1024; i++)
        addKernel<<<1, 10>>>(dev_c, dev_a, dev_b); // Launch a kernel on the GPU with one thread for each element
        if (cudaSuccess != cudaGetLastError()) { printf("addKernel failed: %s\n", cudaGetErrorString(cudaStatus)); goto Error; }
        if (cudaSuccess != cudaDeviceSynchronize()) goto Error; // Waits for the kernel to finish, and returns any errors encountered during the launch
    if (cudaSuccess != cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost)) goto Error; // Copy output vector from GPU buffer to host memory
    cudaStatus = cudaSuccess;

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    return cudaStatus;
}

void runGPU() {
    const int size = 5;
    int* a = (int*)malloc(size * sizeof(int));
    int* b = (int*)malloc(size * sizeof(int));
    int* c = (int*)malloc(size * sizeof(int));
    int i = 0;

    for (i = 0; i < size; i++) {
        a[i] = i;
        b[i] = i * 10;
    }

    if (cudaSuccess != addWithCuda(c, a, b, size)) goto Error; // Add vectors in parallel
    printf("%u\n", c[size - 1]);

    if (cudaSuccess != cudaDeviceReset()) goto Error; // Called for profiling and tracing tools such as Nsight and Visual Profiler to show complete traces.

Error:
    free(a);
    free(b);
    free(c);
}

int main() {
    clock_t begin = clock();
    runGPU();
    clock_t end = clock();
    double times = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("time: %lf\n", times);
    return 0;
}
