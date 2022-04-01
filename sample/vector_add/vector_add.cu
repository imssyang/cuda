
/**
 * Vector addition: C = A + B.
 */

#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>

#define EXIT_ON_ERROR(condition, ...) do { \
    if (condition) { \
        fprintf(stderr, __VA_ARGS__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

__global__ void vectorAdd(const float* A, const float* B, float* C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        C[i] = A[i] + B[i] + 0.0f;
    }
    //printf("[%u](%u-%u-%u) (%f=%f+%f)\n", i, blockDim.x, blockIdx.x, threadIdx.x, C[i], A[i], B[i]);
}

int main(void) {
    cudaError_t err = cudaSuccess;

    int numElements = 500000000;
    size_t size = numElements * sizeof(float);

    float* h_A = (float*)malloc(size); EXIT_ON_ERROR(h_A == NULL, "Alloc h_A fail!");
    float* h_B = (float*)malloc(size); EXIT_ON_ERROR(h_B == NULL, "Alloc h_B fail!");
    float* h_C = (float*)malloc(size); EXIT_ON_ERROR(h_C == NULL, "Alloc h_C fail!");

    for (int i = 0; i < numElements; ++i) {
        h_A[i] = i;      // rand() / (float)RAND_MAX;
        h_B[i] = i * 10; // rand() / (float)RAND_MAX;
    }

    clock_t begin = clock();
    float* d_A = NULL;
    float* d_B = NULL;
    float* d_C = NULL;
    err = cudaMalloc((void**)& d_A, size); EXIT_ON_ERROR(err != cudaSuccess, "Alloc d_A fail: %s", cudaGetErrorString(err));
    err = cudaMalloc((void**)& d_B, size); EXIT_ON_ERROR(err != cudaSuccess, "Alloc d_B fail: %s", cudaGetErrorString(err));
    err = cudaMalloc((void**)& d_C, size); EXIT_ON_ERROR(err != cudaSuccess, "Alloc d_C fail: %s", cudaGetErrorString(err));

    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice); EXIT_ON_ERROR(err != cudaSuccess, "Copy h_A to d_A fail: %s\n", cudaGetErrorString(err));
    err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice); EXIT_ON_ERROR(err != cudaSuccess, "Copy h_B to d_B fail: %s\n", cudaGetErrorString(err));

    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd << <blocksPerGrid, threadsPerBlock >> > (d_A, d_B, d_C, numElements);
    err = cudaGetLastError(); EXIT_ON_ERROR(err != cudaSuccess, "Launch vectorAdd kernel fail: %s\n", cudaGetErrorString(err));

    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost); EXIT_ON_ERROR(err != cudaSuccess, "Copy d_C to h_C fail: %s\n", cudaGetErrorString(err));

    for (int i = 0; i < numElements; ++i) {
        float precision = h_A[i] + h_B[i] - h_C[i];
        EXIT_ON_ERROR(precision > 1e-5, "Verify element[%d](%f+%f-%f=%f>0.00001) fail\n", i, h_A[i], h_B[i], h_C[i], precision);
    }

    err = cudaFree(d_A); EXIT_ON_ERROR(err != cudaSuccess, "Free d_A fail: %s\n", cudaGetErrorString(err));
    err = cudaFree(d_B); EXIT_ON_ERROR(err != cudaSuccess, "Free d_B fail: %s\n", cudaGetErrorString(err));
    err = cudaFree(d_C); EXIT_ON_ERROR(err != cudaSuccess, "Free d_C fail: %s\n", cudaGetErrorString(err));
    clock_t end = clock();
    double duration = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("CUDA kernel launch blocks(%u) threads(%u) for elements(%u) duration(%lf)\n", blocksPerGrid, threadsPerBlock, numElements, duration);

    free(h_A);
    free(h_B);
    free(h_C);
    return 0;
}

/**
CUDA kernel launch blocks(1953125) threads(256) for elements(500000000) duration(2.435000)

[1](1-1-0) (11.000000=1.000000+10.000000)
[4](1-4-0) (44.000000=4.000000+40.000000)
[2](1-2-0) (22.000000=2.000000+20.000000)
[3](1-3-0) (33.000000=3.000000+30.000000)
[0](1-0-0) (0.000000=0.000000+0.000000)
CUDA kernel launch blocks(5) threads(1) for elements(5)

[2](2-1-0) (22.000000=2.000000+20.000000)
[3](2-1-1) (33.000000=3.000000+30.000000)
[4](2-2-0) (44.000000=4.000000+40.000000)
[5](2-2-1) (0.000000=0.000000+0.000000)
[0](2-0-0) (0.000000=0.000000+0.000000)
[1](2-0-1) (11.000000=1.000000+10.000000)
CUDA kernel launch blocks(3) threads(2) for elements(5)

[3](3-1-0) (33.000000=3.000000+30.000000)
[4](3-1-1) (44.000000=4.000000+40.000000)
[5](3-1-2) (0.000000=0.000000+0.000000)
[0](3-0-0) (0.000000=0.000000+0.000000)
[1](3-0-1) (11.000000=1.000000+10.000000)
[2](3-0-2) (22.000000=2.000000+20.000000)
CUDA kernel launch blocks(2) threads(3) for elements(5)

[4](4-1-0) (44.000000=4.000000+40.000000)
[5](4-1-1) (0.000000=0.000000+0.000000)
[6](4-1-2) (0.000000=0.000000+0.000000)
[7](4-1-3) (0.000000=0.000000+0.000000)
[0](4-0-0) (0.000000=0.000000+0.000000)
[1](4-0-1) (11.000000=1.000000+10.000000)
[2](4-0-2) (22.000000=2.000000+20.000000)
[3](4-0-3) (33.000000=3.000000+30.000000)
CUDA kernel launch blocks(2) threads(4) for elements(5)

[0](5-0-0) (0.000000=0.000000+0.000000)
[1](5-0-1) (11.000000=1.000000+10.000000)
[2](5-0-2) (22.000000=2.000000+20.000000)
[3](5-0-3) (33.000000=3.000000+30.000000)
[4](5-0-4) (44.000000=4.000000+40.000000)
CUDA kernel launch blocks(1) threads(5) for elements(5)
*/
