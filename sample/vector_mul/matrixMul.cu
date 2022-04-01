
/**
 * Matrix multiplication (CUDA Kernel) on the device: C = A * B
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

template <typename T>
void check(T result, char const* const func, const char* const file,
    int const line) {
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), cudaGetErrorName(result), func);
        exit(EXIT_FAILURE);
    }
}

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

void printDim3(const char* prefix, const float* data, const dim3& d3, bool onlyOne = true) {
    if (prefix)
        printf("%s (%d,%d):", prefix, d3.x, d3.y);
    if (!onlyOne)
        printf("\n");
    for (int i = 0; i < d3.y; i++) {
        for (int j = 0; j < d3.x; j++) {
            int pos = i * d3.x + j;
            printf("%.0f ", data[pos]);
            if (onlyOne)
                printf("\n");
            return;
        }
        printf("\n");
    }
}

template <int BLOCK_SIZE> __global__ void MatrixMulCUDA(float* C, float* A, float* B, int wA, int wB) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int aBegin = BLOCK_SIZE * by * wA; // Index of the first sub-matrix of A processed by the block
    int bBegin = BLOCK_SIZE * bx;      // Index of the first sub-matrix of B processed by the block
    int aStep = BLOCK_SIZE;            // Step size used to iterate through the sub-matrices of A
    int bStep = BLOCK_SIZE * wB;       // Step size used to iterate through the sub-matrices of B
    int aEnd = aBegin + wA - 1;        // Index of the last sub-matrix of A processed by the block

    // Csub is used to store the element of the block sub-matrix that is computed by the thread
    float Csub = 0;
    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
        // Declaration of the shared memory array As and Bs used to store the sub-matrix
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from device memory to shared memory; each thread loads one element of each matrix
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together; each thread computes one element of the block sub-matrix
#pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Csub += As[ty][k] * Bs[k][tx];
        }

        // Synchronize to make sure that the preceding computation is done before loading two new sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory; each thread writes one element
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    int cPos = c + wB * ty + tx;
    C[cPos] = Csub;
    //printf("block(%u,%u) thread(%u,%u) A(%u-%u:%u) B(%u-%u) C(%u:%u)\n", bx, by, tx, ty, aBegin, aStep, aEnd, bBegin, bStep, cPos, Csub);
}

void MatrixMultiply(int block_size, const dim3& dimsA, const dim3& dimsB) {
    dim3 dimsC(dimsB.x, dimsA.y, 1);

    // Allocate host memory
    float *h_A, *h_B, *h_C;
    unsigned int size_A = dimsA.x * dimsA.y;
    unsigned int size_B = dimsB.x * dimsB.y;
    unsigned int size_C = dimsC.x * dimsC.y;
    unsigned int mem_size_A = sizeof(float) * size_A;
    unsigned int mem_size_B = sizeof(float) * size_B;
    unsigned int mem_size_C = sizeof(float) * size_C;
    checkCudaErrors(cudaMallocHost(&h_A, mem_size_A));
    checkCudaErrors(cudaMallocHost(&h_B, mem_size_B));
    checkCudaErrors(cudaMallocHost(&h_C, mem_size_C));

    // Initialize host memory
    for (int i = 0; i < dimsA.y; i++) {
        for (int j = 0; j < dimsA.x; j++) {
            int pos = i * dimsA.x + j;
            h_A[pos] = (float)(pos + 1);
        }
    }
    printDim3("MatrixA", h_A, dimsA);

    for (int i = 0; i < dimsB.y; i++) {
        for (int j = 0; j < dimsB.x; j++) {
            int pos = i * dimsB.x + j;
            h_B[pos] = (float)((i + 1) * 10 + j);
        }
    }
    printDim3("MatrixB", h_B, dimsB);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_A), mem_size_A));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_B), mem_size_B));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_C), mem_size_C));

    // Allocate non-blocking stream
    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // Copy host memory to device
    checkCudaErrors(cudaMemcpyAsync(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice, stream));

    // Allocate CUDA events that we'll use for timing
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    // Setup execution parameters
    int wA = dimsA.x;
    int wB = dimsB.x;
    dim3 threads(block_size, block_size);
    dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);
    printf("grid(%u,%u) threads(%u,%u)\n", grid.x, grid.y, threads.x, threads.y);

    // Record the start event
    checkCudaErrors(cudaEventRecord(start, stream));

    // Execute the kernel
    int nIter = 300;
    for (int i = 0; i < nIter; i++) {
        MatrixMulCUDA<32> << <grid, threads, 0, stream >> > (d_C, d_A, d_B, wA, wB);
    }

    // Record the stop event
    checkCudaErrors(cudaEventRecord(stop, stream));

    // Wait for the stop event to complete
    checkCudaErrors(cudaEventSynchronize(stop));

    float msecTotal = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    // Compute and print the performance
    float msecPerMatrixMul = msecTotal / nIter;
    double flopsPerMatrixMul = 2.0 * static_cast<double>(dimsA.x) * static_cast<double>(dimsA.y) * static_cast<double>(dimsB.x);
    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
    printf("Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops, WorkgroupSize= %u threads/block\n",
        gigaFlops, msecPerMatrixMul, flopsPerMatrixMul, threads.x * threads.y);

    // Copy result from device to host
    checkCudaErrors(cudaMemcpyAsync(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost, stream));
    checkCudaErrors(cudaStreamSynchronize(stream));
    printDim3("MatrixC", h_C, dimsC);

    // Clean up memory
    checkCudaErrors(cudaFreeHost(h_A));
    checkCudaErrors(cudaFreeHost(h_B));
    checkCudaErrors(cudaFreeHost(h_C));
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    checkCudaErrors(cudaStreamDestroy(stream));
}

int main(void) {
    int block_size = 32;
    dim3 dimsA(5 * 2 * block_size, 5 * 2 * block_size, 1);
    dim3 dimsB(5 * 4 * block_size, 5 * 2 * block_size, 1);
    if (dimsA.x != dimsB.y) {
        printf("Error: outer matrix dimensions must be equal. (%d != %d)\n", dimsA.x, dimsB.y);
        exit(EXIT_FAILURE);
    }

    checkCudaErrors(cudaProfilerStart());
    MatrixMultiply(block_size, dimsA, dimsB);
    checkCudaErrors(cudaProfilerStop());

    exit(EXIT_SUCCESS);
}


/*
NOTE: Results may vary when GPU Boost is enabled.

MatrixA (3,2):
1 2 3
4 5 6
MatrixB (2,3):
10 11
20 21
30 31
grid(2,2) threads(1,1)
block(1,0) thread(0,0) A(0-1:2) B(1-2) C(1:0)
block(0,1) thread(0,0) A(3-1:5) B(0-2) C(2:0)
block(1,1) thread(0,0) A(3-1:5) B(1-2) C(3:0)
block(0,0) thread(0,0) A(0-1:2) B(0-2) C(0:0)
MatrixC (2,2):
140 146
320 335

MatrixA (320,320):1
MatrixB (640,320):10
grid(20,10) threads(32,32)
Performance= 71.21 GFlop/s, Time= 1.841 msec, Size= 131072000 Ops, WorkgroupSize= 1024 threads/block
MatrixC (640,320):109739096
*/
