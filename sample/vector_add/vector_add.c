#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define EXIT_ON_ERROR(condition, ...) do { \
    if (condition) { \
        fprintf(stderr, __VA_ARGS__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

void vectorAdd(const float* A, const float* B, float* C, int i) {
    C[i] = A[i] + B[i] + 0.0f;
}

int main() {
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
    for (int i = 0; i < numElements; i++) {
        vectorAdd(h_A, h_B, h_C, i);
        float precision = h_A[i] + h_B[i] - h_C[i];
        EXIT_ON_ERROR(precision > 1e-5, "Verify element[%d](%f+%f-%f=%f>0.00001) fail\n", i, h_A[i], h_B[i], h_C[i], precision);
    }
    clock_t end = clock();
    double duration = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("CPU kernel launch for elements(%u) duration(%lf)\n", numElements, duration);
    free(h_A);
    free(h_B);
    free(h_C);
    return 0;
}

/*
CPU kernel launch for elements(500000000) duration(9.901000)
*/