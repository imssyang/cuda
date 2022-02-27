#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void addItem(int* c, const int* a, const int* b, const int i) {
    c[i] = a[i] * b[i] / 31415926;
}

int addVector(int* c, const int* a, const int* b, unsigned int size) {
    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;
    int status = 999;
    int i = 0;

    dev_c = (int*)malloc(size * sizeof(int));
    dev_a = (int*)malloc(size * sizeof(int));
    dev_b = (int*)malloc(size * sizeof(int));
    memcpy(dev_a, a, size * sizeof(int));
    memcpy(dev_b, b, size * sizeof(int));

    for (i = 0; i < size; i++)
        addItem(dev_c, dev_a, dev_b, i);

    memcpy(c, dev_c, size * sizeof(int));
    status = 0;

    free(dev_c);
    free(dev_a);
    free(dev_b);
    return status;
}

void runCPU() {
    const int size = 300000000;
    int* a = (int*)malloc(size * sizeof(int));
    int* b = (int*)malloc(size * sizeof(int));
    int* c = (int*)malloc(size * sizeof(int));
    int i = 0;

    for (i = 0; i < size; i++) {
        a[i] = i * 3;
        b[i] = i * 5;
    }

    if (0 != addVector(c, a, b, size)) goto Error; // Add vectors in sequential
    printf("%u\n", c[size-1]);

Error:
    free(a);
    free(b);
    free(c);
}

int main() {
    clock_t begin = clock();
    runCPU();
    clock_t end = clock();
    double times = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("time: %lf\n", times);
    return 0;
}

/*
4294967274
time: 31.254000
*/