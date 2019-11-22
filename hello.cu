/*
    Created based off of Cuda intro tutorial: https://devblogs.nvidia.com/even-easier-introduction-cuda/
    Compile with g++: g++ add.cpp -o add
    Complie with Cuda nvcc: nvcc add.cu -o add_cuda
    * Must rename file to *.cu in order to compile with Cuda
*/


#include <iostream>
#include <math.h>

// CPU Add
void add(int n, float* x, float* y) {
    for (int i = 0; i < n; i++) {
        y[i] = x[i] + y[i];
    }
}

// Single Thread GPU Add
__global__
void addCuda(int n, float* x, float* y) {
    for (int i = 0; i < n; i++) {
        y[i] = x[i] + y[i];
    }
}

// Parallel GPU Add
__global__
void addCudaParallel(int n, float* x, float* y) {
    int index = threadIdx.x;
    int stride = blockDim.x;

    for (int i = index; i < n; i += stride) {
        y[i] = x[i] + y[i];
    }
}

int main(void) {
    int numCalcs = 1 << 20; // 1 Million elements to calculate

    // CPU
    float* x = new float[numCalcs];
    float* y = new float[numCalcs];

    // CUDA
    float *xCuda, *yCuda;
    cudaMallocManaged(&xCuda, numCalcs * sizeof(float));
    cudaMallocManaged(&yCuda, numCalcs * sizeof(float));

    for (int i = 0; i < numCalcs; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    add(numCalcs, x, y); // Run cpu kernal
    addCuda<<<1, 1>>>(numCalcs, xCuda, yCuda);
    addCudaParallel<<<1, 256>>>(numCalcs, xCuda, yCuda);

    // Check for error
    float maxError = 0.0f;
    for (int i = 0; i < numCalcs; i++) {
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    }
    std::cout << "Max Error: " << maxError << std::endl;

    // Free CPU memory
    delete [] x;
    delete [] y;

    // Free Shared CPU & GPU memory
    cudaFree(xCuda);
    cudaFree(yCuda);

    return 0;
}
