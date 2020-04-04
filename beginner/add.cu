/*
    Created based off of Cuda intro tutorial: https://devblogs.nvidia.com/even-easier-introduction-cuda/
    Compile with g++: g++ add.cpp -o add
    Complie with Cuda nvcc: nvcc add.cu -o add_cuda
    * Must rename file to *.cu in order to compile with Cuda
*/


#include <iostream>
#include <string>
#include <math.h>
#include <cuda_profiler_api.h>
using namespace std;

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

// Parallel Block GPU Add
__global__
void addCudaParallelBlock(int n, float* x, float* y) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride) {
        y[i] = x[i] + y[i];
    }
}

void calcError(string label, float *y, float expected, int n) {
    // Check for error
    float maxError = 0.0f;
    float totalError = 0.0f;
    int totalOff = 0;
    for (int i = 0; i < n; i++) {
        maxError = fmax(maxError, fabs(y[i] - expected));
        totalError += y[i] - expected;
        if (y[i] - expected != 0.0) totalOff++;
    }
    std::cout << label << std::endl;
    std::cout << "Max Error: " << maxError << std::endl;
    std::cout << "Total Error: " << totalError << std::endl;
    std::cout << "Total Off: " << totalOff << std::endl;

    // delete &maxError;
    // delete &totalError;
    // delete &totalOff;
}

void reset(float* x, float* y, int numCalcs) {
    for (int i = 0; i < numCalcs; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }
}

int main(void) {
    int numCalcs = 1 << 20; // 1 Million elements to calculate
    float *x, *y;
    cudaMallocManaged(&x, numCalcs*sizeof(float));
    cudaMallocManaged(&y, numCalcs*sizeof(float));
    
    // Threading Parameters
    int blockSize = 256;
    int numBlocks = (numCalcs + blockSize - 1) / blockSize;

    // Sinlge Threaded GPU
    // reset(x, y, numCalcs);
    // addCuda<<<1, 1>>>(numCalcs, x, y); // Run addCuda on GPU
    // cudaDeviceSynchronize(); // Wait for GPU to finish
    // calcError("addCuda()", y, 3.0f, numCalcs); // Calculate errors if any

    // Multithreaded GPU
    // reset(x, y, numCalcs);
    // addCudaParallel<<<1, blockSize>>>(numCalcs, x, y); // Run addCudaParallel on GPU
    // cudaDeviceSynchronize(); // Wait for GPU to finish
    // calcError("addCudaParallel()", y, 3.0f, numCalcs); // Calculate errors if any

    // Multiblock GPU
    reset(x, y, numCalcs);
    addCudaParallelBlock<<<numBlocks, blockSize>>>(numCalcs, x, y); // Run addCudaParallel on GPU
    cudaDeviceSynchronize(); // Wait for GPU to finish
    calcError("addCudaParallelBlock()", y, 3.0f, numCalcs); // Calculate errors if any



    // Free Shared CPU & GPU memory
    cudaFree(x);
    cudaFree(y);

    return 0;
}
