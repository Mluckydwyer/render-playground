#ifndef WANDER_COMMON_H_
#define WANDER_COMMON_H_

// Basic Cuda Error Checkign function
#define checkCudaErrors(val) checkCuda( (val), #val, __FILE__, __LINE__ )

void checkCuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n" << cudaGetErrorString(result) << '\n';
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

#include "vec3.hpp"


#endif // WANDER_COMMON_H_