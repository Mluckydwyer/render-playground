#ifndef RTWEEKEND_H
#define RTWEEKEND_H

#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
#include <curand.h>
#include <curand_kernel.h>

// Usings
using std::shared_ptr;
using std::make_shared;

// Constants
const double infinity = std::numeric_limits<double>::infinity();
const double pi = 3.1415926535897932385;


// Cuda Safe Random Numbers
__device__ curandState init_random(unsigned int seed) {
    curandState s;

    // seed a random number generator
    curand_init(seed, 0, 0, &s);
    return s;
}


// Utility Functions
inline double degrees_to_radians(double degrees) {
    return degrees * pi / 180;
}

__device__ __host__ inline double ffmin(double a, double b) { return a <= b ? a : b; }
__device__ __host__ inline double ffmax(double a, double b) { return a >= b ? a : b; }

__device__  inline double random_double(curandState s) {
    // Returns ar random number from [0,1)
    return curand_uniform(&s) / (RAND_MAX + 1.0);
}

__device__ inline double random_double(double min, double max, curandState s) {
    // Returns a random real in [min,max)
    return min + (max-min)*random_double(s);
}

__host__ inline double random_double_host() {
    // Returns ar random number from [0,1)
    return rand() / (RAND_MAX + 1.0);
}

__host__ inline double random_double_host(double min, double max) {
    // Returns a random real in [min,max)
    return min + (max-min)*random_double_host();
}

__device__ __host__ inline double clamp(double x, double min, double max) {
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

// Common Headers
#include "ray.h"
#include "vec3.h"

#endif