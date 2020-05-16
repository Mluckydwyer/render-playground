#ifndef RAY_H
#define RAY_H

#include "vec3.h"

class ray {
    public:
        __device__ __host__ ray() {}
        __device__ __host__ ray (const vec3& origin, const vec3& direction) : orig(origin), dir(direction) {}

        __device__ __host__ vec3 origin() const {return orig;}
        __device__ __host__ vec3 direction() const {return dir;}

        __device__ __host__
        vec3 at(double t) const {
            return orig + t * dir;
        }
    
    public:
        vec3 orig;
        vec3 dir;
};

#endif