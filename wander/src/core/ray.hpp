#ifndef WANDER_RAY_H_
#define WANDER_RAY_H_

#include "vec3.h"

class ray {
    public:
        __device__ ray() {}
        __device__ ray (const vec3& origin, const vec3& direction) : orig(origin), dir(direction) {}

        __device__ vec3 origin() const {return orig;}
        __device__ vec3 direction() const {return dir;}
        __device__ vec3 at(double time) const {return orig + time * dir;}
    
    public:
        vec3 orig;
        vec3 dir;
};

#endif // WANDER_RAY_H_