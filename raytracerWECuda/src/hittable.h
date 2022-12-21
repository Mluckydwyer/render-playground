#ifndef HITTABLE_H
#define HITTABLE_H

#include "rtweekend.h"
#include "ray.h"
class material;

struct hit_record {
    vec3 p;
    vec3 normal;
    material* mat_ptr;
    double t;
    bool front_face;

    __device__ __host__ inline void set_face_normal(const ray& r, const vec3 outward_normal) {
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
 };

class hittable {
    public:
        // ~hittable() {}
        __host__ __device__ virtual bool hit(const ray& r, double t_min, double t_max, hit_record& rec) const = 0;
};

#endif