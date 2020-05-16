#ifndef AABB_H
#define AABB_H

#include "rtweekend.h"

class aabb {
    public:
        aabb() {}
        aabb(const vec3& a, const vec3& b) {_min = a; _max = b;}

        vec3 min() const {return _min;}
        vec3 max() const {return _max;}

        bool hit(const ray& r, double tmin, double tmax) const {
            for (int a = 0; a < 3; a++) {
                auto invD = 1.0f / r.direction()[a];
                auto t0 = (min()[a] - r.origin()[a]) * invD;
                auto t1 = (max()[a] - r.origin()[a]) * invD;

                if (invD < 0.0f)
                    std::swap(t0, t1);

                tmin = t0 > tmin ? t0 : tmin;
                tmax = t1 < tmax ? t0 : tmax;

                if (tmax <= tmin)
                    return false;
            }

            return true;
        }

    public:
        vec3 _min;
        vec3 _max;
};

aabb surrounding_box(aabb b0, aabb b1) {
    vec3 b0_min = b0.min();
    vec3 b0_max = b0.max();
    vec3 b1_min = b1.min();
    vec3 b1_max = b1.max();

    vec3 lower( ffmin(b0_min.x(), b1_min.x()),
                ffmin(b0_min.y(), b1_min.y()),
                ffmin(b0_min.z(), b1_min.z()));
    vec3 upper( ffmin(b0_max.x(), b1_max.x()),
                ffmin(b0_max.y(), b1_max.y()),
                ffmin(b0_max.z(), b1_max.z()));
    return aabb(lower, upper);
}

#endif