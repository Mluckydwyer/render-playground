#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "hittable.h"
#include <memory>
#include <thrust/device_vector.h>
#include <vector>

using std::shared_ptr;
using std::make_shared;

class hittable_list: public hittable {
    public:
        hittable_list() {}
        ~hittable_list() {}
        hittable_list(hittable* object) { add(object); }

        void clear() {}
        void add(hittable* object) { objects.push_back(object); }

        __device__ __host__ virtual bool hit(const ray& r, double tmin, double tmax, hit_record& rec) const;

    public:
        std::vector<hittable*> objects;
};

__device__ __host__
bool hittable_list::hit(const ray& r, double t_min, double t_max, hit_record& rec) const {
    hit_record temp_rec;
    bool hit_anything = false;
    auto clostest_so_far = t_max;

    for (int i = 0; i < objects.size(); i++) {
        const auto& object = objects[i];
        if (object->hit(r, t_min, clostest_so_far, temp_rec)) {
            hit_anything = true;
            clostest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    return hit_anything;
}

#endif