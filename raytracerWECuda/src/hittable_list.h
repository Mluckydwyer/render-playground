#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "hittable.h"
#include <memory>
#include <vector>

using std::shared_ptr;
using std::make_shared;

class hittable_list: public hittable {
    public:
        hittable_list() {
            //count = 0;
            //total = N;
        }
        ~hittable_list() {}
        hittable_list(hittable* object) {
            add(object);
        }

        void clear() {}
        void add(hittable* object) { objects.push_back(object); }

        __host__ __device__ virtual bool hit(const ray& r, double tmin, double tmax, hit_record& rec) const;

    public:
        //int count;
        //int total;
        //hittable* device_objs[];
        std::vector<hittable*> objects;
        //thrust::device_vector<hittable*> objects;
};

__host__
bool hittable_list::hit(const ray& r, double t_min, double t_max, hit_record& rec) const {
    hit_record temp_rec;
    bool hit_anything = false;
    auto clostest_so_far = t_max;

    // for (int i = 0; i < objects.size(); i++) {
    //     const auto& object = objects[i];
    for (const auto& object : objects) {
        if (object->hit(r, t_min, clostest_so_far, temp_rec)) {
            hit_anything = true;
            clostest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    return hit_anything;
}

// ----- Device Class -----
class hittable_list_device: public hittable {
public:
    hittable_list_device() {}
    hittable_list_device(hittable_list* list) {
        cudaMallocManaged(&device_objs, sizeof(hittable*) * list.objects.size());
        cudaMemcpy(device_objs, list.objects[0], list.objects.size() * sizeof(hittable*));
        size = list.objects.size();
    }
    ~hittable_list_device() { cudaFree(device_objs); }

    void clear() {}

    __host__ __device__ virtual bool hit_device(const ray& r, double tmin, double tmax, hit_record& rec) const;

public:
    hittable *device_objs;
    int size;
};

__device__
bool hittable_list_device::hit_device(const ray& r, double t_min, double t_max, hit_record& rec) const {
    hit_record temp_rec;
    bool hit_anything = false;
    auto clostest_so_far = t_max;

    for (int i = 0; i < size; i++) {
        const auto& object = device_objs[i];
    // for (const auto& object : objects) {
        if (object->hit(r, t_min, clostest_so_far, temp_rec)) {
            hit_anything = true;
            clostest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    return hit_anything;
}

#endif