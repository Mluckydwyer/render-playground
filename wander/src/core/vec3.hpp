#ifndef WANDER_VEC3_H_
#define WANDER_VEC3_H_

#include <iostream>
#include <math.h>
#include <stdlib.h>

class vec3 {
  public:
    __host__ __device__ vec3() : e{0,0,0} {}
    __host__ __device__ vec3(double e0, double e1, double e2) : e{e0, e1, e2} {}
    __host__ __device__ inline double x() const { return e[0]; }
    __host__ __device__ inline double y() const { return e[1]; }
    __host__ __device__ inline double z() const { return e[2]; }
    __host__ __device__ inline double r() const { return e[0]; }
    __host__ __device__ inline double g() const { return e[1]; }
    __host__ __device__ inline double b() const { return e[2]; }

    __host__ __device__ inline const vec3& operator+() const { return *this; }
    __host__ __device__ inline vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
    __host__ __device__ inline double operator[](int i) const { return e[i]; }
    __host__ __device__ inline double& operator[](int i) { return e[i]; }

    __host__ __device__ inline vec3& operator+=(const vec3 &v);
    __host__ __device__ inline vec3& operator-=(const vec3 &v);
    __host__ __device__ inline vec3& operator*=(const vec3 &v);
    __host__ __device__ inline vec3& operator/=(const vec3 &v);
    __host__ __device__ inline vec3& operator*=(const double t);
    __host__ __device__ inline vec3& operator/=(const double t);

    __host__ __device__ inline double length() const { return sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]); }
    __host__ __device__ inline double length_squared() const { return e[0]*e[0] + e[1]*e[1] + e[2]*e[2]; }
    __host__ __device__ inline void make_unit_vector();

  public:
    double e[3];
};

using point3 = vec3; // 3D Point Alias
using color = vec3; // RGB Color Alias

// Utility functions

inline std::istream& operator>>(std::istream &is, vec3 &v) {
  return is >> v.e[0] >> v.e[1] >> v.e[2];
}

inline std::ostream& operator<<(std::ostream &os, const vec3 &v) {
  return os << v.e[0] << " " << v.e[1] << " " << v.e[2];
}

__host__ __device__ inline void vec3::make_unit_vector() {
  double d = 1.0 / sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]);
  e[0] *= d;
  e[1] *= d;
  e[2] *= d;
}

__host__ __device__ inline vec3 operator+(const vec3 &u, const vec3 &v) {
  return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

__host__ __device__ inline vec3 operator-(const vec3 &u, const vec3 &v) {
  return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3 &u, const vec3 &v) {
  return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

__host__ __device__ inline vec3 operator/(const vec3 &u, const vec3 &v) {
  return vec3(u.e[0] / v.e[0], u.e[1] / v.e[1], u.e[2] / v.e[2]);
}

__host__ __device__ inline vec3 operator*(double t, const vec3 &v) {
  return vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3 &v, double t) {
  return vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}

__host__ __device__ inline vec3 operator/(double t, const vec3 &v) {
  return vec3(v.e[0]/t, v.e[1]/t, v.e[2]/t);
}

__host__ __device__ inline vec3 operator/(const vec3 &v, double t) {
  return vec3(v.e[0]/t, v.e[1]/t, v.e[2]/t);
}

__host__ __device__ inline float dot(const vec3 &u, const vec3 &v) {
  return u.e[0]*v.e[0] + u.e[1]*v.e[1]  + u.e[2]*v.e[2];
}

__host__ __device__ inline vec3 cross(const vec3 &u, const vec3 &v) {
  return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
              u.e[2] * v.e[0] - u.e[0] * v.e[2],
              u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

__host__ __device__ inline vec3& vec3::operator+=(const vec3 &v) {
  e[0] += v.e[0];
  e[1] += v.e[1];
  e[2] += v.e[2];
  return *this;
}

__host__ __device__ inline vec3& vec3::operator-=(const vec3 &v) {
  e[0] -= v.e[0];
  e[1] -= v.e[1];
  e[2] -= v.e[2];
  return *this;
}

__host__ __device__ inline vec3& vec3::operator*=(const vec3 &v) {
  e[0] *= v.e[0];
  e[1] *= v.e[1];
  e[2] *= v.e[2];
  return *this;
}

__host__ __device__ inline vec3& vec3::operator/=(const vec3 &v) {
  e[0] /= v.e[0];
  e[1] /= v.e[1];
  e[2] /= v.e[2];
  return *this;
}

__host__ __device__ inline vec3& vec3::operator*=(const double t) {
  e[0] *= t;
  e[1] *= t;
  e[2] *= t;
  return *this;
}

__host__ __device__ inline vec3& vec3::operator/=(const double t) {
  e[0] /= t;
  e[1] /= t;
  e[2] /= t;
  return *this;
}

__host__ __device__ inline vec3 unit_vector(vec3 v) {
  return v / v.length();
}

#endif //WANDER_VEC3_H_
