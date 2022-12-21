#ifndef VEC3_H
#define VEC3_H

#include <iostream>
// #include <math.h>
#include <stdlib.h>

class vec3 {
  public:
    __device__ __host__ vec3() : e{0,0,0} {}
    __device__ __host__ vec3(double e0, double e1, double e2) : e{e0, e1, e2} {}
    // vec3(vec3 v) : e{v.x, v.y, v.z} {}
    __device__ __host__ double x() const { return e[0]; }
    __device__ __host__ double y() const { return e[1]; }
    __device__ __host__ double z() const { return e[2]; }

    __device__ __host__ vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
    __device__ __host__ double operator[](int i) const { return e[i]; }
    __device__ __host__ double& operator[](int i) { return e[i]; }

    __device__ __host__ vec3& operator+=(const vec3 &v) {
      e[0] += v.e[0];
      e[1] += v.e[1];
      e[2] += v.e[2];
      return *this;
    }

    __device__ __host__ vec3& operator*=(const double t) {
      e[0] *= t;
      e[1] *= t;
      e[2] *= t;
      return *this;
    }

    __device__ __host__ vec3& operator/=(const double t) {
      return *this *= 1/t;
    }

    __device__ __host__
    double length() const {
      return sqrt(length_squared());
    }

    __device__ __host__
    double length_squared() const {
      return e[0]*e[0] + e[1]*e[1] + e[2]*e[2]; 
    }

    __host__
    void write_color(std::ostream &out, int samples_per_pixel) {
      // Divide color by number of samples taken and then gamma correct (scale from 0 to 1)
      // For gamma value of 2.0 (^1/gamma power), ^1/2
      auto scale = 1.0 / samples_per_pixel;
      auto r = sqrt(scale * e[0]);
      auto g = sqrt(scale * e[1]);
      auto b = sqrt(scale * e[2]);

      // Write translated color value [0, 255] of each color component
      out << static_cast<int>(256 * clamp(r, 0.0, 0.999)) << ' '
          << static_cast<int>(256 * clamp(g, 0.0, 0.999)) << ' '
          << static_cast<int>(256 * clamp(b, 0.0, 0.999)) << '\n';
    }

    __device__ inline static vec3 random(curandState s) {
      return vec3(random_double(s), random_double(s), random_double(s));
    }

    __device__ inline static vec3 random(double min, double max, curandState s) {
      return vec3(random_double(min, max, s), random_double(min, max, s), random_double(min, max, s));
    }

    __host__ inline static vec3 random_host() {
      return vec3(random_double_host(), random_double_host(), random_double_host());
    }

    __host__ inline static vec3 random(double min, double max) {
      return vec3(random_double_host(min, max), random_double_host(min, max), random_double_host(min, max));
    }


  public:
    double e[3];
};

// Utility functions

__host__
inline std::ostream& operator<<(std::ostream &out, const vec3 &v) {
  return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

__device__ __host__
inline vec3 operator+(const vec3 &u, const vec3 &v) {
  return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

__device__ __host__
inline vec3 operator-(const vec3 &u, const vec3 &v) {
  return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

__device__ __host__
inline vec3 operator*(const vec3 &u, const vec3 &v) {
  return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

__device__ __host__
inline vec3 operator*(float t, const vec3 &v) {
  return vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}

__device__ __host__
inline vec3 operator*(const vec3 &v, float t) {
  return t * v;
}

__device__ __host__
inline vec3 operator/(vec3 v, float t) {
  return (1/t) * v;
}

__device__ __host__
inline double dot(const vec3 &u, const vec3 &v) {
  return u.e[0]*v.e[0]
       + u.e[1]*v.e[1]
       + u.e[2]*v.e[2];
}

__device__ __host__
inline vec3 cross(const vec3 &u, const vec3 &v) {
  return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
              u.e[2] * v.e[0] - u.e[0] * v.e[2],
              u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

__device__ __host__
inline vec3 unit_vector(vec3 v) {
  return v / v.length();
}

// Replacement for Lambertian random_in_unit_sphere implementation
__device__
vec3 random_unit_vector(curandState s) {
  auto a = random_double(0, 2*pi, s);
  auto z = random_double(-1, 1, s);
  auto r = sqrt(1 - z*z);
  return vec3(r*cos(a), r*sin(a), z);
}

__device__
vec3 random_in_unit_sphere(curandState s) {
  while(true) {
    auto p = vec3::random(-1, 1, s);
    if (p.length_squared() < 1) return p;
  }
}

// Intuative Hack (Incorrect distribution)
__device__
vec3 random_in_hemisphere(const vec3& normal, curandState s) {
  vec3 in_unit_sphere = random_in_unit_sphere(s);
  if(dot(in_unit_sphere, normal) > 0.0) return in_unit_sphere; // In the same hemisphere as the normal
  else return -in_unit_sphere;
}

__device__ __host__
vec3 reflect(const vec3& v, const vec3& n) {
  return v - 2*dot(v, n)*n;
}

__device__ __host__
vec3 refract(const vec3& uv, const vec3& n, double etai_over_etat) {
  auto cos_theta = dot(-uv, n);
  vec3 r_out_parallel = etai_over_etat * (uv + cos_theta*n);
  vec3 r_out_perp = -sqrt(1.0 - r_out_parallel.length_squared()) * n;
  return r_out_parallel + r_out_perp;
}

__device__
vec3 random_in_unit_disk(curandState s) {
  while(true) {
    auto p = vec3(random_double(-1, 1, s), random_double(-1, 1, s), 0);
    if (p.length_squared() < 1) return p;
  }
}

__host__
vec3 random_in_unit_disk() {
  while(true) {
    auto p = vec3(random_double_host(-1, 1), random_double_host(-1, 1), 0);
    if (p.length_squared() < 1) return p;
  }
}

#endif
