#include "rtweekend.h"

#include "hittable_list.h"
#include "sphere.h"
#include "camera.h"
#include "material.h"

#include <iostream>

// Command Line Options: http://www.cplusplus.com/articles/DEN36Up4/

cudaError_t checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

__host__ __device__
vec3 ray_color_scatter(const ray& r, const hittable& world, int depth) {
	hit_record rec;

	// If bounce limit is exceeded, no light is gathered
	if (depth <= 0) return vec3(0, 0, 0);

	if (world.hit(r, 0.00001, infinity, rec)) {
		ray scattered;
		vec3 attenuation;

		if (rec.mat_ptr->scatter(r, rec, attenuation, scattered))
			return attenuation * ray_color_scatter(scattered, world, depth-1);
		
		return vec3(0, 0, 0);
	}

	vec3 unit_direction = unit_vector(r.direction());
	auto t = 0.5 * (unit_direction.y() + 1.0);
	return (1.0 - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
}

__global__
void ray_color(vec3 colors[], const ray rays[], const hittable& world, int depth, int samples, int N) {
	int thread_id = (threadIdx.y + blockDim.y * threadIdx.z) * blockDim.x + threadIdx.x;
	int block_id = (blockIdx.y + gridDim.y * blockIdx.z) * gridDim.x + blockIdx.x;
	int block_size = blockDim.x * blockDim.y * blockDim.z;
	int grid_size = gridDim.x * gridDim.y * gridDim.z;
	
	int id = thread_id + block_id * block_size;
	int stride = block_size * grid_size;
		
	hit_record rec;
	vec3 temp_color(0, 0, 0);
	ray r;
	vec3 color;

	for (int i = id; i < N; i += stride) {
		r = rays[i];

		if (world.hit(r, 0.00001, infinity, rec)) {
			ray scattered;
			vec3 attenuation;

			if (rec.mat_ptr->scatter(r, rec, attenuation, scattered)) {
				*color = attenuation * ray_color_scatter(scattered, world, depth-1);
			}
			else {
				color = vec3(0, 0, 0);
			}			
		}
		else {
			vec3 unit_direction = unit_vector(r.direction());
			auto t = 0.5 * (unit_direction.y() + 1.0);
			color = (1.0 - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);			
		}

		colors[i/samples] = colors[i/samples] + color;
	}
}

hittable_list random_scene() {
	hittable_list world;

	// Ground
	lambertian ground_sphere_mat = lambertian(vec3(0.5, 0.5, 0.5));
	sphere ground_sphere = sphere(vec3(0, -1000, 0), 1000, &ground_sphere_mat);
	world.add(&ground_sphere);

	for (int a = -11; a < 11; a++) {
		for (int b = -11; b < 11; b++) {
			auto choose_mat = random_double();
			vec3 center(a + 0.9*random_double(), 0.2, b + 0.9*random_double());

			if ((center - vec3(4, 0.2, 0)).length() > 0.9) {
				if (choose_mat < 0.8) {
					// Diffuse Material
					auto albedo = vec3::random() * vec3::random();
					lambertian diffuse_sphere_mat = lambertian(albedo);
					sphere diffuse_sphere = sphere(center, 0.2, &diffuse_sphere_mat);
					world.add(&diffuse_sphere);
				} else if (choose_mat < 0.95) {
					// Metal Material
					auto albedo = vec3::random(0.5, 1);
					auto fuzz = random_double(0, 0.5);
					metal metal_sphere_mat = metal(albedo, fuzz);
					sphere metal_sphere = sphere(center, 0.2, &metal_sphere_mat);
					world.add(&metal_sphere);
				} else  {
					// Glass Material
					dielectric glass_sphere_mat = dielectric(1.5);
					sphere glass_sphere = sphere(center, 0.2, &glass_sphere_mat);
					world.add(&glass_sphere);
				}
			}
		}
	}

	dielectric main_glass_sphere_mat = dielectric(1.5);
	sphere main_glass_sphere = sphere(vec3(0, 1, 0), 1.0, &main_glass_sphere_mat);
	world.add(&main_glass_sphere);

	lambertian main_diffuse_sphere_mat = lambertian(vec3(0.4, 0.2, 0.1));
	sphere main_diffuse_sphere = sphere(vec3(-4, 1, 0), 1.0, &main_diffuse_sphere_mat);
	world.add(&main_diffuse_sphere);

	metal main_metal_sphere_mat = metal(vec3(0.7, 0.6, 0.5), 0.0);
	sphere main_metal_sphere = sphere(vec3(4, 1, 0), 1.0, &main_metal_sphere_mat);
	world.add(&main_metal_sphere);

	return world;
}

int main() {
	// Output image
	const int image_width = 1920;
	const int image_height = 1080;
	const int samples_per_pixel = 1;
	const int max_bounce_depth = 50;

	vec3 lookfrom(13, 2, 3);
	vec3 lookat(0, 0, 0);
	vec3 vup(0, 1, 0);
	auto aperture = 0.1;
	auto focus_dist = 10.0;

	// Auto Focus
	// auto focus_dist = (lookfrom - lookat).length();
	const auto aspect_ratio = double(image_width) / image_height;
	
	// Generating World
	std::cerr << "Generating World...\n" << std::flush;
	auto world = random_scene();

	// Adding objects to world
	// hittable_list world;

	// Wide FOV
	// auto R = cos(pi/4);
	// world.add(make_shared<sphere>(vec3(-R, 0, -1), R, make_shared<lambertian>(vec3(0.0, 0.0, 1.0))));
	// world.add(make_shared<sphere>(vec3(R, 0, -1), R, make_shared<lambertian>(vec3(1.0, 0.0, 0.0))));

	// Glass
	// world.add(make_shared<sphere>(vec3(0, 0, -1), 0.5, make_shared<lambertian>(vec3(0.1, 0.2, 0.5))));
	// world.add(make_shared<sphere>(vec3(0, -100.5, -1), 100, make_shared<lambertian>(vec3(0.8, 0.8, 0.0))));
	// world.add(make_shared<sphere>(vec3(1, 0, -1), 0.5, make_shared<metal>(vec3(0.8, 0.6, 0.2), 0.3)));
	// world.add(make_shared<sphere>(vec3(-1, 0, -1), 0.5, make_shared<dielectric>(1.5)));
	// world.add(make_shared<sphere>(vec3(-1, 0, -1), -0.45, make_shared<dielectric>(1.5)));

	// Camera
	camera cam(lookfrom, lookat, vup, 20, aspect_ratio, aperture, focus_dist);
	ray *rays;
	vec3 *colors;
	cudaError_t error;

	size_t num_pixels = image_width*image_height;
	size_t num_rays = num_pixels*samples_per_pixel;

	error = cudaMallocManaged(&colors, num_pixels*sizeof(vec3));
	checkCuda(error);

	error = cudaMallocManaged(&rays, num_rays*sizeof(ray));
	checkCuda(error);

	for (int j = image_height - 1; j >= 0; j--) {
		for (int i = 0; i < image_width; i++) {
			vec3 color(0, 0, 0);
			for (int s = 0; s < samples_per_pixel; ++s) {
				auto u = (i + random_double()) / image_width;
				auto v = (j + random_double()) / image_height;
				rays[s + samples_per_pixel * (i + image_width * j)] = cam.get_ray(u, v);
			}
		}
	}

	int numSMs;
	error = cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
	checkCuda(error);

	dim3 grid(32*numSMs, 32*numSMs, 32*numSMs);
	dim3 block(256, 256, 256);

	ray_color<<<grid, block>>>(colors, rays, world, max_bounce_depth, samples_per_pixel, image_height*image_width*samples_per_pixel);

	error = cudaDeviceSynchronize();
	checkCuda(error);
	cudaFree(rays);
	cudaFree(colors);


	for (int j = image_height - 1; j >= 0; j--) {
		std::cerr << "\rScanlines remaining: " << j << ' ' << std::flush;
		for (int i = 0; i < image_width; i++) {
			colors[i + image_width * j].write_color(std::cout, samples_per_pixel);
		}
	}

	// // Rendering
	// std::cout << "P3\n" << image_width << " " << image_height << "\n255\n";
	// for (int j = image_height - 1; j >= 0; j--) {
	// 	std::cerr << "\rScanlines remaining: " << j << ' ' << std::flush;
	// 	for (int i = 0; i < image_width; i++) {
	// 		vec3 color(0, 0, 0);
	// 		for (int s = 0; s < samples_per_pixel; ++s) {
	// 			auto u = (i + random_double()) / image_width;
	// 			auto v = (j + random_double()) / image_height;

	// 			ray r = cam.get_ray(u, v);
	// 			ray_color(&color, r, world, max_bounce_depth);
	// 		}

	// 		color.write_color(std::cout, samples_per_pixel);
	// 	}
	// }

	std::cerr << "\nDone.\n";
}
