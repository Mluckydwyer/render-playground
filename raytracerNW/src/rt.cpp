#include "rtweekend.h"
#include "hittable_list.h"
#include "camera.h"
#include "scenes.h"

#include <iostream>
#include <fstream>

// Command Line Options: http://www.cplusplus.com/articles/DEN36Up4/

vec3 ray_color(const ray& r, const color& background, const hittable& world, int depth) {
	hit_record rec;

	// If bounce limit is exceeded, no light is gathered
	if (depth <= 0) return color(0, 0, 0);

	if (!world.hit(r, 0.00001, infinity, rec))
		return background;

	ray scattered;
	color attenuation;
	color emitted = rec.mat_ptr->emitted(rec.u, rec.v, rec.p);

	if (!rec.mat_ptr->scatter(r, rec, attenuation, scattered))
		return emitted;
	
	return emitted + attenuation * ray_color(scattered, background, world, depth - 1);
	
	// vec3 unit_direction = unit_vector(r.direction());
	// auto t = 0.5 * (unit_direction.y() + 1.0);
	// return (1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
}

int main() {
	// Output image
	const int image_width = 400;
	const int image_height = 400;
	const int samples_per_pixel = 100;
	const int max_bounce_depth = 50;

	vec3 lookfrom(478, 278, -800);
	vec3 lookat(278, 278, 0);
	vec3 vup(0, 1, 0);
	auto aperture = 0.0;
	auto focus_dist = 10.0;
	auto vfov = 40.0;

	// Auto Focus
	// auto focus_dist = (lookfrom - lookat).length();
	const auto aspect_ratio = double(image_width) / image_height;
	const color background(0,0,0);
	
	// Generating World
	std::cerr << "Generating World...\n" << std::flush;
	auto world = book_final_scene();

	std::ofstream output_file;
	output_file.open("image.ppm");

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
	camera cam(lookfrom, lookat, vup, vfov, aspect_ratio, aperture, focus_dist, 0.0, 1.0);

	// Rendering
	output_file << "P3\n" << image_width << " " << image_height << "\n255\n";
	for (int j = image_height - 1; j >= 0; j--) {
		std::cerr << "\rScanlines remaining: " << j << ' ' << std::flush;
		for (int i = 0; i < image_width; i++) {
			color pixel_color(0, 0, 0);
			for (int s = 0; s < samples_per_pixel; ++s) {
				auto u = (i + random_double()) / image_width;
				auto v = (j + random_double()) / image_height;

				ray r = cam.get_ray(u, v);
				pixel_color += ray_color(r, background, world, max_bounce_depth);
			}

			pixel_color.write_color(output_file, samples_per_pixel);
		}
	}

	std::cerr << "\nDone.\n";
}
