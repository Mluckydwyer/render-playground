#include "rtweekend.h"

#include "hittable_list.h"
#include "sphere.h"
#include "camera.h"
#include "material.h"
#include "texture.h"

#include <iostream>

// Command Line Options: http://www.cplusplus.com/articles/DEN36Up4/

vec3 ray_color(const ray& r, const hittable& world, int depth) {
	hit_record rec;

	// If bounce limit is exceeded, no light is gathered
	if (depth <= 0) return vec3(0, 0, 0);

	if (world.hit(r, 0.00001, infinity, rec)) {
		ray scattered;
		vec3 attenuation;

		if (rec.mat_ptr->scatter(r, rec, attenuation, scattered))
			return attenuation * ray_color(scattered, world, depth - 1);
		
		return vec3(0, 0, 0);
	}

	vec3 unit_direction = unit_vector(r.direction());
	auto t = 0.5 * (unit_direction.y() + 1.0);
	return (1.0 - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
}

hittable_list random_scene() {
	hittable_list world;

	// Ground
	world.add(make_shared<sphere>(
		vec3(0, -1000, 0), 1000, make_shared<lambertian>(make_shared<solid_color>(0.5, 0.5, 0.5))
	));

	int i = 0;
	for (int a = -11; a < 11; a++) {
		for (int b = -11; b < 11; b++) {
			auto choose_mat = random_double();
			vec3 center(a + 0.9*random_double(), 0.2, b + 0.9*random_double());

			if ((center - vec3(4, 0.2, 0)).length() > 0.9) {
				if (choose_mat < 0.8) {
					// Diffuse Material
					auto albedo = vec3::random() * vec3::random();
					world.add(make_shared<sphere>(
						center, 0.2, make_shared<lambertian>(albedo)
					));
				} else if (choose_mat < 0.95) {
					// Metal Material
					auto albedo = vec3::random(0.5, 1);
					auto fuzz = random_double(0, 0.5);
					world.add(make_shared<sphere>(
						center, 0.2, make_shared<metal>(albedo, fuzz)
					));
				} else  {
					// Glass Material
					world.add(make_shared<sphere>(
						center, 0.2, make_shared<dielectric>(1.5)
					));
				}
			}
		}
	}

	world.add(make_shared<sphere>(
		vec3(0, 1, 0), 1.0, make_shared<dielectric>(1.5)
	));

	world.add(make_shared<sphere>(
		vec3(-4, 1, 0), 1.0, make_shared<lambertian>(vec3(0.4, 0.2, 0.1))
	));

	world.add(make_shared<sphere>(
		vec3(4, 1, 0), 1.0, make_shared<metal>(vec3(0.7, 0.6, 0.5), 0.0)
	));

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

	// Rendering
	std::cout << "P3\n" << image_width << " " << image_height << "\n255\n";
	for (int j = image_height - 1; j >= 0; j--) {
		std::cerr << "\rScanlines remaining: " << j << ' ' << std::flush;
		for (int i = 0; i < image_width; i++) {
			vec3 color(0, 0, 0);
			for (int s = 0; s < samples_per_pixel; ++s) {
				auto u = (i + random_double()) / image_width;
				auto v = (j + random_double()) / image_height;

				ray r = cam.get_ray(u, v);
				color += ray_color(r, world, max_bounce_depth);
			}

			color.write_color(std::cout, samples_per_pixel);
		}
	}

	std::cerr << "\nDone.\n";
}
