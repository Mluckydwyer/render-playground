#ifndef SCENES_H
#define SCENES_H

#include "hittable_list.h"
#include "sphere.h"
#include "moving_sphere.h"
#include "aarect.h"
#include "box.h"
#include "material.h"
#include "constant_medium.h"
#include "bvh.h"

hittable_list random_scene() {
	hittable_list world;

	// Floor
	// world.add(make_shared<sphere>(
	// 	vec3(0, -1000, 0), 1000, make_shared<lambertian>(make_shared<solid_color>(0.5, 0.5, 0.5))
	// ));
	auto checker = make_shared<checker_texture>(
		make_shared<solid_color>(0.2, 0.3, 0.1),
		make_shared<solid_color>(0.9, 0.9, 0.9)
	);
	world.add(make_shared<sphere>(point3(0,-1000,0), 1000, make_shared<lambertian>(checker)));

	int i = 0;
	for (int a = -10; a < 10; a++) {
		for (int b = -10; b < 10; b++) {
			auto choose_mat = random_double();
			vec3 center(a + 0.9*random_double(), 0.2, b + 0.9*random_double());

			if ((center - vec3(4, 0.2, 0)).length() > 0.9) {
				if (choose_mat < 0.8) {
					// Diffuse Material
					auto albedo = color::random() * color::random();
					world.add(make_shared<moving_sphere>(
						center, center + vec3(0, random_double(0, 0.5), 0),
						0.0, 1.0, 0.2, make_shared<lambertian>(make_shared<solid_color>(albedo))
					));
				} else if (choose_mat < 0.95) {
					// Metal Material
					auto albedo = color::random(0.5, 1);
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
		vec3(-4, 1, 0), 1.0, make_shared<lambertian>(make_shared<solid_color>(0.4, 0.2, 0.1))
	));

	world.add(make_shared<sphere>(
		vec3(4, 1, 0), 1.0, make_shared<metal>(vec3(0.7, 0.6, 0.5), 0.0)
	));

	return world;
}

hittable_list two_spheres() {
	hittable_list objects;

	auto checker = make_shared<checker_texture>(
		make_shared<solid_color>(0.2, 0.3, 0.1),
		make_shared<solid_color>(0.9, 0.9, 0.9)
	);

	objects.add(make_shared<sphere>(point3(0,-10,0), 10, make_shared<lambertian>(checker)));
	objects.add(make_shared<sphere>(point3(0,10,0), 10, make_shared<lambertian>(checker)));

	return objects;
}

hittable_list two_perlin_spheres() {
	hittable_list objects;

	auto pertext = make_shared<noise_texture>(4);
	objects.add(make_shared<sphere>(point3(0,-1000,0), 1000, make_shared<lambertian>(pertext)));
	objects.add(make_shared<sphere>(point3(0,2,0), 2, make_shared<lambertian>(pertext)));

	return objects;
}

hittable_list earth() {
	auto earth_texture = make_shared<image_texture>("textures/earthmap.jpg");
	auto earth_surface = make_shared<lambertian>(earth_texture);
	auto globe = make_shared<sphere>(point3(0,0,0), 2, earth_surface);

	return hittable_list(globe);
}

hittable_list simple_light() {
	hittable_list objects;

	auto pertext = make_shared<noise_texture>(4);
	objects.add(make_shared<sphere>(point3(0,-1000,0), 1000, make_shared<lambertian>(pertext)));
	objects.add(make_shared<sphere>(point3(0,2,0), 2, make_shared<lambertian>(pertext)));

	auto diff_light = make_shared<diffuse_light>(make_shared<solid_color>(4,4,4));
	objects.add(make_shared<sphere>(point3(0,7,0), 2, diff_light));
	objects.add(make_shared<xy_rect>(3, 5, 1, 3, -2, diff_light));

	return objects;
}

hittable_list cornell_box() {
	hittable_list objects;
    
    // Materials
	auto red = make_shared<lambertian>(make_shared<solid_color>(0.65, 0.05, 0.05));
	auto white = make_shared<lambertian>(make_shared<solid_color>(0.73, 0.73, 0.73));
	auto green = make_shared<lambertian>(make_shared<solid_color>(0.12, 0.45, 0.15));
	auto light = make_shared<diffuse_light>(make_shared<solid_color>(7, 7, 7));

    // Walls and Light
	objects.add(make_shared<flip_face>(make_shared<yz_rect>(0, 555, 0, 555, 555, green)));
	objects.add(make_shared<yz_rect>(0, 555, 0, 555, 0, red));
	objects.add(make_shared<xz_rect>(213, 343, 227, 332, 554, light));
	objects.add(make_shared<flip_face>(make_shared<xz_rect>(0, 555, 0, 555, 0, white)));
	objects.add(make_shared<xz_rect>(0, 555, 0, 555, 555, white));
	objects.add(make_shared<flip_face>(make_shared<xy_rect>(0, 555, 0, 555, 555, white)));

    // Boxes
    shared_ptr<hittable> box1 = make_shared<box>(point3(0, 0, 0), point3(165, 330, 165), white);
    box1 = make_shared<rotate_y>(box1, 15);
    box1 = make_shared<translate>(box1, vec3(265, 0, 295));
    objects.add(make_shared<constant_medium>(box1, 0.01, make_shared<solid_color>(0, 0, 0)));

    shared_ptr<hittable> box2 = make_shared<box>(point3(0, 0, 0), point3(165, 165, 165), white);
    box2 = make_shared<rotate_y>(box2, -18);
    box2 = make_shared<translate>(box2, vec3(130, 0, 65));
    objects.add(make_shared<constant_medium>(box2, 0.01, make_shared<solid_color>(1, 1, 1)));

	return objects;
}

hittable_list final_scene() {
    hittable_list boxes1;
    hittable_list objects;
    auto ground = make_shared<lambertian>(make_shared<solid_color>(0.48, 0.83, 0.53));

	// Floor
    const int boxes_per_side = 20;
    for (int i = 0; i < boxes_per_side; i++) {
        for (int j = 0; j < boxes_per_side; j++) {
            auto w = 100.0;
            auto x0 = -1000.0 + j*w;
            auto z0 = -1000.0 + j*w;
            auto y0 = 0.0;
            auto x1 = x0 + w;
            auto y1 = random_double(1, 101);
            auto z1 = z0 + w;

            boxes1.add(make_shared<box>(point3(x0, y0, z0), point3(x1, y1, z1), ground));
        }
    }
    objects.add(make_shared<bvh_node>(boxes1, 0, 1));

	// Light
    auto light = make_shared<diffuse_light>(make_shared<solid_color>(7, 7, 7));
    objects.add(make_shared<xz_rect>(123, 423, 147, 412, 554, light));

	// Moving Sphere
    auto center1 = point3(400, 400, 400);
    auto center2 = center1 + vec3(30, 0, 0);
    auto moving_sphere_material = make_shared<lambertian>(make_shared<solid_color>(0.7, 0.3, 0.1));
    objects.add(make_shared<moving_sphere>(center1, center2, 0, 1, 50, moving_sphere_material));

	// Reflective spheres
	objects.add(make_shared<sphere>(point3(260, 150, 45), 50, make_shared<dielectric>(1.5)));
	objects.add(make_shared<sphere>(point3(0, 150, 145), 50, make_shared<metal>(color(0.8, 0.8, 0.8), 10.0)));

	auto boundary = make_shared<sphere>(point3(360, 150, 145), 70, make_shared<dielectric>(1.5));
	objects.add(boundary);

	// Smokes/Fogs
	objects.add(make_shared<constant_medium>(boundary, 0.2, make_shared<solid_color>(0.2, 0.4, 0.9)));

	boundary = make_shared<sphere>(point3(0, 0, 0), 5000, make_shared<dielectric>(1.5));
	objects.add(make_shared<constant_medium>(boundary, 0.0001, make_shared<solid_color>(1, 1, 1)));

	// Textues (Earth)
	auto earth_material = make_shared<lambertian>(make_shared<image_texture>("textures/earthmap.jpg"));
	objects.add(make_shared<sphere>(point3(400, 200, 400), 100, earth_material));

	// Noise
	auto pertext = make_shared<noise_texture>(0.1);
	objects.add(make_shared<sphere>(point3(220, 280, 300), 80, make_shared<lambertian>(pertext)));

	// Spheres in a box
	hittable_list boxes2;
	auto white = make_shared<lambertian>(make_shared<solid_color>(0.73, 0.73, 0.73));
	int num_spheres = 1000;
	for (int j = 0; j < num_spheres; j++) {
		boxes2.add(make_shared<sphere>(point3::random(0, 165), 10, white));
	}

	// Rotated
	objects.add(make_shared<translate>(make_shared<rotate_y>(make_shared<bvh_node>(boxes2, 0.0, 1.0), 15), vec3(-100, 270, 395)));

    return objects;
}

hittable_list book_final_scene() {
    hittable_list boxes1;
    auto ground = make_shared<lambertian>(make_shared<solid_color>(0.48, 0.83, 0.53));

    const int boxes_per_side = 20;
    for (int i = 0; i < boxes_per_side; i++) {
        for (int j = 0; j < boxes_per_side; j++) {
            auto w = 100.0;
            auto x0 = -1000.0 + i*w;
            auto z0 = -1000.0 + j*w;
            auto y0 = 0.0;
            auto x1 = x0 + w;
            auto y1 = random_double(1,101);
            auto z1 = z0 + w;

            boxes1.add(make_shared<box>(point3(x0,y0,z0), point3(x1,y1,z1), ground));
        }
    }

    hittable_list objects;

    objects.add(make_shared<bvh_node>(boxes1, 0, 1));

    auto light = make_shared<diffuse_light>(make_shared<solid_color>(7, 7, 7));
    objects.add(make_shared<xz_rect>(123, 423, 147, 412, 554, light));

    auto center1 = point3(400, 400, 200);
    auto center2 = center1 + vec3(30,0,0);
    auto moving_sphere_material =
        make_shared<lambertian>(make_shared<solid_color>(0.7, 0.3, 0.1));
    objects.add(make_shared<moving_sphere>(center1, center2, 0, 1, 50, moving_sphere_material));

    objects.add(make_shared<sphere>(point3(260, 150, 45), 50, make_shared<dielectric>(1.5)));
    objects.add(make_shared<sphere>(
        point3(0, 150, 145), 50, make_shared<metal>(color(0.8, 0.8, 0.9), 10.0)
    ));

    auto boundary = make_shared<sphere>(point3(360,150,145), 70, make_shared<dielectric>(1.5));
    objects.add(boundary);
    objects.add(make_shared<constant_medium>(
        boundary, 0.2, make_shared<solid_color>(0.2, 0.4, 0.9)
    ));
    boundary = make_shared<sphere>(point3(0, 0, 0), 5000, make_shared<dielectric>(1.5));
    objects.add(make_shared<constant_medium>(
        boundary, .0001, make_shared<solid_color>(1,1,1)));

    auto emat = make_shared<lambertian>(make_shared<image_texture>("earthmap.jpg"));
    objects.add(make_shared<sphere>(point3(400,200,400), 100, emat));
    auto pertext = make_shared<noise_texture>(0.1);
    objects.add(make_shared<sphere>(point3(220,280,300), 80, make_shared<lambertian>(pertext)));

    hittable_list boxes2;
    auto white = make_shared<lambertian>(make_shared<solid_color>(.73, .73, .73));
    int ns = 1000;
    for (int j = 0; j < ns; j++) {
        boxes2.add(make_shared<sphere>(point3::random(0,165), 10, white));
    }

    objects.add(make_shared<translate>(
        make_shared<rotate_y>(
            make_shared<bvh_node>(boxes2, 0.0, 1.0), 15),
            vec3(-100,270,395)
        )
    );

    return objects;
}

#endif