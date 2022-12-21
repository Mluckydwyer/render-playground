#include <iostream>
#include <fstream>
#include <time.h>

#include "common.cuh"


// Test Render Kernal
__global__ 
void render(float *frame_buffer, int max_x, int max_y) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
	if((i >= max_x) || (j >= max_y)) return;
	
    int pixel_index = j*max_x*3 + i*3;
    frame_buffer[pixel_index + 0] = float(i) / max_x;
    frame_buffer[pixel_index + 1] = float(j) / max_y;
    frame_buffer[pixel_index + 2] = 0.2;
}

int main() {
	// Output image
	const int kImageWidth = 800; // X
	const int kImageHeight = 800; // Y
	const int kRenderBlocksX = 8;
	const int kRenderBlocksY = 8;
	//const int kSamplesPerPixel = 1;
	//const int kMaxBounceDepth = 50;
	//const auto kAspectRatio = double(kImageWidth) / kImageHeight;
	const auto kImageFileName = "image.ppm";
	
	cudaError_t error;

	// Allocate frame buffer memory
	int num_pixels = kImageWidth*kImageHeight;
	size_t frame_buffer_size = 3 * num_pixels * sizeof(float);
	float* frame_buffer;
	error = cudaMallocManaged((void**) &frame_buffer, frame_buffer_size);
	checkCudaErrors(error);

	clock_t start, stop;
	start = clock();

	// Launch Render Kernal
	dim3 blocks(kImageWidth/kRenderBlocksX + 1, kImageHeight/kRenderBlocksY + 1);
	dim3 threads(kRenderBlocksX, kRenderBlocksY);
	render<<<blocks, threads>>>(frame_buffer, kImageWidth, kImageHeight);

	//error = cudaGetLastError();
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	stop = clock();
	double render_time = ((double) (start - stop)) / CLOCKS_PER_SEC;
	std::cerr << "\rRender Completed in: " << render_time << " seconds\n" << std::flush;

	// Output render to file
	std::ofstream output_file;
	output_file.open(kImageFileName);
	output_file << "P3\n" << kImageWidth << " " << kImageHeight << "\n255\n";
	for (int j = kImageHeight - 1; j >= 0; j--) {
		if (j % 100 == 0) std::cerr << "\rScanlines remaining: " << j << ' ' << std::flush;
		for (int i = 0; i < kImageWidth; i++) {
			size_t pixel_index = j*3*kImageWidth + i*3;
			int r = int(255.99 * frame_buffer[pixel_index]);
			int g = int(255.99 * frame_buffer[pixel_index + 1]);
			int b = int(255.99 * frame_buffer[pixel_index + 2]);
			output_file << r << " " << g << " " << b << "\n";
		}
	}

	std::cerr << "\nFile Written to " << kImageFileName << "\n";
}
