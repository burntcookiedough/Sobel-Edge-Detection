// Define block dimensions (must match host code)
#define TILE_SIZE 16
// Shared memory tile size includes halo: extra 1 pixel border on each side
#define SMEM_SIZE (TILE_SIZE + 2)

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath> // For abs()

// Kernel for Sobel edge detection using shared memory tiling
__global__ void sobel_filter(const unsigned char* input, unsigned char* output,
                             int width, int height) {

    // Calculate global indices for the pixel this thread corresponds to
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;

    // Allocate shared memory for the tile + halo region
    // Using extern __shared__ might be slightly more efficient for dynamic sizing,
    // but static allocation is simpler for fixed TILE_SIZE.
    __shared__ unsigned char smem[SMEM_SIZE][SMEM_SIZE];

    // Calculate indices within shared memory (offset by 1 for halo)
    int smem_x = threadIdx.x + 1;
    int smem_y = threadIdx.y + 1;

    // --- Load data into shared memory ---
    // Strategy: Each thread loads its primary pixel. Threads on the edge/corner
    // of the block load the necessary halo pixels.

    // Load the central pixel (within image bounds)
    if (x < width && y < height) {
        smem[smem_y][smem_x] = input[y * width + x];
    } else {
        // Pad with 0 if outside image bounds
        smem[smem_y][smem_x] = 0;
    }

    // Load halo pixels (requires boundary checks for image edges)
    // Top halo row (loaded by threads in the top row of the block)
    if (threadIdx.y == 0) {
        int y_top = y - 1;
        smem[0][smem_x] = (y_top >= 0 && x < width) ? input[y_top * width + x] : 0;
    }
    // Bottom halo row (loaded by threads in the bottom row of the block)
    if (threadIdx.y == TILE_SIZE - 1) {
        int y_bottom = y + 1;
        smem[SMEM_SIZE - 1][smem_x] = (y_bottom < height && x < width) ? input[y_bottom * width + x] : 0;
    }
    // Left halo column (loaded by threads in the left column of the block)
    if (threadIdx.x == 0) {
        int x_left = x - 1;
        smem[smem_y][0] = (x_left >= 0 && y < height) ? input[y * width + x_left] : 0;
    }
    // Right halo column (loaded by threads in the right column of the block)
    if (threadIdx.x == TILE_SIZE - 1) {
        int x_right = x + 1;
        smem[smem_y][SMEM_SIZE - 1] = (x_right < width && y < height) ? input[y * width + x_right] : 0;
    }

    // Load corner halo pixels (loaded by corner threads of the block)
    // Top-left corner
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        int x_left = x - 1, y_top = y - 1;
        smem[0][0] = (x_left >= 0 && y_top >= 0) ? input[y_top * width + x_left] : 0;
    }
    // Top-right corner
    if (threadIdx.x == TILE_SIZE - 1 && threadIdx.y == 0) {
        int x_right = x + 1, y_top = y - 1;
        smem[0][SMEM_SIZE - 1] = (x_right < width && y_top >= 0) ? input[y_top * width + x_right] : 0;
    }
    // Bottom-left corner
    if (threadIdx.x == 0 && threadIdx.y == TILE_SIZE - 1) {
        int x_left = x - 1, y_bottom = y + 1;
        smem[SMEM_SIZE - 1][0] = (x_left >= 0 && y_bottom < height) ? input[y_bottom * width + x_left] : 0;
    }
    // Bottom-right corner
    if (threadIdx.x == TILE_SIZE - 1 && threadIdx.y == TILE_SIZE - 1) {
        int x_right = x + 1, y_bottom = y + 1;
        smem[SMEM_SIZE - 1][SMEM_SIZE - 1] = (x_right < width && y_bottom < height) ? input[y_bottom * width + x_right] : 0;
    }

    // Synchronize threads within the block to ensure all shared memory is loaded
    __syncthreads();

    // --- Perform Sobel calculation using shared memory ---
    // Only calculate for threads corresponding to valid pixels within the output image
    // (i.e., threads whose corresponding pixel has a full 3x3 neighborhood within the loaded smem tile)
    // The condition also ensures we don't write out-of-bounds for the output array.
    if (x < width && y < height) {

        // We use smem indices (smem_x, smem_y) which are offset by +1
        // Gx = [-1 0 1]   Gy = [ 1  2  1]
        //      [-2 0 2]        [ 0  0  0]
        //      [-1 0 1]        [-1 -2 -1]
        int Gx = -smem[smem_y - 1][smem_x - 1] + smem[smem_y - 1][smem_x + 1]
                 - 2 * smem[smem_y][smem_x - 1] + 2 * smem[smem_y][smem_x + 1]
                 - smem[smem_y + 1][smem_x - 1] + smem[smem_y + 1][smem_x + 1];

        int Gy =  smem[smem_y - 1][smem_x - 1] + 2 * smem[smem_y - 1][smem_x] + smem[smem_y - 1][smem_x + 1]
                 - smem[smem_y + 1][smem_x - 1] - 2 * smem[smem_y + 1][smem_x] - smem[smem_y + 1][smem_x + 1];

        // Approximate gradient magnitude: |Gx| + |Gy|
        int gradient = abs(Gx) + abs(Gy);

        // Clamp the result to [0, 255]
        if (gradient > 255) gradient = 255;

        // Write the result to the global output memory
        // NOTE: Pixels on the very border of the image (x=0, y=0, x=width-1, y=height-1)
        // will have gradients calculated using padded '0' values from shared memory,
        // which is a common way to handle boundaries. Alternatively, output could be
        // explicitly set to 0 for borders in host code or kernel.
        output[y * width + x] = (unsigned char)gradient;
    }
    // Threads whose (x, y) fall outside the image dimensions do nothing further.
} 