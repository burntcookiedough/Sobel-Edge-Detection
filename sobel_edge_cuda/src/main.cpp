#include <opencv2/opencv.hpp>
#include <iostream>
#include <cuda_runtime.h> // Added for CUDA Runtime API

// Forward declaration of the CUDA kernel (must match the definition in .cu file)
__global__ void sobel_filter(const unsigned char* input, unsigned char* output, int width, int height);

// Error-checking macro for CUDA API calls (from Plan.txt)
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        // Use std::cerr for error output
        std::cerr << "CUDA Error: " << cudaGetErrorString(code) << " at " << file << ":" << line << std::endl;
        if (abort) exit(code);
    }
}

int main() {
    // Initialize video capture from the default camera
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Unable to open the camera stream." << std::endl;
        return -1;
    }

    // Set desired frame resolution (optional, use dimensions from Plan.txt)
    int frame_width = 640;
    int frame_height = 480;
    cap.set(cv::CAP_PROP_FRAME_WIDTH, frame_width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, frame_height);

    // Check if the resolution was set correctly (some cameras might not support it)
    frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    std::cout << "Actual frame resolution: " << frame_width << "x" << frame_height << std::endl;

    cv::Mat frame, gray;
    cv::Mat edgeImage(frame_height, frame_width, CV_8UC1); // Pre-allocate host memory for result

    // --- CUDA Setup ---
    unsigned char *d_input, *d_output; // Device pointers
    // Calculate size in bytes for a single channel 8-bit image
    size_t numBytes = frame_width * frame_height * sizeof(unsigned char);

    // Allocate GPU memory
    std::cout << "Allocating GPU memory..." << std::endl;
    cudaCheckError( cudaMalloc((void**)&d_input, numBytes) );
    cudaCheckError( cudaMalloc((void**)&d_output, numBytes) );
    std::cout << "GPU memory allocated." << std::endl;

    // Define tile (block) dimensions; must match the kernel's TILE_SIZE (16)
    const int TILE_SIZE = 16;
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    // Calculate grid dimensions to cover the entire image
    dim3 gridSize((frame_width + TILE_SIZE - 1) / TILE_SIZE,
                  (frame_height + TILE_SIZE - 1) / TILE_SIZE);
    std::cout << "Kernel launch config: Grid=" << gridSize.x << "x" << gridSize.y
              << ", Block=" << blockSize.x << "x" << blockSize.y << std::endl;
    // --- End CUDA Setup ---

    std::cout << "Press ESC to exit." << std::endl;

    while (true) {
        cap >> frame; // Capture a frame
        if (frame.empty()) {
            std::cerr << "Error: Captured empty frame." << std::endl;
            break;
        }

        // Convert to grayscale; necessary for Sobel
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // --- CUDA Processing ---
        // 1. Copy grayscale frame from host (gray.data) to device (d_input)
        cudaCheckError( cudaMemcpy(d_input, gray.data, numBytes, cudaMemcpyHostToDevice) );

        // 2. Launch the Sobel edge detection kernel
        sobel_filter<<<gridSize, blockSize>>>(d_input, d_output, frame_width, frame_height);

        // Check for kernel launch errors (optional but recommended)
        cudaError_t kernelError = cudaGetLastError();
        if (kernelError != cudaSuccess) {
             std::cerr << "CUDA Kernel Launch Error: " << cudaGetErrorString(kernelError) << std::endl;
             exit(kernelError); // Exit if kernel launch failed
        }

        // 3. Synchronize device to ensure kernel completion before copying back
        // Not strictly necessary before cudaMemcpy D->H if using default stream (stream 0),
        // but good practice, especially if adding timing later.
        cudaCheckError( cudaDeviceSynchronize() );

        // 4. Copy the computed edge map from device (d_output) back to host (edgeImage.data)
        cudaCheckError( cudaMemcpy(edgeImage.data, d_output, numBytes, cudaMemcpyDeviceToHost) );
        // --- End CUDA Processing ---

        // Display the original grayscale and the edge-detected images
        cv::imshow("Grayscale Feed", gray);
        cv::imshow("Sobel Edges (CUDA)", edgeImage); // Updated window title

        // Break the loop if the user presses the ESC key (ASCII 27)
        if (cv::waitKey(1) == 27) {
            std::cout << "ESC key pressed. Exiting..." << std::endl;
            break;
        }
    }

    // --- CUDA Cleanup ---
    std::cout << "Freeing GPU memory..." << std::endl;
    cudaCheckError( cudaFree(d_input) );
    cudaCheckError( cudaFree(d_output) );
    std::cout << "GPU memory freed." << std::endl;
    // --- End CUDA Cleanup ---

    // Clean up: Release the camera and destroy windows
    cap.release();
    cv::destroyAllWindows();
    std::cout << "Camera released and windows closed." << std::endl;

    return 0;
} 