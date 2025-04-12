# Real-Time Sobel Edge Detection with CUDA

This project implements Sobel edge detection using CUDA for GPU acceleration and OpenCV for video handling, processing a live camera feed.

![Screenshot of running application](placeholder_screenshot.png) <!-- You can replace placeholder_screenshot.png with an actual screenshot if desired -->

## Prerequisites

*   **Hardware:**
    *   A CUDA-capable NVIDIA GPU (Compute Capability 6.0 or higher recommended).
*   **Software:**
    *   **CUDA Toolkit:** Version 11.x or later (tested with 12.8). Installation guide: [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
    *   **OpenCV:** Version 4.x (tested with 4.11.0). Must be built or installed with C++ support. Installation guide: [https://opencv.org/releases/](https://opencv.org/releases/)
    *   **CMake:** Version 3.18 or later.
    *   **C++ Compiler:** A compatible C++ compiler (e.g., MSVC on Windows, g++ on Linux) supported by your chosen CUDA Toolkit version.

## Building

This project uses CMake for building.

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <repository-url>
    cd sobel_edge_cuda
    ```

2.  **Create a build directory:**
    ```bash
    mkdir build
    cd build
    ```

3.  **Configure using CMake:**
    *   You **must** specify the path to your OpenCV installation directory using `OpenCV_DIR`.
    *   Replace `C:/opencv/build` with the actual path on your system.
    ```bash
    cmake .. -DOpenCV_DIR=C:/opencv/build
    ```
    *   *(Optional)* You can specify a different GPU architecture if needed (default is `60` for `sm_60`):
        ```bash
        cmake .. -DOpenCV_DIR=C:/opencv/build -DCMAKE_CUDA_ARCHITECTURES=75
        ```

4.  **Build the project:**
    ```bash
    cmake --build .
    ```
    On Windows with Visual Studio generator, you might want to specify the configuration:
    ```bash
    cmake --build . --config Debug
    # or
    cmake --build . --config Release
    ```

## Running

1.  **Ensure OpenCV DLLs are accessible:**
    *   **Option 1 (Recommended):** Add the OpenCV `bin` directory (e.g., `C:\opencv\build\x64\vc16\bin` or similar, check your install path) to your system's PATH environment variable.
    *   **Option 2:** Copy the required OpenCV DLLs (e.g., `opencv_worldXYZ.dll`, `opencv_videoio_ffmpegXYZ_64.dll` etc., where XYZ is the version) from the OpenCV `bin` directory into the executable directory (`sobel_edge_cuda/build/Debug` or `sobel_edge_cuda/build/Release`).

2.  **Run the executable:**
    Navigate to the build directory (e.g., `sobel_edge_cuda/build`) and run:
    ```bash
    ./Debug/SobelEdgeCUDA.exe  # Or ./Release/SobelEdgeCUDA.exe
    # On Linux: ./Debug/SobelEdgeCUDA or ./Release/SobelEdgeCUDA
    ```

3.  Two windows should appear:
    *   `Grayscale Feed`: The direct grayscale output from the camera.
    *   `Sobel Edges (CUDA)`: The edge-detected output processed by the GPU.

4.  Press the `ESC` key while one of the windows is focused to quit.

## Configuration

*   **CUDA Architecture:** The target GPU compute capability is set in `CMakeLists.txt` via `CMAKE_CUDA_ARCHITECTURES`. The default is `60` (`sm_60`). You can change this value or pass it during CMake configuration (`-DCMAKE_CUDA_ARCHITECTURES=XX`) for better performance on newer GPUs (e.g., `75`, `86`).
*   **Tile Size:** The CUDA kernel uses a fixed tile size (`TILE_SIZE = 16`) defined in `src/sobel_kernel.cu`. This affects performance and shared memory usage. 