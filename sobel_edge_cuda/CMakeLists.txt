cmake_minimum_required(VERSION 3.18)
project(SobelEdgeCUDA LANGUAGES CXX CUDA)

# Set C++ standard if needed
# set(CMAKE_CXX_STANDARD 11)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard if needed (CUDA language support implies C++14 by default in recent CMake)
# set(CMAKE_CUDA_STANDARD 14)
# set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

# Print CUDA version found
message(STATUS "Found OpenCV ${OpenCV_VERSION}")
message(STATUS "Found CUDA ${CUDA_VERSION_STRING}, compiler ${CUDA_NVCC_EXECUTABLE}")

# Specify target CUDA architecture(s)
# Replace '60' with the compute capability of your GPU(s) (e.g., '75', '86')
# See: https://developer.nvidia.com/cuda-gpus
# You can list multiple architectures, e.g., "60;75;86"
set(CMAKE_CUDA_ARCHITECTURES 60) # Default to 6.0 (sm_60) as per Plan.txt
message(STATUS "Setting CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# Specify source files
set(SOURCES src/main.cpp src/sobel_kernel.cu)

# Add executable using both C++ and CUDA sources
add_executable(${PROJECT_NAME} ${SOURCES})

# Explicitly set main.cpp to be compiled with CUDA compiler
set_source_files_properties(src/main.cpp PROPERTIES LANGUAGE CUDA)

# Link libraries - using modern CMake targets
# OpenCV::opencv_core, OpenCV::opencv_highgui, OpenCV::opencv_imgproc are common
# Use ${OpenCV_LIBS} if the specific component targets aren't found/needed
# CUDA::cudart_static or CUDA::cudart (shared) should be available now
# --- Trying explicit CUDA_LIBRARIES linking as CUDA::cudart target isn't found ---
target_link_libraries(${PROJECT_NAME} PRIVATE
    ${OpenCV_LIBS} # Link all found OpenCV libs
    # CUDA::cudart # Link CUDA runtime (usually shared) <-- Problematic
    ${CUDA_LIBRARIES} # Link CUDA runtime library directly
) 