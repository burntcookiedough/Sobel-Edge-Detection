cmake_minimum_required(VERSION 3.10)
project(SobelEdgeCUDA LANGUAGES CXX CUDA)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

# Print CUDA version found
message(STATUS "Found CUDA ${CUDA_VERSION_STRING}, compiler ${CUDA_NVCC_EXECUTABLE}")

# Set CUDA architecture flag (adjust sm_XX based on your GPU capability)
# Example: sm_60 (Pascal), sm_70 (Volta), sm_75 (Turing), sm_86 (Ampere)
# Refer to Plan.txt (Module 1A) or NVIDIA documentation for your GPU's compute capability.
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_60") # Default to sm_60 as per Plan.txt
message(STATUS "Setting CUDA flags: ${CUDA_NVCC_FLAGS}")

# Specify source files
set(SOURCES src/main.cpp src/sobel_kernel.cu)

# Add executable using both C++ and CUDA sources
add_executable(${PROJECT_NAME} ${SOURCES})

# Link libraries
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} CUDA::cudart) 