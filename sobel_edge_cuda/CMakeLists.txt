cmake_minimum_required(VERSION 3.10)
project(SobelEdgeCUDA)

# Find required packages (will be configured later)
find_package(OpenCV REQUIRED)
# find_package(CUDA REQUIRED)

# Specify source files
set(SOURCES src/main.cpp)

# Add executable (will add sources later)
add_executable(${PROJECT_NAME} ${SOURCES})

# Link libraries (will add links later)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS}) 