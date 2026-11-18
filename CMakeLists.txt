
cmake_minimum_required(VERSION 3.20)

project(AdjointNonlinearRayTracing LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

set(CMAKE_CUDA_ARCHITECTURES 86) # RTX 3090

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
)

# OpenGL
find_package(OpenGL REQUIRED)

# GLFW
find_package(glfw3 3.3 REQUIRED)

# CUDA
find_package(CUDAToolkit REQUIRED)

# Dependencies
include(FetchContent)

# GLAD 2
FetchContent_Declare(
    glad
    GIT_REPOSITORY https://github.com/Dav1dde/glad.git
    GIT_TAG v2.0.8
    SOURCE_SUBDIR cmake
)

FetchContent_MakeAvailable(glad)

glad_add_library(
    glad_gl_core_33
    REPRODUCIBLE
    API gl:core=3.3
)

# Matplot++
FetchContent_Declare(
    matplotplusplus
    GIT_REPOSITORY https://github.com/alandefreitas/matplotplusplus
    GIT_TAG v1.2.2
)

FetchContent_MakeAvailable(matplotplusplus)

# Main executable
add_executable(tp_final
    src/main.cpp
)

target_link_libraries(tp_final PRIVATE
    OpenGL::GL
    glfw
    glad_gl_core_33
)

# Analytical Luneburg executable
add_executable(AnalyticalLuneburg
    src/Examples/AnalyticalLuneburg.cpp
    src/Simulation/Tracer.cpp
)

target_link_libraries(AnalyticalLuneburg PRIVATE
    Matplot++::matplot
)

# Optimized Luneburg executable
add_executable(OptimizeLuneburg
    src/Examples/OptimizeLuneburg.cpp

    # Simulation
    src/Simulation/Tracer.cpp

    # Medium
    src/MediumType/VoxelMedium.cpp

    # Loss
    src/Optimization/LossType/LuneburgLoss.cpp

    # Optimizers
    src/Optimization/OptimizationTypes.cpp
    src/Optimization/AdjointStateSolver.cpp
)

target_link_libraries(OptimizeLuneburg PRIVATE
    Matplot++::matplot
    CUDA::cudart
)

set_target_properties(OptimizeLuneburg PROPERTIES
    CUDA_ARCHITECTURES 86
    CUDA_SEPARABLE_COMPILATION ON
)
