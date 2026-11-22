
cmake_minimum_required(VERSION 3.20)

# ------------------------------------------------------------
# Compilers
# ------------------------------------------------------------

set(CMAKE_C_COMPILER /usr/bin/gcc-12)
set(CMAKE_CXX_COMPILER /usr/bin/g++-12)
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-12)

project(AdjointNonlinearRayTracing LANGUAGES C CXX CUDA)

# ------------------------------------------------------------
# Language standards
# ------------------------------------------------------------

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

set(CMAKE_CUDA_ARCHITECTURES 86)

# ------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------

find_package(OpenGL REQUIRED)
find_package(glfw3 3.3 REQUIRED)
find_package(CUDAToolkit REQUIRED)

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

# ------------------------------------------------------------
# Include directories
# ------------------------------------------------------------

set(PROJECT_INCLUDE_DIR
    ${PROJECT_SOURCE_DIR}/include
)

# ------------------------------------------------------------
# CUDA backend
# ------------------------------------------------------------

add_library(raytracing_backend STATIC

    # Simulation
    src/Simulation/Tracer.cu

    # Medium
    src/MediumType/VoxelMedium.cu

    # Optimization
    src/Optimization/OptimizationTypes.cu
    src/Optimization/AdjointStateSolver.cu

    # Loss
    src/Optimization/LossType/LuneburgLoss.cu
)

target_include_directories(raytracing_backend PUBLIC
    ${PROJECT_INCLUDE_DIR}
    ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(raytracing_backend PUBLIC
    CUDA::cudart
)

set_target_properties(raytracing_backend PROPERTIES
    CUDA_ARCHITECTURES 86
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# ------------------------------------------------------------
# Main executable
# ------------------------------------------------------------

add_executable(tp_final
    src/main.cpp
)

target_include_directories(tp_final PRIVATE
    ${PROJECT_INCLUDE_DIR}
)

target_link_libraries(tp_final PRIVATE
    raytracing_backend
    OpenGL::GL
    glfw
    glad_gl_core_33
)

# ------------------------------------------------------------
# Analytical Luneburg executable
# ------------------------------------------------------------

add_executable(AnalyticalLuneburg
    src/Examples/AnalyticalLuneburg.cpp
)

target_include_directories(AnalyticalLuneburg PRIVATE
    ${PROJECT_INCLUDE_DIR}
)

target_link_libraries(AnalyticalLuneburg PRIVATE
    raytracing_backend
    Matplot++::matplot
)

# ------------------------------------------------------------
# Optimized Luneburg executable
# ------------------------------------------------------------

add_executable(OptimizeLuneburg
    src/Examples/OptimizeLuneburg.cu
    src/Examples/OptimizeLuneburgVisualization.cu
)

target_include_directories(OptimizeLuneburg PRIVATE
    ${PROJECT_INCLUDE_DIR}
    ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(OptimizeLuneburg PRIVATE
    raytracing_backend
    Matplot++::matplot
)

set_target_properties(OptimizeLuneburg PROPERTIES
    CUDA_ARCHITECTURES 86
    CUDA_SEPARABLE_COMPILATION ON
)

#target_compile_options(OptimizeLuneburg PRIVATE
#    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-O1>
#)
