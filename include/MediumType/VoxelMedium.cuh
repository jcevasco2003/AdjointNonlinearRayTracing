
#pragma once

#include "Simulation/OptimizableMedium.cuh"

#include <cuda_runtime.h>
#include <glm/glm.hpp>

#include <vector>
#include <cstddef>

// ============================================================
// VoxelMedium
// ============================================================

class VoxelMedium : public OptimizableMedium {
public:

    // --------------------------------------------------------
    // Construction / destruction (host)
    // --------------------------------------------------------

    VoxelMedium(
        const glm::dvec3& origin,
        const glm::dvec3& size,
        const glm::uvec3& resolution,
        double initialEta = 1.0
    );

    ~VoxelMedium();

    VoxelMedium(const VoxelMedium&) = delete;
    VoxelMedium& operator=(const VoxelMedium&) = delete;

    VoxelMedium(VoxelMedium&& other) noexcept;
    VoxelMedium& operator=(VoxelMedium&& other) noexcept;

    // Device
    __device__ VoxelMedium(
        const double3& origin,
        const double3& size,
        const uint3& resolution,
        double* voxels
    );

    // --------------------------------------------------------
    // Medium interface (device)
    // --------------------------------------------------------

    __device__ bool isInside(
        const double3& x
    ) const override;

    __device__ double refractiveIndex(
        const double3& x
    ) const override;

    __device__ double3 gradient(
        const double3& x
    ) const override;

    __device__ void hessian(
        const double3& x,
        double* H
    ) const override;

    double* deviceParameters() const;

    // --------------------------------------------------------
    // OptimizableMedium interface (device)
    // --------------------------------------------------------

    __device__ std::size_t parameterCount() const override;

    __device__ double indexParameterGradient(
        const double3& x,
        std::size_t parameterIndex
    ) const override;

    __device__ double3 gradientParameterGradient(
        const double3& x,
        std::size_t parameterIndex
    ) const override;


    // --------------------------------------------------------
    // Optimized parameter accumulation (device)
    // --------------------------------------------------------

    __device__ void accumulateParameterGradient(
        const double3& x,
        double eta,
        const double3& gradEta,
        const double3& mu,
        double dt,
        double* parameterGradient
    ) const;


    // --------------------------------------------------------
    // GPU object access (host)
    // --------------------------------------------------------

    // Returns a pointer to the device-resident VoxelMedium.
    // The object must be constructed on the GPU.
    VoxelMedium* deviceObject() const;


    // --------------------------------------------------------
    // Explicit CPU <-> GPU transfers (host)
    // --------------------------------------------------------

    std::vector<double> downloadParameters() const;

    void uploadParameters(
        const std::vector<double>& parameters
    );


    // --------------------------------------------------------
    // Grid management (host)
    // --------------------------------------------------------

    void refine(
        const glm::uvec3& newResolution
    );

    glm::uvec3 resolution() const;

    glm::dvec3 voxelSize() const;

    std::size_t parameterCountHost() const override;

    double* deviceParameters() override;


private:

    // --------------------------------------------------------
    // Device object management
    // --------------------------------------------------------

    void createDeviceObject();

    void destroyDeviceObject();

    void allocateDeviceMemory();

    void releaseDeviceMemory();


    // --------------------------------------------------------
    // Host metadata
    // --------------------------------------------------------

    glm::dvec3 origin_;
    glm::dvec3 size_;
    glm::uvec3 resolution_;


    // --------------------------------------------------------
    // Device metadata
    // --------------------------------------------------------

    double3 deviceOrigin_;
    double3 deviceSize_;
    uint3 deviceResolution_;


    // --------------------------------------------------------
    // GPU storage
    // --------------------------------------------------------

    // Voxel parameters: GPU is the source of truth.
    double* d_voxels_ = nullptr;

    // Actual polymorphic object residing on the GPU.
    VoxelMedium* d_self_ = nullptr;
};
