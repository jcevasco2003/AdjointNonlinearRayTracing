#pragma once

#include "Simulation/Ray.cuh"
#include "Simulation/OptimizableMedium.cuh"

#include <cstddef>
#include <cuda_runtime.h>

struct TracerConfig {
    double stepSize = 0.001;
    std::size_t maxSteps = 10000;
};

struct DeviceTraceResult {
    double3 finalPosition;
    double3 finalDirection;

    unsigned int stepsTaken = 0;
    bool completed = false;
};

class Tracer {
public:
    explicit Tracer(const TracerConfig& config);

    // Forward normal: solo estado final
    void forward(
        const Ray* d_rays,
        DeviceTraceResult* d_results,
        const Medium* d_medium,
        std::size_t rayCount,
        cudaStream_t stream = nullptr
    ) const;

    // Forward que también guarda la trayectoria
    void forwardWithTrajectory(
        const Ray* d_rays,
        DeviceTraceResult* d_results,
        double3* d_trajectories,
        const Medium* d_medium,
        std::size_t rayCount,
        cudaStream_t stream = nullptr
    ) const;

    // Stride por rayo: posición inicial + maxSteps
    std::size_t trajectoryStride() const {
        return config_.maxSteps + 1;
    }

private:
    TracerConfig config_;
};