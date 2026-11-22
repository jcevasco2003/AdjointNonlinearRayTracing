
#include "Simulation/Tracer.cuh"

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

// ============================================================
// Kernel: un thread por rayo
// ============================================================

__global__ void traceRaysKernel(
    const Ray* rays,
    DeviceTraceResult* results,
    const Medium* medium,
    double dt,
    unsigned int maxSteps,
    unsigned int rayCount
)
{
    const unsigned int id =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (id >= rayCount)
        return;

    // Estado inicial del rayo
    double3 x = rays[id].position;
    double3 v = rays[id].direction;

    unsigned int steps = 0;

    // Evolución del rayo dentro del medio
    while (medium->isInside(x) && steps < maxSteps)
    {
        const double eta = medium->refractiveIndex(x);
        const double3 gradEta = medium->gradient(x);

        // Integración semi-implícita
        v.x += eta * gradEta.x * dt;
        v.y += eta * gradEta.y * dt;
        v.z += eta * gradEta.z * dt;

        x.x += v.x * dt;
        x.y += v.y * dt;
        x.z += v.z * dt;

        ++steps;
    }

    // El rayo completó el recorrido si salió del medio
    const bool completed = !medium->isInside(x);

    results[id] = {
        x,
        v,
        steps,
        completed
    };
}

__global__ void traceRaysWithTrajectoryKernel(
    const Ray* rays,
    DeviceTraceResult* results,
    double3* trajectories,
    const Medium* medium,
    double dt,
    unsigned int maxSteps,
    unsigned int rayCount
)
{
    const unsigned int id =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (id >= rayCount)
        return;

    const std::size_t stride =
        static_cast<std::size_t>(maxSteps) + 1;

    double3 x = rays[id].position;
    double3 v = rays[id].direction;

    unsigned int steps = 0;

    // Guardar posición inicial
    trajectories[id * stride] = x;

    while (medium->isInside(x) && steps < maxSteps)
    {
        const double eta = medium->refractiveIndex(x);
        const double3 gradEta = medium->gradient(x);

        // Integración semi-implícita
        v.x += eta * gradEta.x * dt;
        v.y += eta * gradEta.y * dt;
        v.z += eta * gradEta.z * dt;

        x.x += v.x * dt;
        x.y += v.y * dt;
        x.z += v.z * dt;

        ++steps;

        // Guardar la posición posterior al paso
        trajectories[id * stride + steps] = x;
    }

    const bool completed = !medium->isInside(x);

    results[id] = {
        x,
        v,
        steps,
        completed
    };
}


// ============================================================
// Wrapper CPU
// ============================================================

Tracer::Tracer(const TracerConfig& config)
    : config_(config)
{
    if (config_.stepSize <= 0.0)
        throw std::invalid_argument("Tracer stepSize must be positive");

    if (config_.maxSteps == 0)
        throw std::invalid_argument("Tracer maxSteps must be positive");

    if (config_.maxSteps >
        static_cast<std::size_t>(UINT_MAX))
    {
        throw std::invalid_argument("Tracer maxSteps exceeds UINT_MAX");
    }
}


void Tracer::forward(
    const Ray* d_rays,
    DeviceTraceResult* d_results,
    const Medium* d_medium,
    std::size_t rayCount,
    cudaStream_t stream
) const
{
    if (rayCount == 0)
        return;

    if (d_rays == nullptr ||
        d_results == nullptr ||
        d_medium == nullptr)
    {
        throw std::invalid_argument(
            "Tracer::forward received a null device pointer"
        );
    }

    if (rayCount > static_cast<std::size_t>(UINT_MAX))
    {
        throw std::invalid_argument(
            "Tracer rayCount exceeds UINT_MAX"
        );
    }

    constexpr unsigned int BLOCK_SIZE = 256;

    const unsigned int blocks =
        static_cast<unsigned int>(
            (rayCount + BLOCK_SIZE - 1) / BLOCK_SIZE
        );

    traceRaysKernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
        d_rays,
        d_results,
        d_medium,
        config_.stepSize,
        static_cast<unsigned int>(config_.maxSteps),
        static_cast<unsigned int>(rayCount)
    );

    const cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess)
    {
        throw std::runtime_error(
            std::string("Tracer kernel launch error: ") +
            cudaGetErrorString(error)
        );
    }
}

void Tracer::forwardWithTrajectory(
    const Ray* d_rays,
    DeviceTraceResult* d_results,
    double3* d_trajectories,
    const Medium* d_medium,
    std::size_t rayCount,
    cudaStream_t stream
) const
{
    if (rayCount == 0)
        return;

    if (d_rays == nullptr ||
        d_results == nullptr ||
        d_trajectories == nullptr ||
        d_medium == nullptr)
    {
        throw std::invalid_argument(
            "Tracer::forwardWithTrajectory received a null device pointer"
        );
    }

    if (rayCount > static_cast<std::size_t>(UINT_MAX))
    {
        throw std::invalid_argument(
            "Tracer rayCount exceeds UINT_MAX"
        );
    }

    constexpr unsigned int BLOCK_SIZE = 256;

    const unsigned int blocks =
        static_cast<unsigned int>(
            (rayCount + BLOCK_SIZE - 1) / BLOCK_SIZE
        );

    traceRaysWithTrajectoryKernel<<<
        blocks,
        BLOCK_SIZE,
        0,
        stream
    >>>(
        d_rays,
        d_results,
        d_trajectories,
        d_medium,
        config_.stepSize,
        static_cast<unsigned int>(config_.maxSteps),
        static_cast<unsigned int>(rayCount)
    );

    const cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess)
    {
        throw std::runtime_error(
            std::string("Tracer trajectory kernel launch error: ") +
            cudaGetErrorString(error)
        );
    }
}
