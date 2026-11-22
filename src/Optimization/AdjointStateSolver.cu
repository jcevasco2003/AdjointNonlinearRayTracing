
#include "Optimization/AdjointStateSolver.cuh"

#include <cuda_runtime.h>

#include <climits>
#include <cmath>
#include <stdexcept>
#include <string>

namespace {

constexpr unsigned int BLOCK_SIZE = 256;

// ============================================================
// CUDA error handling
// ============================================================

void checkCuda(
    cudaError_t error,
    const char* operation
) {
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " +
            cudaGetErrorString(error)
        );
    }
}

// ============================================================
// Vector helpers
// ============================================================

__device__ inline double3 add3(
    const double3& a,
    const double3& b
) {
    return make_double3(
        a.x + b.x,
        a.y + b.y,
        a.z + b.z
    );
}

__device__ inline double3 subtract3(
    const double3& a,
    const double3& b
) {
    return make_double3(
        a.x - b.x,
        a.y - b.y,
        a.z - b.z
    );
}

__device__ inline double3 multiply3(
    const double3& a,
    double scalar
) {
    return make_double3(
        a.x * scalar,
        a.y * scalar,
        a.z * scalar
    );
}

// ============================================================
// Adjoint backward kernel
//
// Algorithm 2: Backward Tracing
// ============================================================

__global__ void adjointBackwardKernel(
    const DeviceTraceResult* forwardResults,
    const double3* deltaX,
    const double3* deltaV,
    double* parameterGradient,
    const OptimizableMedium* medium,
    double dt,
    std::size_t rayCount
) {
    const std::size_t rayId =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (rayId >= rayCount)
        return;

    const DeviceTraceResult forward =
        forwardResults[rayId];

    if (forward.stepsTaken == 0 || !forward.completed)
        return;

    // --------------------------------------------------------
    // Initialization
    //
    // x <- xf
    // v <- vf
    // mu <- deltaV
    // lambda <- deltaX + deltaV * dt
    // --------------------------------------------------------

    double3 x = forward.finalPosition;
    double3 v = forward.finalDirection;

    double3 mu = deltaV[rayId];

    double3 lambda = add3(
        deltaX[rayId],
        multiply3(deltaV[rayId], dt)
    );

    double H[9];

    // --------------------------------------------------------
    // Backward tracing
    // --------------------------------------------------------

    for (std::size_t step = 0;
         step < forward.stepsTaken;
         ++step)
    {
        // ----------------------------------------------------
        // 1. x <- x - v * dt
        // ----------------------------------------------------

        x = subtract3(
            x,
            multiply3(v, dt)
        );

        // ----------------------------------------------------
        // 2. eta, gradEta <- interpolate(eta, x)
        // ----------------------------------------------------

        const double eta =
            medium->refractiveIndex(x);

        const double3 gradEta =
            medium->gradient(x);

        // ----------------------------------------------------
        // 3. v <- v - eta * gradEta * dt
        // ----------------------------------------------------

        v = subtract3(
            v,
            multiply3(gradEta, eta * dt)
        );

        // ----------------------------------------------------
        // 4. H <- Hessian(eta, x)
        // ----------------------------------------------------

        medium->hessian(x, H);

        // ----------------------------------------------------
        // 5. A = gradEta * gradEta^T + eta * H
        //
        // lambda <- lambda + A * mu * dt
        // ----------------------------------------------------

        const double A[9] = {
            gradEta.x * gradEta.x + eta * H[0],
            gradEta.x * gradEta.y + eta * H[1],
            gradEta.x * gradEta.z + eta * H[2],

            gradEta.y * gradEta.x + eta * H[3],
            gradEta.y * gradEta.y + eta * H[4],
            gradEta.y * gradEta.z + eta * H[5],

            gradEta.z * gradEta.x + eta * H[6],
            gradEta.z * gradEta.y + eta * H[7],
            gradEta.z * gradEta.z + eta * H[8]
        };

        const double3 A_mu = make_double3(
            A[0] * mu.x + A[1] * mu.y + A[2] * mu.z,
            A[3] * mu.x + A[4] * mu.y + A[5] * mu.z,
            A[6] * mu.x + A[7] * mu.y + A[8] * mu.z
        );

        lambda = add3(
            lambda,
            multiply3(A_mu, dt)
        );

        // ----------------------------------------------------
        // 6. mu <- mu + lambda * dt
        // ----------------------------------------------------

        mu = add3(
            mu,
            multiply3(lambda, dt)
        );

        // ----------------------------------------------------
        // 7. Accumulate parameter gradient
        //
        // deltaTheta +=
        //
        //   (mu . gradEta) * dN/dTheta
        //
        //   + eta * (mu . grad(dN/dTheta))
        //
        // The medium implementation handles the dt factor.
        // ----------------------------------------------------

        medium->accumulateParameterGradient(
            x,
            eta,
            gradEta,
            mu,
            dt,
            parameterGradient
        );
    }
}

} // namespace


// ============================================================
// AdjointStateSolver
// ============================================================

AdjointStateSolver::AdjointStateSolver(
    const SolverConfig& config
)
    : config_(config)
{
    if (!std::isfinite(config_.stepSize) ||
        config_.stepSize <= 0.0) {

        throw std::invalid_argument(
            "AdjointStateSolver: stepSize must be positive and finite."
        );
    }

    if (config_.maxSteps == 0) {
        throw std::invalid_argument(
            "AdjointStateSolver: maxSteps must be positive."
        );
    }
}


// ============================================================
// Backward
// ============================================================

void AdjointStateSolver::backward(
    const DeviceTraceResult* d_forwardResults,
    const double3* d_deltaX,
    const double3* d_deltaV,
    double* d_parameterGradient,
    const OptimizableMedium* d_medium,
    std::size_t rayCount,
    cudaStream_t stream
) const {

    if (rayCount == 0)
        return;

    if (
        d_forwardResults == nullptr ||
        d_deltaX == nullptr ||
        d_deltaV == nullptr ||
        d_parameterGradient == nullptr ||
        d_medium == nullptr
    ) {
        throw std::invalid_argument(
            "AdjointStateSolver::backward: null device pointer."
        );
    }

    const std::size_t maxRayCount =
        static_cast<std::size_t>(UINT_MAX) * BLOCK_SIZE;

    if (rayCount > maxRayCount) {
        throw std::invalid_argument(
            "AdjointStateSolver::backward: rayCount exceeds supported grid size."
        );
    }

    const unsigned int gridSize =
        static_cast<unsigned int>(
            (rayCount + BLOCK_SIZE - 1) / BLOCK_SIZE
        );

    adjointBackwardKernel<<<
        gridSize,
        BLOCK_SIZE,
        0,
        stream
    >>>(
        d_forwardResults,
        d_deltaX,
        d_deltaV,
        d_parameterGradient,
        d_medium,
        config_.stepSize,
        rayCount
    );

    checkCuda(
        cudaGetLastError(),
        "Failed to launch adjointBackwardKernel"
    );
}
