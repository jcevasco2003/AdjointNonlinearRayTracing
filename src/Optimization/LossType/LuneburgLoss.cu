
#include "Optimization/LossType/LuneburgLoss.cuh"

#include <cuda_runtime.h>

#include <math_constants.h>
#include <cmath>
#include <climits>
#include <stdexcept>
#include <string>

namespace {

constexpr unsigned int BLOCK_SIZE = 256;
constexpr double DENOMINATOR_EPSILON = 1e-12;

// ============================================================
// CUDA error handling
// ============================================================

void checkCuda(cudaError_t error, const char* operation)
{
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string("LuneburgLoss CUDA error in ") +
            operation + ": " +
            cudaGetErrorString(error)
        );
    }
}

// ============================================================
// Vector helpers
// ============================================================

__device__ inline double dot3(
    const double3& a,
    const double3& b
) {
    return a.x * b.x +
           a.y * b.y +
           a.z * b.z;
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
// Batched Luneburg loss
//
// Computes:
//
//   target = center + radius * initialDirection
//
//   t = dot(normal, target - xf) / dot(normal, vf)
//
//   intersection = xf + t * vf
//
//   error = intersection - target
//
//   loss = dot(error, error)
//
// Terminal gradients:
//
//   dL/dxf =  2 * a
//   dL/dvf =  2 * t * a
//
// where:
//
//   a = error - normal * dot(vf,error)/dot(normal,vf)
//
// All gradients are scaled by 1/N, corresponding to
// the gradient of the mean loss.
// ============================================================

__global__ void luneburgLossKernel(
    const double3* initialDirections,
    const double3* finalPositions,
    const double3* finalDirections,
    std::size_t rayCount,
    double3 sphereCenter,
    double sphereRadius,
    double* losses,
    double3* gradientPositions,
    double3* gradientDirections,
    double* meanLoss
) {
    const std::size_t i =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= rayCount)
        return;

    const double invN =
        1.0 / static_cast<double>(rayCount);

    // --------------------------------------------------------
    // 1. Initial direction and target plane
    // --------------------------------------------------------

    const double3 normal = initialDirections[i];

    const double3 target = add3(
        sphereCenter,
        multiply3(normal, sphereRadius)
    );

    const double3 xf = finalPositions[i];
    const double3 vf = finalDirections[i];

    // --------------------------------------------------------
    // 2. Intersect final ray with target plane
    // --------------------------------------------------------

    const double denominator = dot3(normal, vf);

    // A ray parallel (or almost parallel) to the focal plane
    // has no numerically reliable intersection.
    //
    // Do not assign zero loss: that would reward an invalid ray.
    if (fabs(denominator) < DENOMINATOR_EPSILON) {

        losses[i] = CUDART_INF;

        gradientPositions[i] = make_double3(
            0.0, 0.0, 0.0
        );

        gradientDirections[i] = make_double3(
            0.0, 0.0, 0.0
        );

        atomicAdd(meanLoss, CUDART_INF);

        return;
    }

    const double numerator = dot3(
        normal,
        subtract3(target, xf)
    );

    const double t = numerator / denominator;

    const double3 intersection = add3(
        xf,
        multiply3(vf, t)
    );

    // --------------------------------------------------------
    // 3. Squared distance to focal point
    // --------------------------------------------------------

    const double3 error = subtract3(
        intersection,
        target
    );

    const double loss = dot3(error, error);

    losses[i] = loss;

    // --------------------------------------------------------
    // 4. Terminal gradients
    // --------------------------------------------------------

    const double projection =
        dot3(vf, error) / denominator;

    const double3 a = subtract3(
        error,
        multiply3(normal, projection)
    );

    // CPU-equivalent gradients, scaled by 1/N.
    //
    // IMPORTANT: positive signs, matching the CPU version.

    gradientPositions[i] = multiply3(
        a,
        2.0 * invN
    );

    gradientDirections[i] = multiply3(
        a,
        2.0 * t * invN
    );

    // --------------------------------------------------------
    // 5. Mean loss reduction
    // --------------------------------------------------------

    atomicAdd(meanLoss, loss * invN);
}

} // namespace


// ============================================================
// LuneburgLoss
// ============================================================

LuneburgLoss::LuneburgLoss(
    const double3& sphereCenter,
    double sphereRadius
)
    : sphereCenter_(sphereCenter),
      sphereRadius_(sphereRadius)
{
    if (!std::isfinite(sphereRadius_) ||
        sphereRadius_ <= 0.0) {

        throw std::invalid_argument(
            "LuneburgLoss: sphere radius must be positive and finite"
        );
    }

    if (!std::isfinite(sphereCenter_.x) ||
        !std::isfinite(sphereCenter_.y) ||
        !std::isfinite(sphereCenter_.z)) {

        throw std::invalid_argument(
            "LuneburgLoss: sphere center must be finite"
        );
    }
}


// ============================================================
// Evaluate
// ============================================================

void LuneburgLoss::evaluate(
    const double3* d_initialDirections,
    const double3* d_finalPositions,
    const double3* d_finalDirections,
    std::size_t rayCount,
    double* d_losses,
    double3* d_gradientPositions,
    double3* d_gradientDirections,
    double* d_meanLoss,
    cudaStream_t stream
) const {

    if (d_meanLoss == nullptr) {
        throw std::invalid_argument(
            "LuneburgLoss: null mean loss pointer"
        );
    }

    if (rayCount == 0) {
        throw std::invalid_argument(
            "LuneburgLoss: rayCount must be positive"
        );
    }

    if (d_initialDirections == nullptr ||
        d_finalPositions == nullptr ||
        d_finalDirections == nullptr ||
        d_losses == nullptr ||
        d_gradientPositions == nullptr ||
        d_gradientDirections == nullptr) {

        throw std::invalid_argument(
            "LuneburgLoss: null device pointer"
        );
    }

    // The kernel uses an unsigned grid index.
    const std::size_t maxRayCount =
        static_cast<std::size_t>(UINT_MAX) * BLOCK_SIZE;

    if (rayCount > maxRayCount) {
        throw std::invalid_argument(
            "LuneburgLoss: rayCount exceeds supported grid size"
        );
    }

    // --------------------------------------------------------
    // 1. Reset mean loss accumulator
    // --------------------------------------------------------

    checkCuda(
        cudaMemsetAsync(
            d_meanLoss,
            0,
            sizeof(double),
            stream
        ),
        "initialize mean loss"
    );

    // --------------------------------------------------------
    // 2. Launch batched loss kernel
    // --------------------------------------------------------

    const unsigned int blocks =
        static_cast<unsigned int>(
            (rayCount + BLOCK_SIZE - 1) / BLOCK_SIZE
        );

    luneburgLossKernel<<<
        blocks,
        BLOCK_SIZE,
        0,
        stream
    >>>(
        d_initialDirections,
        d_finalPositions,
        d_finalDirections,
        rayCount,
        sphereCenter_,
        sphereRadius_,
        d_losses,
        d_gradientPositions,
        d_gradientDirections,
        d_meanLoss
    );

    checkCuda(
        cudaGetLastError(),
        "launch Luneburg loss kernel"
    );
}
