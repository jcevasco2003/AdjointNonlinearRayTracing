#pragma once

#include <cuda_runtime.h>
#include <cstddef>

class LuneburgLoss {
public:
    LuneburgLoss(
        const double3& sphereCenter,
        double sphereRadius
    );

    void evaluate(
        const double3* d_initialDirections,
        const double3* d_finalPositions,
        const double3* d_finalDirections,
        std::size_t rayCount,
        double* d_losses,
        double3* d_gradientPositions,
        double3* d_gradientDirections,
        double* d_meanLoss,
        cudaStream_t stream = nullptr
    ) const;

private:
    double3 sphereCenter_;
    double sphereRadius_;
};