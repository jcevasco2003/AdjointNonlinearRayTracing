#pragma once

#include "Simulation/Tracer.cuh"
#include "Simulation/OptimizableMedium.cuh"

#include <cstddef>

struct SolverConfig {
    double stepSize = 1e-3;
    std::size_t maxSteps = 100000;
};

class AdjointStateSolver {
public:
    explicit AdjointStateSolver(const SolverConfig& config);

    void backward(
        const DeviceTraceResult* d_forwardResults,
        const double3* d_deltaX,
        const double3* d_deltaV,
        double* d_parameterGradient,
        const OptimizableMedium* d_medium,
        std::size_t rayCount,
        cudaStream_t stream = nullptr
    ) const;

private:
    SolverConfig config_;
};