#pragma once

#include "Simulation/Medium.h"
#include "Optimization/LossFunction.h"

#include <vector>

struct SolverConfig {
    double stepSize = 1e-3;
    std::size_t maxSteps = 100000;
};

struct ForwardResult {
    glm::dvec3 finalPosition;
    glm::dvec3 finalVelocity;
    std::size_t steps = 0;
};

struct AdjointResult {
    ForwardResult forward;
    std::vector<double> parameterGradient;
};

class AdjointStateSolver {
public:
    explicit AdjointStateSolver(const SolverConfig& config)
        : config_(config) {}

    std::vector<double> backward(
        const ForwardResult& forwardResult,
        const glm::dvec3& deltaX,
        const glm::dvec3& deltaV,
        const Medium& medium
    ) const;

private:
    SolverConfig config_;
};