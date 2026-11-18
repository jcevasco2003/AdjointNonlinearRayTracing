
#pragma once

#include <vector>

struct OptimizerConfig {
    int maxIterations = 1000;
    double learningRate = 1e-3;
    double tolerance = 1e-8;
};

struct OptimizationResult {
    std::vector<double> parameters;
    double loss = 0.0;
    int iterations = 0;
    bool converged = false;
};

class Optimizer {
public:
    virtual ~Optimizer() = default;

    virtual std::vector<double> step(
        const std::vector<double>& parameters,
        const std::vector<double>& gradient
    ) = 0;
};
