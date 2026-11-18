#pragma once

#include "Optimization/Optimizer.h"

#include <vector>
#include <cstddef>

// ----------------------------------------------------------------------------------------------------------

struct AdamConfig {
    double learningRate = 1e-3;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double epsilon = 1e-8;
};

class Adam : public Optimizer {
public:
    explicit Adam(const AdamConfig& config = {});

    std::vector<double> step(
        const std::vector<double>& parameters,
        const std::vector<double>& gradient
    ) override;

    void reset();

private:
    AdamConfig config_;
    std::vector<double> firstMoment_;
    std::vector<double> secondMoment_;
    std::size_t timestep_ = 0;
};

// ----------------------------------------------------------------------------------------------------------

struct GradientDescentConfig {
    double learningRate = 1e-3;
};

class GradientDescent : public Optimizer {
public:
    explicit GradientDescent(
        const GradientDescentConfig& config = {}
    );

    std::vector<double> step(
        const std::vector<double>& parameters,
        const std::vector<double>& gradient
    ) override;

private:
    GradientDescentConfig config_;
};