
#pragma once

#include "Optimization/Optimizer.cuh"

#include <cuda_runtime.h>
#include <cstddef>

// ============================================================
// Adam
// ============================================================

struct AdamConfig {
    double learningRate = 1e-3;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double epsilon = 1e-8;
};

class Adam : public Optimizer {
public:

    explicit Adam(const AdamConfig& config = {});
    ~Adam() override;

    Adam(const Adam&) = delete;
    Adam& operator=(const Adam&) = delete;

    void initialize(std::size_t parameterCount) override;

    void step(
        double* d_parameters,
        const double* d_gradient,
        std::size_t parameterCount,
        cudaStream_t stream = nullptr
    ) override;

    void reset() override;

private:

    AdamConfig config_;

    // GPU state
    double* d_firstMoment_ = nullptr;
    double* d_secondMoment_ = nullptr;

    // Host state
    std::size_t parameterCount_ = 0;
    std::size_t timestep_ = 0;
};

// ============================================================
// Gradient Descent
// ============================================================

struct GradientDescentConfig {
    double learningRate = 1e-3;
};

class GradientDescent : public Optimizer {
public:

    explicit GradientDescent(
        const GradientDescentConfig& config = {}
    );

    void initialize(std::size_t parameterCount) override;

    void step(
        double* d_parameters,
        const double* d_gradient,
        std::size_t parameterCount,
        cudaStream_t stream = nullptr
    ) override;

    void reset() override;

private:

    GradientDescentConfig config_;
};

// ============================================================
// Optimization Result
// ============================================================

struct OptimizationResult {
    double loss = 0.0;
    std::size_t iterations = 0;
    bool converged = false;
};
