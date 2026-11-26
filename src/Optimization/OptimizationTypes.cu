#include "Optimization/OptimizationTypes.cuh"

#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>
#include <string>

namespace {

void checkCuda(cudaError_t error, const char* operation)
{
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA error in ") + operation +
            ": " + cudaGetErrorString(error)
        );
    }
}

// ------------------------------------------------------------
// Adam update kernel
// ------------------------------------------------------------

__global__ void adamUpdateKernel(
    double* parameters,
    const double* gradient,
    double* firstMoment,
    double* secondMoment,
    std::size_t count,
    double learningRate,
    double beta1,
    double beta2,
    double epsilon,
    double correction1,
    double correction2
) {
    const std::size_t i =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= count)
        return;

    const double g = gradient[i];

    // First moment
    firstMoment[i] =
        beta1 * firstMoment[i]
        + (1.0 - beta1) * g;

    // Second moment
    secondMoment[i] =
        beta2 * secondMoment[i]
        + (1.0 - beta2) * g * g;

    // Bias correction
    const double mHat =
        firstMoment[i] / correction1;

    const double vHat =
        secondMoment[i] / correction2;

    // Parameter update
    parameters[i] -=
        learningRate * mHat
        / (sqrt(vHat) + epsilon);
}

// ------------------------------------------------------------
// Gradient descent update kernel
// ------------------------------------------------------------

__global__ void gradientDescentKernel(
    double* parameters,
    const double* gradient,
    std::size_t count,
    double learningRate
) {
    const std::size_t i =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= count)
        return;

    parameters[i] -= learningRate * gradient[i];
}


} // namespace


// ============================================================
// Adam
// ============================================================

Adam::Adam(const AdamConfig& config)
    : config_(config)
{
    if (config_.learningRate <= 0.0)
        throw std::invalid_argument(
            "Adam: learning rate must be positive"
        );

    if (config_.beta1 < 0.0 || config_.beta1 >= 1.0 ||
        config_.beta2 < 0.0 || config_.beta2 >= 1.0)
        throw std::invalid_argument(
            "Adam: beta values must be in [0, 1)"
        );

    if (config_.epsilon <= 0.0)
        throw std::invalid_argument(
            "Adam: epsilon must be positive"
        );
}

Adam::~Adam()
{
    if (d_firstMoment_)
        cudaFree(d_firstMoment_);

    if (d_secondMoment_)
        cudaFree(d_secondMoment_);
}

void Adam::initialize(std::size_t parameterCount)
{
    if (parameterCount == 0)
        throw std::invalid_argument(
            "Adam: parameter count must be positive"
        );

    if (parameterCount == parameterCount_ &&
        d_firstMoment_ && d_secondMoment_) {
        return;
    }

    reset();

    checkCuda(
        cudaMalloc(
            &d_firstMoment_,
            parameterCount * sizeof(double)
        ),
        "allocate first moment"
    );

    try {
        checkCuda(
            cudaMalloc(
                &d_secondMoment_,
                parameterCount * sizeof(double)
            ),
            "allocate second moment"
        );
    }
    catch (...) {
        cudaFree(d_firstMoment_);
        d_firstMoment_ = nullptr;
        throw;
    }

    checkCuda(
        cudaMemset(
            d_firstMoment_,
            0,
            parameterCount * sizeof(double)
        ),
        "initialize first moment"
    );

    checkCuda(
        cudaMemset(
            d_secondMoment_,
            0,
            parameterCount * sizeof(double)
        ),
        "initialize second moment"
    );

    parameterCount_ = parameterCount;
    timestep_ = 0;
}

void Adam::step(
    double* d_parameters,
    const double* d_gradient,
    std::size_t parameterCount,
    cudaStream_t stream
) {
    if (!d_parameters || !d_gradient)
        throw std::invalid_argument(
            "Adam: null device pointer"
        );

    if (parameterCount != parameterCount_)
        throw std::invalid_argument(
            "Adam: parameter dimension mismatch; initialize first"
        );

    ++timestep_;

    const double correction1 =
        1.0 - std::pow(
            config_.beta1,
            static_cast<double>(timestep_)
        );

    const double correction2 =
        1.0 - std::pow(
            config_.beta2,
            static_cast<double>(timestep_)
        );

    constexpr int BLOCK_SIZE = 256;

    const int blocks =
        static_cast<int>(
            (parameterCount + BLOCK_SIZE - 1) / BLOCK_SIZE
        );

    adamUpdateKernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
        d_parameters,
        d_gradient,
        d_firstMoment_,
        d_secondMoment_,
        parameterCount,
        config_.learningRate,
        config_.beta1,
        config_.beta2,
        config_.epsilon,
        correction1,
        correction2
    );

    checkCuda(
        cudaGetLastError(),
        "launch Adam update kernel"
    );
}

void Adam::reset()
{
    if (d_firstMoment_) {
        cudaFree(d_firstMoment_);
        d_firstMoment_ = nullptr;
    }

    if (d_secondMoment_) {
        cudaFree(d_secondMoment_);
        d_secondMoment_ = nullptr;
    }

    parameterCount_ = 0;
    timestep_ = 0;
}

void Adam::setLearningRate(double learningRate)
{
    if (!std::isfinite(learningRate) || learningRate <= 0.0) {
        throw std::invalid_argument(
            "Adam: learning rate must be positive"
        );
    }

    config_.learningRate = learningRate;
}



// ============================================================
// Gradient Descent
// ============================================================

GradientDescent::GradientDescent(
    const GradientDescentConfig& config
)
    : config_(config)
{
    if (config_.learningRate <= 0.0)
        throw std::invalid_argument(
            "GradientDescent: learning rate must be positive"
        );
}

void GradientDescent::initialize(std::size_t parameterCount)
{
    if (parameterCount == 0)
        throw std::invalid_argument(
            "GradientDescent: parameter count must be positive"
        );
}

void GradientDescent::step(
    double* d_parameters,
    const double* d_gradient,
    std::size_t parameterCount,
    cudaStream_t stream
) {
    if (!d_parameters || !d_gradient)
        throw std::invalid_argument(
            "GradientDescent: null device pointer"
        );

    if (parameterCount == 0)
        return;

    constexpr int BLOCK_SIZE = 256;

    const int blocks =
        static_cast<int>(
            (parameterCount + BLOCK_SIZE - 1) / BLOCK_SIZE
        );

    gradientDescentKernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
        d_parameters,
        d_gradient,
        parameterCount,
        config_.learningRate
    );

    checkCuda(
        cudaGetLastError(),
        "launch gradient descent kernel"
    );
}

void GradientDescent::reset()
{
    // No internal state.
}