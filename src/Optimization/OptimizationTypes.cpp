
#include "Optimization/OptimizationTypes.h"

#include <cmath>
#include <stdexcept>

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

std::vector<double> Adam::step(
    const std::vector<double>& parameters,
    const std::vector<double>& gradient
)
{
    if (parameters.size() != gradient.size())
        throw std::invalid_argument(
            "Adam: parameters and gradient must have the same size"
        );

    if (timestep_ == 0) {
        firstMoment_.assign(parameters.size(), 0.0);
        secondMoment_.assign(parameters.size(), 0.0);
    }

    if (parameters.size() != firstMoment_.size())
        throw std::invalid_argument(
            "Adam: parameter dimension changed; call reset() first"
        );

    ++timestep_;

    const double beta1 = config_.beta1;
    const double beta2 = config_.beta2;

    const double correction1 =
        1.0 - std::pow(beta1, static_cast<double>(timestep_));

    const double correction2 =
        1.0 - std::pow(beta2, static_cast<double>(timestep_));

    std::vector<double> nextParameters(parameters.size());

    for (std::size_t i = 0; i < parameters.size(); ++i) {

        // First moment
        firstMoment_[i] =
            beta1 * firstMoment_[i]
            + (1.0 - beta1) * gradient[i];

        // Second moment
        secondMoment_[i] =
            beta2 * secondMoment_[i]
            + (1.0 - beta2) * gradient[i] * gradient[i];

        // Bias correction
        const double mHat = firstMoment_[i] / correction1;
        const double vHat = secondMoment_[i] / correction2;

        // Update parameters
        nextParameters[i] =
            parameters[i]
            - config_.learningRate * mHat
              / (std::sqrt(vHat) + config_.epsilon);
    }

    return nextParameters;
}

void Adam::reset()
{
    firstMoment_.clear();
    secondMoment_.clear();
    timestep_ = 0;
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

std::vector<double> GradientDescent::step(
    const std::vector<double>& parameters,
    const std::vector<double>& gradient
)
{
    if (parameters.size() != gradient.size())
        throw std::invalid_argument(
            "GradientDescent: parameters and gradient must have the same size"
        );

    std::vector<double> nextParameters(parameters.size());

    for (std::size_t i = 0; i < parameters.size(); ++i) {
        nextParameters[i] =
            parameters[i] - config_.learningRate * gradient[i];
    }

    return nextParameters;
}
