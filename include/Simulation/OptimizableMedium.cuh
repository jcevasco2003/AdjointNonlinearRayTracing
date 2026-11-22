#pragma once

#include "Simulation/Medium.cuh"

#include <cstddef>

class OptimizableMedium : public Medium {
public:


    virtual std::size_t parameterCountHost() const = 0;

    virtual double* deviceParameters() = 0;


    // Number of optimizable parameters
    __device__ virtual std::size_t parameterCount() const = 0;

    // Derivative of refractive index with respect to parameters
    __device__ virtual double indexParameterGradient(
        const double3& x,
        std::size_t parameterIndex
    ) const = 0;

    // Derivative of spatial gradient with respect to parameters
    __device__ virtual double3 gradientParameterGradient(
        const double3& x,
        std::size_t parameterIndex
    ) const = 0;

    // Accumulate the parameter gradient contribution
    __device__ virtual void accumulateParameterGradient(
        const double3& x,
        double eta,
        const double3& gradEta,
        const double3& mu,
        double dt,
        double* parameterGradient
    ) const = 0;
};