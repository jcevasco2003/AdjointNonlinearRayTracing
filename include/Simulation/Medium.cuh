#pragma once

#include <cuda_runtime.h>

class Medium {
public:
    virtual ~Medium() = default;

    // Geometric domain
    __device__ virtual bool isInside(
        const double3& x
    ) const = 0;

    // Refractive index
    __device__ virtual double refractiveIndex(
        const double3& x
    ) const = 0;

    // Spatial derivatives
    __device__ virtual double3 gradient(
        const double3& x
    ) const = 0;

    __device__ virtual void hessian(
        const double3& x,
        double* H
    ) const = 0;
};