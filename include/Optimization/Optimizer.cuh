#pragma once

#include <cuda_runtime.h>
#include <cstddef>

class Optimizer {
public:

    virtual ~Optimizer() = default;

    // Reserva los buffers internos necesarios en GPU.
    virtual void initialize(std::size_t parameterCount) = 0;

    // Actualiza los parámetros directamente en GPU.
    virtual void step(
        double* d_parameters,
        const double* d_gradient,
        std::size_t parameterCount,
        cudaStream_t stream = nullptr
    ) = 0;

    // Reinicia el estado interno del optimizador.
    virtual void reset() = 0;
};