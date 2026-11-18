#pragma once

#include "Simulation/Ray.h"
#include "Simulation/Medium.h"

#include <vector>
#include <cstddef>
#include <functional>

struct TracerConfig {
    float stepSize = 0.001f;
    std::size_t maxSteps = 10000;
};

struct TraceResult {
    Ray finalState;
    std::size_t stepsTaken = 0;
    bool completed = false;
};

class Tracer {
public:
    using TraceCallback =
        std::function<void(const glm::dvec3&)>;

    explicit Tracer(const TracerConfig& config);

    TraceResult forward(
        const Ray& ray,
        const Medium& medium
    ) const;

    TraceResult forward(
        const Ray& ray,
        const Medium& medium,
        TraceCallback callback
    ) const;

private:
    TracerConfig config_;
};