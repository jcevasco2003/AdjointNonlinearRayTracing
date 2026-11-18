#include "Simulation/Tracer.h"

#include <stdexcept>

Tracer::Tracer(const TracerConfig& config)
    : config_(config)
{
    if (config_.stepSize <= 0.0)
        throw std::invalid_argument("Step size must be positive");

    if (config_.maxSteps == 0)
        throw std::invalid_argument("Max steps must be positive");
}


TraceResult Tracer::forward(
    const Ray& ray,
    const Medium& medium
) const
{
    return forward(ray, medium, nullptr);
}


TraceResult Tracer::forward(
    const Ray& ray,
    const Medium& medium,
    TraceCallback callback
) const
{
    glm::dvec3 x = ray.position;
    glm::dvec3 v = ray.direction;

    const double dt = config_.stepSize;

    std::size_t steps = 0;

    while (medium.isInside(x) && steps < config_.maxSteps)
    {
        const double n = medium.refractiveIndex(x);
        const glm::dvec3 gradient_n = medium.gradient(x);

        v += n * gradient_n * dt;
        x += v * dt;

        ++steps;

        if (callback)
            callback(x);
    }

    return {
        Ray(x, v),
        steps,
        !medium.isInside(x)
    };
}