#pragma once

#include "Simulation/Ray.h"

struct Intersection {
    bool hit = false;

    double distance = 0.0;

    glm::dvec3 position{0.0};
    glm::dvec3 normal{0.0};
};

class Intersectable {
public:
    virtual ~Intersectable() = default;

    virtual Intersection intersect(
        const Ray& ray
    ) const = 0;
};